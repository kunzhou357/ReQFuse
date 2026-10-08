"""Strict paired image indexing, synchronized transforms and resumable batches."""
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, Sampler

from .degradation import LocalDegrader, randint, uniform
from .image import read_image


EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def image_index(directory):
    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {root}")
    index = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in EXTENSIONS:
            key = path.relative_to(root).with_suffix("").as_posix()
            if key in index:
                raise ValueError(f"Duplicate sample ID '{key}' in {root}.")
            index[key] = path
    if not index:
        raise ValueError(f"No supported images in {root}.")
    return index


class PairedFusionDataset(Dataset):
    def __init__(self, visible_dir, infrared_dir, mode="synthetic", patch_size=None,
                 training=False, seed=100, degradation=None, hq_visible_dir=None,
                 hq_infrared_dir=None, mask_visible_dir=None, mask_infrared_dir=None):
        if mode not in ("synthetic","paired","inference"):
            raise ValueError("Dataset mode must be synthetic, paired or inference.")
        if patch_size is not None and (not isinstance(patch_size,int) or patch_size < 1):
            raise ValueError("patch_size must be a positive integer or null.")
        if (hq_visible_dir is None) != (hq_infrared_dir is None):
            raise ValueError("Supply both HQ source directories together.")
        if mode == "paired" and hq_visible_dir is None:
            raise ValueError("Paired training requires HQ visible and infrared references.")
        if mode == "synthetic" and (hq_visible_dir or mask_visible_dir or mask_infrared_dir):
            raise ValueError("Synthetic mode reads HQ directly from visible_dir/infrared_dir; external HQ/masks are unnecessary.")
        self.mode,self.training,self.patch_size,self.seed = mode,training,patch_size,int(seed)
        self.degrader = LocalDegrader(degradation)
        self.context_margin = self.degrader.config["context_margin"] if mode == "synthetic" and patch_size else 0
        self.indices = {"visible":image_index(visible_dir),"infrared":image_index(infrared_dir)}
        for key,path in (("hq_visible",hq_visible_dir),("hq_infrared",hq_infrared_dir),
                         ("mask_visible",mask_visible_dir),("mask_infrared",mask_infrared_dir)):
            if path:
                self.indices[key] = image_index(path)
        keys = set(self.indices["visible"])
        for name,index in self.indices.items():
            if set(index) != keys:
                missing,extra = sorted(keys-set(index)),sorted(set(index)-keys)
                raise ValueError(f"Unpaired {name}: missing={missing[:5]}, extra={extra[:5]} (sample IDs must match).")
        self.sample_ids = sorted(keys)

    def __len__(self):
        return len(self.sample_ids)

    def _transform(self, images, generator):
        shapes = {tuple(x.shape[-2:]) for x in images.values()}
        if len(shapes) != 1:
            raise ValueError(f"Paired sample dimensions differ: {shapes}; images must be registered.")
        h,w = next(iter(shapes))
        if self.patch_size:
            margin = self.context_margin
            if margin:
                images = {k:F.pad(x, (margin,)*4, mode="replicate") for k,x in images.items()}
                h, w = h+2*margin, w+2*margin
            size = self.patch_size+2*margin
            ph,pw = max(0,size-h),max(0,size-w)
            images = {k:F.pad(x,(0,pw,0,ph),mode="replicate") for k,x in images.items()}
            h,w = h+ph,w+pw
            top,left = (randint(generator,0,h-size+1),randint(generator,0,w-size+1)) if self.training else ((h-size)//2,(w-size)//2)
            images = {k:x[:,top:top+size,left:left+size] for k,x in images.items()}
        if self.training:
            for dim in (-1,-2):
                if uniform(generator) < .5:
                    images = {k:x.flip(dim) for k,x in images.items()}
            rotations = randint(generator,0,4)
            images = {k:torch.rot90(x,rotations,(-2,-1)) for k,x in images.items()}
        return images

    def __getitem__(self, item):
        index,draw = item if isinstance(item,tuple) else (int(item),int(item))
        sample_id = self.sample_ids[index]
        # Each draw owns its RNG; worker count and prefetch do not alter augmentation.
        generator = torch.Generator().manual_seed((self.seed+draw*1000003+index*9176) % (2**63-1))
        images = {name:read_image(paths[sample_id],3 if name.endswith("visible") and not name.startswith("mask_") else 1)
                  for name,paths in self.indices.items()}
        images = self._transform(images,generator)
        if self.mode == "synthetic":
            images["hq_visible"],images["hq_infrared"] = images["visible"].clone(),images["infrared"].clone()
            vi,ir,mv,mi = self.degrader(images["visible"],images["infrared"],generator)
            images.update(visible=vi,infrared=ir,mask_visible=mv,mask_infrared=mi)
            if self.context_margin:
                m, size = self.context_margin, self.patch_size
                images = {k:x[:, m:m+size, m:m+size] for k,x in images.items()}
        elif self.mode == "paired":
            # No region annotations means no claim about an intact region.
            for name in ("mask_visible","mask_infrared"):
                images.setdefault(name,torch.ones(1,*images["visible"].shape[-2:]))
        return dict(images,sample_id=sample_id)


class EpochBatchSampler(Sampler):
    """One shuffled pass, retaining the tail; each sample owns an epoch RNG token."""
    def __init__(self, dataset_size, batch_size, epoch=0, seed=100):
        if dataset_size < 1 or batch_size < 1 or epoch < 0:
            raise ValueError("Invalid epoch sampler sizes/index.")
        self.size, self.batch_size, self.epoch, self.seed = dataset_size, batch_size, epoch, seed

    def __len__(self):
        return (self.size+self.batch_size-1)//self.batch_size

    def __iter__(self):
        order = torch.randperm(self.size, generator=torch.Generator().manual_seed(self.seed+self.epoch)).tolist()
        for start in range(0, self.size, self.batch_size):
            yield [(index, self.epoch*self.size+offset)
                   for offset,index in enumerate(order[start:start+self.batch_size], start)]
