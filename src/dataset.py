"""Dataset Factory"""

import os
from typing import Optional

from mindspore.dataset import ImageFolderDataset


def create_dataset(
        name,
        root,
        split: str = 'train',
        shuffle: Optional[bool] = None,
        sampler=None,
        num_shards: Optional[int] = None,
        shard_id: Optional[int] = None,
        num_parallel_workers: Optional[int] = None,
        download: bool = False,
        **kwargs
):
    name = name.lower()
    mindspore_kwargs = dict(shuffle=shuffle, sampler=sampler, num_shards=num_shards, shard_id=shard_id,
                            num_parallel_workers=num_parallel_workers, **kwargs)

    if name == 'image_folder' or name == 'folder' or name == 'imagenet':
        if download:
            raise ValueError("Dataset download is not supported.")
        if os.path.isdir(root):
            root = os.path.join(root, split)
        dataset = ImageFolderDataset(dataset_dir=root, **mindspore_kwargs)
    else:
        assert False, "Unknown dataset"

    return dataset
