
"""Dataset Loader"""

from .transforms import transforms_imagenet_train, transforms_imagenet_eval

import mindspore as ms
import mindspore.dataset.vision as vision
import mindspore.dataset.transforms as transforms


def create_loader(
        dataset,
        batch_size,
        drop_remainder=False,
        is_training=False,
        mixup=0.0,
        num_classes=1000,
        transform=None,
        num_parallel_workers=None,
        python_multiprocessing=False,
        **kwargs
):
    if transform is None:
        if is_training:
            transform = transforms_imagenet_train(**kwargs)
        else:
            transform = transforms_imagenet_eval(**kwargs)
    else:
        transform = transform

    dataset = dataset.map(operations=transform,
                          input_columns='image',
                          num_parallel_workers=num_parallel_workers,
                          python_multiprocessing=python_multiprocessing)

    target_transform = transforms.TypeCast(ms.int32)
    dataset = dataset.map(operations=target_transform,
                          input_columns='label',
                          num_parallel_workers=num_parallel_workers,
                          python_multiprocessing=python_multiprocessing)
    if is_training and mixup > 0:
        one_hot_encode = transforms.OneHot(num_classes)
        dataset = dataset.map(operations=one_hot_encode, input_columns=["label"])

    dataset = dataset.batch(batch_size=batch_size, drop_remainder=drop_remainder)

    if is_training and mixup > 0:
        trans_mixup = vision.MixUpBatch(alpha=mixup)
        dataset = dataset.map(input_columns=["image", "label"], num_parallel_workers=num_parallel_workers,
                              operations=trans_mixup)

    return dataset
