"""Transforms Factory"""

import math

import mindspore.dataset.vision as vision
from mindspore.dataset.vision import Inter
import mindspore.dataset.transforms as c_transforms

# define Auto Augmentation operators
PARAMETER_MAX = 10
DEFAULT_CROP_PCT = 0.875
IMAGENET_DEFAULT_MEAN = [0.485 * 255, 0.456 * 255, 0.406 * 255]
IMAGENET_DEFAULT_STD = [0.229 * 255, 0.224 * 255, 0.225 * 255]


def float_parameter(level, maxval):
    return float(level) * maxval / PARAMETER_MAX


def int_parameter(level, maxval):
    return int(level * maxval / PARAMETER_MAX)


def shear_x(level):
    transforms_list = []
    v = float_parameter(level, 0.3)

    transforms_list.append(vision.RandomAffine(degrees=0, shear=(-v, -v)))
    transforms_list.append(vision.RandomAffine(degrees=0, shear=(v, v)))
    return c_transforms.RandomChoice(transforms_list)


def shear_y(level):
    transforms_list = []
    v = float_parameter(level, 0.3)

    transforms_list.append(vision.RandomAffine(degrees=0, shear=(0, 0, -v, -v)))
    transforms_list.append(vision.RandomAffine(degrees=0, shear=(0, 0, v, v)))
    return c_transforms.RandomChoice(transforms_list)


def translate_x(level):
    transforms_list = []
    v = float_parameter(level, 150 / 331)

    transforms_list.append(vision.RandomAffine(degrees=0, translate=(-v, -v)))
    transforms_list.append(vision.RandomAffine(degrees=0, translate=(v, v)))
    return c_transforms.RandomChoice(transforms_list)


def translate_y(level):
    transforms_list = []
    v = float_parameter(level, 150 / 331)

    transforms_list.append(vision.RandomAffine(degrees=0, translate=(0, 0, -v, -v)))
    transforms_list.append(vision.RandomAffine(degrees=0, translate=(0, 0, v, v)))
    return c_transforms.RandomChoice(transforms_list)


def color_impl(level):
    v = float_parameter(level, 1.8) + 0.1
    return vision.RandomColor(degrees=(v, v))


def rotate_impl(level):
    transforms_list = []
    v = int_parameter(level, 30)

    transforms_list.append(vision.RandomRotation(degrees=(-v, -v)))
    transforms_list.append(vision.RandomRotation(degrees=(v, v)))
    return c_transforms.RandomChoice(transforms_list)


def solarize_impl(level):
    level = int_parameter(level, 256)
    v = 256 - level
    return vision.RandomSolarize(threshold=(0, v))


def posterize_impl(level):
    level = int_parameter(level, 4)
    v = 4 - level
    return vision.RandomPosterize(bits=(v, v))


def contrast_impl(level):
    v = float_parameter(level, 1.8) + 0.1
    return vision.RandomColorAdjust(contrast=(v, v))


def autocontrast_impl(level):
    return vision.AutoContrast()


def sharpness_impl(level):
    v = float_parameter(level, 1.8) + 0.1
    return vision.RandomSharpness(degrees=(v, v))


def brightness_impl(level):
    v = float_parameter(level, 1.8) + 0.1
    return vision.RandomColorAdjust(brightness=(v, v))


# define the Auto Augmentation policy
imagenet_policy = [
    [(posterize_impl(8), 0.4), (rotate_impl(9), 0.6)],
    [(solarize_impl(5), 0.6), (autocontrast_impl(5), 0.6)],
    [(vision.Equalize(), 0.8), (vision.Equalize(), 0.6)],
    [(posterize_impl(7), 0.6), (posterize_impl(6), 0.6)],

    [(vision.Equalize(), 0.4), (solarize_impl(4), 0.2)],
    [(vision.Equalize(), 0.4), (rotate_impl(8), 0.8)],
    [(solarize_impl(3), 0.6), (vision.Equalize(), 0.6)],
    [(posterize_impl(5), 0.8), (vision.Equalize(), 1.0)],
    [(rotate_impl(3), 0.2), (solarize_impl(8), 0.6)],
    [(vision.Equalize(), 0.6), (posterize_impl(6), 0.4)],

    [(rotate_impl(8), 0.8), (color_impl(0), 0.4)],
    [(rotate_impl(9), 0.4), (vision.Equalize(), 0.6)],
    [(vision.Equalize(), 0.0), (vision.Equalize(), 0.8)],
    [(vision.Invert(), 0.6), (vision.Equalize(), 1.0)],
    [(color_impl(4), 0.6), (contrast_impl(8), 1.0)],

    [(rotate_impl(8), 0.8), (color_impl(2), 1.0)],
    [(color_impl(8), 0.8), (solarize_impl(7), 0.8)],
    [(sharpness_impl(7), 0.4), (vision.Invert(), 0.6)],
    [(shear_x(5), 0.6), (vision.Equalize(), 1.0)],
    [(color_impl(0), 0.4), (vision.Equalize(), 0.6)],

    [(vision.Equalize(), 0.4), (solarize_impl(4), 0.2)],
    [(solarize_impl(5), 0.6), (autocontrast_impl(5), 0.6)],
    [(vision.Invert(), 0.6), (vision.Equalize(), 1.0)],
    [(color_impl(4), 0.6), (contrast_impl(8), 1.0)],
    [(vision.Equalize(), 0.8), (vision.Equalize(), 0.6)],
]


def transforms_imagenet_train(
        image_resize=224,
        scale=(0.08, 1.0),
        ratio=(0.75, 1.333),
        hflip=0.5,
        vflip=0.0,
        color_jitter=None,
        auto_augment=False,
        interpolation='bilinear',
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        re_prob=0.,
        re_scale=(0.02, 0.33),
        re_ratio=(0.3, 3.3),
        re_value=0,
        re_max_attempts=10
):
    # Define map operations for training dataset
    if hasattr(Inter, interpolation.upper()):
        interpolation = getattr(Inter, interpolation.upper())
    else:
        interpolation = Inter.BILINEAR

    trans_list = [vision.RandomCropDecodeResize(size=image_resize,
                                                scale=scale,
                                                ratio=ratio,
                                                interpolation=interpolation
                                                )]
    if hflip > 0.:
        trans_list += [vision.RandomHorizontalFlip(prob=hflip)]
    if vflip > 0.:
        trans_list += [vision.RandomVerticalFlip(prob=vflip)]

    if auto_augment:
        trans_list += [vision.RandomSelectSubpolicy(imagenet_policy)]
    elif color_jitter is not None:
        if isinstance(color_jitter, (list, tuple)):
            # color jitter shoulf be a 3-tuple/list for brightness/contrast/saturation
            # or 4 if also augmenting hue
            assert len(color_jitter) in (3, 4)
        else:
            color_jitter = (float(color_jitter),) * 3
        trans_list += [vision.RandomColorAdjust(*color_jitter)]

    trans_list += [
        vision.Normalize(mean=mean, std=std),
        vision.HWC2CHW()
    ]
    if re_prob > 0.:
        trans_list.append(
            vision.RandomErasing(prob=re_prob,
                                 scale=re_scale,
                                 ratio=re_ratio,
                                 value=re_value,
                                 max_attempts=re_max_attempts)
        )

    return trans_list


def transforms_imagenet_eval(
        image_resize=224,
        crop_pct=DEFAULT_CROP_PCT,
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        interpolation='bilinear',

):
    if isinstance(image_resize, (tuple, list)):
        assert len(image_resize) == 2
        if image_resize[-1] == image_resize[-2]:
            scale_size = int(math.floor(image_resize[0] / crop_pct))
        else:
            scale_size = tuple([int(x / crop_pct) for x in image_resize])
    else:
        scale_size = int(math.floor(image_resize / crop_pct))

    # Define map operations for training dataset
    if hasattr(Inter, interpolation.upper()):
        interpolation = getattr(Inter, interpolation.upper())
    else:
        interpolation = Inter.BILINEAR
    trans_list = [
        vision.Decode(),
        vision.Resize(scale_size, interpolation=interpolation),
        vision.CenterCrop(image_resize),
        vision.Normalize(mean=mean, std=std),
        vision.HWC2CHW()

    ]

    return trans_list
