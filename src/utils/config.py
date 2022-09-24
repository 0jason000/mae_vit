
"""config"""

import os
import yaml
import logging
import argparse

logger = logging.getLogger(__name__)


def create_parser():
    # The first arg parser parses out only the --config argument, this argument is used to
    # load a yaml file containing key-values that override the defaults for the main parser below
    parser_config = argparse.ArgumentParser(description='Training Config', add_help=False)
    parser_config.add_argument('-c', '--config', type=str, default='',
                               help='YAML config file specifying default arguments (default='')')

    # The main parser. It inherits the --config argument for better help information.
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training', parents=[parser_config])

    # System parameters
    group = parser.add_argument_group('System parameters')
    group.add_argument('--mode', type=int, default=0,
                       help='Running in GRAPH_MODE(0) or PYNATIVE_MODE(1) (default=0)')
    group.add_argument('--distribute', type=bool, default=True,
                       help='Run distribute (default=True)')

    # Dataset parameters
    group = parser.add_argument_group('Dataset parameters')
    group.add_argument('--dataset', type=str, default='imagenet',
                       help='Type of dataset (default="imagenet")')
    group.add_argument('--data_dir', type=str, help='Path to dataset')
    group.add_argument('--dataset_download', type=bool, default=False,
                       help='Download dataset (default=False)')
    group.add_argument('--num_parallel_workers', type=int, default=8,
                       help='Number of parallel workers (default=8)')
    group.add_argument('--shuffle', type=bool, default=True,
                       help='Whether or not to perform shuffle on the dataset (default="True")')
    group.add_argument('--batch_size', type=int, default=64,
                       help='Number of batch size (default=64)')
    group.add_argument('--drop_remainder', type=bool, default=True,
                       help='Determines whether or not to drop the last block whose data '
                            'row number is less than batch size (default=True)')

    # Augmentation parameters
    group = parser.add_argument_group('Augmentation parameters')
    group.add_argument('--image_resize', type=int, default=224,
                       help='Crop the size of the image (default=224)')
    group.add_argument('--scale', type=tuple, default=(0.08, 1.0),
                       help='Random resize scale (default=(0.08, 1.0))')
    group.add_argument('--ratio', type=tuple, default=(0.75, 1.333),
                       help='Random resize aspect ratio (default=(0.75, 1.333))')
    group.add_argument('--hflip', type=float, default=0.5,
                       help='Horizontal flip training aug probability (default=0.5)')
    group.add_argument('--vflip', type=float, default=0.,
                       help='Vertical flip training aug probability (default=0.)')
    group.add_argument('--color_jitter', type=float, default=None,
                       help='Color jitter factor (default=None)')
    group.add_argument('--interpolation', type=str, default='bilinear',
                       help='Image interpolation mode for resize operator(default="bilinear")')
    group.add_argument('--auto_augment', type=bool, default=False,
                       help='Whether to use auto augmentation (default=False)')
    group.add_argument('--re_prob', type=float, default=0.,
                       help='Probability of performing erasing (default=0.)')
    group.add_argument('--re_scale', type=tuple, default=(0.02, 0.33),
                       help='Range of area scale of the erased area (default=(0.02, 0.33))')
    group.add_argument('--re_ratio', type=tuple, default=(0.3, 3.3),
                       help='Range of aspect ratio of the erased area (default=(0.3, 3.3))')
    group.add_argument('--re_value', default=0,
                       help='Pixel value used to pad the erased area (default=0)')
    group.add_argument('--re_max_attempts', type=int, default=10,
                       help='The maximum number of attempts to propose a valid erased area, '
                            'beyond which the original image will be returned (default=10)')
    group.add_argument('--mean', type=list, default=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                       help='List or tuple of mean values for each channel, '
                            'with respect to channel order (default=[0.485 * 255, 0.456 * 255, 0.406 * 255])')
    group.add_argument('--std', type=list, default=[0.229 * 255, 0.224 * 255, 0.225 * 255],
                       help='List or tuple of mean values for each channel, '
                            'with respect to channel order (default=[0.229 * 255, 0.224 * 255, 0.225 * 255])')
    group.add_argument('--crop_pct', type=float, default=0.875,
                       help='Input image center crop percent (default=0.875)')
    group.add_argument('--mixup', type=float, default=0.,
                       help='Hyperparameter of beta distribution (default=0.)')

    # Model parameters
    group = parser.add_argument_group('Model parameters')
    group.add_argument('--model', type=str,
                       help='Name of model')
    group.add_argument('--num_classes', type=int, default=1000,
                       help='Number of label classes (default=1000)')
    group.add_argument('--pretrained', type=bool, default=False,
                       help='Load pretrained model (default=False)')
    group.add_argument('--ckpt_path', type=str, default='',
                       help='Initialize model from this checkpoint (default='')')
    group.add_argument('--drop_rate', type=float, default=None,
                       help='Drop rate (default=None)')
    group.add_argument('--drop_path_rate', type=float, default=None,
                       help='Drop path rate (default=None)')
    group.add_argument('--amp_level', type=str, default='O0', help='Amp level (default="O0").')
    group.add_argument('--keep_checkpoint_max', type=int, default=10,
                       help='Max number of checkpoint files (default=10)')
    group.add_argument('--ckpt_save_dir', type=str, default="./ckpt",
                       help='Path of checkpoint (default="./ckpt")')
    group.add_argument('--epoch_size', type=int, default=90,
                       help='Train epoch size (default=90)')
    group.add_argument('--dataset_sink_mode', type=bool, default=True,
                       help='The dataset sink mode (default=True).')

    # Optimize parameters
    group = parser.add_argument_group('Optimizer parameters')
    group.add_argument('--opt', type=str, default='momentum',
                       help='Type of optimizer (default="momentum")')
    group.add_argument('--momentum', type=float, default=0.9,
                       help='Hyperparameter of type float, means momentum for the moving average. '
                            'It must be at least 0.0 (default=0.9)')
    group.add_argument('--weight_decay', type=float, default=0.0,
                       help='Weight decay (default=0.0)')
    group.add_argument('--loss_scale', type=float, default=1.0,
                       help='Loss scale (default=1.0)')
    group.add_argument('--use_nesterov', type=bool, default=False,
                       help='Enables the Nesterov momentum (default=False)')
    group.add_argument('--epsilon', type=float, default=1e-10,
                       help='Term added to the denominator to improve numerical stability (default=1e-10)')
    group.add_argument('--use_locking', type=bool, default=False,
                       help='Whether to enable a lock to protect the updating process'
                            'of variable tensors (default=False)')
    group.add_argument('--centered', type=bool, default=False,
                       help='If True, gradients are normalized by the estimated'
                            ' variance of the gradient (default=False)')
    group.add_argument('--decay', type=float, default=0.9,
                       help='Decay rate (default=0.9)')
    group.add_argument('--dampening', type=float, default=0.,
                       help='A floating point value of dampening for momentum (default=0.0)')
    group.add_argument('--filter_bias_and_bn', type=bool, default=True,
                       help='Filter Bias and BatchNorm (default=True)')

    # Scheduler parameters
    group = parser.add_argument_group('Scheduler parameters')
    group.add_argument('--scheduler', type=str, default='const',
                       help='Type of scheduler (default="const")')
    group.add_argument('--lr', type=float,
                       help='learning rate (default=0.01)')
    group.add_argument('--min_lr', type=float,
                       help='The minimum value of learning rate')
    group.add_argument('--max_lr', type=float,
                       help='The maximum value of learning rate')
    group.add_argument('--warmup_epochs', type=int,
                       help='Warmup epochs')
    group.add_argument('--decay_epochs', type=int,
                       help='Decay epochs')

    # Loss parameters
    group = parser.add_argument_group('Loss parameters')
    group.add_argument('--loss', type=str, default='cross_entropy_smooth',
                       help='Type of loss (default="cross_entropy_smooth")')
    group.add_argument('--smooth_factor', type=float, default=0.0,
                       help='Label smoothing (default=0.0)')
    group.add_argument('--factor', type=float, default=0.0,
                       help='Loss factor (default=0.0)')
    group.add_argument('--sparse', type=bool, default=True,
                       help='Specifies whether labels use sparse format or not (default=False)')
    group.add_argument('--reduction', type=str, default='mean',
                       help='Type of reduction to be applied to loss (default="mean")')

    return parser_config, parser


def parse_args():
    parser_config, parser = create_parser()
    # Do we have a config file to parse?
    args_config, remaining = parser_config.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)
            parser.set_defaults(config=args_config.config)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)
    return args


def save_args(args: argparse.Namespace, filepath: str, rank: int = 0) -> None:
    """If in master process, save ``args`` to a YAML file. Otherwise, do nothing.
    Args:
        args (Namespace): The parsed arguments to be saved.
        filepath (str): A filepath ends with ``.yaml``.
        rank (int): Process rank in the distributed training. Defaults to 0.
    """
    assert isinstance(args, argparse.Namespace)
    assert filepath.endswith(".yaml")
    if rank != 0:
        return
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, "w") as f:
        yaml.safe_dump(args.__dict__, f)
    logger.info(f"Args is saved to {filepath}.")
