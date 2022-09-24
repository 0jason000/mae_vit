"""Validation"""

import mindspore as ms
import mindspore.nn as nn
from mindspore import Model

from src.model import create_model
from src.dataset import create_dataset
from src.dataset_loader import create_loader
from src.loss import create_loss
from src.utils.config import parse_args


def validate(args):
    ms.set_context(mode=args.mode)

    # create dataset
    dataset_eval = create_dataset(
        name=args.dataset,
        root=args.data_dir,
        split='val',
        num_parallel_workers=args.num_parallel_workers,
        download=args.dataset_download)

    # load dataset
    loader_eval = create_loader(
        dataset=dataset_eval,
        batch_size=args.batch_size,
        drop_remainder=args.drop_remainder,
        is_training=False,
        num_parallel_workers=args.num_parallel_workers,
        image_resize=args.image_resize,
        crop_pct=args.crop_pct,
        interpolation=args.interpolation,
        mean=args.mean,
        std=args.std,
    )

    # create model
    network = create_model(model_name=args.model,
                           num_classes=args.num_classes,
                           drop_rate=args.drop_rate,
                           drop_path_rate=args.drop_path_rate,
                           pretrained=args.pretrained,
                           checkpoint_path=args.ckpt_path)
    network.set_train(False)

    # create loss
    loss = create_loss(args)

    # Define eval metrics.
    eval_metrics = {'Top_1_Accuracy': nn.Top1CategoricalAccuracy(),
                    'Top_5_Accuracy': nn.Top5CategoricalAccuracy()}

    # init model
    model = Model(network, loss_fn=loss, metrics=eval_metrics)

    # validate
    result = model.eval(loader_eval)
    print(result)


if __name__ == '__main__':
    args = parse_args()
    validate(args)
