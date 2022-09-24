"""Train"""

import mindspore as ms
from mindspore import FixedLossScaleManager, Model, LossMonitor, TimeMonitor, CheckpointConfig, ModelCheckpoint
from mindspore.communication import init, get_rank, get_group_size

from src.model import create_model
from src.dataset import create_dataset
from src.dataset_loader import create_loader
from src.loss import create_loss
from src.optim import create_optimizer
from src.scheduler import create_scheduler
from src.utils.config import parse_args

ms.set_seed(1)


def train(args):
    ms.set_context(mode=args.mode)

    if args.distribute:
        init()
        device_num = get_group_size()
        rank_id = get_rank()
        ms.set_auto_parallel_context(device_num=device_num,
                                     parallel_mode='data_parallel',
                                     gradients_mean=True)
    else:
        device_num = 1
        rank_id = 0

    # create dataset
    dataset_train = create_dataset(
        name=args.dataset,
        root=args.data_dir,
        split='train',
        shuffle=args.shuffle,
        num_shards=device_num,
        shard_id=rank_id,
        num_parallel_workers=args.num_parallel_workers,
        download=args.dataset_download)

    # load dataset
    loader_train = create_loader(
        dataset=dataset_train,
        batch_size=args.batch_size,
        drop_remainder=args.drop_remainder,
        is_training=True,
        mixup=args.mixup,
        num_classes=args.num_classes,
        num_parallel_workers=args.num_parallel_workers,
        image_resize=args.image_resize,
        scale=args.scale,
        ratio=args.ratio,
        hflip=args.hflip,
        vflip=args.vflip,
        color_jitter=args.color_jitter,
        interpolation=args.interpolation,
        auto_augment=args.auto_augment,
        mean=args.mean,
        std=args.std,
        re_prob=args.re_prob,
        re_scale=args.re_scale,
        re_ratio=args.re_ratio,
        re_value=args.re_value,
        re_max_attempts=args.re_max_attempts
    )
    args.steps_per_epoch = loader_train.get_dataset_size()

    # create model
    network = create_model(model_name=args.model,
                           num_classes=args.num_classes,
                           drop_rate=args.drop_rate,
                           drop_path_rate=args.drop_path_rate,
                           pretrained=args.pretrained,
                           checkpoint_path=args.ckpt_path)

    # create loss
    loss = create_loss(args)

    # create learning rate schedule
    lr_scheduler = create_scheduler(args)

    # create optimizer
    optimizer = create_optimizer(args=args,
                                 learning_rate=lr_scheduler,
                                 params=network.trainable_params(),
                                 filter_bias_and_bn=args.filter_bias_and_bn)

    # init model
    if args.loss_scale > 1.0:
        loss_scale_manager = FixedLossScaleManager(loss_scale=args.loss_scale, drop_overflow_update=False)
        model = Model(network, loss_fn=loss, optimizer=optimizer, metrics={'acc'}, amp_level=args.amp_level,
                      loss_scale_manager=loss_scale_manager)
    else:
        model = Model(network, loss_fn=loss, optimizer=optimizer, metrics={'acc'}, amp_level=args.amp_level)

    # callback
    loss_cb = LossMonitor(per_print_times=args.steps_per_epoch)
    time_cb = TimeMonitor(data_size=args.steps_per_epoch)
    callbacks = [loss_cb, time_cb]
    ckpt_config = CheckpointConfig(
        save_checkpoint_steps=args.steps_per_epoch,
        keep_checkpoint_max=args.keep_checkpoint_max)
    ckpt_cb = ModelCheckpoint(prefix=args.model,
                              directory=args.ckpt_save_dir,
                              config=ckpt_config)
    if args.distribute:
        if rank_id == 0:
            callbacks.append(ckpt_cb)
    else:
        callbacks.append(ckpt_cb)

    # train model
    model.train(args.epoch_size, loader_train, callbacks=callbacks, dataset_sink_mode=args.dataset_sink_mode)


if __name__ == '__main__':
    args = parse_args()
    train(args)
