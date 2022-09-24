"""scheduler factory"""

import mindspore.nn as nn
from mindspore.nn.learning_rate_schedule import LearningRateSchedule


class WarmupCosineDecayLR(LearningRateSchedule):
    """ CosineDecayLR """

    def __init__(self,
                 min_lr,
                 max_lr,
                 warmup_epochs,
                 decay_epochs,
                 steps_per_epoch
                 ):
        super(WarmupCosineDecayLR, self).__init__()
        self.warmup_steps = warmup_epochs * steps_per_epoch
        self.decay_steps = decay_epochs * steps_per_epoch
        if self.warmup_steps > 0:
            self.warmup_lr = nn.WarmUpLR(max_lr, self.warmup_steps)
        self.cosine_decay_lr = nn.CosineDecayLR(min_lr, max_lr, self.decay_steps)

    def construct(self, global_step):
        """ construct """
        if self.warmup_steps > 0:
            if global_step > self.warmup_steps:
                lr = self.cosine_decay_lr(global_step - self.warmup_steps)
            else:
                lr = self.warmup_lr(global_step)
        else:
            lr = self.cosine_decay_lr(global_step)
        return lr


def create_scheduler(args):
    if args.scheduler == 'warmup_cosine_decay_lr':
        lr_scheduler = WarmupCosineDecayLR(min_lr=args.min_lr,
                                           max_lr=args.max_lr,
                                           warmup_epochs=args.warmup_epochs,
                                           decay_epochs=args.decay_epochs,
                                           steps_per_epoch=args.steps_per_epoch
                                           )
    elif args.scheduler == 'const':
        lr_scheduler = args.lr

    return lr_scheduler
