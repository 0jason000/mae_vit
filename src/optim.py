"""optimizer factory"""

import mindspore.nn as nn


def init_group_params(params, weight_decay):
    decay_params = []
    no_decay_params = []

    for param in params:
        if 'beta' not in param.name and 'gamma' not in param.name and 'bias' not in param.name:
            decay_params.append(param)
        else:
            no_decay_params.append(param)
    return [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params},
        {'order_params': params}
    ]


def create_optimizer(args, learning_rate, params, filter_bias_and_bn=True):
    weight_decay = args.weight_decay
    if weight_decay and filter_bias_and_bn:
        params = init_group_params(params, weight_decay)

    if args.opt.lower() == 'momentum':
        optimizer = nn.Momentum(params=params,
                                learning_rate=learning_rate,
                                momentum=args.momentum,
                                weight_decay=weight_decay,
                                loss_scale=args.loss_scale,
                                use_nesterov=args.use_nesterov,
                                )
    elif args.opt.lower() == 'rmsprop':
        optimizer = nn.RMSProp(params=params,
                               learning_rate=learning_rate,
                               decay=args.decay,
                               momentum=args.momentum,
                               epsilon=args.epsilon,
                               use_locking=args.use_locking,
                               centered=args.centered,
                               weight_decay=weight_decay,
                               loss_scale=args.loss_scale,
                               )
    elif args.opt.lower() == 'sgd':
        optimizer = nn.SGD(params=params,
                           learning_rate=learning_rate,
                           momentum=args.momentum,
                           dampening=args.dampening,
                           weight_decay=weight_decay,
                           nesterov=args.use_nesterov,
                           loss_scale=args.loss_scale)
    else:
        raise ValueError('Invalid optimizer.')

    return optimizer
