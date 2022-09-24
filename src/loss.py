"""loss factory"""

import mindspore.nn as nn
import mindspore.ops as ops


class CrossEntropySmooth(nn.LossBase):
    """CrossEntropy"""

    def __init__(self, smooth_factor=0., factor=0.):
        super(CrossEntropySmooth, self).__init__()
        self.smoothing = smooth_factor
        self.confidence = 1. - smooth_factor
        self.factor = factor
        self.log_softmax = ops.LogSoftmax()
        self.gather = ops.Gather()
        self.expand = ops.ExpandDims()

    def construct(self, logit, label):
        loss_aux = 0
        if self.factor > 0:
            logit, aux = logit
            auxprobs = self.log_softmax(aux)
            nll_loss_aux = ops.gather_d((-1 * auxprobs), 1, self.expand(label, -1))
            nll_loss_aux = nll_loss_aux.squeeze(1)
            smooth_loss = -auxprobs.mean(axis=-1)
            loss_aux = (self.confidence * nll_loss_aux + self.smoothing * smooth_loss).mean()
        logprobs = self.log_softmax(logit)
        nll_loss_logit = ops.gather_d((-1 * logprobs), 1, self.expand(label, -1))
        nll_loss_logit = nll_loss_logit.squeeze(1)
        smooth_loss = -logprobs.mean(axis=-1)
        loss_logit = (self.confidence * nll_loss_logit + self.smoothing * smooth_loss).mean()
        loss = loss_logit + self.factor * loss_aux
        return loss


def create_loss(args):
    if args.loss == 'cross_entropy_smooth':
        loss = CrossEntropySmooth(smooth_factor=args.smooth_factor,
                                  factor=args.factor)
    elif args.loss == 'softmax_cross_entropy_with_logits':
        loss = nn.SoftmaxCrossEntropyWithLogits(sparse=args.sparse,
                                                reduction=args.reduction)
    return loss
