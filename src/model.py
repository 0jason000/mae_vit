from mindspore import load_checkpoint, load_param_into_net

_model_entrypoints = {}


def is_model(model_name):
    """
    Check if a model name exists
    """
    return model_name in _model_entrypoints


def model_entrypoint(model_name):
    """
    Fetch a model entrypoint for specified model name
    """
    return _model_entrypoints[model_name]


def create_model(
        model_name: str,
        num_classes: int = 1000,
        pretrained=False,
        in_channels: int = 3,
        checkpoint_path: str = '',
        **kwargs):
    """
    Create a model
    """

    model_args = dict(num_classes=num_classes, pretrained=pretrained, in_channels=in_channels)
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    if not is_model(model_name):
        raise RuntimeError('Unknown model (%s)' % model_name)

    create_fn = model_entrypoint(model_name)
    model = create_fn(**model_args, **kwargs)

    if checkpoint_path:
        param_dict = load_checkpoint(checkpoint_path)
        load_param_into_net(model, param_dict)

    return model
