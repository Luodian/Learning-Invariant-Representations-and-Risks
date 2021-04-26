import torch

models = {}
__all__ = ['get_model']
def register_model(name):
    def decorator(cls):
        models[name] = cls
        return cls
    return decorator


def get_model(name, **args):
    net = models[name].create(args)
    if torch.cuda.is_available():
        net = net.cuda()
    return net
