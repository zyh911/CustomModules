import os
import re

import torch
from torchvision.models.densenet import DenseNet


__all__ = ['densenet201', 'densenet169', 'densenet161', 'densenet121']


model_urls = {
    'densenet121': 'densenet121-a639ec97.pth',
    'densenet169': 'densenet169-b2777c0a.pth',
    'densenet201': 'densenet201-c1103571.pth',
    'densenet161': 'densenet161-8d451a50.pth',
}


def _load_state_dict(model, model_url):
    # '.'s are no longer allowed in module names, but previous _DenseLayer
    # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
    # They are also in the checkpoints in model_urls. This pattern is used
    # to find such keys.
    pattern = re.compile(
        r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')

    state_dict = torch.load(model_url)
    for key in list(state_dict.keys()):
        res = pattern.match(key)
        if res:
            new_key = res.group(1) + res.group(2)
            state_dict[new_key] = state_dict[key]
            del state_dict[key]
    model.load_state_dict(state_dict)


def _densenet(arch, model_path, growth_rate, block_config, num_init_features, pretrained,
              **kwargs):
    model = DenseNet(growth_rate, block_config, num_init_features, **kwargs)
    if pretrained:
        _load_state_dict(model, os.path.join(model_path, model_urls[arch]))
    return model


def densenet121(model_path='script/saved_model', pretrained=False, **kwargs):
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        model_path (str): model path string
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """
    return _densenet('densenet121', model_path, 32, (6, 12, 24, 16), 64, pretrained,
                     **kwargs)


def densenet161(model_path='script/saved_model', pretrained=False, **kwargs):
    r"""Densenet-161 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        model_path (str): model path string
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """
    return _densenet('densenet161', model_path, 48, (6, 12, 36, 24), 96, pretrained,
                     **kwargs)


def densenet169(model_path='script/saved_model', pretrained=False, **kwargs):
    r"""Densenet-169 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        model_path (str): model path string
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """
    return _densenet('densenet169', model_path, 32, (6, 12, 32, 32), 64, pretrained,
                     **kwargs)


def densenet201(model_path='script/saved_model', pretrained=False, **kwargs):
    r"""Densenet-201 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        model_path (str): model path string
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """
    return _densenet('densenet201', model_path, 32, (6, 12, 48, 32), 64, pretrained,
                     **kwargs)
