import sys

import torch.nn as nn
from torchvision.models.segmentation.segmentation import _segm_resnet, model_urls
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url


__all__ = ['SegmentationNet']


def load_model(arch_type, backbone, model_path, pretrained, num_classes, **kwargs):
    aux_loss = True
    temp_arch_type = arch_type
    if temp_arch_type == 'deeplabv3':
        temp_arch_type = temp_arch_type[:-2]
    model = _segm_resnet(temp_arch_type, backbone, num_classes, aux_loss, **kwargs)
    if pretrained:
        arch = arch_type + '_' + backbone + '_coco'
        model_url = model_urls[arch]
        if model_url is None:
            raise NotImplementedError('pretrained {} is not supported as of now'.format(arch))
        else:
            state_dict = load_state_dict_from_url(model_url, model_dir=model_path)
            model.load_state_dict(state_dict)
    return model


class SegmentationFuncs:

    @staticmethod
    def fcn_resnet50(model_path='script/saved_model', pretrained=False,
                     num_classes=21, **kwargs):
        return load_model('fcn', 'resnet50', model_path, pretrained, num_classes, **kwargs)

    @staticmethod
    def fcn_resnet101(model_path='script/saved_model', pretrained=False,
                      num_classes=21, **kwargs):
        return load_model('fcn', 'resnet101', model_path, pretrained, num_classes, **kwargs)

    @staticmethod
    def deeplabv3_resnet50(model_path='script/saved_model', pretrained=False,
                           num_classes=21, **kwargs):
        return load_model('deeplabv3', 'resnet50', model_path, pretrained, num_classes, **kwargs)

    @staticmethod
    def deeplabv3_resnet101(model_path='script/saved_model', pretrained=False,
                            num_classes=21, **kwargs):
        return load_model('deeplabv3', 'resnet101', model_path, pretrained, num_classes, **kwargs)


class SegmentationNet(nn.Module):
    def __init__(self, model_type='deeplabv3_resnet101', model_path=None, pretrained=True, num_classes=21):
        super().__init__()
        net_func = getattr(SegmentationFuncs, model_type, None)

        if net_func is None:
            print(f'Error: No such pretrained model {model_type}')
            sys.exit()

        self.model = net_func(pretrained=pretrained, model_path=model_path, num_classes=num_classes)

    def forward(self, input):
        return self.model(input)
