import os

import torch
from torchvision.models.segmentation.segmentation import _segm_resnet


__all__ = ['fcn_resnet50', 'fcn_resnet101', 'deeplabv3_resnet50', 'deeplabv3_resnet101']


model_urls = {
    'fcn_resnet50_coco': None,
    'fcn_resnet101_coco': 'fcn_resnet101_coco-7ecb50ca.pth',
    'deeplabv3_resnet50_coco': None,
    'deeplabv3_resnet101_coco': 'deeplabv3_resnet101_coco-586e9e4e.pth',
}


def _load_model(arch_type, backbone, model_path, pretrained, num_classes, **kwargs):
    aux_loss = True
    model = _segm_resnet(arch_type, backbone, num_classes, aux_loss, **kwargs)
    if pretrained:
        arch = arch_type + '_' + backbone + '_coco'
        model_url = os.path.join(model_path, model_urls[arch])
        if model_url is None:
            raise NotImplementedError('pretrained {} is not supported as of now'.format(arch))
        else:
            state_dict = torch.load(model_url)
            model.load_state_dict(state_dict)
    return model


def fcn_resnet50(model_path='script/saved_model', pretrained=False,
                 num_classes=21, **kwargs):
    """Constructs a Fully-Convolutional Network model with a ResNet-50 backbone.
    Args:
        model_path (str): Model path string
        pretrained (bool): If True, returns a model pre-trained on COCO train2017 which
            contains the same classes as Pascal VOC
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _load_model('fcn', 'resnet50', model_path, pretrained, num_classes, **kwargs)


def fcn_resnet101(model_path='script/saved_model', pretrained=False,
                  num_classes=21, **kwargs):
    """Constructs a Fully-Convolutional Network model with a ResNet-101 backbone.
    Args:
        model_path (str): Model path string
        pretrained (bool): If True, returns a model pre-trained on COCO train2017 which
            contains the same classes as Pascal VOC
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _load_model('fcn', 'resnet101', model_path, pretrained, num_classes, **kwargs)


def deeplabv3_resnet50(model_path='script/saved_model', pretrained=False,
                       num_classes=21, **kwargs):
    """Constructs a DeepLabV3 model with a ResNet-50 backbone.
    Args:
        model_path (str): Model path string
        pretrained (bool): If True, returns a model pre-trained on COCO train2017 which
            contains the same classes as Pascal VOC
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _load_model('deeplabv3', 'resnet50', model_path, pretrained, num_classes, **kwargs)


def deeplabv3_resnet101(model_path='script/saved_model', pretrained=False,
                        num_classes=21, **kwargs):
    """Constructs a DeepLabV3 model with a ResNet-101 backbone.
    Args:
        model_path (str): Model path string
        pretrained (bool): If True, returns a model pre-trained on COCO train2017 which
            contains the same classes as Pascal VOC
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _load_model('deeplabv3', 'resnet101', model_path, pretrained, num_classes, **kwargs)
