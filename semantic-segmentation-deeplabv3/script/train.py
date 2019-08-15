import os
import fire
import torch

from .model import deeplabv3_resnet101, fcn_resnet101
from .smt_fake import smt_fake_model


def entrance_fake(model_type='deeplabv3_resnet101', model_path='script/saved_model', save_path='script/saved_model'):

    os.makedirs(save_path, exist_ok=True)

    if model_type == 'deeplabv3_resnet101':
        model = deeplabv3_resnet101(model_path=model_path, pretrained=True)
    else:
        model = fcn_resnet101(model_path=model_path, pretrained=True)
    torch.save(model.state_dict(), os.path.join(save_path, 'model.pth'))

    smt_fake_model(save_path)

    print('This experiment has been completed.')


if __name__ == '__main__':
    fire.Fire(entrance_fake)
