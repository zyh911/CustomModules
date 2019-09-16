import os
import numpy as np
import logging
from PIL import Image
from io import BytesIO
from argparse import ArgumentParser
import base64
import pandas as pd
import pyarrow.parquet as pq  # noqa: F401
from azureml.studio.modulehost.handler.port_io_handler import OutputHandler
from azureml.studio.common.datatypes import DataTypes
from azureml.studio.common.datatable.data_table import DataTable
from azureml.studio.common.io.data_frame_utils import load_data_frame_from_directory
import torch
from torchvision import transforms
from .model import SegmentationNet
from .smt_fake import smt_fake_file

logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s',
                    datefmt='%d-%M-%Y %H:%M:%S', level=logging.INFO)


class Score:
    def __init__(self, model_path, meta={}):
        # 21 colors to paint
        self.pallete = [
            [255, 255, 255],
            [128, 64, 128],
            [244, 35, 232],
            [70, 70, 70],
            [102, 102, 156],
            [190, 153, 153],
            [153, 153, 153],
            [250, 170, 30],
            [220, 220, 0],
            [107, 142, 35],
            [152, 251, 152],
            [70, 130, 180],
            [220, 20, 60],
            [255, 0, 0],
            [0, 0, 142],
            [0, 0, 70],
            [0, 60, 100],
            [0, 80, 100],
            [0, 0, 230],
            [119, 11, 32],
            [0, 0, 0]
        ]
        self.inference_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])
        self.model = SegmentationNet(model_type=meta['Model type'], model_path=model_path, pretrained=False)
        self.model.load_state_dict(torch.load(os.path.join(model_path, 'model.pth'), map_location='cpu'))
        if meta['Use CUDA'] and torch.cuda.is_available():
            self.model = self.model.cuda()
        self.model.eval()

    def image_to_string(self, image):
        imgByteArr = BytesIO()
        image.save(imgByteArr, format='PNG')
        imgBytes = imgByteArr.getvalue()
        s = base64.b64encode(imgBytes)
        s = s.decode('ascii')
        s = 'data:image/png;base64,' + s
        return s

    def decode_image_str(self, input_str):
        if input_str.startswith('data:'):
            my_index = input_str.find('base64,')
            input_str = input_str[my_index + 7:]
        temp = base64.b64decode(input_str)
        img = Image.open(BytesIO(temp))
        return img

    def run(self, input, meta=None):
        my_list = []
        for i in range(input.shape[0]):
            img = self.decode_image_str(input.iloc[i]['image_string'])
            input_tensor = self.inference_transforms(img)
            input_tensor = input_tensor.unsqueeze(0)
            if torch.cuda.is_available():
                input_tensor = input_tensor.cuda()
            with torch.no_grad():
                output = self.model(input_tensor)['out'][0]
                output_predictions = output.argmax(0).cpu().numpy()
                resultImg = np.zeros((output_predictions.shape[0], output_predictions.shape[1], 3), dtype=np.uint8)
                for idx in range(len(self.pallete)):
                    resultImg[output_predictions == idx] = self.pallete[idx]
                resultImg1 = Image.fromarray(resultImg).resize(img.size)
                resultImg2 = Image.blend(img, resultImg1, 0.5)
                s1 = self.image_to_string(resultImg1)
                s2 = self.image_to_string(resultImg2)
            my_list.append([s1, s2])
        df = pd.DataFrame(my_list, columns=['mask', 'fusion'])
        return df

    def inference(self, data_path, save_path):
        os.makedirs(save_path, exist_ok=True)
        input = load_data_frame_from_directory(data_path).data
        df = self.run(input)
        dt = DataTable(df)
        OutputHandler.handle_output(data=dt, file_path=save_path,
                                    file_name='data.dataset.parquet', data_type=DataTypes.DATASET)


def test(args):
    meta = {'Model type': args.model_type, 'Use CUDA': args.use_cuda}
    score = Score(args.model_path, meta)
    score.inference(data_path=args.data_path, save_path=args.save_path)
    smt_fake_file(args.save_path)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model_type', default='deeplabv3_resnet101', help='Model type')
    parser.add_argument('--model_path', default='ssdeeplabv3/saved_model', help='Model directory')
    parser.add_argument('--data_path', default='ssdeeplabv3/outputs', help='Data directory')
    parser.add_argument('--save_path', default='ssdeeplabv3/outputs2', help='directory to save the results')
    parser.add_argument('--use_cuda', default=False, help='if use cuda')
    test(parser.parse_args())
    logging.info('This experiment has been completed.')