import cv2
import time
import torch
import numpy as np
from torchvision import models

from torch_onnx_trt import torch_2_onnx, onnx_2_trt
from trt_engine import TRT_Engine
from torch_trt import torch_2_trt, load_trt_model


if __name__ == '__main__':
    onnx_path = 'model.onnx'
    trt_engine_path = 'model.trt'
    trt_pth_path = 'model_trt.pth'
    trt_pth_engine_path = 'model_trt.engine'

    device = torch.device('cuda')
    model = models.resnet18(pretrained=True).eval().to(device)

    # convert torch model to trt: torch -> onnx -> trt
    torch_2_onnx(model, onnx_path, device=device)
    onnx_2_trt(onnx_path, trt_engine_path)
    # convert torch model to trt: torch -> trt
    inp = torch.randn((1, 3, 224, 224)).float().to(device)
    torch_2_trt(model, inp, trt_pth_path, save_engine=trt_pth_engine_path)
    print('Convert model done.')

    # Load trt model
    trt_engine = TRT_Engine(trt_engine_path)
    trt_model = load_trt_model(trt_pth_path)
    trt_pth_engine = TRT_Engine(trt_pth_engine_path)

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    x = cv2.imread('1.jpg')[:, :, ::-1]
    x = cv2.resize(x, (224, 224))
    x = (x / 255. - mean) / std
    x = x.transpose((2, 0, 1)).astype(np.float32)
    x_tensor = torch.from_numpy(x).unsqueeze(0).float().to(device)
    print('x shape : ', x.shape)
    print('x tensor shape : ', x_tensor.shape)

    # Inference
    def infer(model, inp, engine=False):
        s = 0
        with torch.no_grad():
            for i in range(100):
                t1 = time.time()
                y = model(inp)
                t2 = time.time()
                s += t2 - t1
            if engine:
                y = np.array(y)
            else:
                y = y.cpu().numpy()
        print('Infer time is : {:.4f}ms'.format(s * 10))
        return y

    y1 = infer(model, x_tensor)
    y2 = infer(trt_model, x_tensor)
    y3 = infer(trt_engine, x, engine=True)
    y4 = infer(trt_pth_engine, x, engine=True)

    print('MSE Error: \n{} \n{} \n{} \n{}'.format(np.mean(np.power(y1-y2, 2)),
                                                  np.mean(np.power(y1-y3, 2)),
                                                  np.mean(np.power(y2-y3, 2)),
                                                  np.mean(np.power(y3-y4, 2))))

    """OUTPUT"""
    """
    onnx version 1.8.1
    ==> Check ONNX Model Passed
    trt version 7.2.3.4
    TRT model exported to model.trt
    TRT model exported to model_trt.pth
    TRT Engine exported to model_trt.engine
    Convert model done.
    Load TRT Engine Done.
    Load TRT Engine Done.
    x shape :  (3, 224, 224)
    x tensor shape :  torch.Size([1, 3, 224, 224])
    Infer time is : 5.5845ms
    Infer time is : 2.7022ms
    Infer time is : 3.3296ms
    Infer time is : 3.2379ms
    MSE Error: 
    2.6493043869812993e-12 
    2.679772202751618e-12 
    1.1912007296353833e-13 
    1.1912007296353833e-13
    """