import torch
import torchvision
from torch2trt import torch2trt
from torch2trt import TRTModule


def torch_2_trt(model, x, trt_model_path, save_engine=None):
    assert isinstance(x, torch.cuda.FloatTensor), print('Invalid Input Type - {}'.format(type(x)))
    assert save_engine is None or isinstance(save_engine, str), print('save_engine is path to trt engine')
    model_trt = torch2trt(model,
                          [x],
                          fp16_mode=False,
                          max_batch_size=1,
                          max_workspace_size=(1 << 32))
    torch.save(model_trt.state_dict(), trt_model_path)
    print('TRT model exported to {}'.format(trt_model_path))

    if save_engine is not None:
        trt_engine_path = trt_model_path.split('.')[0] + '.engine'
        with open(trt_engine_path, 'wb') as f:
            f.write(model_trt.engine.serialize())
        print('TRT Engine exported to {}'.format(trt_engine_path))


def load_trt_model(trt_model_path):
    model_trt = TRTModule()
    model_trt.load_state_dict(torch.load(trt_model_path))
    return model_trt


if __name__ == '__main__':
    import time

    model = torchvision.models.resnet50(pretrained=True).eval().cuda()
    x = torch.randn((1, 3, 224, 224)).float().cuda()
    model_trt = torch2trt(model, [x])

    with torch.no_grad():
        t1 = time.time()
        y = model(x)
        t2 = time.time()
        t3 = time.time()
        y_trt = model_trt(x)
        t4 = time.time()
        print(t4-t3)
        print(t2-t1)
        print(torch.mean(torch.pow(y - y_trt, 2)))
