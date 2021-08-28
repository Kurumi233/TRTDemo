"""
Ref:
    https://github.com/ycdhqzhiai/yolov5_tensorRT/blob/7d6bccb4133fa12269ab8145ee6fde7d8224a67b/onnx_tensorrt.py
    https://github.com/SeanAvery/yolov5-tensorrt/blob/930a3629bb4a7a70d3353938c3f0aab55900af11/python/export_tensorrt.py
"""
import onnx
import torch
import torchvision
import tensorrt as trt


def torch_2_onnx(model, savepath, batch_size=1, input_size=(3, 224, 224), device='cpu'):
    print('onnx version', onnx.__version__)

    input_name = ['input']
    output_name = ['output']
    dummy_input = torch.randn(batch_size, *input_size).float().to(device)

    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            savepath,
            input_names=input_name,
            output_names=output_name,
            verbose=False,
            opset_version=11
        )

    # check
    test = onnx.load(savepath)
    onnx.checker.check_model(test)
    print("==> Check ONNX Model Passed")


def onnx_2_trt(onnx_model_path, trt_model_path='model.trt'):
    TRT_LOGGER = trt.Logger()  # This logger is required to build an engine

    EXPLICIT_BATCH = []
    print('trt version', trt.__version__)
    if trt.__version__[0] >= '7':
        EXPLICIT_BATCH.append(
            1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

    with trt.Builder(TRT_LOGGER) as builder, \
            builder.create_network(*EXPLICIT_BATCH) as network, \
            trt.OnnxParser(network, TRT_LOGGER) as parser:

        builder.max_workspace_size = 1 << 28
        builder.max_batch_size = 1

        with open(onnx_model_path, 'rb') as f:
            if not parser.parse(f.read()):
                for error in range(parser.num_errors):
                    print(parser.get_error(error))

        engine = builder.build_cuda_engine(network)
        with open(trt_model_path, 'wb') as f:
            f.write(engine.serialize())

    print('TRT model exported to {}'.format(trt_model_path))


if __name__ == '__main__':
    model = torchvision.models.resnet18(pretrained=True).cuda()
    torch_2_onnx(model, 'resnet18.onnx')


