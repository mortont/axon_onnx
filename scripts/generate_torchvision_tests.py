import os
from onnx import numpy_helper
import onnxruntime
import torchvision
import torch
import numpy as np

BASE_TEST_PATH = 'test/cases/torchvision/'
NUM_DUMMY_TESTS = 3

if not os.path.exists(BASE_TEST_PATH):
  os.mkdir(BASE_TEST_PATH)

MODELS_TO_TEST = {
  'alexnet': torchvision.models.alexnet,
  'convnext_tiny': torchvision.models.convnext_tiny,
  'convnext_small': torchvision.models.convnext_small,
  'convnext_base': torchvision.models.convnext_base,
  'convnext_large': torchvision.models.convnext_large,
  'efficientnet_b0': torchvision.models.efficientnet_b0,
  'efficientnet_b1': torchvision.models.efficientnet_b1,
  'efficientnet_b2': torchvision.models.efficientnet_b2,
  'efficientnet_b3': torchvision.models.efficientnet_b3,
  'efficientnet_b4': torchvision.models.efficientnet_b4,
  'efficientnet_b5': torchvision.models.efficientnet_b5,
  'efficientnet_b6': torchvision.models.efficientnet_b6,
  'efficientnet_b7': torchvision.models.efficientnet_b7,
  'efficientnet_v2_s': torchvision.models.efficientnet_v2_s,
  'efficientnet_v2_m': torchvision.models.efficientnet_v2_m,
  'efficientnet_v2_l': torchvision.models.efficientnet_v2_l,
  'vgg11': torchvision.models.vgg11,
  'vgg11_bn': torchvision.models.vgg11_bn,
  'vgg13': torchvision.models.vgg13,
  'vgg16': torchvision.models.vgg16,
  'vgg16_bn': torchvision.models.vgg16_bn,
  'vgg19': torchvision.models.vgg19,
  'vgg19_bn': torchvision.models.vgg19_bn,
  'resnet18': torchvision.models.resnet18,
  'resnet34': torchvision.models.resnet34,
  'resnet50': torchvision.models.resnet50,
  'resnet101': torchvision.models.resnet101,
  'resnet152': torchvision.models.resnet152,
  'squeezenet1_0': torchvision.models.squeezenet1_0,
  'squeezenet1_1': torchvision.models.squeezenet1_1,
  'densenet121': torchvision.models.densenet121,
  'densenet161': torchvision.models.densenet161,
  'densenet169': torchvision.models.densenet169,
  'densenet201': torchvision.models.densenet201,
  'inception_v3': torchvision.models.inception_v3,
  'googlenet': torchvision.models.googlenet,
  'shufflenet_v2_x0_5': torchvision.models.shufflenet_v2_x0_5,
  'shufflenet_v2_x1_0': torchvision.models.shufflenet_v2_x1_0,
  # 'shufflenet_v2_x1_5': torchvision.models.shufflenet_v2_x1_5,
  # 'shufflenet_v2_x2_0': torchvision.models.shufflenet_v2_x2_0,
  'mobilenet_v2': torchvision.models.mobilenet_v2,
  'mobilenet_v3_small': torchvision.models.mobilenet_v3_small,
  'mobilenet_v3_large': torchvision.models.mobilenet_v3_large,
  'resnext50_32x4d': torchvision.models.resnext50_32x4d,
  'resnext101_32x8d': torchvision.models.resnext101_32x8d,
  'resnext101_64x4d': torchvision.models.resnext101_64x4d,
  'wide_resnet50_2': torchvision.models.wide_resnet50_2,
  # 'wide_resnet101_2': torchvision.models.wide_resnet101_2,
  'mnasnet0_5': torchvision.models.mnasnet0_5,
  # 'mnasnet0_75': torchvision.models.mnasnet0_75,
  'mnasnet1_0': torchvision.models.mnasnet1_0,
  'mnasnet1_3': torchvision.models.mnasnet1_3,
  # 'swin_t': torchvision.models.swin_t,
  # 'swin_s': torchvision.models.swin_s,
  # 'swin_b': torchvision.models.swin_b,
  'vit_b_16': torchvision.models.vit_b_16,
  'vit_b_32': torchvision.models.vit_b_32,
  'vit_l_16': torchvision.models.vit_l_16,
  # 'vit_l_32': torchvision.models.vit_l_32,
  'vit_h_14': torchvision.models.vit_h_14,
  # Segmentation
  # 'fcn_resnet50': torchvision.models.segmentation.fcn_resnet50,
  # 'fcn_resnet101': torchvision.models.segmentation.fcn_resnet101,
  # 'deeplabv3_resnet50': torchvision.models.segmentation.deeplabv3_resnet50,
  # 'deeplabv3_resnet101': torchvision.models.segmentation.deeplabv3_resnet101,
  # Object Detection
  # 'fasterrcnn_resnet50_fpn': torchvision.models.detection.fasterrcnn_resnet50_fpn,
  # 'retinanet_resnet50_fpn': torchvision.models.detection.retinanet_resnet50_fpn,
  # 'maskrcnn_resnet50_fpn': torchvision.models.detection.maskrcnn_resnet50_fpn,
  # 'keypointrcnn_resnet50_fpn': torchvision.models.detection.keypointrcnn_resnet50_fpn,
  # Video Classification
  # 'r3d_18': torchvision.models.video.r3d_18,
  # 'mc3_18': torchvision.models.video.mc3_18,
  # 'r2plus1d_18': torchvision.models.video.r2plus1d_18
}

for key, model in MODELS_TO_TEST.items():
  model_base_path = os.path.join(BASE_TEST_PATH, key)
  if not os.path.exists(model_base_path):
    os.mkdir(model_base_path)

    net = model(pretrained=True)
    model_onnx_path = os.path.join(model_base_path, 'model.onnx')

    input_names = [ "input_0" ]
    output_names = [ "output_0" ]

    input_shape = (3, 224, 224)
    batch_size = 1

    for i in range(NUM_DUMMY_TESTS):
      dummy_input = torch.randn(batch_size, *input_shape)
      dummy_output = net.eval()(dummy_input)
      dummy_input_array = dummy_input.numpy()

      test_data_path = os.path.join(model_base_path, f'test_data_set_{i}')
      os.mkdir(test_data_path)
      with open(os.path.join(test_data_path, 'input_0.pb'), 'wb') as outfile:
        inp_tensor = numpy_helper.from_array(dummy_input_array)
        outfile.write(inp_tensor.SerializeToString())

      outputs = []
      if torch.is_tensor(dummy_output):
        out = dummy_output.detach().numpy()
        outputs.append(out)
      elif type(dummy_output) is dict:
        for out in dummy_output.values():
          out = out.detach().numpy()
          outputs.append(out)
      else:
        for out_dict in dummy_output:
          for out in out_dict.values():
            out = out.detach().numpy()
            outputs.append(out)

      for j, out in enumerate(outputs):
        with open(os.path.join(test_data_path, f'output_{j}.pb'), 'wb') as outfile:
          out_tensor = numpy_helper.from_array(out)
          outfile.write(out_tensor.SerializeToString())

    dummy_input = torch.randn(batch_size, *input_shape)
    torch.onnx.export(net, dummy_input, model_onnx_path, \
                      verbose=False, input_names=input_names, output_names=output_names, opset_version=11)