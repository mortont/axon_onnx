import os
from onnx import numpy_helper
import torchvision
import torch
import numpy as np

BASE_TEST_PATH = 'test/cases/torchvision/'
NUM_DUMMY_TESTS = 1

if not os.path.exists(BASE_TEST_PATH):
  os.mkdir(BASE_TEST_PATH)

MODELS_TO_TEST = {
  'alexnet': torchvision.models.alexnet,
  'vgg11': torchvision.models.vgg11,
  # 'vgg11_bn': torchvision.models.vgg11_bn,
  # 'vgg13': torchvision.models.vgg13,
  # 'vgg16': torchvision.models.vgg16,
  # 'vgg16_bn': torchvision.models.vgg16_bn,
  # 'vgg19': torchvision.models.vgg19,
  # 'vgg19_bn': torchvision.models.vgg19_bn,
  'resnet18': torchvision.models.resnet18,
  # 'resnet34': torchvision.models.resnet34,
  # 'resnet50': torchvision.models.resnet50,
  # 'resnet101': torchvision.models.resnet101,
  # 'resnet151': torchvision.models.resnet151,
  # 'resnet152': torchvision.models.resnet152,
  'squeezenet1_0': torchvision.models.squeezenet1_0,
  'squeezenet1_1': torchvision.models.squeezenet1_1,
  # 'densenet121': torchvision.models.densenet121,
  # 'densenet161': torchvision.models.densenet161,
  # 'densenet169': torchvision.models.densenet169,
  # 'densenet201': torchvision.models.densenet201,
  # 'inception_v3': torchvision.models.inception_v3,
  'googlenet': torchvision.models.googlenet,
  'shufflenet_v2_x0_5': torchvision.models.shufflenet_v2_x0_5,
  # 'shufflenet_v2_x1_0': torchvision.models.shufflenet_v2_x1_0,
  # 'shufflenet_v2_x1_5': torchvision.models.shufflenet_v2_x1_5,
  # 'shufflenet_v2_x2_0': torchvision.models.shufflenet_v2_x2_0,
  'mobilenet_v2': torchvision.models.mobilenet_v2,
  'resnext50_32x4d': torchvision.models.resnext50_32x4d,
  # 'resnext101_32x8d': torchvision.models.resnext101_32x8d,
  'wide_resnet50_2': torchvision.models.wide_resnet50_2,
  # 'wide_resnet101_2': torchvision.models.wide_resnet101_2,
  'mnasnet0_5': torchvision.models.mnasnet0_5,
  # 'mnasnet0_75': torchvision.models.mnasnet0_75,
  # 'mnasnet1_0': torchvision.models.mnasnet1_0,
  # 'mnasnet1_3': torchvision.models.mnasnet1_3
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
    dummy_input = torch.full((1, 3, 224, 224), 0.5)

    for i in range(NUM_DUMMY_TESTS):
      dummy_output = net(dummy_input)
      dummy_input_array = dummy_input.numpy()
      dummy_output_array = dummy_output.detach().numpy()

      test_data_path = os.path.join(model_base_path, f'test_data_set_{i}')
      os.mkdir(test_data_path)
      with open(os.path.join(test_data_path, 'input_0.pb'), 'wb') as outfile:
        inp_tensor = numpy_helper.from_array(dummy_input_array)
        outfile.write(inp_tensor.SerializeToString())

      with open(os.path.join(test_data_path, 'output_0.pb'), 'wb') as outfile:
        out_tensor = numpy_helper.from_array(dummy_output_array)
        outfile.write(out_tensor.SerializeToString())

    torch.onnx.export(net, dummy_input, model_onnx_path, \
                      verbose=False, input_names=input_names, output_names=output_names)