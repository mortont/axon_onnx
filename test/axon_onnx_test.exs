defmodule AxonOnnxTest do
  use ExUnit.Case
  import OnnxTestHelper

  describe "serializes basic models" do
    test "basic in/out" do
      model = Axon.input({1, 32})
      serialize_and_test_model!(model, num_tests: 3, name: "basic_in_out")
    end
  end

  describe "serializes dense layers" do
    test "dense with defaults" do
      model = Axon.input({1, 32}) |> Axon.dense(10)
      serialize_and_test_model!(model, num_tests: 3, name: "default_dense")
    end

    test "dense with name" do
      model = Axon.input({1, 32}) |> Axon.dense(10, name: "dense")
      serialize_and_test_model!(model, num_tests: 3, name: "dense_with_name")
    end

    test "dense with use_bias false" do
      model = Axon.input({1, 32}) |> Axon.dense(10, name: "dense", use_bias: false)
      serialize_and_test_model!(model, num_tests: 3, name: "dense_no_bias")
    end

    test "dense with activation" do
      model = Axon.input({1, 32}) |> Axon.dense(10, activation: :relu)
      serialize_and_test_model!(model, num_tests: 3, name: "dense_with_activation")
    end

    test "multiple dense layers" do
      model =
        Axon.input({1, 32})
        |> Axon.dense(10, name: "dense_1")
        |> Axon.dense(1, name: "dense_2", use_bias: false)

      serialize_and_test_model!(model, num_tests: 3, name: "multi_dense")
    end
  end

  describe "serializes activations" do
    test "onnx supported activations" do
      # TODO: HardSigmoid implementation differs!
      supported_activations = [
        {:celu, "Celu"},
        {:elu, "Elu"},
        {:exp, "Exp"},
        {:leaky_relu, "LeakyRelu"},
        {:linear, "Identity"},
        {:relu, "Relu"},
        {:sigmoid, "Sigmoid"},
        {:selu, "Selu"},
        {:softmax, "Softmax"},
        {:softplus, "Softplus"},
        {:softsign, "Softsign"},
        {:tanh, "Tanh"}
      ]

      for {op, onnx_op} <- supported_activations do
        model = Axon.input({1, 10}) |> Axon.activation(op)
        serialize_and_test_model!(model, num_tests: 3, name: onnx_op)
      end
    end
  end

  describe "serializes convolution" do
    test "conv1d with defaults" do
      model = Axon.input({1, 3, 3}) |> Axon.conv(10)
      serialize_and_test_model!(model, num_tests: 3, name: "conv1d_defaults")
    end

    test "conv2d with defaults" do
      model = Axon.input({1, 3, 3, 3}) |> Axon.conv(10)
      serialize_and_test_model!(model, num_tests: 3, name: "conv2d_defaults")
    end

    test "conv3d with defaults" do
      model = Axon.input({1, 3, 3, 3, 3}) |> Axon.conv(10)
      serialize_and_test_model!(model, num_tests: 3, name: "conv3d_defaults")
    end

    test "conv with kernel" do
      model = Axon.input({1, 3, 8, 8}) |> Axon.conv(10, kernel_size: {2, 1})
      serialize_and_test_model!(model, num_tests: 3, name: "conv_with_kernel")
    end

    test "conv with strides" do
      model = Axon.input({1, 3, 6, 6}) |> Axon.conv(10, strides: [1, 2])
      serialize_and_test_model!(model, num_tests: 3, name: "conv_with_strides")
    end

    test "conv with same padding" do
      model = Axon.input({1, 3, 6, 6}) |> Axon.conv(10, padding: :same)
      serialize_and_test_model!(model, num_tests: 3, name: "conv_with_same_padding")
    end

    test "conv with padding config" do
      model = Axon.input({1, 3, 6, 6}) |> Axon.conv(10, padding: [{1, 1}, {0, 2}])
      serialize_and_test_model!(model, num_tests: 3, name: "conv_with_padding_config")
    end

    test "conv with activation" do
      model = Axon.input({1, 3, 7, 7}) |> Axon.conv(10, activation: :relu)
      serialize_and_test_model!(model, num_tests: 3, name: "conv_with_activation")
    end

    test "multiple conv layers" do
      model = Axon.input({1, 3, 7, 7}) |> Axon.conv(10, kernel_size: {2, 2}) |> Axon.conv(5)
      serialize_and_test_model!(model, num_tests: 3, name: "multi_conv")
    end
  end

  describe "serializes pooling layers" do
    test "max_pool1d with defaults" do
      model = Axon.input({1, 3, 7}) |> Axon.max_pool()
      serialize_and_test_model!(model, num_tests: 3, name: "max_pool1d_default")
    end

    test "max_pool2d with defaults" do
      model = Axon.input({1, 3, 7, 7}) |> Axon.max_pool()
      serialize_and_test_model!(model, num_tests: 3, name: "max_pool2d_defaults")
    end

    test "max_pool3d with defaults" do
      model = Axon.input({1, 3, 7, 7, 7}) |> Axon.max_pool()
      serialize_and_test_model!(model, num_tests: 3, name: "max_pool3d_defaults")
    end

    test "max_pool with kernel size" do
      model = Axon.input({1, 3, 7, 7}) |> Axon.max_pool(kernel_size: {2, 2})
      serialize_and_test_model!(model, num_tests: 3, name: "max_pool_kernel")
    end

    test "max_pool with strides" do
      model = Axon.input({1, 3, 7, 7}) |> Axon.max_pool(kernel_size: {2, 2}, stirdes: [1, 2])
      serialize_and_test_model!(model, num_tests: 3, name: "max_pool_strides")
    end

    test "max_pool with same padding" do
      model = Axon.input({1, 3, 7, 7}) |> Axon.max_pool(kernel_size: {2, 1}, padding: :same)
      serialize_and_test_model!(model, num_tests: 3, name: "max_pool_same_padding")
    end

    test "max_pool with padding config" do
      model =
        Axon.input({1, 3, 7, 7}) |> Axon.max_pool(kernel_size: {2, 2}, padding: [{1, 1}, {0, 1}])

      serialize_and_test_model!(model, num_tests: 3, name: "max_pool_padding_config")
    end

    test "max_pool with conv" do
      model =
        Axon.input({1, 3, 12, 12})
        |> Axon.conv(16, kernel_size: {2, 2})
        |> Axon.max_pool(kernel_size: {2, 2})

      serialize_and_test_model!(model, num_tests: 3, name: "max_pool_with_conv")
    end

    test "avg_pool1d with defaults" do
      model = Axon.input({1, 3, 7}) |> Axon.avg_pool()
      serialize_and_test_model!(model, num_tests: 3, name: "avg_pool1d_default")
    end

    test "avg_pool2d with defaults" do
      model = Axon.input({1, 3, 7, 7}) |> Axon.avg_pool()
      serialize_and_test_model!(model, num_tests: 3, name: "avg_pool2d_defaults")
    end

    test "avg_pool3d with defaults" do
      model = Axon.input({1, 3, 7, 7, 7}) |> Axon.avg_pool()
      serialize_and_test_model!(model, num_tests: 3, name: "avg_pool3d_defaults")
    end

    test "avg_pool with kernel size" do
      model = Axon.input({1, 3, 7, 7}) |> Axon.avg_pool(kernel_size: {2, 2})
      serialize_and_test_model!(model, num_tests: 3, name: "avg_pool_kernel")
    end

    test "avg_pool with strides" do
      model = Axon.input({1, 3, 7, 7}) |> Axon.avg_pool(kernel_size: {2, 2}, stirdes: [1, 2])
      serialize_and_test_model!(model, num_tests: 3, name: "avg_pool_strides")
    end

    test "avg_pool with same padding" do
      model = Axon.input({1, 3, 7, 7}) |> Axon.avg_pool(kernel_size: {2, 1}, padding: :same)
      serialize_and_test_model!(model, num_tests: 3, name: "avg_pool_same_padding")
    end

    test "avg_pool with padding config" do
      model =
        Axon.input({1, 3, 7, 7})
        |> Axon.avg_pool(kernel_size: {2, 2}, padding: [{1, 1}, {0, 1}])

      serialize_and_test_model!(model, num_tests: 3, name: "avg_pool_padding_config")
    end

    test "avg_pool with conv" do
      model =
        Axon.input({1, 3, 12, 12})
        |> Axon.conv(16, kernel_size: {2, 2})
        |> Axon.avg_pool(kernel_size: {2, 2})

      serialize_and_test_model!(model, num_tests: 3, name: "avg_pool_with_conv")
    end
  end

  describe "deserialize" do
    test "resnets" do
      resnets = [resnet(18)]

      for model <- resnets do
        test_deserialized_model!(model)
      end
    end
  end
end
