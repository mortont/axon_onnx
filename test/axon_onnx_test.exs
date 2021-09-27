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
        model = Axon.input({1, 32}) |> Axon.activation(op)
        serialize_and_test_model!(model, num_tests: 3, name: onnx_op)
      end
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
