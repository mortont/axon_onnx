defmodule DeserializeTest do
  use ExUnit.Case
  import OnnxTestHelper

  describe "node tests" do
    test "Abs" do
      check_onnx_test_case!("node", "test_abs")
    end

    test "Acos" do
      check_onnx_test_case!("node", "test_acos")
      # TODO: https://github.com/elixir-nx/axon/issues/184
      # check_onnx_test_case!("node", "test_acos_example")
    end

    test "Acosh" do
      check_onnx_test_case!("node", "test_acosh")
      # TODO: https://github.com/elixir-nx/axon/issues/184
      # check_onnx_test_case!("node", "test_acosh_example")
    end

    test "Asin" do
      check_onnx_test_case!("node", "test_asin")
      # TODO: https://github.com/elixir-nx/axon/issues/184
      # check_onnx_test_case!("node", "test_asin_example")
    end

    test "Asinh" do
      check_onnx_test_case!("node", "test_asinh")
      # TODO: https://github.com/elixir-nx/axon/issues/184
      # check_onnx_test_case!("node", "test_asinh_example")
    end

    test "Ceil" do
      check_onnx_test_case!("node", "test_ceil")
      # TODO: https://github.com/elixir-nx/axon/issues/184
      # check_onnx_test_case!("node", "test_ceil_example")
    end

    test "Celu" do
      check_onnx_test_case!("node", "test_celu")
      # TODO
      # check_onnx_test_case!("node", "test_celu_expanded")
    end

    test "Cos" do
      check_onnx_test_case!("node", "test_cos")
      # TODO: https://github.com/elixir-nx/axon/issues/184
      # check_onnx_test_case!("node", "test_cos_example")
    end

    test "Cosh" do
      check_onnx_test_case!("node", "test_cosh")
      # TODO: https://github.com/elixir-nx/axon/issues/184
      # check_onnx_test_case!("node", "test_cosh_example")
    end

    test "Erf" do
      check_onnx_test_case!("node", "test_erf")
    end

    test "Exp" do
      check_onnx_test_case!("node", "test_exp")
      # TODO: https://github.com/elixir-nx/axon/issues/184
      # check_onnx_test_case!("node", "test_exp_example")
    end

    test "Floor" do
      check_onnx_test_case!("node", "test_floor")
      # TODO: https://github.com/elixir-nx/axon/issues/184
      # check_onnx_test_case!("node", "test_floor_example")
    end

    test "HardSigmoid" do
      check_onnx_test_case!("node", "test_hardsigmoid")
      check_onnx_test_case!("node", "test_hardsigmoid_default")
    end

    test "HardSwish" do
      check_onnx_test_case!("node", "test_hardswish")
      check_onnx_test_case!("node", "test_hardswish_expanded")
    end

    test "Identity" do
      check_onnx_test_case!("node", "test_identity")
      # TODO: Sequence types
      # check_onnx_test_case!("node", "test_identity_sequence")
    end

    test "LeakyRelu" do
      check_onnx_test_case!("node", "test_leakyrelu")
      check_onnx_test_case!("node", "test_leakyrelu_default")
      # TODO: https://github.com/elixir-nx/axon/issues/184
      # check_onnx_test_case!("node", "test_leakyrelu_example")
    end

    test "Log" do
      check_onnx_test_case!("node", "test_log")
      # TODO: https://github.com/elixir-nx/axon/issues/184
      # check_onnx_test_case!("node", "test_log_example")
    end

    test "LogSoftmax" do
      check_onnx_test_case!("node", "test_logsoftmax_axis_0")
      # TODO: ReduceMax
      # check_onnx_test_case!("node", "test_logsoftmax_axis_0_expanded")
      check_onnx_test_case!("node", "test_logsoftmax_axis_1")
      # TODO: ReduceMax
      # check_onnx_test_case!("node", "test_logsoftmax_axis_1_expanded")
      check_onnx_test_case!("node", "test_logsoftmax_axis_2")
      # TODO: ReduceMax
      # check_onnx_test_case!("node", "test_logsoftmax_axis_2_expanded")
      check_onnx_test_case!("node", "test_logsoftmax_default_axis")
      # TODO: ReduceMax
      # check_onnx_test_case!("node", "test_logsoftmax_default_axis_expanded")
      check_onnx_test_case!("node", "test_logsoftmax_large_number")
      # TODO: ReduceMax
      # check_onnx_test_case!("node", "test_logsoftmax_large_number_expanded")
      check_onnx_test_case!("node", "test_logsoftmax_negative_axis")
      # TODO: ReduceMax
      # check_onnx_test_case!("node", "test_logsoftmax_negative_axis_expanded")
      check_onnx_test_case!("node", "test_logsoftmax_example_1")
      # TODO: ReduceMax
      # check_onnx_test_case!("node", "test_logsoftmax_example_1_expanded")
    end

    test "MaxPool" do
      check_onnx_test_case!("node", "test_maxpool_1d_default")
      check_onnx_test_case!("node", "test_maxpool_2d_default")
      # TODO: https://github.com/elixir-nx/axon/issues/185
      # check_onnx_test_case!("node", "test_maxpool_2d_dilations")
      check_onnx_test_case!("node", "test_maxpool_2d_pads")
      # TODO: Adjust pads for same lower behavior
      # check_onnx_test_case!("node", "test_maxpool_2d_same_lower")
      check_onnx_test_case!("node", "test_maxpool_2d_same_upper")
      check_onnx_test_case!("node", "test_maxpool_2d_strides")
      check_onnx_test_case!("node", "test_maxpool_2d_uint8")
      check_onnx_test_case!("node", "test_maxpool_2d_precomputed_pads")
      check_onnx_test_case!("node", "test_maxpool_2d_precomputed_same_upper")
      check_onnx_test_case!("node", "test_maxpool_2d_precomputed_strides")
      # TODO: ArgMax
      # check_onnx_test_case!("node", "test_maxpool_with_argmax_2d_precomputed_pads")
      # check_onnx_test_case!("node", "test_maxpool_with_argmax_2d_precomputed_strides")
      check_onnx_test_case!("node", "test_maxpool_3d_default")

      # TODO: Reevaluate this behavior
      assert_raise ArgumentError, ~r/invalid ceil_mode/, fn ->
        check_onnx_test_case!("node", "test_maxpool_2d_ceil")
      end
    end

    test "Neg" do
      check_onnx_test_case!("node", "test_neg")
      # TODO: https://github.com/elixir-nx/axon/issues/184
      # check_onnx_test_case!("node", "test_neg_example")
    end

    test "Not" do
      check_onnx_test_case!("node", "test_not_2d")
      check_onnx_test_case!("node", "test_not_3d")
      check_onnx_test_case!("node", "test_not_4d")
    end

    test "Relu" do
      check_onnx_test_case!("node", "test_relu")
    end

    test "Selu" do
      check_onnx_test_case!("node", "test_selu")
      check_onnx_test_case!("node", "test_selu_default")
      # TODO: https://github.com/elixir-nx/axon/issues/184
      # check_onnx_test_case!("node", "test_selu_example")
    end

    test "Sigmoid" do
      check_onnx_test_case!("node", "test_sigmoid")
      # TODO: https://github.com/elixir-nx/axon/issues/184
      # check_onnx_test_case!("node", "test_sigmoid_example")
    end

    test "Sign" do
      # TODO: https://github.com/elixir-nx/axon/issues/184
      # check_onnx_test_case!("node", "test_sign")
    end

    test "Sin" do
      check_onnx_test_case!("node", "test_sin")
      # TODO: https://github.com/elixir-nx/axon/issues/184
      # check_onnx_test_case!("node", "test_sin_example")
    end

    test "Sinh" do
      check_onnx_test_case!("node", "test_sinh")
      # TODO: https://github.com/elixir-nx/axon/issues/184
      # check_onnx_test_case!("node", "test_sinh_example")
    end

    test "Softmax" do
      check_onnx_test_case!("node", "test_softmax_axis_0")
      # TODO: ReduceMax
      # check_onnx_test_case!("node", "test_softmax_axis_0_expanded")
      check_onnx_test_case!("node", "test_softmax_axis_1")
      # TODO: ReduceMax
      # check_onnx_test_case!("node", "test_softmax_axis_1_expanded")
      check_onnx_test_case!("node", "test_softmax_axis_2")
      # TODO: ReduceMax
      # check_onnx_test_case!("node", "test_softmax_axis_2_expanded")
      check_onnx_test_case!("node", "test_softmax_default_axis")
      # TODO: ReduceMax
      # check_onnx_test_case!("node", "test_softmax_default_axis_expanded")
      check_onnx_test_case!("node", "test_softmax_large_number")
      # TODO: ReduceMax
      # check_onnx_test_case!("node", "test_softmax_large_number_expanded")
      check_onnx_test_case!("node", "test_softmax_negative_axis")
      # TODO: ReduceMax
      # check_onnx_test_case!("node", "test_softmax_negative_axis_expanded")
      check_onnx_test_case!("node", "test_softmax_example")
      # TODO: ReduceMax
      # check_onnx_test_case!("node", "test_softmax_example_expanded")
    end

    test "Softsign" do
      check_onnx_test_case!("node", "test_softsign")
      # TODO: https://github.com/elixir-nx/axon/issues/184
      # check_onnx_test_case!("node", "test_softsign_example")
    end

    test "Sqrt" do
      check_onnx_test_case!("node", "test_sqrt")
      # TODO: https://github.com/elixir-nx/axon/issues/184
      # check_onnx_test_case!("node", "test_sqrt_example")
    end

    test "Tan" do
      check_onnx_test_case!("node", "test_tan")
      # TODO: https://github.com/elixir-nx/axon/issues/184
      # check_onnx_test_case!("node", "test_tan_example")
    end

    test "Tanh" do
      check_onnx_test_case!("node", "test_tanh")
      # TODO: https://github.com/elixir-nx/axon/issues/184
      # check_onnx_test_case!("node", "test_tanh_example")
    end
  end
end
