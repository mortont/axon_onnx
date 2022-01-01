defmodule DeserializeTest do
  use ExUnit.Case
  import OnnxTestHelper

  describe "node tests" do
    test "abs" do
      check_onnx_test_case!("node", "test_abs")
    end

    test "acos" do
      check_onnx_test_case!("node", "test_acos")
      # TODO: https://github.com/elixir-nx/axon/issues/184
      # check_onnx_test_case!("node", "test_acos_example")
    end

    test "acosh" do
      check_onnx_test_case!("node", "test_acosh")
      # TODO: https://github.com/elixir-nx/axon/issues/184
      # check_onnx_test_case!("node", "test_acosh_example")
    end

    test "add" do
      check_onnx_test_case!("node", "test_add")
      # TODO: https://github.com/elixir-nx/axon/issues/184
      # check_onnx_test_case!("node", "test_add_bcast")
      check_onnx_test_case!("node", "test_add_uint8")
    end

    test "asin" do
      check_onnx_test_case!("node", "test_asin")
      # TODO: https://github.com/elixir-nx/axon/issues/184
      # check_onnx_test_case!("node", "test_asin_example")
    end
  end
end
