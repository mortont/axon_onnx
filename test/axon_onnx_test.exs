defmodule AxonOnnxTest do
  use ExUnit.Case
  import OnnxTestHelper

  @resnets [
    %OnnxModel{
      category: "vision",
      subcategory: "classification",
      name: "resnet",
      long_name: "resnet18",
      library: "",
      model_version: "1",
      onnx_version: "7"
    },
    %OnnxModel{
      category: "vision",
      subcategory: "classification",
      name: "resnet",
      long_name: "resnet34",
      library: "",
      model_version: "1",
      onnx_version: "7"
    }
  ]

  describe "models" do
    test "resnets" do
      for model <- @resnets do
        test_deserialized_model!(model)
      end
    end
  end
end
