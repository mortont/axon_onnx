defmodule AxonOnnxTest do
  use ExUnit.Case
  import OnnxTestHelper

  describe "serialize" do
    test "basic in/out" do
      model = Axon.input({1, 32})
      serialize_and_test_model!(model, num_tests: 3, name: "basic_in_out")
    end
  end

  describe "deserialize" do
    test "resnets" do
      for model <- [resnet(18)] do
        test_deserialized_model!(model)
      end
    end
  end
end
