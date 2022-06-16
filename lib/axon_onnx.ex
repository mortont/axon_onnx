defmodule AxonOnnx do
  @moduledoc """
  Library for converting to and from Axon/ONNX.

  [ONNX](https://github.com/onnx/onnx) is a Neural Network specification
  supported by most popular deep learning frameworks such as PyTorch
  and TensorFlow. AxonOnnx allows you to convert to and from ONNX
  models via a simple import/export API.

  You can import supported ONNX models using `AxonOnnx.import/2`:

      {model, params} = AxonOnnx.import("model.onnx")

  `model` will be an Axon struct and `params` will be a compatible
  model state.

  You can export supported models using `AxonOnnx.export/3`:

      AxonOnnx.export(model, params)
  """

  @doc """
  Imports an ONNX model from the given path.

  Some models support ONNX `dim_params` which you may specify
  by providing dimension names as a keyword list:

      AxonOnnx.import("model.onnx", batch: 1)

  The imported model will be in the form:

      {model, params} = AxonOnnx.import("model.onnx")
  """
  def import(path, dimensions \\ []), do: AxonOnnx.Deserialize.__import__(path, dimensions)

  @doc """
  Exports an Axon model and parameters to an ONNX model.

  You may optionally specify a `path` to export a model to
  a specific file path:

      AxonOnnx.export(model, params, path: "resnet.onnx")


  """
  def export(%Axon{} = model, params, opts \\ []),
    do: AxonOnnx.Serialize.__export__(model, params, opts)
end
