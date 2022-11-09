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

  You can export supported models using `AxonOnnx.export/4`:

      AxonOnnx.export(model, templates, params)
  """

  @doc """
  Imports an ONNX model from the given path.

  Some models support ONNX `dim_params` which you may specify
  by providing dimension names as a keyword list:

      AxonOnnx.import("model.onnx", batch: 1)

  The imported model will be in the form:

      {model, params} = AxonOnnx.import("model.onnx")
  """
  def import(path, dimensions \\ []) do
    path
    |> File.read!()
    |> AxonOnnx.Deserialize.__load__(dimensions)
  end

  @doc """
  Loads an ONNX model into an Axon model from the given binary.

  Some models support ONNX `dim_params` which you may specify
  by providing dimension names as a keyword list:
      
      onnx = File.read!("model.onnx")
      AxonOnnx.load(onnx, batch: 1)

  The imported model will be in the form:

      {model, params} = AxonOnnx.import(onnx)
  """
  def load(onnx, dimensions \\ []), do: AxonOnnx.Deserialize.__load__(onnx, dimensions)

  @doc """
  Exports an Axon model and parameters to an ONNX model
  with the given input templates.

  You may optionally specify a `path` to export a model to
  a specific file path:

      AxonOnnx.export(model, templates, params, path: "resnet.onnx")
  """
  def export(%Axon{} = model, templates, params, opts \\ []) do
    {encoded, output_name} = AxonOnnx.Serialize.__dump__(model, templates, params, opts)

    fname = opts[:path] || output_name <> ".onnx"

    {:ok, file} = File.open(fname, [:write])
    IO.binwrite(file, encoded)
    File.close(file)
  end

  @doc """
  Dumps an Axon model and parameters into a binary representing
  and ONNX model.
  """
  def dump(%Axon{} = model, templates, params, opts \\ []) do
    {encoded, _} = AxonOnnx.Serialize.__dump__(model, templates, params, opts)
    encoded
  end
end
