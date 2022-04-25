defmodule AxonOnnx do
  @moduledoc """
  Library for converting to and from Axon/ONNX.
  """

  @doc """
  Imports an ONNX model from the given path.
  """
  def import(path, opts \\ []), do: AxonOnnx.Deserialize.__import__(path, opts)

  @doc """
  Exports an Axon model and parameters to an ONNX model.
  """
  def export(%Axon{} = model, params, opts \\ []),
    do: AxonOnnx.Serialize.__export__(model, params, opts)
end
