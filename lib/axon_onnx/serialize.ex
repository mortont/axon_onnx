defmodule AxonOnnx.Serialize do
  alias Onnx.ModelProto, as: Model
  alias Onnx.GraphProto, as: Graph
  alias Onnx.ValueInfoProto, as: Value
  alias Onnx.OperatorSetIdProto, as: Opset
  alias Onnx.TypeProto, as: Type
  alias Onnx.TypeProto.Tensor, as: Placeholder
  alias Onnx.TensorShapeProto, as: Shape
  alias Onnx.TensorShapeProto.Dimension, as: Dimension

  @onnx_ir_version 3
  @onnx_opset_version 8
  @producer_name "AxonOnnx"
  @producer_version "0.1.0-dev"

  # TODO(seanmor5): Multi-output models
  def __export__(%Axon{name: output_name} = axon, params, opts \\ []) do
    fname = opts[:filename] || output_name <> ".onnx"

    onnx_model = to_onnx_model(axon, params, opts)
    encoded = Model.encode!(onnx_model)

    {:ok, file} = File.open(fname, [:write])
    IO.binwrite(file, encoded)
    File.close(file)
  end

  defp to_onnx_model(axon, params, opts) do
    model_version = opts[:version] || 1
    doc_string = opts[:doc_string] || "An Axon Model"

    opset = %Opset{domain: "", version: @onnx_opset_version}

    graph = to_onnx_graph(axon, params)

    %Model{
      ir_version: @onnx_ir_version,
      producer_name: @producer_name,
      producer_version: @producer_version,
      domain: "",
      model_version: model_version,
      doc_string: doc_string,
      graph: graph,
      opset_import: [opset]
    }
  end

  defp to_onnx_graph(%Axon{name: output_name} = axon, params_or_initializers) do
    {inputs, param_names, nodes} = to_onnx(axon, [], [], [])
    # Building the initializers with Tensors will result in a bunch of expensive
    # copies, so we instead accumulate names and then use them to build initializers
    # later
    # initializers = to_initializers(params_or_initializers, param_names)

    %Graph{
      node: nodes,
      name: output_name,
      input: inputs,
      output: [to_value_info(axon)]
    }
  end

  defp to_onnx(%Axon{op: :input} = axon, inputs, param_names, nodes) do
    input_value = to_value_info(axon)
    {[input_value | inputs], param_names, nodes}
  end

  defp to_value_info(%Axon{name: name, output_shape: shape}) do
    input_type = %Type{value: {:tensor_type, to_placeholder(shape)}}
    %Value{name: name, type: input_type}
  end

  defp to_placeholder(shape) do
    %Placeholder{shape: to_tensor_shape_proto(shape), elem_type: 1}
  end

  defp to_tensor_shape_proto(shape) do
    dims =
      shape
      |> Tuple.to_list()
      |> Enum.map(&%Dimension{value: {:dim_value, &1}})

    %Shape{dim: dims}
  end
end
