defmodule AxonOnnx.Serialize do
  alias Onnx.ModelProto, as: Model
  alias Onnx.GraphProto, as: Graph
  alias Onnx.NodeProto, as: Node
  alias Onnx.ValueInfoProto, as: Value
  alias Onnx.AttributeProto, as: Attribute
  alias Onnx.OperatorSetIdProto, as: Opset
  alias Onnx.TypeProto, as: Type
  alias Onnx.TypeProto.Tensor, as: Placeholder
  alias Onnx.TensorShapeProto, as: Shape
  alias Onnx.TensorShapeProto.Dimension, as: Dimension

  @onnx_ir_version 3
  @onnx_opset_version 13
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
    initializers = to_initializers(params_or_initializers, param_names)

    # Parameters need to be specified as graph inputs as well
    updated_inputs =
      param_names
      |> Enum.reduce(
        inputs,
        fn x, acc ->
          param_value = to_value_info(x, Nx.shape(params_or_initializers[x]))
          [param_value | acc]
        end
      )

    %Graph{
      node: Enum.reverse(nodes),
      name: output_name,
      input: updated_inputs,
      output: [to_value_info(axon)],
      initializer: initializers
    }
  end

  defp to_onnx(%Axon{op: :input} = axon, inputs, param_names, nodes) do
    input_value = to_value_info(axon)
    {[input_value | inputs], param_names, nodes}
  end

  ## Linear

  defp to_onnx(
         %Axon{
           op: :dense,
           name: name,
           parent: %Axon{name: inp_name} = parent,
           params: params,
           opts: [use_bias: use_bias]
         },
         inputs,
         param_names,
         nodes
       ) do
    {inputs, param_names, nodes} = to_onnx(parent, inputs, param_names, nodes)

    %{name: k_name} = params["kernel"]

    {node_inputs, updated_param_names} =
      if use_bias do
        %{name: b_name} = params["bias"]
        {[inp_name, k_name, b_name], [k_name, b_name | param_names]}
      else
        {[inp_name, k_name], [k_name | param_names]}
      end

    node = %Node{
      input: node_inputs,
      output: [name],
      name: name,
      op_type: "Gemm"
    }

    {inputs, updated_param_names, [node | nodes]}
  end

  ## Convolution

  defp to_onnx(
         %Axon{
           op: :conv,
           name: name,
           parent: %Axon{name: inp_name} = parent,
           params: params,
           opts: opts
         },
         inputs,
         param_names,
         nodes
       ) do
    {inputs, param_names, nodes} = to_onnx(parent, inputs, param_names, nodes)

    use_bias = opts[:use_bias]
    strides = opts[:strides]
    padding = opts[:padding]

    strides_attr = to_attr("strides", :INTS, strides)

    padding_attr =
      case padding do
        :valid ->
          to_attr("auto_pad", :STRING, "VALID")

        :same ->
          to_attr("auto_pad", :STRING, "SAME_UPPER")

        padding when is_list(padding) ->
          {pad_begins, pad_ends} = Enum.unzip(padding)
          to_attr("pads", :INTS, pad_begins ++ pad_ends)
      end

    # TODO: Dilations

    %{name: k_name} = params["kernel"]

    {node_inputs, updated_param_names} =
      if use_bias do
        %{name: b_name} = params["bias"]
        {[inp_name, k_name, b_name], [k_name, b_name | param_names]}
      else
        {[inp_name, k_name], [k_name | param_names]}
      end

    node = %Node{
      input: node_inputs,
      output: [name],
      name: name,
      attribute: [strides_attr, padding_attr],
      op_type: "Conv"
    }

    {inputs, updated_param_names, [node | nodes]}
  end

  ## Activations

  @supported_activations [
    {:celu, "Celu"},
    {:elu, "Elu"},
    {:exp, "Exp"},
    {:hard_sigmoid, "HardSigmoid"},
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

  for {op, onnx_op} <- @supported_activations do
    defp to_onnx(
           %Axon{op: unquote(op), name: name, parent: %Axon{name: input_name} = parent},
           inputs,
           param_names,
           nodes
         ) do
      {inputs, param_names, nodes} = to_onnx(parent, inputs, param_names, nodes)

      node_inputs = [input_name]

      node = %Node{
        input: node_inputs,
        output: [name],
        name: name,
        op_type: unquote(onnx_op)
      }

      {inputs, param_names, [node | nodes]}
    end
  end

  defp to_attr(name, type, value) do
    case type do
      :INTS ->
        %Attribute{name: name, type: :INTS, ints: value}

      :STRING ->
        %Attribute{name: name, type: :STRING, s: value}
    end
  end

  defp to_initializers(params_or_initializers, param_names) do
    param_names
    |> Enum.map(fn param ->
      nx_to_tensor_proto(param, params_or_initializers[param])
    end)
  end

  defp to_value_info(%Axon{name: name, output_shape: shape}) do
    input_type = %Type{value: {:tensor_type, to_placeholder(shape)}}
    %Value{name: name, type: input_type}
  end

  defp to_value_info(param_name, shape) do
    input_type = %Type{value: {:tensor_type, to_placeholder(shape)}}
    %Value{name: param_name, type: input_type}
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

  defp nx_to_tensor_proto(param_name, tensor) do
    dims = Nx.shape(tensor) |> Tuple.to_list()
    # TODO: fix
    data_type = 1
    raw_data = Nx.to_binary(tensor)
    %Onnx.TensorProto{name: param_name, dims: dims, data_type: data_type, raw_data: raw_data}
  end
end
