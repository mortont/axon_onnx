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

  ## Pooling

  @supported_pooling [:max_pool, :avg_pool]

  defp to_onnx(
         %Axon{op: pool, name: name, parent: %Axon{name: inp_name} = parent, opts: opts},
         inputs,
         param_names,
         nodes
       )
       when pool in @supported_pooling do
    {inputs, param_names, nodes} = to_onnx(parent, inputs, param_names, nodes)

    kernel_size = opts[:kernel_size]
    strides = opts[:strides]
    padding = opts[:padding]

    strides_attr = to_attr("strides", :INTS, strides)
    kernel_shape_attr = to_attr("kernel_shape", :INTS, Tuple.to_list(kernel_size))

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

    {op_type, count_include_pad_attr} =
      case pool do
        :max_pool ->
          {"MaxPool", []}

        :avg_pool ->
          {"AveragePool", [to_attr("count_include_pad", :INT, 1)]}
      end

    node_inputs = [inp_name]

    node = %Node{
      input: node_inputs,
      output: [name],
      name: name,
      attribute: [padding_attr, strides_attr, kernel_shape_attr | count_include_pad_attr],
      op_type: op_type
    }

    {inputs, param_names, [node | nodes]}
  end

  ## Global Pooling

  @supported_global_pooling [:global_avg_pool, :global_lp_pool, :global_max_pool]

  defp to_onnx(
         %Axon{
           op: pool,
           name: name,
           parent: %Axon{name: inp_name, output_shape: shape} = parent,
           opts: opts
         },
         inputs,
         param_names,
         nodes
       )
       when pool in @supported_global_pooling do
    {inputs, param_names, nodes} = to_onnx(parent, inputs, param_names, nodes)

    keep_axes = opts[:keep_axes]

    {op_type, attrs} =
      case pool do
        :global_avg_pool ->
          {"GlobalAveragePool", []}

        :global_lp_pool ->
          {"GlobalLpPool", [to_attr("p", :INT, opts[:norm])]}

        :global_max_pool ->
          {"GlobalMaxPool", []}
      end

    node_inputs = [inp_name]

    nodes =
      if keep_axes do
        node = %Node{
          input: node_inputs,
          output: [name],
          name: name,
          attribute: attrs,
          op_type: op_type
        }

        [node | nodes]
      else
        pre_squeeze_name = name <> "_pre_squeeze"

        pre_squeeze_node = %Node{
          input: node_inputs,
          output: [pre_squeeze_name],
          name: pre_squeeze_name,
          attribute: attrs,
          op_type: op_type
        }

        constant_name = name <> "_squeeze_axes"
        axes = Enum.to_list(2..(Nx.rank(shape) - 1)//1)
        axes_tensor = nx_to_tensor_proto(constant_name, Nx.tensor(axes))
        value_attr = to_attr("value", :TENSOR, axes_tensor)

        constant_node = %Node{
          output: [constant_name],
          name: constant_name,
          attribute: [value_attr],
          op_type: "Constant"
        }

        node = %Node{
          input: [pre_squeeze_name, constant_name],
          output: [name],
          name: name,
          op_type: "Squeeze"
        }

        [node, constant_node, pre_squeeze_node | nodes]
      end

    {inputs, param_names, nodes}
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
      :INT ->
        %Attribute{name: name, type: :INT, i: value}

      :INTS ->
        %Attribute{name: name, type: :INTS, ints: value}

      :STRING ->
        %Attribute{name: name, type: :STRING, s: value}

      :TENSOR ->
        %Attribute{name: name, type: :TENSOR, t: value}
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
    data_type =
      case Nx.type(tensor) do
        {:f, 32} ->
          1

        {:s, 64} ->
          7
      end

    raw_data = Nx.to_binary(tensor)
    %Onnx.TensorProto{name: param_name, dims: dims, data_type: data_type, raw_data: raw_data}
  end
end
