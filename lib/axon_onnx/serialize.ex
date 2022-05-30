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
  def __export__(%Axon{} = axon, params, opts \\ []) do
    %Model{graph: %Graph{name: output_name}} = onnx_model = to_onnx_model(axon, params, opts)
    fname = opts[:filename] || output_name <> ".onnx"

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

  defp to_onnx_graph(%Axon{id: id, op: op, name: output_name_fn} = axon, params_or_initializers) do
    {inputs, param_names, nodes, op_counts, cache} = to_onnx(axon, [], [], [], %{}, %{})

    output_name =
      case cache do
        %{^id => name} ->
          name

        %{} ->
          output_name_fn.(op, op_counts)
      end

    # Flatten params_or_initializers so it's no longer nested
    # TODO: This is going to be expensive, find a better way
    params_or_initializers =
      params_or_initializers
      |> Enum.reduce(%{}, fn {layer_name, params}, acc ->
        params
        |> Enum.reduce(acc, fn {param_name, v}, acc ->
          Map.put(acc, layer_name <> "_" <> param_name, v)
        end)
      end)

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

    {output, _, _} = to_value_info(axon, op_counts, cache)

    %Graph{
      node: Enum.reverse(nodes),
      name: output_name,
      input: updated_inputs,
      output: [output],
      initializer: initializers
    }
  end

  defp to_onnx(%Axon{op: :input} = axon, inputs, param_names, nodes, op_counts, cache) do
    {input_value, op_counts, cache} = to_value_info(axon, op_counts, cache)
    {[input_value | inputs], param_names, nodes, op_counts, cache}
  end

  ## Linear

  defp to_onnx(
         %Axon{
           id: id,
           op: :dense,
           name: name_fn,
           parent: [%Axon{id: inp_id} = parent],
           parameters: params
         },
         inputs,
         param_names,
         nodes,
         op_counts,
         cache
       ) do
    {inputs, param_names, nodes, op_counts, cache} =
      to_onnx(parent, inputs, param_names, nodes, op_counts, cache)

    inp_name = cache[inp_id]

    {name, op_counts, cache} =
      case cache do
        %{^id => name} ->
          {name, op_counts, cache}

        %{} ->
          name = name_fn.(:dense, op_counts)
          op_counts = Map.update(op_counts, :dense, 1, fn x -> x + 1 end)
          cache = Map.put(cache, id, name)
          {name, op_counts, cache}
      end

    updated_param_names = Enum.map(params, fn %{name: p_name} ->
      name <> "_" <> p_name
    end)

    node = %Node{
      input: [inp_name | updated_param_names],
      output: [name],
      name: name,
      op_type: "Gemm"
    }

    {inputs, updated_param_names ++ param_names, [node | nodes], op_counts, cache}
  end

  ## Convolution

  defp to_onnx(
         %Axon{
           id: id,
           op: :conv,
           name: name_fn,
           parent: [%Axon{id: inp_id} = parent],
           parameters: params,
           opts: opts
         },
         inputs,
         param_names,
         nodes,
         op_counts,
         cache
       ) do
    {inputs, param_names, nodes, op_counts, cache} =
      to_onnx(parent, inputs, param_names, nodes, op_counts, cache)

    inp_name = cache[inp_id]

    {name, op_counts, cache} =
      case cache do
        %{^id => name} ->
          {name, op_counts, cache}

        %{} ->
          name = name_fn.(:conv, op_counts)
          op_counts = Map.update(op_counts, :conv, 1, fn x -> x + 1 end)
          cache = Map.put(cache, id, name)
          {name, op_counts, cache}
      end

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

    updated_param_names = Enum.map(params, fn %{name: p_name} ->
      name <> "_" <> p_name
    end)

    node = %Node{
      input: [inp_name | updated_param_names],
      output: [name],
      name: name,
      attribute: [strides_attr, padding_attr],
      op_type: "Conv"
    }

    {inputs, updated_param_names ++ param_names, [node | nodes], op_counts, cache}
  end

  ## Pooling

  @supported_pooling [:max_pool, :avg_pool, :lp_pool]

  defp to_onnx(
         %Axon{id: id, op: pool, name: name_fn, parent: [%Axon{id: inp_id} = parent], opts: opts},
         inputs,
         param_names,
         nodes,
         op_counts,
         cache
       )
       when pool in @supported_pooling do
    {inputs, param_names, nodes, op_counts, cache} =
      to_onnx(parent, inputs, param_names, nodes, op_counts, cache)

    inp_name = cache[inp_id]

    {name, op_counts, cache} =
      case cache do
        %{^id => name} ->
          {name, op_counts, cache}

        %{} ->
          name = name_fn.(pool, op_counts)
          op_counts = Map.update(op_counts, pool, 1, fn x -> x + 1 end)
          cache = Map.put(cache, id, name)
          {name, op_counts, cache}
      end

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

    {op_type, extra_attrs} =
      case pool do
        :lp_pool ->
          p_attr = to_attr("p", :INT, opts[:norm])
          {"LpPool", [p_attr]}

        :max_pool ->
          {"MaxPool", []}

        :avg_pool ->
          count_include_pad_attr = to_attr("count_include_pad", :INT, 1)
          {"AveragePool", [count_include_pad_attr]}
      end

    node_inputs = [inp_name]

    node = %Node{
      input: node_inputs,
      output: [name],
      name: name,
      attribute: [padding_attr, strides_attr, kernel_shape_attr | extra_attrs],
      op_type: op_type
    }

    {inputs, param_names, [node | nodes], op_counts, cache}
  end

  ## Global Pooling

  @supported_global_pooling [:global_avg_pool, :global_lp_pool, :global_max_pool]

  defp to_onnx(
         %Axon{
           id: id,
           op: pool,
           name: name_fn,
           parent: [%Axon{id: inp_id, output_shape: shape} = parent],
           opts: opts
         },
         inputs,
         param_names,
         nodes,
         op_counts,
         cache
       )
       when pool in @supported_global_pooling do
    {inputs, param_names, nodes, op_counts, cache} =
      to_onnx(parent, inputs, param_names, nodes, op_counts, cache)

    inp_name = cache[inp_id]

    {name, op_counts, cache} =
      case cache do
        %{^id => name} ->
          {name, op_counts, cache}

        %{} ->
          name = name_fn.(pool, op_counts)
          op_counts = Map.update(op_counts, pool, 1, fn x -> x + 1 end)
          cache = Map.put(cache, id, name)
          {name, op_counts, cache}
      end

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

    {inputs, param_names, nodes, op_counts, cache}
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
           %Axon{id: id, op: unquote(op), name: name_fn, parent: [%Axon{id: inp_id} = parent]},
           inputs,
           param_names,
           nodes,
           op_counts,
           cache
         ) do
      {inputs, param_names, nodes, op_counts, cache} =
        to_onnx(parent, inputs, param_names, nodes, op_counts, cache)

      input_name = cache[inp_id]

      {name, op_counts, cache} =
        case cache do
          %{^id => name} ->
            {name, op_counts, cache}

          %{} ->
            name = name_fn.(unquote(op), op_counts)
            op_counts = Map.update(op_counts, unquote(op), 1, fn x -> x + 1 end)
            cache = Map.put(cache, id, name)
            {name, op_counts, cache}
        end

      node_inputs = [input_name]

      node = %Node{
        input: node_inputs,
        output: [name],
        name: name,
        op_type: unquote(onnx_op)
      }

      {inputs, param_names, [node | nodes], op_counts, cache}
    end
  end

  ## Stochastic

  @supported_dropout_layers [:dropout, :spatial_droput, :feature_alpha_dropout, :alpha_dropout]

  defp to_onnx(
         %Axon{
           id: id,
           op: op,
           name: name_fn,
           parent: [%Axon{id: inp_id} = parent]
         },
         inputs,
         param_names,
         nodes,
         op_counts,
         cache
       )
       when op in @supported_dropout_layers do
    {inputs, param_names, nodes, op_counts, cache} =
      to_onnx(parent, inputs, param_names, nodes, op_counts, cache)

    input_name = cache[inp_id]

    {name, op_counts, cache} =
      case cache do
        %{^id => name} ->
          {name, op_counts, cache}

        %{} ->
          name = name_fn.(op, op_counts)
          op_counts = Map.update(op_counts, op, 1, fn x -> x + 1 end)
          cache = Map.put(cache, id, name)
          {name, op_counts, cache}
      end

    # For now just forward with an identity
    node = %Node{
      input: [input_name],
      output: [name],
      name: name,
      op_type: "Identity"
    }

    # Just forward to the next layer
    {inputs, param_names, [node | nodes], op_counts, cache}
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

  defp to_value_info(%Axon{id: id, op: op, name: name_fn, output_shape: shape}, op_counts, cache) do
    {name, op_counts, cache} =
      case cache do
        %{^id => name} ->
          {name, op_counts, cache}

        %{} ->
          name = name_fn.(op, op_counts)
          op_counts = Map.update(op_counts, op, 1, fn x -> x + 1 end)
          cache = Map.put(cache, id, name)
          {name, op_counts, cache}
      end

    input_type = %Type{value: {:tensor_type, to_placeholder(shape)}}
    {%Value{name: name, type: input_type}, op_counts, cache}
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
      |> Enum.map(fn
        nil ->
          %Dimension{value: {:dim_param, 1}}

        value ->
          %Dimension{value: {:dim_value, value}}
      end)

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
