defmodule AxonOnnx.Deserialize do
  alias Onnx.ModelProto, as: Model
  alias Onnx.GraphProto, as: Graph
  alias Onnx.ValueInfoProto, as: Value
  alias Onnx.AttributeProto, as: Attribute
  alias Onnx.NodeProto, as: Node
  alias Onnx.TypeProto, as: Type
  alias Onnx.TensorProto, as: Tensor
  alias Onnx.TypeProto.Tensor, as: Placeholder
  alias Onnx.TensorShapeProto, as: Shape
  alias Onnx.TensorShapeProto.Dimension, as: Dimension

  require Logger

  # TODO(seanmor5): Currently we do a lot of potentially expensive operations
  # eagerly (especially when manipulating parameters), we can potentially make
  # them part of the model or alternatively return an initialization function
  # which can be JIT-compiled.

  # TODO(seanmor5): The current approach builds a lot of intermediate graphs,
  # instead we should only keep graphs which are specified as outputs and override
  # all other graphs so they are GC'ed

  # TODO(seanmor5): Some operations occur strictly on parameters (e.g. reshape, unsqueeze,
  # etc.), so we need to change all of these cases to handle instances where the only
  # input is a parameter which is an Nx expression rather than a model

  # TODO(seanmor5): Because some operations act on parameter inputs which don't have a
  # parameterized equivalent operation in Axon (e.g. add, multiply, etc.), we need
  # a way to implement them that still builds an Axon model but preserves the parameters

  # TODO(seanmor5): Because there are multiple versions of the protocol, there are also
  # multiple versions of each function. It's not that unreasonable to try to support every
  # version, but it just makes for a lot of annoying edge cases. Standardize around a minimum
  # supported version for guaranteed compatibility

  def __import__(file, opts \\ []) do
    file
    |> File.read!()
    |> Model.decode!()
    |> to_axon(opts)
  end

  defp to_axon(%Model{graph: %Graph{node: nodes} = graph}, opts) do
    dimensions = opts[:dimensions] || []
    dimensions = Enum.map(dimensions, &Atom.to_string/1)

    params = get_params(graph)
    inputs = get_inputs(graph, params, dimensions)
    outputs = get_outputs(graph)
    {nodes, params} = get_nodes(nodes, inputs, params, %{})
    {hd(Enum.map(outputs, fn name -> nodes[name] end)), params}
  end

  defp get_inputs(%Graph{input: inputs}, params, dimensions) do
    Enum.reduce(inputs, %{}, fn %Value{name: name, type: %Type{value: value}}, acc ->
      if Map.has_key?(params, name) do
        acc
      else
        case value do
          {:tensor_type, %Placeholder{} = tensor} ->
            input_shape = shape!(tensor, dimensions)

            input_shape =
              if tuple_size(input_shape) == 1,
                do: Tuple.insert_at(input_shape, 0, nil),
                else: input_shape

            Map.put(acc, name, Axon.input(input_shape))

          unsupported ->
            raise ArgumentError, "unsupported input type #{inspect(unsupported)}"
        end
      end
    end)
  end

  defp get_params(%Graph{initializer: initializer}) do
    Enum.reduce(initializer, %{}, fn %Tensor{name: name} = tensor, params ->
      Map.put(params, name, tensor!(tensor))
    end)
  end

  defp get_outputs(%Graph{output: outputs}) do
    Enum.map(outputs, fn %Value{name: name} -> name end)
  end

  defp get_nodes(pruned_nodes, inp, params, used_params) do
    Enum.reduce(pruned_nodes, {inp, used_params}, fn %Node{op_type: op_type} = op_node,
                                                     {axon, used_params} ->
      case op_type do
        "Abs" ->
          to_axon_nx(op_node, axon, params, used_params, &Nx.abs/1)

        "Acos" ->
          to_axon_nx(op_node, axon, params, used_params, &Nx.acos/1)

        "Acosh" ->
          to_axon_nx(op_node, axon, params, used_params, &Nx.acosh/1)

        "Add" ->
          to_axon_binary_op(op_node, axon, params, used_params, :add)

        "ArgMax" ->
          to_axon_reduction(op_node, axon, params, used_params, &Nx.argmax/2)

        "ArgMin" ->
          to_axon_reduction(op_node, axon, params, used_params, &Nx.argmin/2)

        "Asin" ->
          to_axon_nx(op_node, axon, params, used_params, &Nx.asin/1)

        "Asinh" ->
          to_axon_nx(op_node, axon, params, used_params, &Nx.asinh/1)

        "Atan" ->
          to_axon_nx(op_node, axon, params, used_params, &Nx.atan/1)

        "Atanh" ->
          to_axon_nx(op_node, axon, params, used_params, &Nx.atanh/1)

        "BatchNormalization" ->
          to_axon_batch_norm(op_node, axon, params, used_params)

        "Ceil" ->
          to_axon_nx(op_node, axon, params, used_params, &Nx.ceil/1)

        "Celu" ->
          to_axon_activation(op_node, axon, params, used_params, :celu, alpha: {"alpha", 1.0})

        "Constant" ->
          to_axon_constant(op_node, axon, params, used_params)

        "Concat" ->
          to_axon_concat(op_node, axon, params, used_params)

        "Conv" ->
          to_axon_conv(op_node, axon, params, used_params)

        "Cos" ->
          to_axon_nx(op_node, axon, params, used_params, &Nx.cos/1)

        "Cosh" ->
          to_axon_nx(op_node, axon, params, used_params, &Nx.cosh/1)

        "Div" ->
          to_axon_binary_op(op_node, axon, params, used_params, fn {x, y} -> Nx.divide(x, y) end)

        "Elu" ->
          to_axon_activation(op_node, axon, params, used_params, :elu, alpha: {"alpha", 1.0})

        "Equal" ->
          to_axon_binary_op(op_node, axon, params, used_params, fn {x, y} -> Nx.equal(x, y) end)

        "Erf" ->
          to_axon_nx(op_node, axon, params, used_params, &Nx.erf/1)

        "Exp" ->
          to_axon_activation(op_node, axon, params, used_params, :exp)

        "Flatten" ->
          to_axon_flatten(op_node, axon, params, used_params)

        "Floor" ->
          to_axon_nx(op_node, axon, params, used_params, &Nx.floor/1)

        "Gemm" ->
          to_axon_dense(op_node, axon, params, used_params)

        "GlobalAveragePool" ->
          to_axon_global_pool(op_node, axon, params, used_params)

        "GlobalLpPool" ->
          to_axon_global_pool(op_node, axon, params, used_params)

        "GlobalMaxPool" ->
          to_axon_global_pool(op_node, axon, params, used_params)

        "Greater" ->
          to_axon_binary_op(op_node, axon, params, used_params, fn {x, y} -> Nx.greater(x, y) end)

        "GreaterOrEqual" ->
          to_axon_binary_op(op_node, axon, params, used_params, fn {x, y} ->
            Nx.greater_equal(x, y)
          end)

        "HardSigmoid" ->
          to_axon_activation(op_node, axon, params, used_params, :hard_sigmoid,
            alpha: {"alpha", 0.2},
            beta: {"beta", 0.5}
          )

        "HardSwish" ->
          # TODO: Consider adding to Axon
          to_axon_nx(op_node, axon, params, used_params, fn x ->
            alpha = Nx.divide(1, 6)
            beta = Nx.tensor(0.5)

            alpha
            |> Nx.multiply(x)
            |> Nx.add(beta)
            |> Nx.min(1)
            |> Nx.max(0)
            |> Nx.multiply(x)
          end)

        "Identity" ->
          to_axon_nx(op_node, axon, params, used_params, & &1)

        "LeakyRelu" ->
          to_axon_activation(op_node, axon, params, used_params, :leaky_relu,
            alpha: {"alpha", 0.01}
          )

        "Less" ->
          to_axon_binary_op(op_node, axon, params, used_params, fn {x, y} -> Nx.less(x, y) end)

        "LessOrEqual" ->
          to_axon_binary_op(op_node, axon, params, used_params, fn {x, y} ->
            Nx.less_equal(x, y)
          end)

        "Log" ->
          to_axon_nx(op_node, axon, params, used_params, &Nx.log/1)

        "LogSoftmax" ->
          to_axon_activation(op_node, axon, params, used_params, :log_softmax, axis: {"axis", -1})

        "MatMul" ->
          to_axon_dense(op_node, axon, params, used_params)

        "Mod" ->
          # TODO(seanmor5): Support fmod option
          to_axon_binary_op(op_node, axon, params, used_params, fn {x, y} ->
            Nx.remainder(x, y)
          end)

        "Mul" ->
          to_axon_binary_op(op_node, axon, params, used_params, :multiply)

        "Neg" ->
          to_axon_nx(op_node, axon, params, used_params, &Nx.negate/1)

        "Not" ->
          to_axon_nx(op_node, axon, params, used_params, &Nx.logical_not/1)

        "Or" ->
          to_axon_binary_op(op_node, axon, params, used_params, fn {x, y} ->
            Nx.logical_or(x, y)
          end)

        "Pow" ->
          to_axon_binary_op(op_node, axon, params, used_params, fn {x, y} -> Nx.power(x, y) end)

        "ReduceMax" ->
          to_axon_reduction(op_node, axon, params, used_params, &Nx.reduce_max/2)

        "ReduceMin" ->
          to_axon_reduction(op_node, axon, params, used_params, &Nx.reduce_min/2)

        "ReduceProd" ->
          to_axon_reduction(op_node, axon, params, used_params, &Nx.product/2)

        "Relu" ->
          to_axon_activation(op_node, axon, params, used_params, :relu)

        "Reshape" ->
          to_axon_reshape(op_node, axon, params, used_params)

        "Round" ->
          to_axon_nx(op_node, axon, params, used_params, &Nx.round/1)

        "Selu" ->
          to_axon_activation(op_node, axon, params, used_params, :selu,
            alpha: {"alpha", 1.67326319217681884765625},
            gamma: {"gamma", 1.05070102214813232421875}
          )

        "Shape" ->
          to_axon_nx(op_node, axon, params, used_params, fn x ->
            x
            |> Nx.shape()
            |> Tuple.to_list()
            |> Nx.tensor(backend: Nx.Defn.Expr)
          end)

        "Sigmoid" ->
          to_axon_activation(op_node, axon, params, used_params, :sigmoid)

        "Sign" ->
          to_axon_nx(op_node, axon, params, used_params, &Nx.sign/1)

        "Sin" ->
          to_axon_nx(op_node, axon, params, used_params, &Nx.sin/1)

        "Sinh" ->
          to_axon_nx(op_node, axon, params, used_params, &Nx.sinh/1)

        "Size" ->
          to_axon_nx(op_node, axon, params, used_params, fn x ->
            x
            |> Nx.size()
            |> Nx.tensor(backend: Nx.Defn.Expr)
          end)

        "Softmax" ->
          to_axon_activation(op_node, axon, params, used_params, :softmax, axis: {"axis", -1})

        "Softplus" ->
          to_axon_activation(op_node, axon, params, used_params, :softplus)

        "Softsign" ->
          to_axon_activation(op_node, axon, params, used_params, :softsign)

        "Split" ->
          to_axon_split(op_node, axon, params, used_params)

        "Sqrt" ->
          to_axon_nx(op_node, axon, params, used_params, &Nx.sqrt/1)

        "Sub" ->
          to_axon_binary_op(op_node, axon, params, used_params, :subtract)

        "Tan" ->
          to_axon_nx(op_node, axon, params, used_params, &Nx.tan/1)

        "Tanh" ->
          to_axon_activation(op_node, axon, params, used_params, :tanh)

        "Transpose" ->
          to_axon_transpose(op_node, axon, params, used_params)

        "Unsqueeze" ->
          to_axon_unsqueeze(op_node, axon, params, used_params)

        "Upsample" ->
          to_axon_upsample(op_node, axon, params, used_params)

        "Xor" ->
          to_axon_binary_op(op_node, axon, params, used_params, fn {x, y} ->
            Nx.logical_xor(x, y)
          end)

        "MaxPool" ->
          to_axon_max_pool(op_node, axon, params, used_params)

        "Pad" ->
          to_axon_pad(op_node, axon, params, used_params)

        op ->
          raise "unsupported #{op} op in graph"
      end
    end)
  end

  # Builds a generic Nx layer by applying the given operation
  # to the input. Most of these functions are generic element-wise
  # operations such as Abs, Acos, etc.
  #
  # TODO(seanmor5): Replace with Axon.layer when we have better shape
  # inference
  defp to_axon_nx(%Node{input: [input], output: [output_name]}, axon, _params, used_params, fun) do
    axon_input = axon!(input, axon)
    updated_axon = Map.put(axon, output_name, Axon.nx(axon_input, fun, name: output_name))
    {updated_axon, used_params}
  end

  # Builds a generic Nx layer by applying the given reduction operation
  # to the input.
  #
  # TODO(seanmor5): Replace with Axon.layer when we have better shape
  # inference
  defp to_axon_reduction(
         %Node{input: [input], attribute: attrs, output: [output_name]},
         axon,
         _params,
         used_params,
         reduce_fun
       ) do
    reduce_options = options!(attrs)

    axes = reduce_options["axes"]
    keepdims = reduce_options["keepdims"]
    keep_axes = if keepdims == 1, do: true, else: false

    axon_input = axon!(input, axon)

    updated_axon =
      Map.put(
        axon,
        output_name,
        Axon.nx(axon_input, reduce_fun,
          name: output_name,
          opts: [axes: axes, keep_axes: keep_axes]
        )
      )

    {updated_axon, used_params}
  end

  # Builds an Axon dense layer from an ONNX MatMul or GEMM Node. MatMul
  # nodes do not account for bias (they're treated as a separate operation
  # in the graph). GEMM Nodes are a bit more in-depth.
  #
  # TODO(seanmor5): Handle alpha, beta attrs
  defp to_axon_dense(
         %Node{op_type: op_type, input: inputs, output: [output_name], attribute: attrs},
         axon,
         params,
         used_params
       ) do
    [input, weight | maybe_bias] = inputs

    input = axon!(input, axon)
    weight = param!(weight, params)

    case op_type do
      "MatMul" ->
        {_, units} = Nx.shape(weight)

        updated_axon =
          Map.put(
            axon,
            output_name,
            Axon.dense(input, units, use_bias: false, name: output_name)
          )

        updated_params = Map.put(used_params, output_name <> "_kernel", weight)
        {updated_axon, updated_params}

      "Gemm" ->
        dense_options = options!(attrs)

        # TODO(seanmor5): Handle alpha, beta
        _alpha = dense_options["alpha"]
        _beta = dense_options["beta"]

        trans_a = dense_options["transA"]
        trans_b = dense_options["transB"]

        input =
          if trans_a == 1 do
            Nx.transpose(input)
          else
            input
          end

        weight =
          if trans_b == 1 do
            Nx.transpose(weight)
          else
            weight
          end

        {_, units} = Nx.shape(weight)

        updated_axon =
          Map.put(
            axon,
            output_name,
            Axon.dense(input, units, use_bias: maybe_bias != [], name: output_name)
          )

        updated_params =
          if maybe_bias == [] do
            Map.put(used_params, output_name <> "_kernel", weight)
          else
            [bias] = maybe_bias
            bias = param!(bias, params)

            used_params
            |> Map.put(output_name <> "_kernel", weight)
            |> Map.put(output_name <> "_bias", bias)
          end

        {updated_axon, updated_params}
    end
  end

  # Builds an Axon layer from an element-wise binary operation. Binary
  # op is either an atom representing one of Axon's legitimate Binary op
  # layers, or a function to be used in a custom layer.
  #
  # TODO(seanmor5): Verify broadcasting semantics
  defp to_axon_binary_op(
         %Node{input: [x, y], output: [output_name]},
         axon,
         _params,
         used_params,
         binary_op
       ) do
    inp1 = axon!(x, axon)
    inp2 = axon!(y, axon)

    updated_axon =
      case binary_op do
        op when is_atom(op) ->
          Map.put(axon, output_name, apply(Axon, op, [inp1, inp2, [name: output_name]]))

        fun when is_function(fun, 2) ->
          # TODO(seanmor5): Use Axon.layer when shape inference improves
          Map.put(axon, output_name, Axon.nx({inp1, inp2}, fun, name: output_name))
      end

    {updated_axon, used_params}
  end

  defp to_axon_max_pool(
         %Node{op_type: "MaxPool", input: [inp], attribute: attrs, output: [output_name]},
         axon,
         _params,
         used_params
       ) do
    max_pool_options = options!(attrs)

    kernel_shape = max_pool_options["kernel_shape"]
    ceil_mode = max_pool_options["ceil_mode"] || 0
    auto_pad = max_pool_options["auto_pad"] || "NOTSET"
    storage_order = max_pool_options["storage_order"]
    pads = max_pool_options["pads"]
    strides = max_pool_options["strides"]
    dilations = max_pool_options["dilations"]

    # Kernel size is a list of integers
    kernel_size = List.to_tuple(kernel_shape)

    # Axon only supports default ceil_mode right now
    if ceil_mode != 0 do
      raise ArgumentError,
            "invalid ceil_mode #{inspect(ceil_mode)}, Axon only supports" <>
              " ceil_mode of 0"
    end

    # Storage Order is not an Axon concern
    if storage_order do
      Logger.warning(
        "Storage order is not supported by Axon and is instead a backend-specific" <>
          " detail. Your model might behave differently from the imported version if" <>
          " the storage order differs"
      )
    end

    # Axon default strides are equal to the kernel shape (Keras behavior)
    # where as strides default to 1 in ONNX
    strides =
      if strides do
        strides
      else
        List.duplicate(1, tuple_size(kernel_size))
      end

    # Compute padding from auto_pad and pads attributes
    padding_config = padding!(auto_pad, pads)

    inp = axon!(inp, axon)

    updated_axon =
      Map.put(
        axon,
        output_name,
        Axon.max_pool(inp,
          kernel_size: kernel_size,
          strides: strides,
          padding: padding_config,
          dilations: dilations,
          name: output_name
        )
      )

    {updated_axon, used_params}
  end

  defp to_axon_conv(%Node{op_type: "Conv"} = conv_node, axon, params, used_params) do
    %{attribute: attrs, input: input, output: [output_name]} = conv_node

    conv_options = options!(attrs)

    auto_pad = conv_options["auto_pad"]
    # dilations = conv_options["dilations"]
    group = conv_options["group"]
    kernel_shape = conv_options["kernel_shape"]
    pads = conv_options["pads"]
    strides = conv_options["strides"]

    padding_config = padding!(auto_pad, pads)
    kernel_size = List.to_tuple(kernel_shape)

    [inp, kernel | maybe_bias] = input

    axon_inp = axon!(inp, axon)

    # Parameters can either be embedded in the graph as constants or
    # passed as parameters
    {axon_kernel, units} =
      cond do
        Map.has_key?(params, kernel) ->
          kernel = params[kernel]
          {kernel, elem(Nx.shape(kernel), 0)}

        Map.has_key?(axon, kernel) ->
          %{opts: [value: kernel]} = axon[kernel]
          {kernel, elem(Nx.shape(kernel), 0)}

        true ->
          raise "unable to find kernel for conv"
      end

    updated_params = Map.put(used_params, output_name <> "_kernel", axon_kernel)

    updated_params =
      if maybe_bias == [] do
        updated_params
      else
        [bias] = maybe_bias
        axon_bias = params[bias]
        Map.put(updated_params, output_name <> "_bias", axon_bias)
      end

    updated_axon =
      Map.put(
        axon,
        output_name,
        Axon.conv(axon_inp, units,
          kernel_size: kernel_size,
          feature_group_size: group,
          padding: padding_config,
          strides: strides,
          use_bias: maybe_bias != [],
          name: output_name
        )
      )

    {updated_axon, updated_params}
  end

  defp to_axon_pad(
         %Node{op_type: "Pad", input: inputs, output: [output_name], attribute: attrs},
         axon,
         params,
         used_params
       ) do
    pad_options = options!(attrs)

    case pad_options["mode"] do
      "constant" ->
        :ok

      nil ->
        :ok

      mode ->
        raise "unsupported padding mode #{inspect(mode)}"
    end

    [data, pads | maybe_constant] = inputs

    inp = axon!(data, axon)
    # TODO(seanmor5): Pads should probably be scrubbed from the graph
    # and parameters
    pads = param!(pads, params)

    padding_config =
      pads.ints
      |> Enum.chunk_every(2)
      |> Enum.zip()

    constant_value =
      case maybe_constant do
        [] ->
          0

        [value] ->
          tensor!(value)
      end

    updated_axon =
      Map.put(axon, output_name, Axon.pad(inp, padding_config, constant_value, name: output_name))

    {updated_axon, used_params}
  end

  # TODO(seanmor5): Mean and variance
  defp to_axon_batch_norm(
         %Node{
           op_type: "BatchNormalization",
           input: [inp, gamma, beta, _mean, _var],
           output: [output_name]
         },
         axon,
         params,
         used_params
       ) do
    inp = axon!(inp, axon)

    gamma = param!(gamma, params)
    beta = param!(beta, params)

    updated_axon = Map.put(axon, output_name, Axon.batch_norm(inp, name: output_name))

    updated_params =
      used_params
      |> Map.put(output_name <> "_gamma", gamma)
      |> Map.put(output_name <> "_beta", beta)

    {updated_axon, updated_params}
  end

  # Builds an axon activation layer with the given activation function.
  # `activation` must be a legitimate Axon activation. `activation` functions
  # are all element-wise with 1 input. Optionally has activation options.
  defp to_axon_activation(
         %Node{attribute: attrs, input: [inp], output: [output_name]},
         axon,
         _params,
         used_params,
         activation,
         opts \\ []
       ) do
    attrs = options!(attrs)

    opts =
      Enum.map(opts, fn {k, {name, default}} ->
        if attrs[name] do
          {k, attrs[name]}
        else
          {k, default}
        end
      end)

    opts = [name: output_name] ++ opts
    inp = axon!(inp, axon)
    {Map.put(axon, output_name, Axon.activation(inp, activation, opts)), used_params}
  end

  # Builds an Axon layer which returns a new layer with input values 
  # concatenated on the given axis 
  defp to_axon_concat(
         %Node{attribute: attrs, input: inputs, output: [output_name]},
         axon,
         _params,
         used_params
       )
       when is_list(inputs) do
    inputs = for inp <- inputs, do: axon!(inp, axon)
    %{"axis" => axis} = options!(attrs)

    {Map.put(axon, output_name, Axon.concatenate(inputs, axis: axis, name: output_name)),
     used_params}
  end

  # Builds an Axon layer which returns a new layer upsampling the input
  # layer. The scale activation layer must contain 1.0 as the first two values
  # Each dimension value of the output layer is:
  # output_dimension = floor(input_dimension * scale)
  defp to_axon_upsample(
         %Node{attribute: attrs, input: [inp, scale], output: [output_name]},
         axon,
         params,
         used_params
       ) do
    %Axon{output_shape: shape} = inp = axon!(inp, axon)
    %{"mode" => mode} = options!(attrs)
    scale = param!(scale, params)

    # Ignoring the first two 1.0 values to obtain the same dimension of scale_values 
    [_, _ | shape] = Tuple.to_list(shape)

    # Converting mode from string to atom to ensure Axon init and predict works correctly
    method =
      cond do
        is_binary(mode) -> String.to_atom(mode)
        is_atom(mode) -> mode
        true -> raise ArgumentError, "unsupported mode type. Must be string or atom, got: #{mode}"
      end

    output_shape =
      case Nx.to_flat_list(scale) do
        [1.0, 1.0 | scale_values] ->
          scale_values
          |> Enum.zip_with(shape, fn x, y -> floor(x * y) end)
          |> List.to_tuple()

        [s1, s2 | _] ->
          raise ArgumentError,
                "unspported scale format, first two scale values must be 1, got #{s1} and #{s2}"
      end

    {Map.put(
       axon,
       output_name,
       Axon.resize(inp, output_shape, method: method, name: output_name)
     ), used_params}
  end

  defp to_axon_split(
         %Node{attribute: attrs, input: [inp], output: output_names},
         axon,
         _params,
         used_params
       ) do
    inp = axon!(inp, axon)
    %{"axis" => axis, "split" => split_sizes} = options!(attrs)

    split_layers = Axon.split(inp, split_sizes, axis: axis, name: output_names)

    updated_axon =
      Enum.reduce(Tuple.to_list(split_layers), axon, fn output, new_axon ->
        Map.put(new_axon, output.name, output)
      end)

    {updated_axon, used_params}
  end

  # Builds an Axon layer which returns a new layer with input values 
  # concatenated on the given axis 
  defp to_axon_concat(
         %Node{attribute: attrs, input: inputs, output: [output_name]},
         axon,
         params,
         used_params
       )
       when is_list(inputs) do
    inputs = for inp <- inputs, do: input_or_param!(inp, params, axon, used_params)
    %{"axis" => axis} = options!(attrs)

    {Map.put(axon, output_name, Axon.concatenate(inputs, axis: axis, name: output_name)),
     used_params}
  end

  # Builds an Axon layer which returns a new layer upsampling the input
  # layer. The scale activation layer must contain 1.0 as the first two values
  # Each dimension value of the output layer is:
  # output_dimension = floor(input_dimension * scale)
  defp to_axon_upsample(
         %Node{attribute: attrs, input: [inp, scale], output: [output_name]},
         axon,
         params,
         used_params
       ) do
    %Axon{output_shape: shape} = inp = input_or_param!(inp, params, axon, used_params)
    %{"mode" => mode} = options!(attrs)
    scale = input_or_param!(scale, params, axon, used_params)

    # Ignoring the first two 1.0 values to obtain the same dimension of scale_values 
    [_, _ | shape] = Tuple.to_list(shape)

    # Converting mode from string to atom to ensure Axon init and predict works correctly
    method =
      cond do
        is_binary(mode) -> String.to_atom(mode)
        is_atom(mode) -> mode
        true -> raise ArgumentError, "unsupported mode type. Must be string or atom, got: #{mode}"
      end

    output_shape =
      case Nx.to_flat_list(scale) do
        [1.0, 1.0 | scale_values] ->
          scale_values
          |> Enum.zip_with(shape, fn x, y -> floor(x * y) end)
          |> List.to_tuple()

        [s1, s2 | _] ->
          raise ArgumentError,
                "unspported scale format, first two scale values must be 1, got #{s1} and #{s2}"
      end

    {Map.put(
       axon,
       output_name,
       Axon.resize(inp, output_shape, method: method, name: output_name)
     ), used_params}
  end

  defp to_axon_split(
         %Node{attribute: attrs, input: [inp], output: output_names},
         axon,
         params,
         used_params
       ) do
    inp = input_or_param!(inp, params, axon, used_params)
    %{"axis" => axis, "split" => split_sizes} = options!(attrs)

    split_layers = Axon.split(inp, split_sizes, axis: axis, name: output_names)

    updated_axon =
      Enum.reduce(Tuple.to_list(split_layers), axon, fn output, new_axon ->
        Map.put(new_axon, output.name, output)
      end)

    {updated_axon, used_params}
  end

  defp to_axon_global_pool(
         %Node{op_type: op_type, attribute: attrs, input: [inp], output: [output_name]},
         axon,
         _params,
         used_params
       ) do
    inp = axon!(inp, axon)

    updated_axon =
      case op_type do
        "GlobalAveragePool" ->
          Map.put(axon, output_name, Axon.global_avg_pool(inp, name: output_name, keep_axes: true))

        "GlobalMaxPool" ->
          Map.put(axon, output_name, Axon.global_max_pool(inp, name: output_name, keep_axes: true))

        "GlobalLpPool" ->
          lp_pool_options = options!(attrs)

          Map.put(
            axon,
            output_name,
            Axon.global_lp_pool(inp, norm: lp_pool_options["p"], name: output_name)
          )
      end

    {updated_axon, used_params}
  end

  # Builds an Axon layer which returns a constant with the given
  # value. Constants are embedded in custom layers which just yield
  # the value of the constant here. They are not treated as parameters
  defp to_axon_constant(
         %Node{op_type: "Constant", attribute: attrs, output: [output_name]},
         axon,
         _,
         used_params
       ) do
    constant_options = options!(attrs)

    const =
      cond do
        constant_options["sparse_value"] ->
          raise ArgumentError, "sparse tensors are not supported"

        constant_options["value"] ->
          Axon.constant(tensor!(constant_options["value"]), namme: output_name)

        constant_options["value_float"] ->
          Axon.constant(Nx.tensor(constant_options["value_float"], type: {:f, 32}),
            name: output_name
          )

        constant_options["value_floats"] ->
          Axon.constant(Nx.tensor(constant_options["value_floats"], type: {:f, 32}),
            name: output_name
          )

        constant_options["value_int"] ->
          Axon.constant(Nx.tensor(constant_options["value_int"], type: {:s, 64}),
            name: output_name
          )

        constant_options["value_ints"] ->
          Axon.constant(Nx.tensor(constant_options["value_ints"], type: {:s, 64}),
            name: output_name
          )

        constant_options["value_string"] or constant_options["value_strings"] ->
          raise ArgumentError, "string tensors are not supported"

        true ->
          raise ArgumentError, "invalid constant tensor type"
      end

    updated_axon = Map.put(axon, output_name, const)

    {updated_axon, used_params}
  end

  defp to_axon_reshape(
         %Node{op_type: "Reshape", input: [inp], attribute: attrs, output: [output_name]},
         axon,
         _params,
         used_params
       ) do
    reshape_options = options!(attrs)

    inp = axon!(inp, axon)

    new_shape =
      reshape_options["shape"]
      |> List.to_tuple()

    {Map.put(axon, output_name, Axon.reshape(inp, new_shape, name: output_name)), used_params}
  end

  defp to_axon_flatten(
         %Node{op_type: "Flatten", input: [inp], output: [output_name]},
         axon,
         _params,
         used_params
       ) do
    inp = axon!(inp, axon)

    {Map.put(axon, output_name, Axon.flatten(inp, name: output_name)), used_params}
  end

  # Builds an Axon transpose layer. Transpose is given by
  # the perm option in Node attribute.
  defp to_axon_transpose(
         %Node{op_type: "Transpose", input: [input], attribute: attrs, output: [output_name]},
         axon,
         _params,
         used_params
       ) do
    inp = axon!(input, axon)

    transpose_options = options!(attrs)

    permutation = transpose_options["perm"]

    updated_axon = Map.put(axon, output_name, Axon.transpose(inp, permutation, name: output_name))

    {updated_axon, used_params}
  end

  # Builds an unsqueeze layer using a custom Nx layer with the given
  # input and axes.
  #
  # TODO(seanmor5): Use Axon.layer
  defp to_axon_unsqueeze(
         %Node{op_type: "Unsqueeze", input: [input], attribute: attrs, output: [output_name]},
         axon,
         _params,
         used_params
       ) do
    unsqueeze_options = options!(attrs)

    inp = axon!(input, axon)

    axes = unsqueeze_options["axes"]

    fun = fn input ->
      Enum.reduce(axes, input, fn axis, x -> Nx.new_axis(x, axis) end)
    end

    case inp do
      %Nx.Tensor{} = tensor ->
        updated_params = Map.put(used_params, output_name, fun.(tensor))
        {axon, updated_params}

      %Axon{} = model ->
        updated_axon = Map.put(axon, output_name, Axon.nx(model, fun, name: output_name))
        {updated_axon, used_params}
    end
  end

  # TODO(seanmor5): Handle segments
  def tensor!(%Tensor{data_type: dtype, dims: dims} = tensor) do
    shape = List.to_tuple(dims)

    case dtype do
      1 ->
        to_nx_tensor(tensor.float_data, tensor.raw_data, {:f, 32}, shape)

      2 ->
        to_nx_tensor(tensor.int32_data, tensor.raw_data, {:u, 8}, shape)

      3 ->
        to_nx_tensor(tensor.int32_data, tensor.raw_data, {:s, 8}, shape)

      4 ->
        to_nx_tensor(tensor.int32_data, tensor.raw_data, {:u, 16}, shape)

      5 ->
        to_nx_tensor(tensor.int32_data, tensor.raw_data, {:s, 16}, shape)

      6 ->
        to_nx_tensor(tensor.int32_data, tensor.raw_data, {:s, 32}, shape)

      7 ->
        to_nx_tensor(tensor.int64_data, tensor.raw_data, {:s, 64}, shape)

      8 ->
        raise "unsupported Nx tensor type: string"

      9 ->
        to_nx_tensor(tensor.int32_data, tensor.raw_data, {:u, 8}, shape)

      10 ->
        to_nx_tensor(tensor.int32_data, tensor.raw_data, {:f, 16}, shape)

      11 ->
        to_nx_tensor(tensor.double_data, tensor.raw_data, {:f, 64}, shape)

      12 ->
        to_nx_tensor(tensor.uint64_data, tensor.raw_data, {:u, 32}, shape)

      13 ->
        to_nx_tensor(tensor.uint64_data, tensor.raw_data, {:u, 64}, shape)

      14 ->
        # TODO(seanmor5): When complex is supported, tensor.float_data
        raise "unsupported Nx tensor type: C64"

      15 ->
        # TODO(seanmor5): When complex is supported, tensor.double_data
        raise "unsupported Nx tensor type: C128"

      16 ->
        to_nx_tensor([], tensor.raw_data, {:bf, 16}, shape)
    end
  end

  defp to_nx_tensor([], <<>>, _, _) do
    raise "unsupported empty Nx tensor"
  end

  defp to_nx_tensor([], raw, type, shape) do
    raw
    |> Nx.from_binary(type)
    |> Nx.reshape(shape)
  end

  defp to_nx_tensor(data, _, type, shape) do
    data
    |> Nx.tensor(type: type)
    |> Nx.reshape(shape)
  end

  # Retrieves value `name` from current built graphs, asserting
  # that the given input is a graph operation and not a parameter
  defp axon!(name, axon) do
    if Map.has_key?(axon, name) do
      axon[name]
    else
      raise CompileError,
            "unable to build model from ONNX graph, expected value #{name}" <>
              " to be a graph input, but it was not present in built" <>
              " graphs"
    end
  end

  defp param!(name, params) do
    if Map.has_key?(params, name) do
      params[name]
    else
      raise CompileError,
            "unable to build model from ONNX graph, expected value #{name}" <>
              " to be a parameter input, but it was not present in" <>
              " initializers"
    end
  end

  defp padding!(auto_pad, pads) do
    case auto_pad do
      val when val == "NOTSET" or val == nil ->
        if pads do
          pads
          |> Enum.chunk_every(2)
          |> Enum.zip()
        else
          :valid
        end

      val when val == "SAME_UPPER" or val == "SAME_LOWER" ->
        :same

      "VALID" ->
        :valid
    end
  end

  defp options!(attrs) when is_list(attrs) do
    Enum.reduce(attrs, %{}, fn %Attribute{type: type, name: name} = attr, options ->
      case type do
        :FLOAT ->
          Map.put(options, name, attr.f)

        :INT ->
          Map.put(options, name, attr.i)

        :STRING ->
          Map.put(options, name, attr.s)

        :TENSOR ->
          Map.put(options, name, attr.t)

        :GRAPH ->
          Map.put(options, name, attr.g)

        :SPARSE_TENSOR ->
          Map.put(options, name, attr.sparse_tensor)

        :TYPE_PROTO ->
          Map.put(options, name, attr.tp)

        :FLOATS ->
          Map.put(options, name, attr.floats)

        :INTS ->
          Map.put(options, name, attr.ints)

        :STRINGS ->
          Map.put(options, name, attr.strings)

        :TENSORS ->
          Map.put(options, name, attr.tensors)

        :GRAPHS ->
          Map.put(options, name, attr.graphs)

        :SPARSE_TENSORS ->
          Map.put(options, name, attr.sparse_tensors)

        :TYPE_PROTOS ->
          Map.put(options, name, attr.type_protos)
      end
    end)
  end

  defp shape!(%Placeholder{shape: %Shape{dim: dims}}, dim_params) do
    dims
    |> Enum.map(fn %Dimension{value: value} ->
      case value do
        {:dim_value, val} ->
          val

        {:dim_param, key} ->
          unless Map.has_key?(dim_params, key) do
            raise "dimension #{inspect(key)} not found in provided dimensions," <>
                    " you must specify unknown dimension shapes at import time"
          end

          dim_params[key]

        _ ->
          raise ArgumentError, "unsupported dimension type"
      end
    end)
    |> List.to_tuple()
  end
end
