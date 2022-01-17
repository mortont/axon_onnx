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

  defp to_axon(%Model{graph: %Graph{} = graph}, opts) do
    dimensions = opts[:dimensions] || []

    {graph, params} = graph_to_axon(graph, dimensions)

    case graph do
      [graph] ->
        # single-output
        {graph, params}

      graph when is_list(graph) ->
        # multi-output
        {List.to_tuple(graph), params}
    end
  end

  def graph_to_axon(%Graph{node: nodes} = graph, dimensions) do
    params = get_params(graph)
    inputs = get_inputs(graph, params, dimensions)
    outputs = get_outputs(graph)
    {nodes, params} = get_nodes(nodes, inputs, params, %{})
    {Enum.map(outputs, fn name -> nodes[name] end), params}
  end

  defp get_inputs(%Graph{input: inputs}, params, dimensions) do
    Enum.reduce(inputs, %{}, fn %Value{name: name, type: %Type{value: value}}, acc ->
      if Map.has_key?(params, name) do
        acc
      else
        case value do
          {:tensor_type, %Placeholder{} = tensor} ->
            input_shape = shape!(tensor, dimensions)
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

        "And" ->
          to_axon_binary_op(op_node, axon, params, used_params, &Nx.logical_and/2)

        "ArgMax" ->
          to_axon_reduction(op_node, axon, params, used_params, &Nx.argmax/2, :axis)

        "ArgMin" ->
          to_axon_reduction(op_node, axon, params, used_params, &Nx.argmin/2, :axis)

        "Asin" ->
          to_axon_nx(op_node, axon, params, used_params, &Nx.asin/1)

        "Asinh" ->
          to_axon_nx(op_node, axon, params, used_params, &Nx.asinh/1)

        "Atan" ->
          to_axon_nx(op_node, axon, params, used_params, &Nx.atan/1)

        "Atanh" ->
          to_axon_nx(op_node, axon, params, used_params, &Nx.atanh/1)

        "AveragePool" ->
          to_axon_avg_pool(op_node, axon, params, used_params)

        "BatchNormalization" ->
          to_axon_batch_norm(op_node, axon, params, used_params)

        "BitShift" ->
          %Node{attribute: attrs} = op_node

          to_axon_binary_op(op_node, axon, params, used_params, fn x, y ->
            shift_options = options!(attrs)

            case shift_options["direction"] do
              "LEFT" ->
                Nx.left_shift(x, y)

              "RIGHT" ->
                Nx.right_shift(x, y)
            end
          end)

        "Cast" ->
          to_axon_cast(op_node, axon, params, used_params)

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
          to_axon_binary_op(op_node, axon, params, used_params, &Nx.divide/2)

        "Elu" ->
          to_axon_activation(op_node, axon, params, used_params, :elu, alpha: {"alpha", 1.0})

        "Equal" ->
          to_axon_binary_op(op_node, axon, params, used_params, &Nx.equal/2)

        "Erf" ->
          to_axon_nx(op_node, axon, params, used_params, &Nx.erf/1)

        "Exp" ->
          to_axon_activation(op_node, axon, params, used_params, :exp)

        "Flatten" ->
          to_axon_flatten(op_node, axon, params, used_params)

        "Floor" ->
          to_axon_nx(op_node, axon, params, used_params, &Nx.floor/1)

        "Gather" ->
          to_axon_gather(op_node, axon, params, used_params)

        "Gemm" ->
          to_axon_dense(op_node, axon, params, used_params)

        "GlobalAveragePool" ->
          to_axon_global_pool(op_node, axon, params, used_params)

        "GlobalLpPool" ->
          to_axon_global_pool(op_node, axon, params, used_params)

        "GlobalMaxPool" ->
          to_axon_global_pool(op_node, axon, params, used_params)

        "Greater" ->
          to_axon_binary_op(op_node, axon, params, used_params, &Nx.greater/2)

        "GreaterOrEqual" ->
          to_axon_binary_op(op_node, axon, params, used_params, &Nx.greater_equal/2)

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

        "If" ->
          to_axon_cond(op_node, axon, params, used_params)

        "LeakyRelu" ->
          to_axon_activation(op_node, axon, params, used_params, :leaky_relu,
            alpha: {"alpha", 0.01}
          )

        "Less" ->
          to_axon_binary_op(op_node, axon, params, used_params, &Nx.less/2)

        "LessOrEqual" ->
          to_axon_binary_op(op_node, axon, params, used_params, &Nx.less_equal/2)

        "Log" ->
          to_axon_nx(op_node, axon, params, used_params, &Nx.log/1)

        "LogSoftmax" ->
          to_axon_activation(op_node, axon, params, used_params, :log_softmax, axis: {"axis", -1})

        "LRN" ->
          to_axon_lrn(op_node, axon, params, used_params)

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
          to_axon_binary_op(op_node, axon, params, used_params, &Nx.logical_or/2)

        "Pow" ->
          to_axon_binary_op(op_node, axon, params, used_params, &Nx.power/2)

        "ReduceMax" ->
          to_axon_reduction(op_node, axon, params, used_params, &Nx.reduce_max/2, :axes)

        "ReduceMean" ->
          to_axon_reduction(op_node, axon, params, used_params, &Nx.mean/2, :axes)

        "ReduceMin" ->
          to_axon_reduction(op_node, axon, params, used_params, &Nx.reduce_min/2, :axes)

        "ReduceProd" ->
          to_axon_reduction(op_node, axon, params, used_params, &Nx.product/2, :axes)

        "ReduceLogSum" ->
          # TODO: I think there is a stability problem here
          to_axon_reduction(op_node, axon, params, used_params, &Nx.log(Nx.sum(&1, &2)), :axes)

        "ReduceLogSumExp" ->
          # TODO: I think there is a stability problem here
          to_axon_reduction(
            op_node,
            axon,
            params,
            used_params,
            fn x, opts ->
              x
              |> Nx.exp()
              |> Nx.sum(opts)
              |> Nx.log()
            end,
            :axes
          )

        "ReduceSumSquare" ->
          to_axon_reduction(
            op_node,
            axon,
            params,
            used_params,
            &Nx.sum(Nx.power(&1, 2), &2),
            :axes
          )

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

        "Sum" ->
          to_axon_sum(op_node, axon, params, used_params)

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
          to_axon_binary_op(op_node, axon, params, used_params, &Nx.logical_xor/2)

        "MaxPool" ->
          to_axon_max_pool(op_node, axon, params, used_params)

        "Pad" ->
          to_axon_pad(op_node, axon, params, used_params)

        op ->
          raise "unsupported #{op} op in graph"
      end
    end)
  end

  defp to_axon_cast(
         %Node{op_type: "Cast", attribute: attrs, input: [input], output: [output_name]},
         axon,
         _params,
         used_params
       ) do
    cast_options = options!(attrs)
    inp = axon!(input, axon)

    fun = fn x ->
      case cast_options["to"] do
        1 ->
          Nx.as_type(x, {:f, 32})

        2 ->
          Nx.as_type(x, {:u, 8})

        3 ->
          Nx.as_type(x, {:s, 8})

        4 ->
          Nx.as_type(x, {:u, 16})

        5 ->
          Nx.as_type(x, {:s, 16})

        6 ->
          Nx.as_type(x, {:s, 32})

        7 ->
          Nx.as_type(x, {:s, 64})

        8 ->
          raise ArgumentError, "unsupported STRING type"

        9 ->
          raise ArgumentError, "unsupported BOOL type"

        10 ->
          Nx.as_type(x, {:f, 16})

        11 ->
          Nx.as_type(x, {:f, 64})

        12 ->
          Nx.as_type(x, {:u, 32})

        13 ->
          Nx.as_type(x, {:u, 64})

        14 ->
          raise ArgumentError, "unsupported COMPLEX type"

        15 ->
          raise ArgumentError, "unsupported COMPLEX type"

        16 ->
          Nx.as_type(x, {:bf, 16})
      end
    end

    updated_axon = Map.put(axon, output_name, Axon.nx(inp, fun, name: output_name))
    {updated_axon, used_params}
  end

  defp to_axon_lrn(
         %Node{input: [input], attribute: attrs, output: [output_name]},
         axon,
         _params,
         used_params
       ) do
    inp = axon!(input, axon)
    lrn_options = options!(attrs)

    alpha = lrn_options["alpha"] || 0.0001
    beta = lrn_options["beta"] || 0.75
    bias = lrn_options["bias"] || 1.0
    size = lrn_options["size"]

    axes = Enum.to_list(0..(size - 1))

    fun = fn x ->
      squares = Nx.power(x, 2)
      sum_squares = Nx.sum(squares, axes: axes, keep_axes: true)
      denom = Nx.power(Nx.add(bias, Nx.divide(alpha, Nx.multiply(size, sum_squares))), beta)
      Nx.divide(x, denom)
    end

    updated_axon = Map.put(axon, output_name, Axon.nx(inp, fun, name: output_name))
    {updated_axon, used_params}
  end

  defp to_axon_gather(
         %Node{op_type: "Gather", input: [x, ind], output: [output_name], attribute: attrs},
         axon,
         _params,
         used_params
       ) do
    gather_options = options!(attrs)

    axis = gather_options["axis"]

    %Axon{output_shape: shape} = inp = axon!(x, axon)
    inp_names = List.duplicate(nil, Nx.rank(shape))
    %Axon{output_shape: indices_shape} = indices = axon!(ind, axon)
    ind_names = List.duplicate(nil, Nx.rank(indices_shape))
    output_shape = Nx.Shape.take(shape, inp_names, indices_shape, ind_names, axis)

    fun = fn x, indices ->
      Nx.take(x, indices, axis: axis)
    end

    layer = Axon.layer([inp, indices], fun, output_shape, %{}, output_name)
    updated_axon = Map.put(axon, output_name, layer)
    {updated_axon, used_params}
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

  # Builds a generic reduction layer by applying the given reduction operation
  # to the input in a custom layer.
  defp to_axon_reduction(
         %Node{input: [input], attribute: attrs, output: [output_name]},
         axon,
         _params,
         used_params,
         reduce_fun,
         axis_or_axes
       ) do
    reduce_options = options!(attrs)

    %Axon{output_shape: shape} = axon_input = axon!(input, axon)

    keepdims = reduce_options["keepdims"] || 1
    keep_axes = if keepdims == 1, do: true, else: false

    axes =
      if axis_or_axes == :axis do
        axis = reduce_options["axis"] || 0
        Nx.Shape.normalize_axis(shape, axis, List.duplicate(nil, Nx.rank(shape) - 1))
      else
        axes = reduce_options["axes"] || Nx.axes(shape)
        Nx.Shape.normalize_axes(shape, axes, List.duplicate(nil, Nx.rank(shape) - 1))
      end

    opts =
      if axis_or_axes == :axis do
        last_index = reduce_options["select_last_index"] || 0
        tie_break = if last_index == 0, do: :low, else: :high
        [keep_axis: keep_axes, axis: axes, tie_break: tie_break]
      else
        [keep_axes: keep_axes, axes: axes]
      end

    out_shape =
      if keep_axes do
        Enum.reduce(List.wrap(axes), shape, fn x, shape -> put_elem(shape, x, 1) end)
      else
        shape = for i <- Nx.axes(shape), i not in List.wrap(axes), do: i
        List.to_tuple(shape)
      end

    layer = Axon.layer(axon_input, reduce_fun, out_shape, %{}, output_name, opts)

    updated_axon = Map.put(axon, output_name, layer)

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
    %Axon{output_shape: s1} = inp1 = axon!(x, axon)
    %Axon{output_shape: s2} = inp2 = axon!(y, axon)

    updated_axon =
      case binary_op do
        op when is_atom(op) ->
          Map.put(axon, output_name, apply(Axon, op, [inp1, inp2, [name: output_name]]))

        fun when is_function(fun, 2) ->
          # TODO: Must fix Axon.layer with no parameters
          out_shape = Axon.Shape.element_wise([s1, s2])

          Map.put(
            axon,
            output_name,
            Axon.layer([inp1, inp2], fun, out_shape, %{}, name: output_name)
          )
      end

    {updated_axon, used_params}
  end

  defp to_axon_sum(%Node{input: inputs, output: [output_name]}, axon, _params, used_params) do
    axons = for input <- inputs, do: axon!(input, axon)
    updated_axon = Map.put(axon, output_name, Axon.add(axons, name: output_name))
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

  defp to_axon_avg_pool(
         %Node{op_type: "AveragePool", input: [inp], attribute: attrs, output: [output_name]},
         axon,
         _params,
         used_params
       ) do
    avg_pool_options = options!(attrs)

    kernel_shape = avg_pool_options["kernel_shape"]
    ceil_mode = avg_pool_options["ceil_mode"] || 0
    auto_pad = avg_pool_options["auto_pad"] || "NOTSET"
    _count_include_pad = avg_pool_options["count_include_pad"] || 0
    pads = avg_pool_options["pads"]
    strides = avg_pool_options["strides"]
    dilations = avg_pool_options["dilations"]

    # Kernel size is a list of integers
    kernel_size = List.to_tuple(kernel_shape)

    # Axon only supports default ceil_mode right now
    if ceil_mode != 0 do
      raise ArgumentError,
            "invalid ceil_mode #{inspect(ceil_mode)}, Axon only supports" <>
              " ceil_mode of 0"
    end

    # Axon only supports count_include_pad == 1
    # if count_include_pad != 1 do
    #   raise ArgumentError, "invalid count_include_pad #{inspect(count_include_pad)}," <>
    #                           " Axon only supports mode 1"
    # end

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
        Axon.avg_pool(inp,
          kernel_size: kernel_size,
          strides: strides,
          padding: padding_config,
          dilations: dilations,
          name: output_name
        )
      )

    {updated_axon, used_params}
  end

  defp to_axon_conv(
         %Node{op_type: "Conv", attribute: attrs, input: input, output: [output_name]},
         axon,
         params,
         used_params
       ) do
    conv_options = options!(attrs)

    kernel_shape = conv_options["kernel_shape"]
    auto_pad = conv_options["auto_pad"] || "NOTSET"
    dilations = conv_options["dilations"]
    group = conv_options["group"]
    pads = conv_options["pads"]
    strides = conv_options["strides"]

    padding_config = padding!(auto_pad, pads)

    [inp, kernel | maybe_bias] = input

    axon_inp = axon!(inp, axon)
    kernel = param!(kernel, params)

    {units, kernel_size} =
      if kernel_shape do
        full_shape = Nx.shape(kernel)
        units = elem(full_shape, 0)

        shape =
          full_shape
          |> Tuple.delete_at(0)
          |> Tuple.delete_at(0)

        {units, shape}
      else
        full_shape = Nx.shape(kernel)
        units = elem(full_shape, 0)

        shape =
          full_shape
          |> Tuple.delete_at(0)
          |> Tuple.delete_at(0)

        {units, shape}
      end

    updated_params =
      if maybe_bias == [] do
        Map.put(used_params, output_name, %{"kernel" => kernel})
      else
        [bias] = maybe_bias
        bias = param!(bias, params)
        Map.put(used_params, output_name, %{"kernel" => kernel, "bias" => bias})
      end

    updated_axon =
      Map.put(
        axon,
        output_name,
        Axon.conv(
          axon_inp,
          units,
          kernel_size: kernel_size,
          feature_group_size: group,
          kernel_dilation: dilations,
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

    updated_params = Map.put(used_params, output_name, %{"gamma" => gamma, "beta" => beta})

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

    updated_axon =
      Map.put(axon, output_name, Axon.concatenate(inputs, axis: axis, name: output_name))

    {updated_axon, used_params}
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
          Map.put(
            axon,
            output_name,
            Axon.global_avg_pool(inp, name: output_name, keep_axes: true)
          )

        "GlobalMaxPool" ->
          Map.put(
            axon,
            output_name,
            Axon.global_max_pool(inp, name: output_name, keep_axes: true)
          )

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

  # TODO: This currently won't pass any Node tests because reshape
  # value is read in as an input, how do we handle that?
  defp to_axon_reshape(
         %Node{op_type: "Reshape", input: [inp, shape], attribute: attrs, output: [output_name]},
         axon,
         params,
         used_params
       ) do
    reshape_options = options!(attrs)

    allowzero = reshape_options["allowzero"] || 0

    inp = axon!(inp, axon)
    # Reshape is a constant value input that MUST be known
    # ahead of time so we can build a static graph, we can't
    # support any other reshape types
    shape = constant!(shape, axon, params)

    # We currently do not support zero sized dimensions
    if allowzero == 1 do
      Logger.warning(
        "Nx does not support zero-sized dimensions. If your reshape" <>
          " operation contains a zero-sized dimension, it will fail"
      )
    end

    new_shape =
      shape
      |> Nx.to_flat_list()
      |> List.to_tuple()

    {Map.put(
       axon,
       output_name,
       Axon.reshape(inp, new_shape, name: output_name, ignore_batch?: false)
     ), used_params}
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
  # the perm option in Node attribute. ONNX does not ignore
  # batch dimensions, so that option is always false.
  defp to_axon_transpose(
         %Node{op_type: "Transpose", input: [input], attribute: attrs, output: [output_name]},
         axon,
         _params,
         used_params
       ) do
    transpose_options = options!(attrs)

    %Axon{output_shape: shape} = inp = axon!(input, axon)

    rank = Nx.rank(shape)

    permutation = transpose_options["perm"] || Enum.to_list((rank - 1)..0//-1)

    updated_axon =
      Map.put(
        axon,
        output_name,
        Axon.transpose(inp, permutation, name: output_name, ignore_batch?: false)
      )

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

  # Builds a conditional `If` layer.
  defp to_axon_cond(
         %Node{op_type: "If", input: [input], attribute: attrs, output: outputs},
         axon,
         _params,
         used_params
       ) do
    cond_options = options!(attrs)

    inp = axon!(input, axon)

    else_branch = cond_options["else_branch"]
    then_branch = cond_options["then_branch"]

    # TODO: Don't match
    {[else_graph], else_params} = graph_to_axon(else_branch, [])
    {[then_graph], then_params} = graph_to_axon(then_branch, [])

    updated_axon =
      outputs
      |> Enum.reduce(axon, fn out_name, axon ->
        Map.put(axon, out_name, Axon.cond(inp, & &1, then_graph, else_graph))
      end)

    updated_params =
      else_params
      |> Map.merge(then_params)
      |> Map.merge(used_params)

    {updated_axon, updated_params}
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
      raise ArgumentError,
            "unable to build model from ONNX graph, expected value #{name}" <>
              " to be a graph input, but it was not present in built" <>
              " graphs"
    end
  end

  defp param!(name, params) do
    if Map.has_key?(params, name) do
      params[name]
    else
      raise ArgumentError,
            "unable to build model from ONNX graph, expected value #{name}" <>
              " to be a parameter input, but it was not present in" <>
              " initializers"
    end
  end

  defp constant!(name, axon, params) do
    cond do
      Map.has_key?(axon, name) ->
        case axon[name] do
          %Axon{op: :constant, opts: [value: shape]} ->
            shape

          %Axon{op: op} ->
            raise ArgumentError,
                  "unable to build model from ONNX graph, expected value #{name}" <>
                    " to be constant value, but was #{inspect(op)}"
        end

      Map.has_key?(params, name) ->
        params[name]

      true ->
        raise ArgumentError,
              "unable to build model from ONNX graph, could not find constant" <>
                " value #{name} in subgraphs or parameters"
    end
  end

  defp padding!(auto_pad, pads) do
    case auto_pad do
      val when val == "NOTSET" or val == nil ->
        case pads do
          [l1, u1] ->
            [{l1, u1}]

          pads when is_list(pads) ->
            pads
            |> Enum.chunk_every(2)
            |> Enum.zip()

          nil ->
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
