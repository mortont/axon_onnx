defmodule AxonOnnx.Layers do
  @moduledoc false

  import Nx.Defn

  # Implementations of various ONNX operations as Axon
  # layers

  def gemm_with_bias_layer(inp, units, opts \\ []) do
    opts = Keyword.validate!(opts, [:alpha, :beta, :name])

    kernel_param = Axon.param("kernel", &Axon.Shape.dense_kernel(&1, units))
    bias_param = Axon.param("bias", &Axon.Shape.dense_bias(&1, units))

    alpha = Nx.to_number(opts[:alpha])
    beta = Nx.to_number(opts[:beta])

    Axon.layer(&gemm_with_bias_impl/4, [inp, kernel_param, bias_param],
      name: opts[:name],
      op_name: :gemm,
      alpha: alpha,
      beta: beta
    )
  end

  defnp gemm_with_bias_impl(inp, kernel, bias, opts \\ []) do
    opts = keyword!(opts, alpha: 1.0, beta: 1.0, mode: :train)
    bias = Nx.multiply(bias, opts[:beta])
    Axon.Layers.dense(inp, kernel, bias) |> Nx.multiply(opts[:alpha])
  end

  def numpy_matmul_layer(
        %Axon{} = a,
        %Axon{} = b,
        opts \\ []
      ) do
    opts = Keyword.validate!(opts, [:name])
    Axon.layer(&numpy_matmul_impl/3, [a, b], name: opts[:name], op_name: :numpy_matmul)
  end

  defnp numpy_matmul_impl(a, b, _opts) do
    {out_a_shape, c1_dims, b1_dims, out_b_shape, c2_dims, b2_dims} =
      transform({Nx.shape(a), Nx.shape(b)}, fn
        {{}, {}} ->
          {{}, [], [], {}, [], []}

        {{_} = a, {_} = b} ->
          {a, [0], [], b, [0], []}

        {{_, _} = a, {_, _} = b} ->
          {a, [1], [], b, [0], []}

        {a_shape, b_shape} ->
          # TODO: This should broadcast both sides, not just one
          batch_dims = Enum.to_list(0..(Nx.rank(a_shape) - 3))

          b_shape =
            if Elixir.Kernel.==(Nx.rank(b_shape), Nx.rank(a_shape)) do
              b_shape
            else
              Enum.reduce(Enum.reverse(batch_dims), b_shape, fn dim, shape ->
                Tuple.insert_at(shape, 0, elem(a_shape, dim))
              end)
            end

          {a_shape, [Nx.rank(a_shape) - 1], batch_dims, b_shape, [Nx.rank(b_shape) - 2],
           batch_dims}
      end)

    a = Nx.broadcast(a, out_a_shape)
    b = Nx.broadcast(b, out_b_shape)

    Nx.dot(a, c1_dims, b1_dims, b, c2_dims, b2_dims)
  end

  def gather_layer(
        %Axon{} = x,
        %Axon{} = ind,
        axis,
        output_name
      ) do
    Axon.layer(&gather_impl/3, [x, ind], name: output_name, op_name: :gather, axis: axis)
  end

  defnp gather_impl(x, indices, opts) do
    opts = keyword!(opts, [:axis, mode: :train])
    Nx.take(x, Nx.as_type(indices, {:s, 64}), axis: opts[:axis])
  end

  def slice_layer(inp, starts, ends, axes, steps, output_name) do
    case inp do
      %Axon{op: :constant, opts: [value: v]} ->
        new_value = slice_impl(v, starts: starts, ends: ends, axes: axes, steps: steps)
        Axon.constant(new_value, name: output_name)

      %Nx.Tensor{} = value ->
        new_value = slice_impl(value, starts: starts, ends: ends, axes: axes, steps: steps)
        Axon.constant(new_value, name: output_name)

      %Axon{} = inp ->
        Axon.layer(&slice_impl/2, inp,
          name: output_name,
          starts: starts,
          ends: ends,
          axes: axes,
          steps: steps
        )
    end
  end

  defnp slice_impl(x, opts \\ []) do
    opts = keyword!(opts, [:starts, :ends, :axes, :steps, mode: :train])

    shape = Nx.shape(x)
    rank = Nx.rank(shape)

    axes =
      transform({x, opts[:axes]}, fn
        {x, nil} -> Nx.axes(x)
        {_x, axes} -> axes
      end)

    steps =
      transform({rank, opts[:steps]}, fn
        {rank, nil} -> List.duplicate(1, rank)
        {_rank, steps} -> steps
      end)

    transform({x, opts[:starts], opts[:ends], axes, steps}, fn
      {x, starts, ends, axes, steps} ->
        [starts, ends, axes, steps]
        |> Enum.zip()
        |> Enum.reduce(x, &do_slice(shape, &1, &2))
    end)
  end

  defp do_slice(shape, {start, stop, axis, stride}, acc) do
    start = if start < 0, do: start + elem(shape, axis), else: start

    start =
      if stride < 0,
        do: clamp_to_range(start, 0, elem(shape, axis) - 1),
        else: clamp_to_range(start, 0, elem(shape, axis))

    stop = if stop < 0, do: stop + elem(shape, axis), else: stop

    stop =
      if stride < 0,
        do: clamp_to_range(stop, -1, elem(shape, axis) - 1),
        else: clamp_to_range(stop, 0, elem(shape, axis))

    if stride < 0 do
      len = start - stop

      acc
      |> Nx.reverse(axes: [axis])
      |> Nx.slice_along_axis(start, len, axis: axis, strides: abs(stride))
    else
      len = stop - start
      Nx.slice_along_axis(acc, start, len, axis: axis, strides: stride)
    end
  end

  defp clamp_to_range(val, min, max) do
    floor(min(max(min, val), max))
  end

  def trainable_binary_layer(%Axon{} = input, %Nx.Tensor{} = param, opts \\ []) do
    param_shape = Nx.shape(param)
    kernel = Axon.param("kernel", fn _ -> param_shape end)

    Axon.layer(&trainable_binary_impl/3, [input, kernel],
      binary_op: opts[:binary_op],
      name: opts[:name],
      op_name: opts[:op_name]
    )
  end

  defnp trainable_binary_impl(input, param, opts \\ []) do
    opts = keyword!(opts, [:binary_op, mode: :train])

    transform({input, param, opts[:binary_op]}, fn {input, param, binary_op} ->
      if is_atom(binary_op) do
        apply(Nx, binary_op, [input, param])
      else
        apply(binary_op, [input, param])
      end
    end)
  end

  def cast_layer(input, opts \\ []) do
    Axon.layer(&cast_impl/2, [input], type: opts[:type], op_name: :cast, name: opts[:name])
  end

  defnp cast_impl(input, opts \\ []) do
    opts = keyword!(opts, [:type, mode: :train])
    Nx.as_type(input, opts[:type])
  end

  def lrn_layer(input, opts \\ []) do
    {name, opts} = Keyword.pop(opts, :name)
    Axon.layer(&AxonOnnx.Shared.lrn/2, [input], [name: name, op_name: :lrn] ++ opts)
  end

  def cumulative_sum_layer(input, opts \\ []) do
    {name, opts} = Keyword.pop(opts, :name)
    Axon.layer(&cumulative_sum_impl/2, [input], [name: name, op_name: :cumulative_sum] ++ opts)
  end

  defnp cumulative_sum_impl(input, opts \\ []) do
    opts = keyword!(opts, [:axis, mode: :train])
    Nx.cumulative_sum(input, axis: opts[:axis])
  end

  def expand_layer(input, expand_shape, opts \\ []) do
    Axon.layer(&expand_impl/2, [input], name: opts[:name], op_name: :expand, shape: expand_shape)
  end

  defnp expand_impl(input, opts \\ []) do
    opts = keyword!(opts, [:shape, mode: :train])
    Nx.multiply(input, Nx.broadcast(1, opts[:shape]))
  end
end
