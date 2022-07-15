defmodule AxonOnnx.Shared do
  @moduledoc false

  # defn implementations of ONNX operators and shared
  # helpers for converting between onnx and axon

  import Nx.Defn

  # Numerical helpers

  defn hardswish(x) do
    alpha = Nx.divide(1, 6)
    beta = Nx.tensor(0.5)

    alpha
    |> Nx.multiply(x)
    |> Nx.add(beta)
    |> Nx.min(1)
    |> Nx.max(0)
    |> Nx.multiply(x)
  end

  defn reciprocal(x) do
    1 / x
  end

  defn(identity(x), do: x)

  defn logsum(x, opts \\ []) do
    opts = keyword!(opts, [:axes, keep_axes: false])

    x |> Nx.sum(opts) |> Nx.log()
  end

  defn logsumexp(x, opts \\ []) do
    opts = keyword!(opts, [:axes, keep_axes: false])

    x |> Nx.exp() |> Nx.sum(opts) |> Nx.log()
  end

  defn sumsquare(x, opts \\ []) do
    opts = keyword!(opts, [:axes, keep_axes: false])

    x |> Nx.power(2) |> Nx.sum(opts)
  end

  defn l1_norm(x, opts \\ []) do
    x |> Nx.abs() |> Nx.sum(opts)
  end

  defn l2_norm(x, opts \\ []) do
    x |> Nx.power(2) |> Nx.sum(opts)
  end

  defn lrn(x, opts \\ []) do
    opts = keyword!(opts, [:size, alpha: 1.0e-4, beta: 0.75, bias: 1.0])
    size = opts[:size]
    axes = transform(size, &Enum.to_list(0..(&1 - 1)))
    alpha = opts[:alpha]
    beta = opts[:beta]
    bias = opts[:bias]

    squares = Nx.power(x, 2)
    sum_squares = Nx.sum(squares, axes: axes, keep_axes: true)
    denom = Nx.power(Nx.add(bias, Nx.divide(alpha, Nx.multiply(size, sum_squares))), beta)
    Nx.divide(x, denom)
  end

  defn(mean(x, y), do: Nx.divide(Nx.add(x, y), 2))
  # Layer helpers

  def trainable_binary_layer(%Axon{} = input, %Nx.Tensor{} = param, op, name) do
    param_shape = Nx.shape(param)

    kernel = Axon.param("kernel", fn _ -> param_shape end)

    fun = fn x, kernel, _opts ->
      if is_atom(op) do
        apply(Nx, op, [x, kernel])
      else
        apply(op, [x, kernel])
      end
    end

    Axon.layer(fun, [input, kernel], name: name)
  end

  def numpy_matmul_layer(
        %Axon{} = a,
        %Axon{} = b,
        output_name
      ) do
    Axon.layer(&numpy_matmul/3, [a, b], name: output_name, op_name: :numpy_matmul)
  end

  defnp numpy_matmul(a, b, _opts) do
    {c1_dims, b1_dims, c2_dims, b2_dims} =
      transform({Nx.shape(a), Nx.shape(b)}, fn
        {{}, {}} ->
          {[], [], [], []}

        {{_}, {_}} ->
          {[0], [], [0], []}

        {{_, _}, {_, _}} ->
          {[1], [], [0], []}

        {a_shape, b_shape} ->
          batch_dims = Enum.to_list(0..(Nx.rank(a_shape) - 3))
          {[Nx.rank(a_shape) - 1], batch_dims, [Nx.rank(b_shape) - 2], batch_dims}
      end)

    Nx.dot(a, c1_dims, b1_dims, b, c2_dims, b2_dims)
  end

  def gather_layer(
        %Axon{} = x,
        %Axon{} = ind,
        axis,
        output_name
      ) do
    fun = fn x, indices, _opts ->
      Nx.take(x, Nx.as_type(indices, {:s, 64}), axis: axis)
    end

    Axon.layer(fun, [x, ind], name: output_name)
  end

  def slice_layer(inp, starts, ends, axes, steps, output_name, axon, used_params) do
    fun = fn x ->
      shape = Nx.shape(x)
      rank = Nx.rank(shape)
      axes = if axes, do: axes, else: Nx.axes(x)
      axes = axes |> Enum.map(fn x -> if x < 0, do: x + rank, else: x end)
      steps = if steps, do: steps, else: List.duplicate(1, rank)

      [starts, ends, axes, steps]
      |> Enum.zip()
      |> Enum.reduce(x, &do_slice(shape, &1, &2))
    end

    case inp do
      %Axon{op: :constant, opts: [value: v]} ->
        new_value = fun.(v)
        layer = Axon.constant(new_value, name: output_name)
        updated_axon = Map.put(axon, output_name, layer)
        {updated_axon, used_params}

      %Axon{} = inp ->
        layer = Axon.nx(inp, fun, name: output_name)
        updated_axon = Map.put(axon, output_name, layer)
        {updated_axon, used_params}

      %Nx.Tensor{} = param ->
        shape = Nx.shape(param)
        param_slice(shape, starts, ends, axes, steps, output_name, axon, param, used_params)
    end
  end

  def param_slice(shape, starts, ends, axes, steps, output_name, axon, param, used_params) do
    fun = fn _x, kernel, _opts ->
      [starts, ends, axes, steps]
      |> Enum.zip()
      |> Enum.reduce(kernel, &do_slice(shape, &1, &2))
    end

    kernel = Axon.param("kernel", fn _ -> shape end)
    # empty layer
    inp = Axon.container({})
    layer = Axon.layer(fun, [inp, kernel], name: output_name)

    updated_axon = Map.put(axon, output_name, layer)
    updated_params = Map.put(used_params, output_name, %{"kernel" => param})

    {updated_axon, updated_params}
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

    len = stop - start
    Nx.slice_along_axis(acc, start, len, axis: axis, strides: stride)
  end

  defp clamp_to_range(val, min, max) do
    min(max(min, val), max)
  end

  # Conversion helpers

  def constant?(%{op: :constant}), do: true
  def constant?(%Nx.Tensor{}), do: true
  def constant?(_), do: false

  def get_value(%{op: :constant, opts: [value: v]}), do: v
  def get_value(%Nx.Tensor{} = v), do: v

  def onnx_type_to_nx_type(1), do: {:f, 32}
  def onnx_type_to_nx_type(2), do: {:u, 8}
  def onnx_type_to_nx_type(3), do: {:s, 8}
  def onnx_type_to_nx_type(4), do: {:u, 16}
  def onnx_type_to_nx_type(5), do: {:s, 16}
  def onnx_type_to_nx_type(6), do: {:s, 32}
  def onnx_type_to_nx_type(7), do: {:s, 64}
  def onnx_type_to_nx_type(8), do: raise(ArgumentError, "unsupported STRING type")
  def onnx_type_to_nx_type(9), do: {:u, 8}
  def onnx_type_to_nx_type(10), do: {:f, 16}
  def onnx_type_to_nx_type(11), do: {:f, 64}
  def onnx_type_to_nx_type(12), do: {:u, 32}
  def onnx_type_to_nx_type(13), do: {:u, 64}
  def onnx_type_to_nx_type(14), do: {:c, 64}
  def onnx_type_to_nx_type(15), do: {:c, 128}
  def onnx_type_to_nx_type(16), do: {:bf, 16}
end
