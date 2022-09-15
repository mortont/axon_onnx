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
    opts = keyword!(opts, [:size, mode: :train, alpha: 1.0e-4, beta: 0.75, bias: 1.0])
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
