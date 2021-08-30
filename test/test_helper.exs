ExUnit.start()

defmodule OnnxTestHelper do
  @moduledoc """
  Helpers for running ONNX's suite of tests on imported models.
  """
  require Axon
  require Logger

  @hub_url "https://github.com/onnx/models/raw/master"
  @cache_dir Path.join([File.cwd!(), ".test-cache"])

  @doc """
  Tests model against provided test cases.
  """
  def test_deserialized_model!(model_path, model_name, model_version, cache_dir \\ @cache_dir) do
    model_name_and_version = model_name <> model_version
    :ok = download_and_extract(model_path, model_name, model_version, cache_dir)

    abs_path_to_model = Path.wildcard(Path.join([cache_dir, model_path, model_name, "**", "#{model_name_and_version}.onnx"])) |> IO.inspect
    model_dir = Path.dirname(abs_path_to_model)

    {model, params} = AxonOnnx.Deserialize.__import__(abs_path_to_model)

    expected_inputs_and_outputs =
      model_dir
      |> File.ls!()
      |> Enum.filter(&File.dir?(Path.join(model_dir, &1)))
      |> Enum.map(fn test_dirs ->
          input_paths = Path.wildcard(Path.join([model_dir, test_dirs, "input_*.pb"]))
          output_paths = Path.wildcard(Path.join([model_dir, test_dirs, "output_*.pb"]))

          input_paths
          |> Enum.zip_with(output_paths, fn inp, out ->
              {pb_to_tensor(inp), pb_to_tensor(out)}
          end)
      end)
      |> List.flatten()

    expected_inputs_and_outputs
    |> Enum.map(fn {expected_in, expected_out} ->
        pred = Axon.predict(model, params, expected_in, compiler: EXLA)
        unless Nx.all_close?(pred, expected_out) do
          raise "expected #{inspect(pred)} to be within tolerance of #{inspect(expected_out)}"
        end
    end)
  end

  # Downloads and extracts the model and test cases at the given
  # path to the test cache.
  defp download_and_extract(model_path, model_name, model_version, cache_dir \\ @cache_dir) do
    model_name_and_version = model_name <> "-" <> model_version
    full_path =
      case Path.wildcard(Path.join([cache_dir, model_path, model_name, "**", "#{model_name_and_version}.onnx"])) do
        [] ->
          Path.join([cache_dir, model_path, model_name, "resnet18v1"]) |> IO.inspect
        [path] ->
          path
      end
    full_url = "#{@hub_url}/vision/classification/resnet/model/resnet18-v1-7.tar.gz"

    if File.exists?(full_path) do
      Logger.info "Using cached model #{full_path}"
    else
      Logger.info "Downloading from #{full_url}"
      %{body: response} = Req.get!(full_url)
      File.mkdir_p!(full_path)
      response
      |> Enum.map(fn {fname, bytes} ->
        with :ok <- File.mkdir_p(Path.join([full_path, Path.dirname(fname)])) do
          File.write!(Path.join([full_path, fname]), bytes)
        end
      end)
    end

    :ok
  end

  # Parses the protobuf file into an Nx tensor.
  def pb_to_tensor(pb_path) do
    pb_path
    |> File.read!()
    |> Onnx.TensorProto.decode!()
    |> tensor!()
  end

  defp tensor!(%Onnx.TensorProto{data_type: dtype, dims: dims} = tensor) do
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
end