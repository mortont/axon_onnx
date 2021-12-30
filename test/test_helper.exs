ExUnit.start()

defmodule OnnxTestHelper do
  @moduledoc """
  Helpers for running ONNX's suite of tests on imported models.
  """
  require Axon
  require Logger

  @cache_dir Path.join([File.cwd!(), ".test-cache"])

  @doc """
  Serializes and tests model against test cases.
  This function will generate N cases and serialize them
  along with the model, storing them in the test cache. It invokes
  The script `check_onnx_model.py` to ensure ONNX runtime results
  are consistent with Axon results.
  """
  def serialize_and_test_model!(%Axon{name: name} = axon_model, opts \\ []) do
    num_cases = opts[:num_tests] || 5
    model_name = opts[:name] || name
    cache_dir = Path.join([@cache_dir, model_name])
    File.mkdir_p!(cache_dir)

    model_path = Path.join([cache_dir, "#{model_name}.onnx"])

    {input_shape, _} = Axon.get_model_signature(axon_model)
    params = Axon.init(axon_model, compiler: EXLA)

    Enum.each(1..num_cases//1, fn n ->
      test_path = Path.join([cache_dir, "test_data_set_#{n}"])
      File.mkdir_p!(test_path)

      inp = Nx.random_uniform(input_shape, type: {:f, 32})
      out = Axon.predict(axon_model, params, inp)

      nx_to_tensor_proto(inp, Path.join([test_path, "input_0.pb"]))
      nx_to_tensor_proto(out, Path.join([test_path, "output_0.pb"]))
    end)

    AxonOnnx.Serialize.__export__(axon_model, params, filename: model_path)
    # Run check script
    {_, exit_code} =
      System.cmd("python3", ["scripts/check_onnx_model.py", model_path], into: IO.stream())

    unless exit_code == 0 do
      raise "Model serialization failed for #{model_name}"
    end
  end

  # TODO: Maybe this should be in utils
  defp nx_to_tensor_proto(tensor, path) do
    dims = Nx.shape(tensor) |> Tuple.to_list()
    # TODO: fix
    data_type = 1
    raw_data = Nx.to_binary(tensor)
    tp = %Onnx.TensorProto{dims: dims, data_type: data_type, raw_data: raw_data}

    encoded_tp = Onnx.TensorProto.encode!(tp)
    {:ok, file} = File.open(path, [:write])
    IO.binwrite(file, encoded_tp)
    File.close(file)
  end

  @doc """
  Tests model agains given ONNX test case.
  """
  def check_onnx_test_case!(type, test_name) do
    test_path = Path.join(["test", "cases", type, test_name])
    model_path = Path.join([test_path, "model.onnx"])
    data_paths = Path.wildcard(Path.join([test_path, "test_data_set_*"]))

    {model, params} = AxonOnnx.Deserialize.__import__(model_path)

    data_paths
    |> Enum.map(fn data_path ->
      input_paths = Path.wildcard(Path.join([data_path, "input_*.pb"]))
      output_paths = Path.wildcard(Path.join([data_path, "output_*.pb"]))

      inp_tensors = Enum.map(input_paths, &pb_to_tensor/1)
      out_tensors = Enum.map(output_paths, &pb_to_tensor/1)

      # TODO: Update with better container support
      actual_outputs =
        case inp_tensors do
          [input] ->
            Axon.predict(model, params, input)

          [_ | _] = inputs ->
            Axon.predict(model, params, List.to_tuple(inputs))
        end

      case out_tensors do
        [expected_output] ->
          assert_all_close!(expected_output, actual_outputs)

        [_ | _] = expected_outputs ->
          Enum.zip_with(expected_outputs, Tuple.to_list(actual_outputs), &assert_all_close!/2)
      end
    end)
  end

  defp assert_all_close!(x, y) do
    unless Nx.all_close(x, y) == Nx.tensor(1, type: {:u, 8}) do
      raise "expected #{inspect(x)} to be within tolerance of #{inspect(y)}"
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

# Set EXLA as default compiler
Nx.Defn.default_options(compiler: EXLA)
