ExUnit.start()

defmodule OnnxModel do
  alias __MODULE__, as: M
  @hub_url "https://github.com/onnx/models/raw/master"

  defstruct [
    :category,
    :subcategory,
    :name,
    :long_name,
    :model_version,
    :onnx_version,
    :library
  ]

  def to_url(%M{
        category: cat,
        subcategory: sub,
        name: name,
        long_name: long_name,
        model_version: model_version,
        onnx_version: onnx_version,
        library: library
      }) do
    model_name = name_or_long_name(name, long_name)
    model_version = version_string(model_version, onnx_version)
    model_library = library_name(library)

    fname = "#{model_name}#{model_library}#{model_version}.tar.gz"
    "#{@hub_url}/#{cat}/#{sub}/#{name}/model/#{fname}"
  end

  def to_short_path(%M{
        category: cat,
        subcategory: sub,
        name: name
      }) do
    Path.join([cat, sub, name])
  end

  def to_long_path(
        %M{
          name: name,
          long_name: long_name,
          model_version: model_version,
          onnx_version: onnx_version,
          library: library
        } = model
      ) do
    model_name = name_or_long_name(name, long_name)
    model_version = version_string(model_version, onnx_version)
    model_library = library_name(library)

    fname = "#{model_name}#{model_library}#{model_version}"
    Path.join([to_short_path(model), fname])
  end

  def to_onnx_path(
        %M{
          name: name,
          long_name: long_name,
          model_version: model_version,
          onnx_version: onnx_version,
          library: library
        } = model
      ) do
    model_name = name_or_long_name(name, long_name)
    model_version = version_string(model_version, onnx_version)
    model_library = library_name(library)

    fname = "#{model_name}#{model_library}#{model_version}.onnx"
    Path.join([to_long_path(model), fname])
  end

  defp name_or_long_name(name, long_name) do
    case long_name do
      "" ->
        name

      long_name ->
        long_name
    end
  end

  defp library_name(library) do
    case library do
      "" ->
        "-"

      lib_name ->
        "-#{lib_name}-"
    end
  end

  defp version_string(model_version, onnx_version) do
    case model_version do
      "" ->
        onnx_version

      version ->
        "v#{version}-#{onnx_version}"
    end
  end
end

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
      out = Axon.predict(axon_model, params, inp, compiler: EXLA)

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
  Returns an OnnxModel struct for a specific ResNet.
  """
  def resnet(depth) do
    %OnnxModel{
      category: "vision",
      subcategory: "classification",
      name: "resnet",
      long_name: "resnet#{depth}",
      library: "",
      model_version: "1",
      onnx_version: "7"
    }
  end

  @doc """
  Tests model against provided test cases.
  """
  def test_deserialized_model!(%OnnxModel{} = onnx_model, cache_dir \\ @cache_dir) do
    :ok = download_and_extract(onnx_model, cache_dir)
    model_dir = Path.join([cache_dir, OnnxModel.to_long_path(onnx_model)])
    model_path = Path.join([cache_dir, OnnxModel.to_onnx_path(onnx_model)])

    Logger.debug("Converting model #{onnx_model.long_name} from ONNX to Axon")
    {model, params} = AxonOnnx.Deserialize.__import__(model_path)

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
    |> Enum.with_index(fn {expected_in, expected_out}, i ->
      Logger.debug("#{onnx_model.name}: Test case #{i + 1}...")
      pred = Axon.predict(model, params, expected_in, compiler: EXLA)

      unless Nx.all_close?(pred, expected_out) do
        raise "test failed for model #{onnx_model.name}, expected #{inspect(pred)} " <>
                "to be within tolerance of #{inspect(expected_out)}"
      end
    end)

    :ok
  end

  # Downloads and extracts the model and test cases at the given
  # path to the test cache.
  defp download_and_extract(%OnnxModel{} = model, cache_dir \\ @cache_dir) do
    dir_path = Path.join([cache_dir, OnnxModel.to_short_path(model)])
    full_path = Path.join([cache_dir, OnnxModel.to_onnx_path(model)])
    full_url = OnnxModel.to_url(model)

    if File.exists?(full_path) do
      Logger.debug("Using cached model #{full_path}")
    else
      Logger.debug("Downloading from #{full_url}")
      %{body: response} = Req.get!(full_url)
      File.mkdir_p!(dir_path)

      response
      |> Enum.map(fn {fname, bytes} ->
        with :ok <- File.mkdir_p(Path.join([dir_path, Path.dirname(fname)])) do
          File.write!(Path.join([dir_path, fname]), bytes)
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
