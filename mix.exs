defmodule AxonOnnx.MixProject do
  use Mix.Project

  def project do
    [
      app: :axon_onnx,
      version: "0.1.0",
      elixir: "~> 1.13",
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      aliases: aliases()
    ]
  end

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      extra_applications: [:logger]
    ]
  end

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      {:protox, "~> 1.6.10"},
      {:nx, "~> 0.2.1", nx_opts()},
      {:exla, "~> 0.2.1", exla_opts()},
      {:req, "~> 0.1.0", only: :test},
      {:jason, "~> 1.2", only: :test},
      {:axon, "~> 0.1.0-dev", axon_opts()}
    ]
  end

  defp axon_opts do
    if path = System.get_env("AXON_PATH") do
      [path: path]
    else
      [github: "elixir-nx/axon", branch: "main"]
    end
  end

  defp nx_opts do
    if path = System.get_env("AXON_NX_PATH") do
      [path: path, override: true]
    else
      [github: "elixir-nx/nx", sparse: "nx", override: true, branch: "v0.2"]
    end
  end

  defp exla_opts do
    if path = System.get_env("EXLA_PATH") do
      [path: path, override: true, only: :test]
    else
      [github: "elixir-nx/exla", sparse: "exla", branch: "v0.2", only: :test]
    end
  end

  defp aliases() do
    [
      generate_protobuf:
        "protox.generate --generate-defs-funs=false --keep-unknown-fields=false --multiple-files --output-path=./lib/onnx ./scripts/onnx.proto"
    ]
  end
end
