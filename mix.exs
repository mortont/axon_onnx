defmodule AxonOnnx.MixProject do
  use Mix.Project

  @source_url "https://github.com/elixir-nx/axon_onnx"
  @version "0.4.0"

  def project do
    [
      app: :axon_onnx,
      version: @version,
      name: "AxonOnnx",
      elixir: "~> 1.13",
      elixirc_paths: elixirc_paths(Mix.env()),
      deps: deps(),
      docs: docs(),
      description: "Convert models between Axon/ONNX",
      package: package(),
      preferred_cli_env: [
        docs: :docs,
        "hex.publish": :docs
      ],
      aliases: aliases()
    ]
  end

  defp elixirc_paths(:test), do: ~w(lib test/support)
  defp elixirc_paths(_), do: ~w(lib)

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      extra_applications: [:logger]
    ]
  end

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      {:axon, "~> 0.5", axon_opts()},
      {:protox, "~> 1.6.10"},
      {:nx, "~> 0.5", nx_opts()},
      {:exla, "~> 0.5", [only: :test] ++ exla_opts()},
      {:req, "~> 0.1.0", only: :test},
      {:jason, "~> 1.2", only: :test},
      {:ex_doc, "~> 0.23", only: :docs}
    ]
  end

  defp package do
    [
      maintainers: ["Sean Moriarity"],
      licenses: ["Apache-2.0"],
      links: %{"GitHub" => @source_url}
    ]
  end

  defp docs do
    [
      main: "AxonOnnx",
      source_ref: "v#{@version}",
      source_url: @source_url
    ]
  end

  defp axon_opts do
    if path = System.get_env("AXON_PATH") do
      [path: path]
    else
      []
    end
  end

  defp nx_opts do
    if path = System.get_env("AXON_NX_PATH") do
      [path: path, override: true]
    else
      []
    end
  end

  defp exla_opts do
    if path = System.get_env("AXON_EXLA_PATH") do
      [path: path]
    else
      []
    end
  end

  defp aliases() do
    [
      generate_protobuf:
        "protox.generate --generate-defs-funs=false --keep-unknown-fields=false --multiple-files --output-path=./lib/onnx ./scripts/onnx.proto"
    ]
  end
end
