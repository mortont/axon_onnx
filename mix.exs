defmodule AxonOnnx.MixProject do
  use Mix.Project

  def project do
    [
      app: :axon_onnx,
      version: "0.1.0",
      elixir: "~> 1.12",
      start_permanent: Mix.env() == :prod,
      deps: deps()
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
      {:protox, "~> 1.4.0"},
      {:nx, "~> 0.1.0-dev", github: "elixir-nx/nx", sparse: "nx", override: true},
      {:exla, "~> 0.1.0-dev",
       github: "elixir-nx/nx", sparse: "exla", override: true, only: :test},
      {:req, "~> 0.1.0", only: :test},
      {:axon, "~> 0.1.0-dev", axon_opts()}
    ]
  end

  defp axon_opts do
    if path = System.get_env("AXON_PATH") do
      [path: path]
    else
      [github: "elixir-nx/axon"]
    end
  end
end
