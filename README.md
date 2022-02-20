# AxonOnnx

**TODO: Add description**

## Installation

If [available in Hex](https://hex.pm/docs/publish), the package can be installed
by adding `axon_onnx` to your list of dependencies in `mix.exs`:

```elixir
def deps do
  [
    {:axon_onnx, github: "elixir-nx/axon_onnx"}
  ]
end
```

Additionally, axon_onnx uses [protox](https://github.com/ahamez/protox) for the protocol buffers used in ONNX file format.
The requirements for protox includes the following:
- protoc >= 3.0 *This dependency is only required at compile-time*
  `protox` uses Google's `protoc` (>= 3.0) to parse `.proto` files. It must be available in `$PATH`.
  👉 You can download it [here](https://github.com/google/protobuf) or you can install it with your favorite package manager (`brew install protobuf`, `apt install protobuf-compiler`, etc.).

Documentation can be generated with [ExDoc](https://github.com/elixir-lang/ex_doc)
and published on [HexDocs](https://hexdocs.pm). Once published, the docs can
be found at [https://hexdocs.pm/axon_onnx](https://hexdocs.pm/axon_onnx).

