# AxonONNX

Easily convert models between ONNX and Axon.

## Installation

AxonONNX is currently in development. You can use it as a `git` dependency:

```elixir
def deps do
  [
    {:axon_onnx, github: "elixir-nx/axon_onnx"}
  ]
end
```

Additionally, AxonONNX uses [protox](https://github.com/ahamez/protox) for
parsing protocol buffers files (.proto) within ONNX. You'll also need:

- `protoc >= 3.0`. It must be installed in your system and available in your
  `$PATH`. *This dependency is only required at compile-time*.
  👉 You can download it [here](https://github.com/google/protobuf) or you can
  install it with your favorite package manager (`brew install protobuf`,
  `apt install protobuf-compiler`, etc.).
