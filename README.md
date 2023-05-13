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

### Protobuf

Additionally, AxonONNX uses [protox](https://github.com/ahamez/protox) to generate code for
parsing protocol buffers data within ONNX.

If you want to regenerate this code, you can invoke the following command:

```sh
mix generate_protobuf
```

Note that you'll need to install `protoc` (>= 3.0). It must available in `$PATH`. You can download it
[here](https://github.com/google/protobuf) or you can install it with your favorite package manager
(`brew install protobuf`, `apt install protobuf-compiler`, etc.).

## Development & Test

Axxon_onnx requires an Elixir runtime.

The `onnx` python package is required to run the unit tests: `pip3 install onnx`. The test cases
will be generated automatically upon first `mix test`.

## License

Copyright (c) 2021 Sean Moriarity

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
