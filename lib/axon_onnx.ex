defmodule AxonOnnx do
  use Protox,
    files: [
      :filename.join([:code.priv_dir(:axon_onnx), "onnx.proto"])
    ]
end
