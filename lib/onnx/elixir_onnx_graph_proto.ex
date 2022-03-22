# credo:disable-for-this-file
defmodule Onnx.GraphProto do
  @moduledoc false
  defstruct node: [],
            name: "",
            initializer: [],
            doc_string: "",
            input: [],
            output: [],
            value_info: [],
            quantization_annotation: [],
            sparse_initializer: []

  (
    (
      @spec encode(struct) :: {:ok, iodata} | {:error, any}
      def encode(msg) do
        try do
          {:ok, encode!(msg)}
        rescue
          e in [Protox.EncodingError, Protox.RequiredFieldsError] -> {:error, e}
        end
      end

      @spec encode!(struct) :: iodata | no_return
      def encode!(msg) do
        []
        |> encode_node(msg)
        |> encode_name(msg)
        |> encode_initializer(msg)
        |> encode_doc_string(msg)
        |> encode_input(msg)
        |> encode_output(msg)
        |> encode_value_info(msg)
        |> encode_quantization_annotation(msg)
        |> encode_sparse_initializer(msg)
      end
    )

    []

    [
      defp encode_node(acc, msg) do
        try do
          case msg.node do
            [] ->
              acc

            values ->
              [
                acc,
                Enum.reduce(values, [], fn value, acc ->
                  [acc, "\n", Protox.Encode.encode_message(value)]
                end)
              ]
          end
        rescue
          ArgumentError ->
            reraise Protox.EncodingError.new(:node, "invalid field value"), __STACKTRACE__
        end
      end,
      defp encode_name(acc, msg) do
        try do
          if msg.name == "" do
            acc
          else
            [acc, "\x12", Protox.Encode.encode_string(msg.name)]
          end
        rescue
          ArgumentError ->
            reraise Protox.EncodingError.new(:name, "invalid field value"), __STACKTRACE__
        end
      end,
      defp encode_initializer(acc, msg) do
        try do
          case msg.initializer do
            [] ->
              acc

            values ->
              [
                acc,
                Enum.reduce(values, [], fn value, acc ->
                  [acc, "*", Protox.Encode.encode_message(value)]
                end)
              ]
          end
        rescue
          ArgumentError ->
            reraise Protox.EncodingError.new(:initializer, "invalid field value"), __STACKTRACE__
        end
      end,
      defp encode_doc_string(acc, msg) do
        try do
          if msg.doc_string == "" do
            acc
          else
            [acc, "R", Protox.Encode.encode_string(msg.doc_string)]
          end
        rescue
          ArgumentError ->
            reraise Protox.EncodingError.new(:doc_string, "invalid field value"), __STACKTRACE__
        end
      end,
      defp encode_input(acc, msg) do
        try do
          case msg.input do
            [] ->
              acc

            values ->
              [
                acc,
                Enum.reduce(values, [], fn value, acc ->
                  [acc, "Z", Protox.Encode.encode_message(value)]
                end)
              ]
          end
        rescue
          ArgumentError ->
            reraise Protox.EncodingError.new(:input, "invalid field value"), __STACKTRACE__
        end
      end,
      defp encode_output(acc, msg) do
        try do
          case msg.output do
            [] ->
              acc

            values ->
              [
                acc,
                Enum.reduce(values, [], fn value, acc ->
                  [acc, "b", Protox.Encode.encode_message(value)]
                end)
              ]
          end
        rescue
          ArgumentError ->
            reraise Protox.EncodingError.new(:output, "invalid field value"), __STACKTRACE__
        end
      end,
      defp encode_value_info(acc, msg) do
        try do
          case msg.value_info do
            [] ->
              acc

            values ->
              [
                acc,
                Enum.reduce(values, [], fn value, acc ->
                  [acc, "j", Protox.Encode.encode_message(value)]
                end)
              ]
          end
        rescue
          ArgumentError ->
            reraise Protox.EncodingError.new(:value_info, "invalid field value"), __STACKTRACE__
        end
      end,
      defp encode_quantization_annotation(acc, msg) do
        try do
          case msg.quantization_annotation do
            [] ->
              acc

            values ->
              [
                acc,
                Enum.reduce(values, [], fn value, acc ->
                  [acc, "r", Protox.Encode.encode_message(value)]
                end)
              ]
          end
        rescue
          ArgumentError ->
            reraise Protox.EncodingError.new(:quantization_annotation, "invalid field value"),
                    __STACKTRACE__
        end
      end,
      defp encode_sparse_initializer(acc, msg) do
        try do
          case msg.sparse_initializer do
            [] ->
              acc

            values ->
              [
                acc,
                Enum.reduce(values, [], fn value, acc ->
                  [acc, "z", Protox.Encode.encode_message(value)]
                end)
              ]
          end
        rescue
          ArgumentError ->
            reraise Protox.EncodingError.new(:sparse_initializer, "invalid field value"),
                    __STACKTRACE__
        end
      end
    ]

    []
  )

  (
    (
      @spec decode(binary) :: {:ok, struct} | {:error, any}
      def decode(bytes) do
        try do
          {:ok, decode!(bytes)}
        rescue
          e in [Protox.DecodingError, Protox.IllegalTagError, Protox.RequiredFieldsError] ->
            {:error, e}
        end
      end

      (
        @spec decode!(binary) :: struct | no_return
        def decode!(bytes) do
          parse_key_value(bytes, struct(Onnx.GraphProto))
        end
      )
    )

    (
      @spec parse_key_value(binary, struct) :: struct
      defp parse_key_value(<<>>, msg) do
        msg
      end

      defp parse_key_value(bytes, msg) do
        {field, rest} =
          case Protox.Decode.parse_key(bytes) do
            {0, _, _} ->
              raise %Protox.IllegalTagError{}

            {1, _, bytes} ->
              {len, bytes} = Protox.Varint.decode(bytes)
              {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)
              {[node: msg.node ++ [Onnx.NodeProto.decode!(delimited)]], rest}

            {2, _, bytes} ->
              {len, bytes} = Protox.Varint.decode(bytes)
              {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)
              {[name: delimited], rest}

            {5, _, bytes} ->
              {len, bytes} = Protox.Varint.decode(bytes)
              {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)
              {[initializer: msg.initializer ++ [Onnx.TensorProto.decode!(delimited)]], rest}

            {10, _, bytes} ->
              {len, bytes} = Protox.Varint.decode(bytes)
              {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)
              {[doc_string: delimited], rest}

            {11, _, bytes} ->
              {len, bytes} = Protox.Varint.decode(bytes)
              {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)
              {[input: msg.input ++ [Onnx.ValueInfoProto.decode!(delimited)]], rest}

            {12, _, bytes} ->
              {len, bytes} = Protox.Varint.decode(bytes)
              {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)
              {[output: msg.output ++ [Onnx.ValueInfoProto.decode!(delimited)]], rest}

            {13, _, bytes} ->
              {len, bytes} = Protox.Varint.decode(bytes)
              {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)
              {[value_info: msg.value_info ++ [Onnx.ValueInfoProto.decode!(delimited)]], rest}

            {14, _, bytes} ->
              {len, bytes} = Protox.Varint.decode(bytes)
              {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)

              {[
                 quantization_annotation:
                   msg.quantization_annotation ++ [Onnx.TensorAnnotation.decode!(delimited)]
               ], rest}

            {15, _, bytes} ->
              {len, bytes} = Protox.Varint.decode(bytes)
              {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)

              {[
                 sparse_initializer:
                   msg.sparse_initializer ++ [Onnx.SparseTensorProto.decode!(delimited)]
               ], rest}

            {tag, wire_type, rest} ->
              {_, rest} = Protox.Decode.parse_unknown(tag, wire_type, rest)
              {[], rest}
          end

        msg_updated = struct(msg, field)
        parse_key_value(rest, msg_updated)
      end
    )

    []
  )

  (
    @spec json_decode(iodata(), keyword()) :: {:ok, struct()} | {:error, any()}
    def json_decode(input, opts \\ []) do
      try do
        {:ok, json_decode!(input, opts)}
      rescue
        e in Protox.JsonDecodingError -> {:error, e}
      end
    end

    @spec json_decode!(iodata(), keyword()) :: struct() | no_return()
    def json_decode!(input, opts \\ []) do
      {json_library_wrapper, json_library} = Protox.JsonLibrary.get_library(opts, :decode)

      Protox.JsonDecode.decode!(
        input,
        Onnx.GraphProto,
        &json_library_wrapper.decode!(json_library, &1)
      )
    end

    @spec json_encode(struct(), keyword()) :: {:ok, iodata()} | {:error, any()}
    def json_encode(msg, opts \\ []) do
      try do
        {:ok, json_encode!(msg, opts)}
      rescue
        e in Protox.JsonEncodingError -> {:error, e}
      end
    end

    @spec json_encode!(struct(), keyword()) :: iodata() | no_return()
    def json_encode!(msg, opts \\ []) do
      {json_library_wrapper, json_library} = Protox.JsonLibrary.get_library(opts, :encode)
      Protox.JsonEncode.encode!(msg, &json_library_wrapper.encode!(json_library, &1))
    end
  )

  []

  (
    @spec fields_defs() :: list(Protox.Field.t())
    def fields_defs() do
      [
        %{
          __struct__: Protox.Field,
          json_name: "node",
          kind: :unpacked,
          label: :repeated,
          name: :node,
          tag: 1,
          type: {:message, Onnx.NodeProto}
        },
        %{
          __struct__: Protox.Field,
          json_name: "name",
          kind: {:scalar, ""},
          label: :optional,
          name: :name,
          tag: 2,
          type: :string
        },
        %{
          __struct__: Protox.Field,
          json_name: "initializer",
          kind: :unpacked,
          label: :repeated,
          name: :initializer,
          tag: 5,
          type: {:message, Onnx.TensorProto}
        },
        %{
          __struct__: Protox.Field,
          json_name: "docString",
          kind: {:scalar, ""},
          label: :optional,
          name: :doc_string,
          tag: 10,
          type: :string
        },
        %{
          __struct__: Protox.Field,
          json_name: "input",
          kind: :unpacked,
          label: :repeated,
          name: :input,
          tag: 11,
          type: {:message, Onnx.ValueInfoProto}
        },
        %{
          __struct__: Protox.Field,
          json_name: "output",
          kind: :unpacked,
          label: :repeated,
          name: :output,
          tag: 12,
          type: {:message, Onnx.ValueInfoProto}
        },
        %{
          __struct__: Protox.Field,
          json_name: "valueInfo",
          kind: :unpacked,
          label: :repeated,
          name: :value_info,
          tag: 13,
          type: {:message, Onnx.ValueInfoProto}
        },
        %{
          __struct__: Protox.Field,
          json_name: "quantizationAnnotation",
          kind: :unpacked,
          label: :repeated,
          name: :quantization_annotation,
          tag: 14,
          type: {:message, Onnx.TensorAnnotation}
        },
        %{
          __struct__: Protox.Field,
          json_name: "sparseInitializer",
          kind: :unpacked,
          label: :repeated,
          name: :sparse_initializer,
          tag: 15,
          type: {:message, Onnx.SparseTensorProto}
        }
      ]
    end

    [
      @spec(field_def(atom) :: {:ok, Protox.Field.t()} | {:error, :no_such_field}),
      (
        def field_def(:node) do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "node",
             kind: :unpacked,
             label: :repeated,
             name: :node,
             tag: 1,
             type: {:message, Onnx.NodeProto}
           }}
        end

        def field_def("node") do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "node",
             kind: :unpacked,
             label: :repeated,
             name: :node,
             tag: 1,
             type: {:message, Onnx.NodeProto}
           }}
        end

        []
      ),
      (
        def field_def(:name) do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "name",
             kind: {:scalar, ""},
             label: :optional,
             name: :name,
             tag: 2,
             type: :string
           }}
        end

        def field_def("name") do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "name",
             kind: {:scalar, ""},
             label: :optional,
             name: :name,
             tag: 2,
             type: :string
           }}
        end

        []
      ),
      (
        def field_def(:initializer) do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "initializer",
             kind: :unpacked,
             label: :repeated,
             name: :initializer,
             tag: 5,
             type: {:message, Onnx.TensorProto}
           }}
        end

        def field_def("initializer") do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "initializer",
             kind: :unpacked,
             label: :repeated,
             name: :initializer,
             tag: 5,
             type: {:message, Onnx.TensorProto}
           }}
        end

        []
      ),
      (
        def field_def(:doc_string) do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "docString",
             kind: {:scalar, ""},
             label: :optional,
             name: :doc_string,
             tag: 10,
             type: :string
           }}
        end

        def field_def("docString") do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "docString",
             kind: {:scalar, ""},
             label: :optional,
             name: :doc_string,
             tag: 10,
             type: :string
           }}
        end

        def field_def("doc_string") do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "docString",
             kind: {:scalar, ""},
             label: :optional,
             name: :doc_string,
             tag: 10,
             type: :string
           }}
        end
      ),
      (
        def field_def(:input) do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "input",
             kind: :unpacked,
             label: :repeated,
             name: :input,
             tag: 11,
             type: {:message, Onnx.ValueInfoProto}
           }}
        end

        def field_def("input") do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "input",
             kind: :unpacked,
             label: :repeated,
             name: :input,
             tag: 11,
             type: {:message, Onnx.ValueInfoProto}
           }}
        end

        []
      ),
      (
        def field_def(:output) do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "output",
             kind: :unpacked,
             label: :repeated,
             name: :output,
             tag: 12,
             type: {:message, Onnx.ValueInfoProto}
           }}
        end

        def field_def("output") do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "output",
             kind: :unpacked,
             label: :repeated,
             name: :output,
             tag: 12,
             type: {:message, Onnx.ValueInfoProto}
           }}
        end

        []
      ),
      (
        def field_def(:value_info) do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "valueInfo",
             kind: :unpacked,
             label: :repeated,
             name: :value_info,
             tag: 13,
             type: {:message, Onnx.ValueInfoProto}
           }}
        end

        def field_def("valueInfo") do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "valueInfo",
             kind: :unpacked,
             label: :repeated,
             name: :value_info,
             tag: 13,
             type: {:message, Onnx.ValueInfoProto}
           }}
        end

        def field_def("value_info") do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "valueInfo",
             kind: :unpacked,
             label: :repeated,
             name: :value_info,
             tag: 13,
             type: {:message, Onnx.ValueInfoProto}
           }}
        end
      ),
      (
        def field_def(:quantization_annotation) do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "quantizationAnnotation",
             kind: :unpacked,
             label: :repeated,
             name: :quantization_annotation,
             tag: 14,
             type: {:message, Onnx.TensorAnnotation}
           }}
        end

        def field_def("quantizationAnnotation") do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "quantizationAnnotation",
             kind: :unpacked,
             label: :repeated,
             name: :quantization_annotation,
             tag: 14,
             type: {:message, Onnx.TensorAnnotation}
           }}
        end

        def field_def("quantization_annotation") do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "quantizationAnnotation",
             kind: :unpacked,
             label: :repeated,
             name: :quantization_annotation,
             tag: 14,
             type: {:message, Onnx.TensorAnnotation}
           }}
        end
      ),
      (
        def field_def(:sparse_initializer) do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "sparseInitializer",
             kind: :unpacked,
             label: :repeated,
             name: :sparse_initializer,
             tag: 15,
             type: {:message, Onnx.SparseTensorProto}
           }}
        end

        def field_def("sparseInitializer") do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "sparseInitializer",
             kind: :unpacked,
             label: :repeated,
             name: :sparse_initializer,
             tag: 15,
             type: {:message, Onnx.SparseTensorProto}
           }}
        end

        def field_def("sparse_initializer") do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "sparseInitializer",
             kind: :unpacked,
             label: :repeated,
             name: :sparse_initializer,
             tag: 15,
             type: {:message, Onnx.SparseTensorProto}
           }}
        end
      ),
      def field_def(_) do
        {:error, :no_such_field}
      end
    ]
  )

  []

  (
    @spec required_fields() :: []
    def required_fields() do
      []
    end
  )

  (
    @spec syntax() :: atom()
    def syntax() do
      :proto3
    end
  )

  [
    @spec(default(atom) :: {:ok, boolean | integer | String.t() | float} | {:error, atom}),
    def default(:node) do
      {:error, :no_default_value}
    end,
    def default(:name) do
      {:ok, ""}
    end,
    def default(:initializer) do
      {:error, :no_default_value}
    end,
    def default(:doc_string) do
      {:ok, ""}
    end,
    def default(:input) do
      {:error, :no_default_value}
    end,
    def default(:output) do
      {:error, :no_default_value}
    end,
    def default(:value_info) do
      {:error, :no_default_value}
    end,
    def default(:quantization_annotation) do
      {:error, :no_default_value}
    end,
    def default(:sparse_initializer) do
      {:error, :no_default_value}
    end,
    def default(_) do
      {:error, :no_such_field}
    end
  ]
end
