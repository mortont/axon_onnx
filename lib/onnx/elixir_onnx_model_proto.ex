# credo:disable-for-this-file
defmodule Onnx.ModelProto do
  @moduledoc false
  defstruct ir_version: 0,
            producer_name: "",
            producer_version: "",
            domain: "",
            model_version: 0,
            doc_string: "",
            graph: nil,
            opset_import: [],
            metadata_props: [],
            training_info: []

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
        |> encode_ir_version(msg)
        |> encode_producer_name(msg)
        |> encode_producer_version(msg)
        |> encode_domain(msg)
        |> encode_model_version(msg)
        |> encode_doc_string(msg)
        |> encode_graph(msg)
        |> encode_opset_import(msg)
        |> encode_metadata_props(msg)
        |> encode_training_info(msg)
      end
    )

    []

    [
      defp encode_ir_version(acc, msg) do
        try do
          if msg.ir_version == 0 do
            acc
          else
            [acc, "\b", Protox.Encode.encode_int64(msg.ir_version)]
          end
        rescue
          ArgumentError ->
            reraise Protox.EncodingError.new(:ir_version, "invalid field value"), __STACKTRACE__
        end
      end,
      defp encode_producer_name(acc, msg) do
        try do
          if msg.producer_name == "" do
            acc
          else
            [acc, "\x12", Protox.Encode.encode_string(msg.producer_name)]
          end
        rescue
          ArgumentError ->
            reraise Protox.EncodingError.new(:producer_name, "invalid field value"),
                    __STACKTRACE__
        end
      end,
      defp encode_producer_version(acc, msg) do
        try do
          if msg.producer_version == "" do
            acc
          else
            [acc, "\x1A", Protox.Encode.encode_string(msg.producer_version)]
          end
        rescue
          ArgumentError ->
            reraise Protox.EncodingError.new(:producer_version, "invalid field value"),
                    __STACKTRACE__
        end
      end,
      defp encode_domain(acc, msg) do
        try do
          if msg.domain == "" do
            acc
          else
            [acc, "\"", Protox.Encode.encode_string(msg.domain)]
          end
        rescue
          ArgumentError ->
            reraise Protox.EncodingError.new(:domain, "invalid field value"), __STACKTRACE__
        end
      end,
      defp encode_model_version(acc, msg) do
        try do
          if msg.model_version == 0 do
            acc
          else
            [acc, "(", Protox.Encode.encode_int64(msg.model_version)]
          end
        rescue
          ArgumentError ->
            reraise Protox.EncodingError.new(:model_version, "invalid field value"),
                    __STACKTRACE__
        end
      end,
      defp encode_doc_string(acc, msg) do
        try do
          if msg.doc_string == "" do
            acc
          else
            [acc, "2", Protox.Encode.encode_string(msg.doc_string)]
          end
        rescue
          ArgumentError ->
            reraise Protox.EncodingError.new(:doc_string, "invalid field value"), __STACKTRACE__
        end
      end,
      defp encode_graph(acc, msg) do
        try do
          if msg.graph == nil do
            acc
          else
            [acc, ":", Protox.Encode.encode_message(msg.graph)]
          end
        rescue
          ArgumentError ->
            reraise Protox.EncodingError.new(:graph, "invalid field value"), __STACKTRACE__
        end
      end,
      defp encode_opset_import(acc, msg) do
        try do
          case msg.opset_import do
            [] ->
              acc

            values ->
              [
                acc,
                Enum.reduce(values, [], fn value, acc ->
                  [acc, "B", Protox.Encode.encode_message(value)]
                end)
              ]
          end
        rescue
          ArgumentError ->
            reraise Protox.EncodingError.new(:opset_import, "invalid field value"), __STACKTRACE__
        end
      end,
      defp encode_metadata_props(acc, msg) do
        try do
          case msg.metadata_props do
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
            reraise Protox.EncodingError.new(:metadata_props, "invalid field value"),
                    __STACKTRACE__
        end
      end,
      defp encode_training_info(acc, msg) do
        try do
          case msg.training_info do
            [] ->
              acc

            values ->
              [
                acc,
                Enum.reduce(values, [], fn value, acc ->
                  [acc, "\xA2\x01", Protox.Encode.encode_message(value)]
                end)
              ]
          end
        rescue
          ArgumentError ->
            reraise Protox.EncodingError.new(:training_info, "invalid field value"),
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
          parse_key_value(bytes, struct(Onnx.ModelProto))
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
              {value, rest} = Protox.Decode.parse_int64(bytes)
              {[ir_version: value], rest}

            {2, _, bytes} ->
              {len, bytes} = Protox.Varint.decode(bytes)
              {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)
              {[producer_name: delimited], rest}

            {3, _, bytes} ->
              {len, bytes} = Protox.Varint.decode(bytes)
              {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)
              {[producer_version: delimited], rest}

            {4, _, bytes} ->
              {len, bytes} = Protox.Varint.decode(bytes)
              {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)
              {[domain: delimited], rest}

            {5, _, bytes} ->
              {value, rest} = Protox.Decode.parse_int64(bytes)
              {[model_version: value], rest}

            {6, _, bytes} ->
              {len, bytes} = Protox.Varint.decode(bytes)
              {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)
              {[doc_string: delimited], rest}

            {7, _, bytes} ->
              {len, bytes} = Protox.Varint.decode(bytes)
              {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)

              {[graph: Protox.MergeMessage.merge(msg.graph, Onnx.GraphProto.decode!(delimited))],
               rest}

            {8, _, bytes} ->
              {len, bytes} = Protox.Varint.decode(bytes)
              {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)

              {[opset_import: msg.opset_import ++ [Onnx.OperatorSetIdProto.decode!(delimited)]],
               rest}

            {14, _, bytes} ->
              {len, bytes} = Protox.Varint.decode(bytes)
              {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)

              {[
                 metadata_props:
                   msg.metadata_props ++ [Onnx.StringStringEntryProto.decode!(delimited)]
               ], rest}

            {20, _, bytes} ->
              {len, bytes} = Protox.Varint.decode(bytes)
              {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)

              {[training_info: msg.training_info ++ [Onnx.TrainingInfoProto.decode!(delimited)]],
               rest}

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
        Onnx.ModelProto,
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
          json_name: "irVersion",
          kind: {:scalar, 0},
          label: :optional,
          name: :ir_version,
          tag: 1,
          type: :int64
        },
        %{
          __struct__: Protox.Field,
          json_name: "producerName",
          kind: {:scalar, ""},
          label: :optional,
          name: :producer_name,
          tag: 2,
          type: :string
        },
        %{
          __struct__: Protox.Field,
          json_name: "producerVersion",
          kind: {:scalar, ""},
          label: :optional,
          name: :producer_version,
          tag: 3,
          type: :string
        },
        %{
          __struct__: Protox.Field,
          json_name: "domain",
          kind: {:scalar, ""},
          label: :optional,
          name: :domain,
          tag: 4,
          type: :string
        },
        %{
          __struct__: Protox.Field,
          json_name: "modelVersion",
          kind: {:scalar, 0},
          label: :optional,
          name: :model_version,
          tag: 5,
          type: :int64
        },
        %{
          __struct__: Protox.Field,
          json_name: "docString",
          kind: {:scalar, ""},
          label: :optional,
          name: :doc_string,
          tag: 6,
          type: :string
        },
        %{
          __struct__: Protox.Field,
          json_name: "graph",
          kind: {:scalar, nil},
          label: :optional,
          name: :graph,
          tag: 7,
          type: {:message, Onnx.GraphProto}
        },
        %{
          __struct__: Protox.Field,
          json_name: "opsetImport",
          kind: :unpacked,
          label: :repeated,
          name: :opset_import,
          tag: 8,
          type: {:message, Onnx.OperatorSetIdProto}
        },
        %{
          __struct__: Protox.Field,
          json_name: "metadataProps",
          kind: :unpacked,
          label: :repeated,
          name: :metadata_props,
          tag: 14,
          type: {:message, Onnx.StringStringEntryProto}
        },
        %{
          __struct__: Protox.Field,
          json_name: "trainingInfo",
          kind: :unpacked,
          label: :repeated,
          name: :training_info,
          tag: 20,
          type: {:message, Onnx.TrainingInfoProto}
        }
      ]
    end

    [
      @spec(field_def(atom) :: {:ok, Protox.Field.t()} | {:error, :no_such_field}),
      (
        def field_def(:ir_version) do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "irVersion",
             kind: {:scalar, 0},
             label: :optional,
             name: :ir_version,
             tag: 1,
             type: :int64
           }}
        end

        def field_def("irVersion") do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "irVersion",
             kind: {:scalar, 0},
             label: :optional,
             name: :ir_version,
             tag: 1,
             type: :int64
           }}
        end

        def field_def("ir_version") do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "irVersion",
             kind: {:scalar, 0},
             label: :optional,
             name: :ir_version,
             tag: 1,
             type: :int64
           }}
        end
      ),
      (
        def field_def(:producer_name) do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "producerName",
             kind: {:scalar, ""},
             label: :optional,
             name: :producer_name,
             tag: 2,
             type: :string
           }}
        end

        def field_def("producerName") do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "producerName",
             kind: {:scalar, ""},
             label: :optional,
             name: :producer_name,
             tag: 2,
             type: :string
           }}
        end

        def field_def("producer_name") do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "producerName",
             kind: {:scalar, ""},
             label: :optional,
             name: :producer_name,
             tag: 2,
             type: :string
           }}
        end
      ),
      (
        def field_def(:producer_version) do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "producerVersion",
             kind: {:scalar, ""},
             label: :optional,
             name: :producer_version,
             tag: 3,
             type: :string
           }}
        end

        def field_def("producerVersion") do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "producerVersion",
             kind: {:scalar, ""},
             label: :optional,
             name: :producer_version,
             tag: 3,
             type: :string
           }}
        end

        def field_def("producer_version") do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "producerVersion",
             kind: {:scalar, ""},
             label: :optional,
             name: :producer_version,
             tag: 3,
             type: :string
           }}
        end
      ),
      (
        def field_def(:domain) do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "domain",
             kind: {:scalar, ""},
             label: :optional,
             name: :domain,
             tag: 4,
             type: :string
           }}
        end

        def field_def("domain") do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "domain",
             kind: {:scalar, ""},
             label: :optional,
             name: :domain,
             tag: 4,
             type: :string
           }}
        end

        []
      ),
      (
        def field_def(:model_version) do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "modelVersion",
             kind: {:scalar, 0},
             label: :optional,
             name: :model_version,
             tag: 5,
             type: :int64
           }}
        end

        def field_def("modelVersion") do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "modelVersion",
             kind: {:scalar, 0},
             label: :optional,
             name: :model_version,
             tag: 5,
             type: :int64
           }}
        end

        def field_def("model_version") do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "modelVersion",
             kind: {:scalar, 0},
             label: :optional,
             name: :model_version,
             tag: 5,
             type: :int64
           }}
        end
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
             tag: 6,
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
             tag: 6,
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
             tag: 6,
             type: :string
           }}
        end
      ),
      (
        def field_def(:graph) do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "graph",
             kind: {:scalar, nil},
             label: :optional,
             name: :graph,
             tag: 7,
             type: {:message, Onnx.GraphProto}
           }}
        end

        def field_def("graph") do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "graph",
             kind: {:scalar, nil},
             label: :optional,
             name: :graph,
             tag: 7,
             type: {:message, Onnx.GraphProto}
           }}
        end

        []
      ),
      (
        def field_def(:opset_import) do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "opsetImport",
             kind: :unpacked,
             label: :repeated,
             name: :opset_import,
             tag: 8,
             type: {:message, Onnx.OperatorSetIdProto}
           }}
        end

        def field_def("opsetImport") do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "opsetImport",
             kind: :unpacked,
             label: :repeated,
             name: :opset_import,
             tag: 8,
             type: {:message, Onnx.OperatorSetIdProto}
           }}
        end

        def field_def("opset_import") do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "opsetImport",
             kind: :unpacked,
             label: :repeated,
             name: :opset_import,
             tag: 8,
             type: {:message, Onnx.OperatorSetIdProto}
           }}
        end
      ),
      (
        def field_def(:metadata_props) do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "metadataProps",
             kind: :unpacked,
             label: :repeated,
             name: :metadata_props,
             tag: 14,
             type: {:message, Onnx.StringStringEntryProto}
           }}
        end

        def field_def("metadataProps") do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "metadataProps",
             kind: :unpacked,
             label: :repeated,
             name: :metadata_props,
             tag: 14,
             type: {:message, Onnx.StringStringEntryProto}
           }}
        end

        def field_def("metadata_props") do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "metadataProps",
             kind: :unpacked,
             label: :repeated,
             name: :metadata_props,
             tag: 14,
             type: {:message, Onnx.StringStringEntryProto}
           }}
        end
      ),
      (
        def field_def(:training_info) do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "trainingInfo",
             kind: :unpacked,
             label: :repeated,
             name: :training_info,
             tag: 20,
             type: {:message, Onnx.TrainingInfoProto}
           }}
        end

        def field_def("trainingInfo") do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "trainingInfo",
             kind: :unpacked,
             label: :repeated,
             name: :training_info,
             tag: 20,
             type: {:message, Onnx.TrainingInfoProto}
           }}
        end

        def field_def("training_info") do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "trainingInfo",
             kind: :unpacked,
             label: :repeated,
             name: :training_info,
             tag: 20,
             type: {:message, Onnx.TrainingInfoProto}
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
    def default(:ir_version) do
      {:ok, 0}
    end,
    def default(:producer_name) do
      {:ok, ""}
    end,
    def default(:producer_version) do
      {:ok, ""}
    end,
    def default(:domain) do
      {:ok, ""}
    end,
    def default(:model_version) do
      {:ok, 0}
    end,
    def default(:doc_string) do
      {:ok, ""}
    end,
    def default(:graph) do
      {:ok, nil}
    end,
    def default(:opset_import) do
      {:error, :no_default_value}
    end,
    def default(:metadata_props) do
      {:error, :no_default_value}
    end,
    def default(:training_info) do
      {:error, :no_default_value}
    end,
    def default(_) do
      {:error, :no_such_field}
    end
  ]
end
