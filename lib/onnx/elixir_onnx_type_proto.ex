# credo:disable-for-this-file
defmodule Onnx.TypeProto do
  @moduledoc false
  defstruct value: nil, denotation: ""

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
        [] |> encode_value(msg) |> encode_denotation(msg)
      end
    )

    [
      defp encode_value(acc, msg) do
        case msg.value do
          nil -> acc
          {:tensor_type, _field_value} -> encode_tensor_type(acc, msg)
          {:sequence_type, _field_value} -> encode_sequence_type(acc, msg)
          {:map_type, _field_value} -> encode_map_type(acc, msg)
          {:sparse_tensor_type, _field_value} -> encode_sparse_tensor_type(acc, msg)
          {:optional_type, _field_value} -> encode_optional_type(acc, msg)
        end
      end
    ]

    [
      defp encode_tensor_type(acc, msg) do
        try do
          {_, child_field_value} = msg.value
          [acc, "\n", Protox.Encode.encode_message(child_field_value)]
        rescue
          ArgumentError ->
            reraise Protox.EncodingError.new(:tensor_type, "invalid field value"), __STACKTRACE__
        end
      end,
      defp encode_sequence_type(acc, msg) do
        try do
          {_, child_field_value} = msg.value
          [acc, "\"", Protox.Encode.encode_message(child_field_value)]
        rescue
          ArgumentError ->
            reraise Protox.EncodingError.new(:sequence_type, "invalid field value"),
                    __STACKTRACE__
        end
      end,
      defp encode_map_type(acc, msg) do
        try do
          {_, child_field_value} = msg.value
          [acc, "*", Protox.Encode.encode_message(child_field_value)]
        rescue
          ArgumentError ->
            reraise Protox.EncodingError.new(:map_type, "invalid field value"), __STACKTRACE__
        end
      end,
      defp encode_denotation(acc, msg) do
        try do
          if msg.denotation == "" do
            acc
          else
            [acc, "2", Protox.Encode.encode_string(msg.denotation)]
          end
        rescue
          ArgumentError ->
            reraise Protox.EncodingError.new(:denotation, "invalid field value"), __STACKTRACE__
        end
      end,
      defp encode_sparse_tensor_type(acc, msg) do
        try do
          {_, child_field_value} = msg.value
          [acc, "B", Protox.Encode.encode_message(child_field_value)]
        rescue
          ArgumentError ->
            reraise Protox.EncodingError.new(:sparse_tensor_type, "invalid field value"),
                    __STACKTRACE__
        end
      end,
      defp encode_optional_type(acc, msg) do
        try do
          {_, child_field_value} = msg.value
          [acc, "J", Protox.Encode.encode_message(child_field_value)]
        rescue
          ArgumentError ->
            reraise Protox.EncodingError.new(:optional_type, "invalid field value"),
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
          parse_key_value(bytes, struct(Onnx.TypeProto))
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

              {[
                 case msg.value do
                   {:tensor_type, previous_value} ->
                     {:value,
                      {:tensor_type,
                       Protox.MergeMessage.merge(
                         previous_value,
                         Onnx.TypeProto.Tensor.decode!(delimited)
                       )}}

                   _ ->
                     {:value, {:tensor_type, Onnx.TypeProto.Tensor.decode!(delimited)}}
                 end
               ], rest}

            {4, _, bytes} ->
              {len, bytes} = Protox.Varint.decode(bytes)
              {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)

              {[
                 case msg.value do
                   {:sequence_type, previous_value} ->
                     {:value,
                      {:sequence_type,
                       Protox.MergeMessage.merge(
                         previous_value,
                         Onnx.TypeProto.Sequence.decode!(delimited)
                       )}}

                   _ ->
                     {:value, {:sequence_type, Onnx.TypeProto.Sequence.decode!(delimited)}}
                 end
               ], rest}

            {5, _, bytes} ->
              {len, bytes} = Protox.Varint.decode(bytes)
              {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)

              {[
                 case msg.value do
                   {:map_type, previous_value} ->
                     {:value,
                      {:map_type,
                       Protox.MergeMessage.merge(
                         previous_value,
                         Onnx.TypeProto.Map.decode!(delimited)
                       )}}

                   _ ->
                     {:value, {:map_type, Onnx.TypeProto.Map.decode!(delimited)}}
                 end
               ], rest}

            {6, _, bytes} ->
              {len, bytes} = Protox.Varint.decode(bytes)
              {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)
              {[denotation: Protox.Decode.validate_string!(delimited)], rest}

            {8, _, bytes} ->
              {len, bytes} = Protox.Varint.decode(bytes)
              {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)

              {[
                 case msg.value do
                   {:sparse_tensor_type, previous_value} ->
                     {:value,
                      {:sparse_tensor_type,
                       Protox.MergeMessage.merge(
                         previous_value,
                         Onnx.TypeProto.SparseTensor.decode!(delimited)
                       )}}

                   _ ->
                     {:value,
                      {:sparse_tensor_type, Onnx.TypeProto.SparseTensor.decode!(delimited)}}
                 end
               ], rest}

            {9, _, bytes} ->
              {len, bytes} = Protox.Varint.decode(bytes)
              {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)

              {[
                 case msg.value do
                   {:optional_type, previous_value} ->
                     {:value,
                      {:optional_type,
                       Protox.MergeMessage.merge(
                         previous_value,
                         Onnx.TypeProto.Optional.decode!(delimited)
                       )}}

                   _ ->
                     {:value, {:optional_type, Onnx.TypeProto.Optional.decode!(delimited)}}
                 end
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
        Onnx.TypeProto,
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
          json_name: "tensorType",
          kind: {:oneof, :value},
          label: :optional,
          name: :tensor_type,
          tag: 1,
          type: {:message, Onnx.TypeProto.Tensor}
        },
        %{
          __struct__: Protox.Field,
          json_name: "sequenceType",
          kind: {:oneof, :value},
          label: :optional,
          name: :sequence_type,
          tag: 4,
          type: {:message, Onnx.TypeProto.Sequence}
        },
        %{
          __struct__: Protox.Field,
          json_name: "mapType",
          kind: {:oneof, :value},
          label: :optional,
          name: :map_type,
          tag: 5,
          type: {:message, Onnx.TypeProto.Map}
        },
        %{
          __struct__: Protox.Field,
          json_name: "denotation",
          kind: {:scalar, ""},
          label: :optional,
          name: :denotation,
          tag: 6,
          type: :string
        },
        %{
          __struct__: Protox.Field,
          json_name: "sparseTensorType",
          kind: {:oneof, :value},
          label: :optional,
          name: :sparse_tensor_type,
          tag: 8,
          type: {:message, Onnx.TypeProto.SparseTensor}
        },
        %{
          __struct__: Protox.Field,
          json_name: "optionalType",
          kind: {:oneof, :value},
          label: :optional,
          name: :optional_type,
          tag: 9,
          type: {:message, Onnx.TypeProto.Optional}
        }
      ]
    end

    [
      @spec(field_def(atom) :: {:ok, Protox.Field.t()} | {:error, :no_such_field}),
      (
        def field_def(:tensor_type) do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "tensorType",
             kind: {:oneof, :value},
             label: :optional,
             name: :tensor_type,
             tag: 1,
             type: {:message, Onnx.TypeProto.Tensor}
           }}
        end

        def field_def("tensorType") do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "tensorType",
             kind: {:oneof, :value},
             label: :optional,
             name: :tensor_type,
             tag: 1,
             type: {:message, Onnx.TypeProto.Tensor}
           }}
        end

        def field_def("tensor_type") do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "tensorType",
             kind: {:oneof, :value},
             label: :optional,
             name: :tensor_type,
             tag: 1,
             type: {:message, Onnx.TypeProto.Tensor}
           }}
        end
      ),
      (
        def field_def(:sequence_type) do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "sequenceType",
             kind: {:oneof, :value},
             label: :optional,
             name: :sequence_type,
             tag: 4,
             type: {:message, Onnx.TypeProto.Sequence}
           }}
        end

        def field_def("sequenceType") do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "sequenceType",
             kind: {:oneof, :value},
             label: :optional,
             name: :sequence_type,
             tag: 4,
             type: {:message, Onnx.TypeProto.Sequence}
           }}
        end

        def field_def("sequence_type") do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "sequenceType",
             kind: {:oneof, :value},
             label: :optional,
             name: :sequence_type,
             tag: 4,
             type: {:message, Onnx.TypeProto.Sequence}
           }}
        end
      ),
      (
        def field_def(:map_type) do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "mapType",
             kind: {:oneof, :value},
             label: :optional,
             name: :map_type,
             tag: 5,
             type: {:message, Onnx.TypeProto.Map}
           }}
        end

        def field_def("mapType") do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "mapType",
             kind: {:oneof, :value},
             label: :optional,
             name: :map_type,
             tag: 5,
             type: {:message, Onnx.TypeProto.Map}
           }}
        end

        def field_def("map_type") do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "mapType",
             kind: {:oneof, :value},
             label: :optional,
             name: :map_type,
             tag: 5,
             type: {:message, Onnx.TypeProto.Map}
           }}
        end
      ),
      (
        def field_def(:denotation) do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "denotation",
             kind: {:scalar, ""},
             label: :optional,
             name: :denotation,
             tag: 6,
             type: :string
           }}
        end

        def field_def("denotation") do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "denotation",
             kind: {:scalar, ""},
             label: :optional,
             name: :denotation,
             tag: 6,
             type: :string
           }}
        end

        []
      ),
      (
        def field_def(:sparse_tensor_type) do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "sparseTensorType",
             kind: {:oneof, :value},
             label: :optional,
             name: :sparse_tensor_type,
             tag: 8,
             type: {:message, Onnx.TypeProto.SparseTensor}
           }}
        end

        def field_def("sparseTensorType") do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "sparseTensorType",
             kind: {:oneof, :value},
             label: :optional,
             name: :sparse_tensor_type,
             tag: 8,
             type: {:message, Onnx.TypeProto.SparseTensor}
           }}
        end

        def field_def("sparse_tensor_type") do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "sparseTensorType",
             kind: {:oneof, :value},
             label: :optional,
             name: :sparse_tensor_type,
             tag: 8,
             type: {:message, Onnx.TypeProto.SparseTensor}
           }}
        end
      ),
      (
        def field_def(:optional_type) do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "optionalType",
             kind: {:oneof, :value},
             label: :optional,
             name: :optional_type,
             tag: 9,
             type: {:message, Onnx.TypeProto.Optional}
           }}
        end

        def field_def("optionalType") do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "optionalType",
             kind: {:oneof, :value},
             label: :optional,
             name: :optional_type,
             tag: 9,
             type: {:message, Onnx.TypeProto.Optional}
           }}
        end

        def field_def("optional_type") do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "optionalType",
             kind: {:oneof, :value},
             label: :optional,
             name: :optional_type,
             tag: 9,
             type: {:message, Onnx.TypeProto.Optional}
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
    def default(:tensor_type) do
      {:error, :no_default_value}
    end,
    def default(:sequence_type) do
      {:error, :no_default_value}
    end,
    def default(:map_type) do
      {:error, :no_default_value}
    end,
    def default(:denotation) do
      {:ok, ""}
    end,
    def default(:sparse_tensor_type) do
      {:error, :no_default_value}
    end,
    def default(:optional_type) do
      {:error, :no_default_value}
    end,
    def default(_) do
      {:error, :no_such_field}
    end
  ]

  (
    @spec file_options() :: struct()
    def file_options() do
      file_options = %{
        __struct__: Protox.Google.Protobuf.FileOptions,
        __uf__: [],
        cc_enable_arenas: nil,
        cc_generic_services: nil,
        csharp_namespace: nil,
        deprecated: nil,
        go_package: nil,
        java_generate_equals_and_hash: nil,
        java_generic_services: nil,
        java_multiple_files: nil,
        java_outer_classname: nil,
        java_package: nil,
        java_string_check_utf8: nil,
        objc_class_prefix: nil,
        optimize_for: :LITE_RUNTIME,
        php_class_prefix: nil,
        php_generic_services: nil,
        php_metadata_namespace: nil,
        php_namespace: nil,
        py_generic_services: nil,
        ruby_package: nil,
        swift_prefix: nil,
        uninterpreted_option: []
      }

      case function_exported?(Google.Protobuf.FileOptions, :decode!, 1) do
        true ->
          bytes =
            file_options |> Protox.Google.Protobuf.FileOptions.encode!() |> :binary.list_to_bin()

          apply(Google.Protobuf.FileOptions, :decode!, [bytes])

        false ->
          file_options
      end
    end
  )
end
