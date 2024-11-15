# credo:disable-for-this-file
defmodule Onnx.SparseTensorProto do
  @moduledoc false
  defstruct values: nil, indices: nil, dims: []

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
        [] |> encode_values(msg) |> encode_indices(msg) |> encode_dims(msg)
      end
    )

    []

    [
      defp encode_values(acc, msg) do
        try do
          if msg.values == nil do
            acc
          else
            [acc, "\n", Protox.Encode.encode_message(msg.values)]
          end
        rescue
          ArgumentError ->
            reraise Protox.EncodingError.new(:values, "invalid field value"), __STACKTRACE__
        end
      end,
      defp encode_indices(acc, msg) do
        try do
          if msg.indices == nil do
            acc
          else
            [acc, "\x12", Protox.Encode.encode_message(msg.indices)]
          end
        rescue
          ArgumentError ->
            reraise Protox.EncodingError.new(:indices, "invalid field value"), __STACKTRACE__
        end
      end,
      defp encode_dims(acc, msg) do
        try do
          case msg.dims do
            [] ->
              acc

            values ->
              [
                acc,
                "\x1A",
                (
                  {bytes, len} =
                    Enum.reduce(values, {[], 0}, fn value, {acc, len} ->
                      value_bytes = :binary.list_to_bin([Protox.Encode.encode_int64(value)])
                      {[acc, value_bytes], len + byte_size(value_bytes)}
                    end)

                  [Protox.Varint.encode(len), bytes]
                )
              ]
          end
        rescue
          ArgumentError ->
            reraise Protox.EncodingError.new(:dims, "invalid field value"), __STACKTRACE__
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
          parse_key_value(bytes, struct(Onnx.SparseTensorProto))
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
                 values:
                   Protox.MergeMessage.merge(msg.values, Onnx.TensorProto.decode!(delimited))
               ], rest}

            {2, _, bytes} ->
              {len, bytes} = Protox.Varint.decode(bytes)
              {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)

              {[
                 indices:
                   Protox.MergeMessage.merge(msg.indices, Onnx.TensorProto.decode!(delimited))
               ], rest}

            {3, 2, bytes} ->
              {len, bytes} = Protox.Varint.decode(bytes)
              {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)
              {[dims: msg.dims ++ Protox.Decode.parse_repeated_int64([], delimited)], rest}

            {3, _, bytes} ->
              {value, rest} = Protox.Decode.parse_int64(bytes)
              {[dims: msg.dims ++ [value]], rest}

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
        Onnx.SparseTensorProto,
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
          json_name: "values",
          kind: {:scalar, nil},
          label: :optional,
          name: :values,
          tag: 1,
          type: {:message, Onnx.TensorProto}
        },
        %{
          __struct__: Protox.Field,
          json_name: "indices",
          kind: {:scalar, nil},
          label: :optional,
          name: :indices,
          tag: 2,
          type: {:message, Onnx.TensorProto}
        },
        %{
          __struct__: Protox.Field,
          json_name: "dims",
          kind: :packed,
          label: :repeated,
          name: :dims,
          tag: 3,
          type: :int64
        }
      ]
    end

    [
      @spec(field_def(atom) :: {:ok, Protox.Field.t()} | {:error, :no_such_field}),
      (
        def field_def(:values) do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "values",
             kind: {:scalar, nil},
             label: :optional,
             name: :values,
             tag: 1,
             type: {:message, Onnx.TensorProto}
           }}
        end

        def field_def("values") do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "values",
             kind: {:scalar, nil},
             label: :optional,
             name: :values,
             tag: 1,
             type: {:message, Onnx.TensorProto}
           }}
        end

        []
      ),
      (
        def field_def(:indices) do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "indices",
             kind: {:scalar, nil},
             label: :optional,
             name: :indices,
             tag: 2,
             type: {:message, Onnx.TensorProto}
           }}
        end

        def field_def("indices") do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "indices",
             kind: {:scalar, nil},
             label: :optional,
             name: :indices,
             tag: 2,
             type: {:message, Onnx.TensorProto}
           }}
        end

        []
      ),
      (
        def field_def(:dims) do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "dims",
             kind: :packed,
             label: :repeated,
             name: :dims,
             tag: 3,
             type: :int64
           }}
        end

        def field_def("dims") do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "dims",
             kind: :packed,
             label: :repeated,
             name: :dims,
             tag: 3,
             type: :int64
           }}
        end

        []
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
    def default(:values) do
      {:ok, nil}
    end,
    def default(:indices) do
      {:ok, nil}
    end,
    def default(:dims) do
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
