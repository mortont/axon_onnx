# credo:disable-for-this-file
defmodule Onnx.TensorShapeProto.Dimension do
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
          {:dim_value, _field_value} -> encode_dim_value(acc, msg)
          {:dim_param, _field_value} -> encode_dim_param(acc, msg)
        end
      end
    ]

    [
      defp encode_dim_value(acc, msg) do
        try do
          {_, child_field_value} = msg.value
          [acc, "\b", Protox.Encode.encode_int64(child_field_value)]
        rescue
          ArgumentError ->
            reraise Protox.EncodingError.new(:dim_value, "invalid field value"), __STACKTRACE__
        end
      end,
      defp encode_dim_param(acc, msg) do
        try do
          {_, child_field_value} = msg.value
          [acc, "\x12", Protox.Encode.encode_string(child_field_value)]
        rescue
          ArgumentError ->
            reraise Protox.EncodingError.new(:dim_param, "invalid field value"), __STACKTRACE__
        end
      end,
      defp encode_denotation(acc, msg) do
        try do
          if msg.denotation == "" do
            acc
          else
            [acc, "\x1A", Protox.Encode.encode_string(msg.denotation)]
          end
        rescue
          ArgumentError ->
            reraise Protox.EncodingError.new(:denotation, "invalid field value"), __STACKTRACE__
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
          parse_key_value(bytes, struct(Onnx.TensorShapeProto.Dimension))
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
              {[value: {:dim_value, value}], rest}

            {2, _, bytes} ->
              {len, bytes} = Protox.Varint.decode(bytes)
              {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)
              {[value: {:dim_param, Protox.Decode.validate_string!(delimited)}], rest}

            {3, _, bytes} ->
              {len, bytes} = Protox.Varint.decode(bytes)
              {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)
              {[denotation: Protox.Decode.validate_string!(delimited)], rest}

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
        Onnx.TensorShapeProto.Dimension,
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
          json_name: "dimValue",
          kind: {:oneof, :value},
          label: :optional,
          name: :dim_value,
          tag: 1,
          type: :int64
        },
        %{
          __struct__: Protox.Field,
          json_name: "dimParam",
          kind: {:oneof, :value},
          label: :optional,
          name: :dim_param,
          tag: 2,
          type: :string
        },
        %{
          __struct__: Protox.Field,
          json_name: "denotation",
          kind: {:scalar, ""},
          label: :optional,
          name: :denotation,
          tag: 3,
          type: :string
        }
      ]
    end

    [
      @spec(field_def(atom) :: {:ok, Protox.Field.t()} | {:error, :no_such_field}),
      (
        def field_def(:dim_value) do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "dimValue",
             kind: {:oneof, :value},
             label: :optional,
             name: :dim_value,
             tag: 1,
             type: :int64
           }}
        end

        def field_def("dimValue") do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "dimValue",
             kind: {:oneof, :value},
             label: :optional,
             name: :dim_value,
             tag: 1,
             type: :int64
           }}
        end

        def field_def("dim_value") do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "dimValue",
             kind: {:oneof, :value},
             label: :optional,
             name: :dim_value,
             tag: 1,
             type: :int64
           }}
        end
      ),
      (
        def field_def(:dim_param) do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "dimParam",
             kind: {:oneof, :value},
             label: :optional,
             name: :dim_param,
             tag: 2,
             type: :string
           }}
        end

        def field_def("dimParam") do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "dimParam",
             kind: {:oneof, :value},
             label: :optional,
             name: :dim_param,
             tag: 2,
             type: :string
           }}
        end

        def field_def("dim_param") do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "dimParam",
             kind: {:oneof, :value},
             label: :optional,
             name: :dim_param,
             tag: 2,
             type: :string
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
             tag: 3,
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
             tag: 3,
             type: :string
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
    def default(:dim_value) do
      {:error, :no_default_value}
    end,
    def default(:dim_param) do
      {:error, :no_default_value}
    end,
    def default(:denotation) do
      {:ok, ""}
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
