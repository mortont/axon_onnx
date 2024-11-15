# credo:disable-for-this-file
defmodule Onnx.TensorAnnotation do
  @moduledoc false
  defstruct tensor_name: "", quant_parameter_tensor_names: []

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
        [] |> encode_tensor_name(msg) |> encode_quant_parameter_tensor_names(msg)
      end
    )

    []

    [
      defp encode_tensor_name(acc, msg) do
        try do
          if msg.tensor_name == "" do
            acc
          else
            [acc, "\n", Protox.Encode.encode_string(msg.tensor_name)]
          end
        rescue
          ArgumentError ->
            reraise Protox.EncodingError.new(:tensor_name, "invalid field value"), __STACKTRACE__
        end
      end,
      defp encode_quant_parameter_tensor_names(acc, msg) do
        try do
          case msg.quant_parameter_tensor_names do
            [] ->
              acc

            values ->
              [
                acc,
                Enum.reduce(values, [], fn value, acc ->
                  [acc, "\x12", Protox.Encode.encode_message(value)]
                end)
              ]
          end
        rescue
          ArgumentError ->
            reraise Protox.EncodingError.new(
                      :quant_parameter_tensor_names,
                      "invalid field value"
                    ),
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
          parse_key_value(bytes, struct(Onnx.TensorAnnotation))
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
              {[tensor_name: Protox.Decode.validate_string!(delimited)], rest}

            {2, _, bytes} ->
              {len, bytes} = Protox.Varint.decode(bytes)
              {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)

              {[
                 quant_parameter_tensor_names:
                   msg.quant_parameter_tensor_names ++
                     [Onnx.StringStringEntryProto.decode!(delimited)]
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
        Onnx.TensorAnnotation,
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
          json_name: "tensorName",
          kind: {:scalar, ""},
          label: :optional,
          name: :tensor_name,
          tag: 1,
          type: :string
        },
        %{
          __struct__: Protox.Field,
          json_name: "quantParameterTensorNames",
          kind: :unpacked,
          label: :repeated,
          name: :quant_parameter_tensor_names,
          tag: 2,
          type: {:message, Onnx.StringStringEntryProto}
        }
      ]
    end

    [
      @spec(field_def(atom) :: {:ok, Protox.Field.t()} | {:error, :no_such_field}),
      (
        def field_def(:tensor_name) do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "tensorName",
             kind: {:scalar, ""},
             label: :optional,
             name: :tensor_name,
             tag: 1,
             type: :string
           }}
        end

        def field_def("tensorName") do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "tensorName",
             kind: {:scalar, ""},
             label: :optional,
             name: :tensor_name,
             tag: 1,
             type: :string
           }}
        end

        def field_def("tensor_name") do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "tensorName",
             kind: {:scalar, ""},
             label: :optional,
             name: :tensor_name,
             tag: 1,
             type: :string
           }}
        end
      ),
      (
        def field_def(:quant_parameter_tensor_names) do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "quantParameterTensorNames",
             kind: :unpacked,
             label: :repeated,
             name: :quant_parameter_tensor_names,
             tag: 2,
             type: {:message, Onnx.StringStringEntryProto}
           }}
        end

        def field_def("quantParameterTensorNames") do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "quantParameterTensorNames",
             kind: :unpacked,
             label: :repeated,
             name: :quant_parameter_tensor_names,
             tag: 2,
             type: {:message, Onnx.StringStringEntryProto}
           }}
        end

        def field_def("quant_parameter_tensor_names") do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "quantParameterTensorNames",
             kind: :unpacked,
             label: :repeated,
             name: :quant_parameter_tensor_names,
             tag: 2,
             type: {:message, Onnx.StringStringEntryProto}
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
    def default(:tensor_name) do
      {:ok, ""}
    end,
    def default(:quant_parameter_tensor_names) do
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
