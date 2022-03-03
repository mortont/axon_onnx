# credo:disable-for-this-file
defmodule Onnx.TypeProto.SparseTensor do
  @moduledoc false
  defstruct elem_type: 0, shape: nil

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
        [] |> encode_elem_type(msg) |> encode_shape(msg)
      end
    )

    []

    [
      defp encode_elem_type(acc, msg) do
        try do
          if msg.elem_type == 0 do
            acc
          else
            [acc, "\b", Protox.Encode.encode_int32(msg.elem_type)]
          end
        rescue
          ArgumentError ->
            reraise Protox.EncodingError.new(:elem_type, "invalid field value"), __STACKTRACE__
        end
      end,
      defp encode_shape(acc, msg) do
        try do
          if msg.shape == nil do
            acc
          else
            [acc, "\x12", Protox.Encode.encode_message(msg.shape)]
          end
        rescue
          ArgumentError ->
            reraise Protox.EncodingError.new(:shape, "invalid field value"), __STACKTRACE__
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
          parse_key_value(bytes, struct(Onnx.TypeProto.SparseTensor))
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
              {value, rest} = Protox.Decode.parse_int32(bytes)
              {[elem_type: value], rest}

            {2, _, bytes} ->
              {len, bytes} = Protox.Varint.decode(bytes)
              {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)

              {[
                 shape:
                   Protox.MergeMessage.merge(msg.shape, Onnx.TensorShapeProto.decode!(delimited))
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
        Onnx.TypeProto.SparseTensor,
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
          json_name: "elemType",
          kind: {:scalar, 0},
          label: :optional,
          name: :elem_type,
          tag: 1,
          type: :int32
        },
        %{
          __struct__: Protox.Field,
          json_name: "shape",
          kind: {:scalar, nil},
          label: :optional,
          name: :shape,
          tag: 2,
          type: {:message, Onnx.TensorShapeProto}
        }
      ]
    end

    [
      @spec(field_def(atom) :: {:ok, Protox.Field.t()} | {:error, :no_such_field}),
      (
        def field_def(:elem_type) do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "elemType",
             kind: {:scalar, 0},
             label: :optional,
             name: :elem_type,
             tag: 1,
             type: :int32
           }}
        end

        def field_def("elemType") do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "elemType",
             kind: {:scalar, 0},
             label: :optional,
             name: :elem_type,
             tag: 1,
             type: :int32
           }}
        end

        def field_def("elem_type") do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "elemType",
             kind: {:scalar, 0},
             label: :optional,
             name: :elem_type,
             tag: 1,
             type: :int32
           }}
        end
      ),
      (
        def field_def(:shape) do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "shape",
             kind: {:scalar, nil},
             label: :optional,
             name: :shape,
             tag: 2,
             type: {:message, Onnx.TensorShapeProto}
           }}
        end

        def field_def("shape") do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "shape",
             kind: {:scalar, nil},
             label: :optional,
             name: :shape,
             tag: 2,
             type: {:message, Onnx.TensorShapeProto}
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
    def default(:elem_type) do
      {:ok, 0}
    end,
    def default(:shape) do
      {:ok, nil}
    end,
    def default(_) do
      {:error, :no_such_field}
    end
  ]
end
