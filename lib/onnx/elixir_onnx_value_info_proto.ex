# credo:disable-for-this-file
defmodule Onnx.ValueInfoProto do
  @moduledoc false
  defstruct name: "", type: nil, doc_string: ""

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
        [] |> encode_name(msg) |> encode_type(msg) |> encode_doc_string(msg)
      end
    )

    []

    [
      defp encode_name(acc, msg) do
        try do
          if msg.name == "" do
            acc
          else
            [acc, "\n", Protox.Encode.encode_string(msg.name)]
          end
        rescue
          ArgumentError ->
            reraise Protox.EncodingError.new(:name, "invalid field value"), __STACKTRACE__
        end
      end,
      defp encode_type(acc, msg) do
        try do
          if msg.type == nil do
            acc
          else
            [acc, "\x12", Protox.Encode.encode_message(msg.type)]
          end
        rescue
          ArgumentError ->
            reraise Protox.EncodingError.new(:type, "invalid field value"), __STACKTRACE__
        end
      end,
      defp encode_doc_string(acc, msg) do
        try do
          if msg.doc_string == "" do
            acc
          else
            [acc, "\x1A", Protox.Encode.encode_string(msg.doc_string)]
          end
        rescue
          ArgumentError ->
            reraise Protox.EncodingError.new(:doc_string, "invalid field value"), __STACKTRACE__
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
          parse_key_value(bytes, struct(Onnx.ValueInfoProto))
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
              {[name: delimited], rest}

            {2, _, bytes} ->
              {len, bytes} = Protox.Varint.decode(bytes)
              {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)

              {[type: Protox.MergeMessage.merge(msg.type, Onnx.TypeProto.decode!(delimited))],
               rest}

            {3, _, bytes} ->
              {len, bytes} = Protox.Varint.decode(bytes)
              {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)
              {[doc_string: delimited], rest}

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
        Onnx.ValueInfoProto,
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
          json_name: "name",
          kind: {:scalar, ""},
          label: :optional,
          name: :name,
          tag: 1,
          type: :string
        },
        %{
          __struct__: Protox.Field,
          json_name: "type",
          kind: {:scalar, nil},
          label: :optional,
          name: :type,
          tag: 2,
          type: {:message, Onnx.TypeProto}
        },
        %{
          __struct__: Protox.Field,
          json_name: "docString",
          kind: {:scalar, ""},
          label: :optional,
          name: :doc_string,
          tag: 3,
          type: :string
        }
      ]
    end

    [
      @spec(field_def(atom) :: {:ok, Protox.Field.t()} | {:error, :no_such_field}),
      (
        def field_def(:name) do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "name",
             kind: {:scalar, ""},
             label: :optional,
             name: :name,
             tag: 1,
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
             tag: 1,
             type: :string
           }}
        end

        []
      ),
      (
        def field_def(:type) do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "type",
             kind: {:scalar, nil},
             label: :optional,
             name: :type,
             tag: 2,
             type: {:message, Onnx.TypeProto}
           }}
        end

        def field_def("type") do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "type",
             kind: {:scalar, nil},
             label: :optional,
             name: :type,
             tag: 2,
             type: {:message, Onnx.TypeProto}
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
             tag: 3,
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
             tag: 3,
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
             tag: 3,
             type: :string
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
    def default(:name) do
      {:ok, ""}
    end,
    def default(:type) do
      {:ok, nil}
    end,
    def default(:doc_string) do
      {:ok, ""}
    end,
    def default(_) do
      {:error, :no_such_field}
    end
  ]
end