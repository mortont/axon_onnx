# credo:disable-for-this-file
defmodule Onnx.NodeProto do
  @moduledoc false
  defstruct input: [],
            output: [],
            name: "",
            op_type: "",
            attribute: [],
            doc_string: "",
            domain: ""

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
        |> encode_input(msg)
        |> encode_output(msg)
        |> encode_name(msg)
        |> encode_op_type(msg)
        |> encode_attribute(msg)
        |> encode_doc_string(msg)
        |> encode_domain(msg)
      end
    )

    []

    [
      defp encode_input(acc, msg) do
        try do
          case msg.input do
            [] ->
              acc

            values ->
              [
                acc,
                Enum.reduce(values, [], fn value, acc ->
                  [acc, "\n", Protox.Encode.encode_string(value)]
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
                  [acc, "\x12", Protox.Encode.encode_string(value)]
                end)
              ]
          end
        rescue
          ArgumentError ->
            reraise Protox.EncodingError.new(:output, "invalid field value"), __STACKTRACE__
        end
      end,
      defp encode_name(acc, msg) do
        try do
          if msg.name == "" do
            acc
          else
            [acc, "\x1A", Protox.Encode.encode_string(msg.name)]
          end
        rescue
          ArgumentError ->
            reraise Protox.EncodingError.new(:name, "invalid field value"), __STACKTRACE__
        end
      end,
      defp encode_op_type(acc, msg) do
        try do
          if msg.op_type == "" do
            acc
          else
            [acc, "\"", Protox.Encode.encode_string(msg.op_type)]
          end
        rescue
          ArgumentError ->
            reraise Protox.EncodingError.new(:op_type, "invalid field value"), __STACKTRACE__
        end
      end,
      defp encode_attribute(acc, msg) do
        try do
          case msg.attribute do
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
            reraise Protox.EncodingError.new(:attribute, "invalid field value"), __STACKTRACE__
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
      defp encode_domain(acc, msg) do
        try do
          if msg.domain == "" do
            acc
          else
            [acc, ":", Protox.Encode.encode_string(msg.domain)]
          end
        rescue
          ArgumentError ->
            reraise Protox.EncodingError.new(:domain, "invalid field value"), __STACKTRACE__
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
          parse_key_value(bytes, struct(Onnx.NodeProto))
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
              {[input: msg.input ++ [delimited]], rest}

            {2, _, bytes} ->
              {len, bytes} = Protox.Varint.decode(bytes)
              {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)
              {[output: msg.output ++ [delimited]], rest}

            {3, _, bytes} ->
              {len, bytes} = Protox.Varint.decode(bytes)
              {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)
              {[name: delimited], rest}

            {4, _, bytes} ->
              {len, bytes} = Protox.Varint.decode(bytes)
              {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)
              {[op_type: delimited], rest}

            {5, _, bytes} ->
              {len, bytes} = Protox.Varint.decode(bytes)
              {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)
              {[attribute: msg.attribute ++ [Onnx.AttributeProto.decode!(delimited)]], rest}

            {6, _, bytes} ->
              {len, bytes} = Protox.Varint.decode(bytes)
              {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)
              {[doc_string: delimited], rest}

            {7, _, bytes} ->
              {len, bytes} = Protox.Varint.decode(bytes)
              {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)
              {[domain: delimited], rest}

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
        Onnx.NodeProto,
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
          json_name: "input",
          kind: :unpacked,
          label: :repeated,
          name: :input,
          tag: 1,
          type: :string
        },
        %{
          __struct__: Protox.Field,
          json_name: "output",
          kind: :unpacked,
          label: :repeated,
          name: :output,
          tag: 2,
          type: :string
        },
        %{
          __struct__: Protox.Field,
          json_name: "name",
          kind: {:scalar, ""},
          label: :optional,
          name: :name,
          tag: 3,
          type: :string
        },
        %{
          __struct__: Protox.Field,
          json_name: "opType",
          kind: {:scalar, ""},
          label: :optional,
          name: :op_type,
          tag: 4,
          type: :string
        },
        %{
          __struct__: Protox.Field,
          json_name: "attribute",
          kind: :unpacked,
          label: :repeated,
          name: :attribute,
          tag: 5,
          type: {:message, Onnx.AttributeProto}
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
          json_name: "domain",
          kind: {:scalar, ""},
          label: :optional,
          name: :domain,
          tag: 7,
          type: :string
        }
      ]
    end

    [
      @spec(field_def(atom) :: {:ok, Protox.Field.t()} | {:error, :no_such_field}),
      (
        def field_def(:input) do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "input",
             kind: :unpacked,
             label: :repeated,
             name: :input,
             tag: 1,
             type: :string
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
             tag: 1,
             type: :string
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
             tag: 2,
             type: :string
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
             tag: 2,
             type: :string
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
             tag: 3,
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
             tag: 3,
             type: :string
           }}
        end

        []
      ),
      (
        def field_def(:op_type) do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "opType",
             kind: {:scalar, ""},
             label: :optional,
             name: :op_type,
             tag: 4,
             type: :string
           }}
        end

        def field_def("opType") do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "opType",
             kind: {:scalar, ""},
             label: :optional,
             name: :op_type,
             tag: 4,
             type: :string
           }}
        end

        def field_def("op_type") do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "opType",
             kind: {:scalar, ""},
             label: :optional,
             name: :op_type,
             tag: 4,
             type: :string
           }}
        end
      ),
      (
        def field_def(:attribute) do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "attribute",
             kind: :unpacked,
             label: :repeated,
             name: :attribute,
             tag: 5,
             type: {:message, Onnx.AttributeProto}
           }}
        end

        def field_def("attribute") do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "attribute",
             kind: :unpacked,
             label: :repeated,
             name: :attribute,
             tag: 5,
             type: {:message, Onnx.AttributeProto}
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
        def field_def(:domain) do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "domain",
             kind: {:scalar, ""},
             label: :optional,
             name: :domain,
             tag: 7,
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
             tag: 7,
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
    def default(:input) do
      {:error, :no_default_value}
    end,
    def default(:output) do
      {:error, :no_default_value}
    end,
    def default(:name) do
      {:ok, ""}
    end,
    def default(:op_type) do
      {:ok, ""}
    end,
    def default(:attribute) do
      {:error, :no_default_value}
    end,
    def default(:doc_string) do
      {:ok, ""}
    end,
    def default(:domain) do
      {:ok, ""}
    end,
    def default(_) do
      {:error, :no_such_field}
    end
  ]
end
