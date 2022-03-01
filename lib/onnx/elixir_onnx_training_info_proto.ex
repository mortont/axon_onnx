# credo:disable-for-this-file
defmodule Onnx.TrainingInfoProto do
  @moduledoc false
  defstruct initialization: nil, algorithm: nil, initialization_binding: [], update_binding: []

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
        |> encode_initialization(msg)
        |> encode_algorithm(msg)
        |> encode_initialization_binding(msg)
        |> encode_update_binding(msg)
      end
    )

    []

    [
      defp encode_initialization(acc, msg) do
        try do
          if msg.initialization == nil do
            acc
          else
            [acc, "\n", Protox.Encode.encode_message(msg.initialization)]
          end
        rescue
          ArgumentError ->
            reraise Protox.EncodingError.new(:initialization, "invalid field value"),
                    __STACKTRACE__
        end
      end,
      defp encode_algorithm(acc, msg) do
        try do
          if msg.algorithm == nil do
            acc
          else
            [acc, "\x12", Protox.Encode.encode_message(msg.algorithm)]
          end
        rescue
          ArgumentError ->
            reraise Protox.EncodingError.new(:algorithm, "invalid field value"), __STACKTRACE__
        end
      end,
      defp encode_initialization_binding(acc, msg) do
        try do
          case msg.initialization_binding do
            [] ->
              acc

            values ->
              [
                acc,
                Enum.reduce(values, [], fn value, acc ->
                  [acc, "\x1A", Protox.Encode.encode_message(value)]
                end)
              ]
          end
        rescue
          ArgumentError ->
            reraise Protox.EncodingError.new(:initialization_binding, "invalid field value"),
                    __STACKTRACE__
        end
      end,
      defp encode_update_binding(acc, msg) do
        try do
          case msg.update_binding do
            [] ->
              acc

            values ->
              [
                acc,
                Enum.reduce(values, [], fn value, acc ->
                  [acc, "\"", Protox.Encode.encode_message(value)]
                end)
              ]
          end
        rescue
          ArgumentError ->
            reraise Protox.EncodingError.new(:update_binding, "invalid field value"),
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
          parse_key_value(bytes, struct(Onnx.TrainingInfoProto))
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
                 initialization:
                   Protox.MergeMessage.merge(
                     msg.initialization,
                     Onnx.GraphProto.decode!(delimited)
                   )
               ], rest}

            {2, _, bytes} ->
              {len, bytes} = Protox.Varint.decode(bytes)
              {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)

              {[
                 algorithm:
                   Protox.MergeMessage.merge(msg.algorithm, Onnx.GraphProto.decode!(delimited))
               ], rest}

            {3, _, bytes} ->
              {len, bytes} = Protox.Varint.decode(bytes)
              {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)

              {[
                 initialization_binding:
                   msg.initialization_binding ++ [Onnx.StringStringEntryProto.decode!(delimited)]
               ], rest}

            {4, _, bytes} ->
              {len, bytes} = Protox.Varint.decode(bytes)
              {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)

              {[
                 update_binding:
                   msg.update_binding ++ [Onnx.StringStringEntryProto.decode!(delimited)]
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
        Onnx.TrainingInfoProto,
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
          json_name: "initialization",
          kind: {:scalar, nil},
          label: :optional,
          name: :initialization,
          tag: 1,
          type: {:message, Onnx.GraphProto}
        },
        %{
          __struct__: Protox.Field,
          json_name: "algorithm",
          kind: {:scalar, nil},
          label: :optional,
          name: :algorithm,
          tag: 2,
          type: {:message, Onnx.GraphProto}
        },
        %{
          __struct__: Protox.Field,
          json_name: "initializationBinding",
          kind: :unpacked,
          label: :repeated,
          name: :initialization_binding,
          tag: 3,
          type: {:message, Onnx.StringStringEntryProto}
        },
        %{
          __struct__: Protox.Field,
          json_name: "updateBinding",
          kind: :unpacked,
          label: :repeated,
          name: :update_binding,
          tag: 4,
          type: {:message, Onnx.StringStringEntryProto}
        }
      ]
    end

    [
      @spec(field_def(atom) :: {:ok, Protox.Field.t()} | {:error, :no_such_field}),
      (
        def field_def(:initialization) do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "initialization",
             kind: {:scalar, nil},
             label: :optional,
             name: :initialization,
             tag: 1,
             type: {:message, Onnx.GraphProto}
           }}
        end

        def field_def("initialization") do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "initialization",
             kind: {:scalar, nil},
             label: :optional,
             name: :initialization,
             tag: 1,
             type: {:message, Onnx.GraphProto}
           }}
        end

        []
      ),
      (
        def field_def(:algorithm) do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "algorithm",
             kind: {:scalar, nil},
             label: :optional,
             name: :algorithm,
             tag: 2,
             type: {:message, Onnx.GraphProto}
           }}
        end

        def field_def("algorithm") do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "algorithm",
             kind: {:scalar, nil},
             label: :optional,
             name: :algorithm,
             tag: 2,
             type: {:message, Onnx.GraphProto}
           }}
        end

        []
      ),
      (
        def field_def(:initialization_binding) do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "initializationBinding",
             kind: :unpacked,
             label: :repeated,
             name: :initialization_binding,
             tag: 3,
             type: {:message, Onnx.StringStringEntryProto}
           }}
        end

        def field_def("initializationBinding") do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "initializationBinding",
             kind: :unpacked,
             label: :repeated,
             name: :initialization_binding,
             tag: 3,
             type: {:message, Onnx.StringStringEntryProto}
           }}
        end

        def field_def("initialization_binding") do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "initializationBinding",
             kind: :unpacked,
             label: :repeated,
             name: :initialization_binding,
             tag: 3,
             type: {:message, Onnx.StringStringEntryProto}
           }}
        end
      ),
      (
        def field_def(:update_binding) do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "updateBinding",
             kind: :unpacked,
             label: :repeated,
             name: :update_binding,
             tag: 4,
             type: {:message, Onnx.StringStringEntryProto}
           }}
        end

        def field_def("updateBinding") do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "updateBinding",
             kind: :unpacked,
             label: :repeated,
             name: :update_binding,
             tag: 4,
             type: {:message, Onnx.StringStringEntryProto}
           }}
        end

        def field_def("update_binding") do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "updateBinding",
             kind: :unpacked,
             label: :repeated,
             name: :update_binding,
             tag: 4,
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
    def default(:initialization) do
      {:ok, nil}
    end,
    def default(:algorithm) do
      {:ok, nil}
    end,
    def default(:initialization_binding) do
      {:error, :no_default_value}
    end,
    def default(:update_binding) do
      {:error, :no_default_value}
    end,
    def default(_) do
      {:error, :no_such_field}
    end
  ]
end