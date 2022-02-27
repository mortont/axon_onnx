# credo:disable-for-this-file
defmodule Onnx.TensorProto do
  @moduledoc false
  defstruct dims: [],
            data_type: 0,
            segment: nil,
            float_data: [],
            int32_data: [],
            string_data: [],
            int64_data: [],
            name: "",
            raw_data: "",
            double_data: [],
            uint64_data: [],
            doc_string: "",
            external_data: [],
            data_location: :DEFAULT

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
        |> encode_dims(msg)
        |> encode_data_type(msg)
        |> encode_segment(msg)
        |> encode_float_data(msg)
        |> encode_int32_data(msg)
        |> encode_string_data(msg)
        |> encode_int64_data(msg)
        |> encode_name(msg)
        |> encode_raw_data(msg)
        |> encode_double_data(msg)
        |> encode_uint64_data(msg)
        |> encode_doc_string(msg)
        |> encode_external_data(msg)
        |> encode_data_location(msg)
      end
    )

    []

    [
      defp encode_dims(acc, msg) do
        try do
          case msg.dims do
            [] ->
              acc

            values ->
              [
                acc,
                "\n",
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
      end,
      defp encode_data_type(acc, msg) do
        try do
          if msg.data_type == 0 do
            acc
          else
            [acc, "\x10", Protox.Encode.encode_int32(msg.data_type)]
          end
        rescue
          ArgumentError ->
            reraise Protox.EncodingError.new(:data_type, "invalid field value"), __STACKTRACE__
        end
      end,
      defp encode_segment(acc, msg) do
        try do
          if msg.segment == nil do
            acc
          else
            [acc, "\x1A", Protox.Encode.encode_message(msg.segment)]
          end
        rescue
          ArgumentError ->
            reraise Protox.EncodingError.new(:segment, "invalid field value"), __STACKTRACE__
        end
      end,
      defp encode_float_data(acc, msg) do
        try do
          case msg.float_data do
            [] ->
              acc

            values ->
              [
                acc,
                "\"",
                (
                  {bytes, len} =
                    Enum.reduce(values, {[], 0}, fn value, {acc, len} ->
                      value_bytes = :binary.list_to_bin([Protox.Encode.encode_float(value)])
                      {[acc, value_bytes], len + byte_size(value_bytes)}
                    end)

                  [Protox.Varint.encode(len), bytes]
                )
              ]
          end
        rescue
          ArgumentError ->
            reraise Protox.EncodingError.new(:float_data, "invalid field value"), __STACKTRACE__
        end
      end,
      defp encode_int32_data(acc, msg) do
        try do
          case msg.int32_data do
            [] ->
              acc

            values ->
              [
                acc,
                "*",
                (
                  {bytes, len} =
                    Enum.reduce(values, {[], 0}, fn value, {acc, len} ->
                      value_bytes = :binary.list_to_bin([Protox.Encode.encode_int32(value)])
                      {[acc, value_bytes], len + byte_size(value_bytes)}
                    end)

                  [Protox.Varint.encode(len), bytes]
                )
              ]
          end
        rescue
          ArgumentError ->
            reraise Protox.EncodingError.new(:int32_data, "invalid field value"), __STACKTRACE__
        end
      end,
      defp encode_string_data(acc, msg) do
        try do
          case msg.string_data do
            [] ->
              acc

            values ->
              [
                acc,
                Enum.reduce(values, [], fn value, acc ->
                  [acc, "2", Protox.Encode.encode_bytes(value)]
                end)
              ]
          end
        rescue
          ArgumentError ->
            reraise Protox.EncodingError.new(:string_data, "invalid field value"), __STACKTRACE__
        end
      end,
      defp encode_int64_data(acc, msg) do
        try do
          case msg.int64_data do
            [] ->
              acc

            values ->
              [
                acc,
                ":",
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
            reraise Protox.EncodingError.new(:int64_data, "invalid field value"), __STACKTRACE__
        end
      end,
      defp encode_name(acc, msg) do
        try do
          if msg.name == "" do
            acc
          else
            [acc, "B", Protox.Encode.encode_string(msg.name)]
          end
        rescue
          ArgumentError ->
            reraise Protox.EncodingError.new(:name, "invalid field value"), __STACKTRACE__
        end
      end,
      defp encode_raw_data(acc, msg) do
        try do
          if msg.raw_data == "" do
            acc
          else
            [acc, "J", Protox.Encode.encode_bytes(msg.raw_data)]
          end
        rescue
          ArgumentError ->
            reraise Protox.EncodingError.new(:raw_data, "invalid field value"), __STACKTRACE__
        end
      end,
      defp encode_double_data(acc, msg) do
        try do
          case msg.double_data do
            [] ->
              acc

            values ->
              [
                acc,
                "R",
                (
                  {bytes, len} =
                    Enum.reduce(values, {[], 0}, fn value, {acc, len} ->
                      value_bytes = :binary.list_to_bin([Protox.Encode.encode_double(value)])
                      {[acc, value_bytes], len + byte_size(value_bytes)}
                    end)

                  [Protox.Varint.encode(len), bytes]
                )
              ]
          end
        rescue
          ArgumentError ->
            reraise Protox.EncodingError.new(:double_data, "invalid field value"), __STACKTRACE__
        end
      end,
      defp encode_uint64_data(acc, msg) do
        try do
          case msg.uint64_data do
            [] ->
              acc

            values ->
              [
                acc,
                "Z",
                (
                  {bytes, len} =
                    Enum.reduce(values, {[], 0}, fn value, {acc, len} ->
                      value_bytes = :binary.list_to_bin([Protox.Encode.encode_uint64(value)])
                      {[acc, value_bytes], len + byte_size(value_bytes)}
                    end)

                  [Protox.Varint.encode(len), bytes]
                )
              ]
          end
        rescue
          ArgumentError ->
            reraise Protox.EncodingError.new(:uint64_data, "invalid field value"), __STACKTRACE__
        end
      end,
      defp encode_doc_string(acc, msg) do
        try do
          if msg.doc_string == "" do
            acc
          else
            [acc, "b", Protox.Encode.encode_string(msg.doc_string)]
          end
        rescue
          ArgumentError ->
            reraise Protox.EncodingError.new(:doc_string, "invalid field value"), __STACKTRACE__
        end
      end,
      defp encode_external_data(acc, msg) do
        try do
          case msg.external_data do
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
            reraise Protox.EncodingError.new(:external_data, "invalid field value"),
                    __STACKTRACE__
        end
      end,
      defp encode_data_location(acc, msg) do
        try do
          if msg.data_location == :DEFAULT do
            acc
          else
            [
              acc,
              "p",
              msg.data_location
              |> Onnx.TensorProto.DataLocation.encode()
              |> Protox.Encode.encode_enum()
            ]
          end
        rescue
          ArgumentError ->
            reraise Protox.EncodingError.new(:data_location, "invalid field value"),
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
          parse_key_value(bytes, struct(Onnx.TensorProto))
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

            {1, 2, bytes} ->
              {len, bytes} = Protox.Varint.decode(bytes)
              {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)
              {[dims: msg.dims ++ Protox.Decode.parse_repeated_int64([], delimited)], rest}

            {1, _, bytes} ->
              {value, rest} = Protox.Decode.parse_int64(bytes)
              {[dims: msg.dims ++ [value]], rest}

            {2, _, bytes} ->
              {value, rest} = Protox.Decode.parse_int32(bytes)
              {[data_type: value], rest}

            {3, _, bytes} ->
              {len, bytes} = Protox.Varint.decode(bytes)
              {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)

              {[
                 segment:
                   Protox.MergeMessage.merge(
                     msg.segment,
                     Onnx.TensorProto.Segment.decode!(delimited)
                   )
               ], rest}

            {4, 2, bytes} ->
              {len, bytes} = Protox.Varint.decode(bytes)
              {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)

              {[float_data: msg.float_data ++ Protox.Decode.parse_repeated_float([], delimited)],
               rest}

            {4, _, bytes} ->
              {value, rest} = Protox.Decode.parse_float(bytes)
              {[float_data: msg.float_data ++ [value]], rest}

            {5, 2, bytes} ->
              {len, bytes} = Protox.Varint.decode(bytes)
              {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)

              {[int32_data: msg.int32_data ++ Protox.Decode.parse_repeated_int32([], delimited)],
               rest}

            {5, _, bytes} ->
              {value, rest} = Protox.Decode.parse_int32(bytes)
              {[int32_data: msg.int32_data ++ [value]], rest}

            {6, _, bytes} ->
              {len, bytes} = Protox.Varint.decode(bytes)
              {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)
              {[string_data: msg.string_data ++ [delimited]], rest}

            {7, 2, bytes} ->
              {len, bytes} = Protox.Varint.decode(bytes)
              {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)

              {[int64_data: msg.int64_data ++ Protox.Decode.parse_repeated_int64([], delimited)],
               rest}

            {7, _, bytes} ->
              {value, rest} = Protox.Decode.parse_int64(bytes)
              {[int64_data: msg.int64_data ++ [value]], rest}

            {8, _, bytes} ->
              {len, bytes} = Protox.Varint.decode(bytes)
              {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)
              {[name: delimited], rest}

            {9, _, bytes} ->
              {len, bytes} = Protox.Varint.decode(bytes)
              {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)
              {[raw_data: delimited], rest}

            {10, 2, bytes} ->
              {len, bytes} = Protox.Varint.decode(bytes)
              {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)

              {[
                 double_data:
                   msg.double_data ++ Protox.Decode.parse_repeated_double([], delimited)
               ], rest}

            {10, _, bytes} ->
              {value, rest} = Protox.Decode.parse_double(bytes)
              {[double_data: msg.double_data ++ [value]], rest}

            {11, 2, bytes} ->
              {len, bytes} = Protox.Varint.decode(bytes)
              {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)

              {[
                 uint64_data:
                   msg.uint64_data ++ Protox.Decode.parse_repeated_uint64([], delimited)
               ], rest}

            {11, _, bytes} ->
              {value, rest} = Protox.Decode.parse_uint64(bytes)
              {[uint64_data: msg.uint64_data ++ [value]], rest}

            {12, _, bytes} ->
              {len, bytes} = Protox.Varint.decode(bytes)
              {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)
              {[doc_string: delimited], rest}

            {13, _, bytes} ->
              {len, bytes} = Protox.Varint.decode(bytes)
              {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)

              {[
                 external_data:
                   msg.external_data ++ [Onnx.StringStringEntryProto.decode!(delimited)]
               ], rest}

            {14, _, bytes} ->
              {value, rest} = Protox.Decode.parse_enum(bytes, Onnx.TensorProto.DataLocation)
              {[data_location: value], rest}

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
        Onnx.TensorProto,
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
          json_name: "dims",
          kind: :packed,
          label: :repeated,
          name: :dims,
          tag: 1,
          type: :int64
        },
        %{
          __struct__: Protox.Field,
          json_name: "dataType",
          kind: {:scalar, 0},
          label: :optional,
          name: :data_type,
          tag: 2,
          type: :int32
        },
        %{
          __struct__: Protox.Field,
          json_name: "segment",
          kind: {:scalar, nil},
          label: :optional,
          name: :segment,
          tag: 3,
          type: {:message, Onnx.TensorProto.Segment}
        },
        %{
          __struct__: Protox.Field,
          json_name: "floatData",
          kind: :packed,
          label: :repeated,
          name: :float_data,
          tag: 4,
          type: :float
        },
        %{
          __struct__: Protox.Field,
          json_name: "int32Data",
          kind: :packed,
          label: :repeated,
          name: :int32_data,
          tag: 5,
          type: :int32
        },
        %{
          __struct__: Protox.Field,
          json_name: "stringData",
          kind: :unpacked,
          label: :repeated,
          name: :string_data,
          tag: 6,
          type: :bytes
        },
        %{
          __struct__: Protox.Field,
          json_name: "int64Data",
          kind: :packed,
          label: :repeated,
          name: :int64_data,
          tag: 7,
          type: :int64
        },
        %{
          __struct__: Protox.Field,
          json_name: "name",
          kind: {:scalar, ""},
          label: :optional,
          name: :name,
          tag: 8,
          type: :string
        },
        %{
          __struct__: Protox.Field,
          json_name: "rawData",
          kind: {:scalar, ""},
          label: :optional,
          name: :raw_data,
          tag: 9,
          type: :bytes
        },
        %{
          __struct__: Protox.Field,
          json_name: "doubleData",
          kind: :packed,
          label: :repeated,
          name: :double_data,
          tag: 10,
          type: :double
        },
        %{
          __struct__: Protox.Field,
          json_name: "uint64Data",
          kind: :packed,
          label: :repeated,
          name: :uint64_data,
          tag: 11,
          type: :uint64
        },
        %{
          __struct__: Protox.Field,
          json_name: "docString",
          kind: {:scalar, ""},
          label: :optional,
          name: :doc_string,
          tag: 12,
          type: :string
        },
        %{
          __struct__: Protox.Field,
          json_name: "externalData",
          kind: :unpacked,
          label: :repeated,
          name: :external_data,
          tag: 13,
          type: {:message, Onnx.StringStringEntryProto}
        },
        %{
          __struct__: Protox.Field,
          json_name: "dataLocation",
          kind: {:scalar, :DEFAULT},
          label: :optional,
          name: :data_location,
          tag: 14,
          type: {:enum, Onnx.TensorProto.DataLocation}
        }
      ]
    end

    [
      @spec(field_def(atom) :: {:ok, Protox.Field.t()} | {:error, :no_such_field}),
      (
        def field_def(:dims) do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "dims",
             kind: :packed,
             label: :repeated,
             name: :dims,
             tag: 1,
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
             tag: 1,
             type: :int64
           }}
        end

        []
      ),
      (
        def field_def(:data_type) do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "dataType",
             kind: {:scalar, 0},
             label: :optional,
             name: :data_type,
             tag: 2,
             type: :int32
           }}
        end

        def field_def("dataType") do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "dataType",
             kind: {:scalar, 0},
             label: :optional,
             name: :data_type,
             tag: 2,
             type: :int32
           }}
        end

        def field_def("data_type") do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "dataType",
             kind: {:scalar, 0},
             label: :optional,
             name: :data_type,
             tag: 2,
             type: :int32
           }}
        end
      ),
      (
        def field_def(:segment) do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "segment",
             kind: {:scalar, nil},
             label: :optional,
             name: :segment,
             tag: 3,
             type: {:message, Onnx.TensorProto.Segment}
           }}
        end

        def field_def("segment") do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "segment",
             kind: {:scalar, nil},
             label: :optional,
             name: :segment,
             tag: 3,
             type: {:message, Onnx.TensorProto.Segment}
           }}
        end

        []
      ),
      (
        def field_def(:float_data) do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "floatData",
             kind: :packed,
             label: :repeated,
             name: :float_data,
             tag: 4,
             type: :float
           }}
        end

        def field_def("floatData") do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "floatData",
             kind: :packed,
             label: :repeated,
             name: :float_data,
             tag: 4,
             type: :float
           }}
        end

        def field_def("float_data") do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "floatData",
             kind: :packed,
             label: :repeated,
             name: :float_data,
             tag: 4,
             type: :float
           }}
        end
      ),
      (
        def field_def(:int32_data) do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "int32Data",
             kind: :packed,
             label: :repeated,
             name: :int32_data,
             tag: 5,
             type: :int32
           }}
        end

        def field_def("int32Data") do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "int32Data",
             kind: :packed,
             label: :repeated,
             name: :int32_data,
             tag: 5,
             type: :int32
           }}
        end

        def field_def("int32_data") do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "int32Data",
             kind: :packed,
             label: :repeated,
             name: :int32_data,
             tag: 5,
             type: :int32
           }}
        end
      ),
      (
        def field_def(:string_data) do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "stringData",
             kind: :unpacked,
             label: :repeated,
             name: :string_data,
             tag: 6,
             type: :bytes
           }}
        end

        def field_def("stringData") do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "stringData",
             kind: :unpacked,
             label: :repeated,
             name: :string_data,
             tag: 6,
             type: :bytes
           }}
        end

        def field_def("string_data") do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "stringData",
             kind: :unpacked,
             label: :repeated,
             name: :string_data,
             tag: 6,
             type: :bytes
           }}
        end
      ),
      (
        def field_def(:int64_data) do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "int64Data",
             kind: :packed,
             label: :repeated,
             name: :int64_data,
             tag: 7,
             type: :int64
           }}
        end

        def field_def("int64Data") do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "int64Data",
             kind: :packed,
             label: :repeated,
             name: :int64_data,
             tag: 7,
             type: :int64
           }}
        end

        def field_def("int64_data") do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "int64Data",
             kind: :packed,
             label: :repeated,
             name: :int64_data,
             tag: 7,
             type: :int64
           }}
        end
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
             tag: 8,
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
             tag: 8,
             type: :string
           }}
        end

        []
      ),
      (
        def field_def(:raw_data) do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "rawData",
             kind: {:scalar, ""},
             label: :optional,
             name: :raw_data,
             tag: 9,
             type: :bytes
           }}
        end

        def field_def("rawData") do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "rawData",
             kind: {:scalar, ""},
             label: :optional,
             name: :raw_data,
             tag: 9,
             type: :bytes
           }}
        end

        def field_def("raw_data") do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "rawData",
             kind: {:scalar, ""},
             label: :optional,
             name: :raw_data,
             tag: 9,
             type: :bytes
           }}
        end
      ),
      (
        def field_def(:double_data) do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "doubleData",
             kind: :packed,
             label: :repeated,
             name: :double_data,
             tag: 10,
             type: :double
           }}
        end

        def field_def("doubleData") do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "doubleData",
             kind: :packed,
             label: :repeated,
             name: :double_data,
             tag: 10,
             type: :double
           }}
        end

        def field_def("double_data") do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "doubleData",
             kind: :packed,
             label: :repeated,
             name: :double_data,
             tag: 10,
             type: :double
           }}
        end
      ),
      (
        def field_def(:uint64_data) do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "uint64Data",
             kind: :packed,
             label: :repeated,
             name: :uint64_data,
             tag: 11,
             type: :uint64
           }}
        end

        def field_def("uint64Data") do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "uint64Data",
             kind: :packed,
             label: :repeated,
             name: :uint64_data,
             tag: 11,
             type: :uint64
           }}
        end

        def field_def("uint64_data") do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "uint64Data",
             kind: :packed,
             label: :repeated,
             name: :uint64_data,
             tag: 11,
             type: :uint64
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
             tag: 12,
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
             tag: 12,
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
             tag: 12,
             type: :string
           }}
        end
      ),
      (
        def field_def(:external_data) do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "externalData",
             kind: :unpacked,
             label: :repeated,
             name: :external_data,
             tag: 13,
             type: {:message, Onnx.StringStringEntryProto}
           }}
        end

        def field_def("externalData") do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "externalData",
             kind: :unpacked,
             label: :repeated,
             name: :external_data,
             tag: 13,
             type: {:message, Onnx.StringStringEntryProto}
           }}
        end

        def field_def("external_data") do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "externalData",
             kind: :unpacked,
             label: :repeated,
             name: :external_data,
             tag: 13,
             type: {:message, Onnx.StringStringEntryProto}
           }}
        end
      ),
      (
        def field_def(:data_location) do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "dataLocation",
             kind: {:scalar, :DEFAULT},
             label: :optional,
             name: :data_location,
             tag: 14,
             type: {:enum, Onnx.TensorProto.DataLocation}
           }}
        end

        def field_def("dataLocation") do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "dataLocation",
             kind: {:scalar, :DEFAULT},
             label: :optional,
             name: :data_location,
             tag: 14,
             type: {:enum, Onnx.TensorProto.DataLocation}
           }}
        end

        def field_def("data_location") do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "dataLocation",
             kind: {:scalar, :DEFAULT},
             label: :optional,
             name: :data_location,
             tag: 14,
             type: {:enum, Onnx.TensorProto.DataLocation}
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
    def default(:dims) do
      {:error, :no_default_value}
    end,
    def default(:data_type) do
      {:ok, 0}
    end,
    def default(:segment) do
      {:ok, nil}
    end,
    def default(:float_data) do
      {:error, :no_default_value}
    end,
    def default(:int32_data) do
      {:error, :no_default_value}
    end,
    def default(:string_data) do
      {:error, :no_default_value}
    end,
    def default(:int64_data) do
      {:error, :no_default_value}
    end,
    def default(:name) do
      {:ok, ""}
    end,
    def default(:raw_data) do
      {:ok, ""}
    end,
    def default(:double_data) do
      {:error, :no_default_value}
    end,
    def default(:uint64_data) do
      {:error, :no_default_value}
    end,
    def default(:doc_string) do
      {:ok, ""}
    end,
    def default(:external_data) do
      {:error, :no_default_value}
    end,
    def default(:data_location) do
      {:ok, :DEFAULT}
    end,
    def default(_) do
      {:error, :no_such_field}
    end
  ]
end