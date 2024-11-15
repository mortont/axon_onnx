# credo:disable-for-this-file
defmodule Onnx.AttributeProto do
  @moduledoc false
  defstruct name: "",
            f: 0.0,
            i: 0,
            s: "",
            t: nil,
            g: nil,
            floats: [],
            ints: [],
            strings: [],
            tensors: [],
            graphs: [],
            doc_string: "",
            tp: nil,
            type_protos: [],
            type: :UNDEFINED,
            ref_attr_name: "",
            sparse_tensor: nil,
            sparse_tensors: []

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
        |> encode_name(msg)
        |> encode_f(msg)
        |> encode_i(msg)
        |> encode_s(msg)
        |> encode_t(msg)
        |> encode_g(msg)
        |> encode_floats(msg)
        |> encode_ints(msg)
        |> encode_strings(msg)
        |> encode_tensors(msg)
        |> encode_graphs(msg)
        |> encode_doc_string(msg)
        |> encode_tp(msg)
        |> encode_type_protos(msg)
        |> encode_type(msg)
        |> encode_ref_attr_name(msg)
        |> encode_sparse_tensor(msg)
        |> encode_sparse_tensors(msg)
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
      defp encode_f(acc, msg) do
        try do
          if msg.f == 0.0 do
            acc
          else
            [acc, "\x15", Protox.Encode.encode_float(msg.f)]
          end
        rescue
          ArgumentError ->
            reraise Protox.EncodingError.new(:f, "invalid field value"), __STACKTRACE__
        end
      end,
      defp encode_i(acc, msg) do
        try do
          if msg.i == 0 do
            acc
          else
            [acc, "\x18", Protox.Encode.encode_int64(msg.i)]
          end
        rescue
          ArgumentError ->
            reraise Protox.EncodingError.new(:i, "invalid field value"), __STACKTRACE__
        end
      end,
      defp encode_s(acc, msg) do
        try do
          if msg.s == "" do
            acc
          else
            [acc, "\"", Protox.Encode.encode_bytes(msg.s)]
          end
        rescue
          ArgumentError ->
            reraise Protox.EncodingError.new(:s, "invalid field value"), __STACKTRACE__
        end
      end,
      defp encode_t(acc, msg) do
        try do
          if msg.t == nil do
            acc
          else
            [acc, "*", Protox.Encode.encode_message(msg.t)]
          end
        rescue
          ArgumentError ->
            reraise Protox.EncodingError.new(:t, "invalid field value"), __STACKTRACE__
        end
      end,
      defp encode_g(acc, msg) do
        try do
          if msg.g == nil do
            acc
          else
            [acc, "2", Protox.Encode.encode_message(msg.g)]
          end
        rescue
          ArgumentError ->
            reraise Protox.EncodingError.new(:g, "invalid field value"), __STACKTRACE__
        end
      end,
      defp encode_floats(acc, msg) do
        try do
          case msg.floats do
            [] ->
              acc

            values ->
              [
                acc,
                ":",
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
            reraise Protox.EncodingError.new(:floats, "invalid field value"), __STACKTRACE__
        end
      end,
      defp encode_ints(acc, msg) do
        try do
          case msg.ints do
            [] ->
              acc

            values ->
              [
                acc,
                "B",
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
            reraise Protox.EncodingError.new(:ints, "invalid field value"), __STACKTRACE__
        end
      end,
      defp encode_strings(acc, msg) do
        try do
          case msg.strings do
            [] ->
              acc

            values ->
              [
                acc,
                Enum.reduce(values, [], fn value, acc ->
                  [acc, "J", Protox.Encode.encode_bytes(value)]
                end)
              ]
          end
        rescue
          ArgumentError ->
            reraise Protox.EncodingError.new(:strings, "invalid field value"), __STACKTRACE__
        end
      end,
      defp encode_tensors(acc, msg) do
        try do
          case msg.tensors do
            [] ->
              acc

            values ->
              [
                acc,
                Enum.reduce(values, [], fn value, acc ->
                  [acc, "R", Protox.Encode.encode_message(value)]
                end)
              ]
          end
        rescue
          ArgumentError ->
            reraise Protox.EncodingError.new(:tensors, "invalid field value"), __STACKTRACE__
        end
      end,
      defp encode_graphs(acc, msg) do
        try do
          case msg.graphs do
            [] ->
              acc

            values ->
              [
                acc,
                Enum.reduce(values, [], fn value, acc ->
                  [acc, "Z", Protox.Encode.encode_message(value)]
                end)
              ]
          end
        rescue
          ArgumentError ->
            reraise Protox.EncodingError.new(:graphs, "invalid field value"), __STACKTRACE__
        end
      end,
      defp encode_doc_string(acc, msg) do
        try do
          if msg.doc_string == "" do
            acc
          else
            [acc, "j", Protox.Encode.encode_string(msg.doc_string)]
          end
        rescue
          ArgumentError ->
            reraise Protox.EncodingError.new(:doc_string, "invalid field value"), __STACKTRACE__
        end
      end,
      defp encode_tp(acc, msg) do
        try do
          if msg.tp == nil do
            acc
          else
            [acc, "r", Protox.Encode.encode_message(msg.tp)]
          end
        rescue
          ArgumentError ->
            reraise Protox.EncodingError.new(:tp, "invalid field value"), __STACKTRACE__
        end
      end,
      defp encode_type_protos(acc, msg) do
        try do
          case msg.type_protos do
            [] ->
              acc

            values ->
              [
                acc,
                Enum.reduce(values, [], fn value, acc ->
                  [acc, "z", Protox.Encode.encode_message(value)]
                end)
              ]
          end
        rescue
          ArgumentError ->
            reraise Protox.EncodingError.new(:type_protos, "invalid field value"), __STACKTRACE__
        end
      end,
      defp encode_type(acc, msg) do
        try do
          if msg.type == :UNDEFINED do
            acc
          else
            [
              acc,
              "\xA0\x01",
              msg.type
              |> Onnx.AttributeProto.AttributeType.encode()
              |> Protox.Encode.encode_enum()
            ]
          end
        rescue
          ArgumentError ->
            reraise Protox.EncodingError.new(:type, "invalid field value"), __STACKTRACE__
        end
      end,
      defp encode_ref_attr_name(acc, msg) do
        try do
          if msg.ref_attr_name == "" do
            acc
          else
            [acc, "\xAA\x01", Protox.Encode.encode_string(msg.ref_attr_name)]
          end
        rescue
          ArgumentError ->
            reraise Protox.EncodingError.new(:ref_attr_name, "invalid field value"),
                    __STACKTRACE__
        end
      end,
      defp encode_sparse_tensor(acc, msg) do
        try do
          if msg.sparse_tensor == nil do
            acc
          else
            [acc, "\xB2\x01", Protox.Encode.encode_message(msg.sparse_tensor)]
          end
        rescue
          ArgumentError ->
            reraise Protox.EncodingError.new(:sparse_tensor, "invalid field value"),
                    __STACKTRACE__
        end
      end,
      defp encode_sparse_tensors(acc, msg) do
        try do
          case msg.sparse_tensors do
            [] ->
              acc

            values ->
              [
                acc,
                Enum.reduce(values, [], fn value, acc ->
                  [acc, "\xBA\x01", Protox.Encode.encode_message(value)]
                end)
              ]
          end
        rescue
          ArgumentError ->
            reraise Protox.EncodingError.new(:sparse_tensors, "invalid field value"),
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
          parse_key_value(bytes, struct(Onnx.AttributeProto))
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
              {[name: Protox.Decode.validate_string!(delimited)], rest}

            {2, _, bytes} ->
              {value, rest} = Protox.Decode.parse_float(bytes)
              {[f: value], rest}

            {3, _, bytes} ->
              {value, rest} = Protox.Decode.parse_int64(bytes)
              {[i: value], rest}

            {4, _, bytes} ->
              {len, bytes} = Protox.Varint.decode(bytes)
              {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)
              {[s: delimited], rest}

            {5, _, bytes} ->
              {len, bytes} = Protox.Varint.decode(bytes)
              {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)
              {[t: Protox.MergeMessage.merge(msg.t, Onnx.TensorProto.decode!(delimited))], rest}

            {6, _, bytes} ->
              {len, bytes} = Protox.Varint.decode(bytes)
              {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)
              {[g: Protox.MergeMessage.merge(msg.g, Onnx.GraphProto.decode!(delimited))], rest}

            {7, 2, bytes} ->
              {len, bytes} = Protox.Varint.decode(bytes)
              {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)
              {[floats: msg.floats ++ Protox.Decode.parse_repeated_float([], delimited)], rest}

            {7, _, bytes} ->
              {value, rest} = Protox.Decode.parse_float(bytes)
              {[floats: msg.floats ++ [value]], rest}

            {8, 2, bytes} ->
              {len, bytes} = Protox.Varint.decode(bytes)
              {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)
              {[ints: msg.ints ++ Protox.Decode.parse_repeated_int64([], delimited)], rest}

            {8, _, bytes} ->
              {value, rest} = Protox.Decode.parse_int64(bytes)
              {[ints: msg.ints ++ [value]], rest}

            {9, _, bytes} ->
              {len, bytes} = Protox.Varint.decode(bytes)
              {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)
              {[strings: msg.strings ++ [delimited]], rest}

            {10, _, bytes} ->
              {len, bytes} = Protox.Varint.decode(bytes)
              {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)
              {[tensors: msg.tensors ++ [Onnx.TensorProto.decode!(delimited)]], rest}

            {11, _, bytes} ->
              {len, bytes} = Protox.Varint.decode(bytes)
              {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)
              {[graphs: msg.graphs ++ [Onnx.GraphProto.decode!(delimited)]], rest}

            {13, _, bytes} ->
              {len, bytes} = Protox.Varint.decode(bytes)
              {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)
              {[doc_string: Protox.Decode.validate_string!(delimited)], rest}

            {14, _, bytes} ->
              {len, bytes} = Protox.Varint.decode(bytes)
              {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)
              {[tp: Protox.MergeMessage.merge(msg.tp, Onnx.TypeProto.decode!(delimited))], rest}

            {15, _, bytes} ->
              {len, bytes} = Protox.Varint.decode(bytes)
              {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)
              {[type_protos: msg.type_protos ++ [Onnx.TypeProto.decode!(delimited)]], rest}

            {20, _, bytes} ->
              {value, rest} = Protox.Decode.parse_enum(bytes, Onnx.AttributeProto.AttributeType)
              {[type: value], rest}

            {21, _, bytes} ->
              {len, bytes} = Protox.Varint.decode(bytes)
              {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)
              {[ref_attr_name: Protox.Decode.validate_string!(delimited)], rest}

            {22, _, bytes} ->
              {len, bytes} = Protox.Varint.decode(bytes)
              {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)

              {[
                 sparse_tensor:
                   Protox.MergeMessage.merge(
                     msg.sparse_tensor,
                     Onnx.SparseTensorProto.decode!(delimited)
                   )
               ], rest}

            {23, _, bytes} ->
              {len, bytes} = Protox.Varint.decode(bytes)
              {delimited, rest} = Protox.Decode.parse_delimited(bytes, len)

              {[
                 sparse_tensors: msg.sparse_tensors ++ [Onnx.SparseTensorProto.decode!(delimited)]
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
        Onnx.AttributeProto,
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
          json_name: "f",
          kind: {:scalar, 0.0},
          label: :optional,
          name: :f,
          tag: 2,
          type: :float
        },
        %{
          __struct__: Protox.Field,
          json_name: "i",
          kind: {:scalar, 0},
          label: :optional,
          name: :i,
          tag: 3,
          type: :int64
        },
        %{
          __struct__: Protox.Field,
          json_name: "s",
          kind: {:scalar, ""},
          label: :optional,
          name: :s,
          tag: 4,
          type: :bytes
        },
        %{
          __struct__: Protox.Field,
          json_name: "t",
          kind: {:scalar, nil},
          label: :optional,
          name: :t,
          tag: 5,
          type: {:message, Onnx.TensorProto}
        },
        %{
          __struct__: Protox.Field,
          json_name: "g",
          kind: {:scalar, nil},
          label: :optional,
          name: :g,
          tag: 6,
          type: {:message, Onnx.GraphProto}
        },
        %{
          __struct__: Protox.Field,
          json_name: "floats",
          kind: :packed,
          label: :repeated,
          name: :floats,
          tag: 7,
          type: :float
        },
        %{
          __struct__: Protox.Field,
          json_name: "ints",
          kind: :packed,
          label: :repeated,
          name: :ints,
          tag: 8,
          type: :int64
        },
        %{
          __struct__: Protox.Field,
          json_name: "strings",
          kind: :unpacked,
          label: :repeated,
          name: :strings,
          tag: 9,
          type: :bytes
        },
        %{
          __struct__: Protox.Field,
          json_name: "tensors",
          kind: :unpacked,
          label: :repeated,
          name: :tensors,
          tag: 10,
          type: {:message, Onnx.TensorProto}
        },
        %{
          __struct__: Protox.Field,
          json_name: "graphs",
          kind: :unpacked,
          label: :repeated,
          name: :graphs,
          tag: 11,
          type: {:message, Onnx.GraphProto}
        },
        %{
          __struct__: Protox.Field,
          json_name: "docString",
          kind: {:scalar, ""},
          label: :optional,
          name: :doc_string,
          tag: 13,
          type: :string
        },
        %{
          __struct__: Protox.Field,
          json_name: "tp",
          kind: {:scalar, nil},
          label: :optional,
          name: :tp,
          tag: 14,
          type: {:message, Onnx.TypeProto}
        },
        %{
          __struct__: Protox.Field,
          json_name: "typeProtos",
          kind: :unpacked,
          label: :repeated,
          name: :type_protos,
          tag: 15,
          type: {:message, Onnx.TypeProto}
        },
        %{
          __struct__: Protox.Field,
          json_name: "type",
          kind: {:scalar, :UNDEFINED},
          label: :optional,
          name: :type,
          tag: 20,
          type: {:enum, Onnx.AttributeProto.AttributeType}
        },
        %{
          __struct__: Protox.Field,
          json_name: "refAttrName",
          kind: {:scalar, ""},
          label: :optional,
          name: :ref_attr_name,
          tag: 21,
          type: :string
        },
        %{
          __struct__: Protox.Field,
          json_name: "sparseTensor",
          kind: {:scalar, nil},
          label: :optional,
          name: :sparse_tensor,
          tag: 22,
          type: {:message, Onnx.SparseTensorProto}
        },
        %{
          __struct__: Protox.Field,
          json_name: "sparseTensors",
          kind: :unpacked,
          label: :repeated,
          name: :sparse_tensors,
          tag: 23,
          type: {:message, Onnx.SparseTensorProto}
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
        def field_def(:f) do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "f",
             kind: {:scalar, 0.0},
             label: :optional,
             name: :f,
             tag: 2,
             type: :float
           }}
        end

        def field_def("f") do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "f",
             kind: {:scalar, 0.0},
             label: :optional,
             name: :f,
             tag: 2,
             type: :float
           }}
        end

        []
      ),
      (
        def field_def(:i) do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "i",
             kind: {:scalar, 0},
             label: :optional,
             name: :i,
             tag: 3,
             type: :int64
           }}
        end

        def field_def("i") do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "i",
             kind: {:scalar, 0},
             label: :optional,
             name: :i,
             tag: 3,
             type: :int64
           }}
        end

        []
      ),
      (
        def field_def(:s) do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "s",
             kind: {:scalar, ""},
             label: :optional,
             name: :s,
             tag: 4,
             type: :bytes
           }}
        end

        def field_def("s") do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "s",
             kind: {:scalar, ""},
             label: :optional,
             name: :s,
             tag: 4,
             type: :bytes
           }}
        end

        []
      ),
      (
        def field_def(:t) do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "t",
             kind: {:scalar, nil},
             label: :optional,
             name: :t,
             tag: 5,
             type: {:message, Onnx.TensorProto}
           }}
        end

        def field_def("t") do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "t",
             kind: {:scalar, nil},
             label: :optional,
             name: :t,
             tag: 5,
             type: {:message, Onnx.TensorProto}
           }}
        end

        []
      ),
      (
        def field_def(:g) do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "g",
             kind: {:scalar, nil},
             label: :optional,
             name: :g,
             tag: 6,
             type: {:message, Onnx.GraphProto}
           }}
        end

        def field_def("g") do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "g",
             kind: {:scalar, nil},
             label: :optional,
             name: :g,
             tag: 6,
             type: {:message, Onnx.GraphProto}
           }}
        end

        []
      ),
      (
        def field_def(:floats) do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "floats",
             kind: :packed,
             label: :repeated,
             name: :floats,
             tag: 7,
             type: :float
           }}
        end

        def field_def("floats") do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "floats",
             kind: :packed,
             label: :repeated,
             name: :floats,
             tag: 7,
             type: :float
           }}
        end

        []
      ),
      (
        def field_def(:ints) do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "ints",
             kind: :packed,
             label: :repeated,
             name: :ints,
             tag: 8,
             type: :int64
           }}
        end

        def field_def("ints") do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "ints",
             kind: :packed,
             label: :repeated,
             name: :ints,
             tag: 8,
             type: :int64
           }}
        end

        []
      ),
      (
        def field_def(:strings) do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "strings",
             kind: :unpacked,
             label: :repeated,
             name: :strings,
             tag: 9,
             type: :bytes
           }}
        end

        def field_def("strings") do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "strings",
             kind: :unpacked,
             label: :repeated,
             name: :strings,
             tag: 9,
             type: :bytes
           }}
        end

        []
      ),
      (
        def field_def(:tensors) do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "tensors",
             kind: :unpacked,
             label: :repeated,
             name: :tensors,
             tag: 10,
             type: {:message, Onnx.TensorProto}
           }}
        end

        def field_def("tensors") do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "tensors",
             kind: :unpacked,
             label: :repeated,
             name: :tensors,
             tag: 10,
             type: {:message, Onnx.TensorProto}
           }}
        end

        []
      ),
      (
        def field_def(:graphs) do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "graphs",
             kind: :unpacked,
             label: :repeated,
             name: :graphs,
             tag: 11,
             type: {:message, Onnx.GraphProto}
           }}
        end

        def field_def("graphs") do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "graphs",
             kind: :unpacked,
             label: :repeated,
             name: :graphs,
             tag: 11,
             type: {:message, Onnx.GraphProto}
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
             tag: 13,
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
             tag: 13,
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
             tag: 13,
             type: :string
           }}
        end
      ),
      (
        def field_def(:tp) do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "tp",
             kind: {:scalar, nil},
             label: :optional,
             name: :tp,
             tag: 14,
             type: {:message, Onnx.TypeProto}
           }}
        end

        def field_def("tp") do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "tp",
             kind: {:scalar, nil},
             label: :optional,
             name: :tp,
             tag: 14,
             type: {:message, Onnx.TypeProto}
           }}
        end

        []
      ),
      (
        def field_def(:type_protos) do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "typeProtos",
             kind: :unpacked,
             label: :repeated,
             name: :type_protos,
             tag: 15,
             type: {:message, Onnx.TypeProto}
           }}
        end

        def field_def("typeProtos") do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "typeProtos",
             kind: :unpacked,
             label: :repeated,
             name: :type_protos,
             tag: 15,
             type: {:message, Onnx.TypeProto}
           }}
        end

        def field_def("type_protos") do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "typeProtos",
             kind: :unpacked,
             label: :repeated,
             name: :type_protos,
             tag: 15,
             type: {:message, Onnx.TypeProto}
           }}
        end
      ),
      (
        def field_def(:type) do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "type",
             kind: {:scalar, :UNDEFINED},
             label: :optional,
             name: :type,
             tag: 20,
             type: {:enum, Onnx.AttributeProto.AttributeType}
           }}
        end

        def field_def("type") do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "type",
             kind: {:scalar, :UNDEFINED},
             label: :optional,
             name: :type,
             tag: 20,
             type: {:enum, Onnx.AttributeProto.AttributeType}
           }}
        end

        []
      ),
      (
        def field_def(:ref_attr_name) do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "refAttrName",
             kind: {:scalar, ""},
             label: :optional,
             name: :ref_attr_name,
             tag: 21,
             type: :string
           }}
        end

        def field_def("refAttrName") do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "refAttrName",
             kind: {:scalar, ""},
             label: :optional,
             name: :ref_attr_name,
             tag: 21,
             type: :string
           }}
        end

        def field_def("ref_attr_name") do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "refAttrName",
             kind: {:scalar, ""},
             label: :optional,
             name: :ref_attr_name,
             tag: 21,
             type: :string
           }}
        end
      ),
      (
        def field_def(:sparse_tensor) do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "sparseTensor",
             kind: {:scalar, nil},
             label: :optional,
             name: :sparse_tensor,
             tag: 22,
             type: {:message, Onnx.SparseTensorProto}
           }}
        end

        def field_def("sparseTensor") do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "sparseTensor",
             kind: {:scalar, nil},
             label: :optional,
             name: :sparse_tensor,
             tag: 22,
             type: {:message, Onnx.SparseTensorProto}
           }}
        end

        def field_def("sparse_tensor") do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "sparseTensor",
             kind: {:scalar, nil},
             label: :optional,
             name: :sparse_tensor,
             tag: 22,
             type: {:message, Onnx.SparseTensorProto}
           }}
        end
      ),
      (
        def field_def(:sparse_tensors) do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "sparseTensors",
             kind: :unpacked,
             label: :repeated,
             name: :sparse_tensors,
             tag: 23,
             type: {:message, Onnx.SparseTensorProto}
           }}
        end

        def field_def("sparseTensors") do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "sparseTensors",
             kind: :unpacked,
             label: :repeated,
             name: :sparse_tensors,
             tag: 23,
             type: {:message, Onnx.SparseTensorProto}
           }}
        end

        def field_def("sparse_tensors") do
          {:ok,
           %{
             __struct__: Protox.Field,
             json_name: "sparseTensors",
             kind: :unpacked,
             label: :repeated,
             name: :sparse_tensors,
             tag: 23,
             type: {:message, Onnx.SparseTensorProto}
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
    def default(:f) do
      {:ok, 0.0}
    end,
    def default(:i) do
      {:ok, 0}
    end,
    def default(:s) do
      {:ok, ""}
    end,
    def default(:t) do
      {:ok, nil}
    end,
    def default(:g) do
      {:ok, nil}
    end,
    def default(:floats) do
      {:error, :no_default_value}
    end,
    def default(:ints) do
      {:error, :no_default_value}
    end,
    def default(:strings) do
      {:error, :no_default_value}
    end,
    def default(:tensors) do
      {:error, :no_default_value}
    end,
    def default(:graphs) do
      {:error, :no_default_value}
    end,
    def default(:doc_string) do
      {:ok, ""}
    end,
    def default(:tp) do
      {:ok, nil}
    end,
    def default(:type_protos) do
      {:error, :no_default_value}
    end,
    def default(:type) do
      {:ok, :UNDEFINED}
    end,
    def default(:ref_attr_name) do
      {:ok, ""}
    end,
    def default(:sparse_tensor) do
      {:ok, nil}
    end,
    def default(:sparse_tensors) do
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
