# credo:disable-for-this-file
defmodule Onnx.AttributeProto.AttributeType do
  @moduledoc false
  (
    defstruct []

    (
      @spec default() :: :UNDEFINED
      def default() do
        :UNDEFINED
      end
    )

    @spec encode(atom()) :: integer() | atom()
    [
      (
        def encode(:UNDEFINED) do
          0
        end

        def encode("UNDEFINED") do
          0
        end
      ),
      (
        def encode(:FLOAT) do
          1
        end

        def encode("FLOAT") do
          1
        end
      ),
      (
        def encode(:INT) do
          2
        end

        def encode("INT") do
          2
        end
      ),
      (
        def encode(:STRING) do
          3
        end

        def encode("STRING") do
          3
        end
      ),
      (
        def encode(:TENSOR) do
          4
        end

        def encode("TENSOR") do
          4
        end
      ),
      (
        def encode(:GRAPH) do
          5
        end

        def encode("GRAPH") do
          5
        end
      ),
      (
        def encode(:SPARSE_TENSOR) do
          11
        end

        def encode("SPARSE_TENSOR") do
          11
        end
      ),
      (
        def encode(:TYPE_PROTO) do
          13
        end

        def encode("TYPE_PROTO") do
          13
        end
      ),
      (
        def encode(:FLOATS) do
          6
        end

        def encode("FLOATS") do
          6
        end
      ),
      (
        def encode(:INTS) do
          7
        end

        def encode("INTS") do
          7
        end
      ),
      (
        def encode(:STRINGS) do
          8
        end

        def encode("STRINGS") do
          8
        end
      ),
      (
        def encode(:TENSORS) do
          9
        end

        def encode("TENSORS") do
          9
        end
      ),
      (
        def encode(:GRAPHS) do
          10
        end

        def encode("GRAPHS") do
          10
        end
      ),
      (
        def encode(:SPARSE_TENSORS) do
          12
        end

        def encode("SPARSE_TENSORS") do
          12
        end
      ),
      (
        def encode(:TYPE_PROTOS) do
          14
        end

        def encode("TYPE_PROTOS") do
          14
        end
      )
    ]

    def encode(x) do
      x
    end

    @spec decode(integer()) :: atom() | integer()
    [
      def decode(0) do
        :UNDEFINED
      end,
      def decode(1) do
        :FLOAT
      end,
      def decode(2) do
        :INT
      end,
      def decode(3) do
        :STRING
      end,
      def decode(4) do
        :TENSOR
      end,
      def decode(5) do
        :GRAPH
      end,
      def decode(6) do
        :FLOATS
      end,
      def decode(7) do
        :INTS
      end,
      def decode(8) do
        :STRINGS
      end,
      def decode(9) do
        :TENSORS
      end,
      def decode(10) do
        :GRAPHS
      end,
      def decode(11) do
        :SPARSE_TENSOR
      end,
      def decode(12) do
        :SPARSE_TENSORS
      end,
      def decode(13) do
        :TYPE_PROTO
      end,
      def decode(14) do
        :TYPE_PROTOS
      end
    ]

    def decode(x) do
      x
    end

    @spec constants() :: [{integer(), atom()}]
    def constants() do
      [
        {0, :UNDEFINED},
        {1, :FLOAT},
        {2, :INT},
        {3, :STRING},
        {4, :TENSOR},
        {5, :GRAPH},
        {11, :SPARSE_TENSOR},
        {13, :TYPE_PROTO},
        {6, :FLOATS},
        {7, :INTS},
        {8, :STRINGS},
        {9, :TENSORS},
        {10, :GRAPHS},
        {12, :SPARSE_TENSORS},
        {14, :TYPE_PROTOS}
      ]
    end

    @spec has_constant?(any()) :: boolean()
    (
      [
        def has_constant?(:UNDEFINED) do
          true
        end,
        def has_constant?(:FLOAT) do
          true
        end,
        def has_constant?(:INT) do
          true
        end,
        def has_constant?(:STRING) do
          true
        end,
        def has_constant?(:TENSOR) do
          true
        end,
        def has_constant?(:GRAPH) do
          true
        end,
        def has_constant?(:SPARSE_TENSOR) do
          true
        end,
        def has_constant?(:TYPE_PROTO) do
          true
        end,
        def has_constant?(:FLOATS) do
          true
        end,
        def has_constant?(:INTS) do
          true
        end,
        def has_constant?(:STRINGS) do
          true
        end,
        def has_constant?(:TENSORS) do
          true
        end,
        def has_constant?(:GRAPHS) do
          true
        end,
        def has_constant?(:SPARSE_TENSORS) do
          true
        end,
        def has_constant?(:TYPE_PROTOS) do
          true
        end
      ]

      def has_constant?(_) do
        false
      end
    )
  )
end