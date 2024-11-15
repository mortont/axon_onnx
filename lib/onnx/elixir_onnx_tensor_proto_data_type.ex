# credo:disable-for-this-file
defmodule Onnx.TensorProto.DataType do
  @moduledoc false
  (
    defstruct []

    (
      @spec default() :: :UNDEFINED
      def default() do
        :UNDEFINED
      end
    )

    @spec encode(atom() | String.t()) :: integer() | atom()
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
        def encode(:UINT8) do
          2
        end

        def encode("UINT8") do
          2
        end
      ),
      (
        def encode(:INT8) do
          3
        end

        def encode("INT8") do
          3
        end
      ),
      (
        def encode(:UINT16) do
          4
        end

        def encode("UINT16") do
          4
        end
      ),
      (
        def encode(:INT16) do
          5
        end

        def encode("INT16") do
          5
        end
      ),
      (
        def encode(:INT32) do
          6
        end

        def encode("INT32") do
          6
        end
      ),
      (
        def encode(:INT64) do
          7
        end

        def encode("INT64") do
          7
        end
      ),
      (
        def encode(:STRING) do
          8
        end

        def encode("STRING") do
          8
        end
      ),
      (
        def encode(:BOOL) do
          9
        end

        def encode("BOOL") do
          9
        end
      ),
      (
        def encode(:FLOAT16) do
          10
        end

        def encode("FLOAT16") do
          10
        end
      ),
      (
        def encode(:DOUBLE) do
          11
        end

        def encode("DOUBLE") do
          11
        end
      ),
      (
        def encode(:UINT32) do
          12
        end

        def encode("UINT32") do
          12
        end
      ),
      (
        def encode(:UINT64) do
          13
        end

        def encode("UINT64") do
          13
        end
      ),
      (
        def encode(:COMPLEX64) do
          14
        end

        def encode("COMPLEX64") do
          14
        end
      ),
      (
        def encode(:COMPLEX128) do
          15
        end

        def encode("COMPLEX128") do
          15
        end
      ),
      (
        def encode(:BFLOAT16) do
          16
        end

        def encode("BFLOAT16") do
          16
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
        :UINT8
      end,
      def decode(3) do
        :INT8
      end,
      def decode(4) do
        :UINT16
      end,
      def decode(5) do
        :INT16
      end,
      def decode(6) do
        :INT32
      end,
      def decode(7) do
        :INT64
      end,
      def decode(8) do
        :STRING
      end,
      def decode(9) do
        :BOOL
      end,
      def decode(10) do
        :FLOAT16
      end,
      def decode(11) do
        :DOUBLE
      end,
      def decode(12) do
        :UINT32
      end,
      def decode(13) do
        :UINT64
      end,
      def decode(14) do
        :COMPLEX64
      end,
      def decode(15) do
        :COMPLEX128
      end,
      def decode(16) do
        :BFLOAT16
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
        {2, :UINT8},
        {3, :INT8},
        {4, :UINT16},
        {5, :INT16},
        {6, :INT32},
        {7, :INT64},
        {8, :STRING},
        {9, :BOOL},
        {10, :FLOAT16},
        {11, :DOUBLE},
        {12, :UINT32},
        {13, :UINT64},
        {14, :COMPLEX64},
        {15, :COMPLEX128},
        {16, :BFLOAT16}
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
        def has_constant?(:UINT8) do
          true
        end,
        def has_constant?(:INT8) do
          true
        end,
        def has_constant?(:UINT16) do
          true
        end,
        def has_constant?(:INT16) do
          true
        end,
        def has_constant?(:INT32) do
          true
        end,
        def has_constant?(:INT64) do
          true
        end,
        def has_constant?(:STRING) do
          true
        end,
        def has_constant?(:BOOL) do
          true
        end,
        def has_constant?(:FLOAT16) do
          true
        end,
        def has_constant?(:DOUBLE) do
          true
        end,
        def has_constant?(:UINT32) do
          true
        end,
        def has_constant?(:UINT64) do
          true
        end,
        def has_constant?(:COMPLEX64) do
          true
        end,
        def has_constant?(:COMPLEX128) do
          true
        end,
        def has_constant?(:BFLOAT16) do
          true
        end
      ]

      def has_constant?(_) do
        false
      end
    )
  )
end
