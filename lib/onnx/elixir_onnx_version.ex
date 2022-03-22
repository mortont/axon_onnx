# credo:disable-for-this-file
defmodule Onnx.Version do
  @moduledoc false
  (
    defstruct []

    (
      @spec default() :: :_START_VERSION
      def default() do
        :_START_VERSION
      end
    )

    @spec encode(atom()) :: integer() | atom()
    [
      (
        def encode(:_START_VERSION) do
          0
        end

        def encode("_START_VERSION") do
          0
        end
      ),
      (
        def encode(:IR_VERSION_2017_10_10) do
          1
        end

        def encode("IR_VERSION_2017_10_10") do
          1
        end
      ),
      (
        def encode(:IR_VERSION_2017_10_30) do
          2
        end

        def encode("IR_VERSION_2017_10_30") do
          2
        end
      ),
      (
        def encode(:IR_VERSION_2017_11_3) do
          3
        end

        def encode("IR_VERSION_2017_11_3") do
          3
        end
      ),
      (
        def encode(:IR_VERSION_2019_1_22) do
          4
        end

        def encode("IR_VERSION_2019_1_22") do
          4
        end
      ),
      (
        def encode(:IR_VERSION_2019_3_18) do
          5
        end

        def encode("IR_VERSION_2019_3_18") do
          5
        end
      ),
      (
        def encode(:IR_VERSION_2019_9_19) do
          6
        end

        def encode("IR_VERSION_2019_9_19") do
          6
        end
      ),
      (
        def encode(:IR_VERSION_2020_5_8) do
          7
        end

        def encode("IR_VERSION_2020_5_8") do
          7
        end
      ),
      (
        def encode(:IR_VERSION) do
          8
        end

        def encode("IR_VERSION") do
          8
        end
      )
    ]

    def encode(x) do
      x
    end

    @spec decode(integer()) :: atom() | integer()
    [
      def decode(0) do
        :_START_VERSION
      end,
      def decode(1) do
        :IR_VERSION_2017_10_10
      end,
      def decode(2) do
        :IR_VERSION_2017_10_30
      end,
      def decode(3) do
        :IR_VERSION_2017_11_3
      end,
      def decode(4) do
        :IR_VERSION_2019_1_22
      end,
      def decode(5) do
        :IR_VERSION_2019_3_18
      end,
      def decode(6) do
        :IR_VERSION_2019_9_19
      end,
      def decode(7) do
        :IR_VERSION_2020_5_8
      end,
      def decode(8) do
        :IR_VERSION
      end
    ]

    def decode(x) do
      x
    end

    @spec constants() :: [{integer(), atom()}]
    def constants() do
      [
        {0, :_START_VERSION},
        {1, :IR_VERSION_2017_10_10},
        {2, :IR_VERSION_2017_10_30},
        {3, :IR_VERSION_2017_11_3},
        {4, :IR_VERSION_2019_1_22},
        {5, :IR_VERSION_2019_3_18},
        {6, :IR_VERSION_2019_9_19},
        {7, :IR_VERSION_2020_5_8},
        {8, :IR_VERSION}
      ]
    end

    @spec has_constant?(any()) :: boolean()
    (
      [
        def has_constant?(:_START_VERSION) do
          true
        end,
        def has_constant?(:IR_VERSION_2017_10_10) do
          true
        end,
        def has_constant?(:IR_VERSION_2017_10_30) do
          true
        end,
        def has_constant?(:IR_VERSION_2017_11_3) do
          true
        end,
        def has_constant?(:IR_VERSION_2019_1_22) do
          true
        end,
        def has_constant?(:IR_VERSION_2019_3_18) do
          true
        end,
        def has_constant?(:IR_VERSION_2019_9_19) do
          true
        end,
        def has_constant?(:IR_VERSION_2020_5_8) do
          true
        end,
        def has_constant?(:IR_VERSION) do
          true
        end
      ]

      def has_constant?(_) do
        false
      end
    )
  )
end
