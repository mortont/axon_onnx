# credo:disable-for-this-file
defmodule Onnx.TensorProto.DataLocation do
  @moduledoc false
  (
    defstruct []

    (
      @spec default() :: :DEFAULT
      def default() do
        :DEFAULT
      end
    )

    @spec encode(atom()) :: integer() | atom()
    [
      (
        def encode(:DEFAULT) do
          0
        end

        def encode("DEFAULT") do
          0
        end
      ),
      (
        def encode(:EXTERNAL) do
          1
        end

        def encode("EXTERNAL") do
          1
        end
      )
    ]

    def encode(x) do
      x
    end

    @spec decode(integer()) :: atom() | integer()
    [
      def decode(0) do
        :DEFAULT
      end,
      def decode(1) do
        :EXTERNAL
      end
    ]

    def decode(x) do
      x
    end

    @spec constants() :: [{integer(), atom()}]
    def constants() do
      [{0, :DEFAULT}, {1, :EXTERNAL}]
    end

    @spec has_constant?(any()) :: boolean()
    (
      [
        def has_constant?(:DEFAULT) do
          true
        end,
        def has_constant?(:EXTERNAL) do
          true
        end
      ]

      def has_constant?(_) do
        false
      end
    )
  )
end