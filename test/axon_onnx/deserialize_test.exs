defmodule DeserializeTest do
  use ExUnit.Case
  import OnnxTestHelper

  describe "node tests" do
    test "Abs" do
      check_onnx_test_case!("node", "test_abs")
    end

    test "Acos" do
      check_onnx_test_case!("node", "test_acos")
      check_onnx_test_case!("node", "test_acos_example")
    end

    test "Acosh" do
      check_onnx_test_case!("node", "test_acosh")
      check_onnx_test_case!("node", "test_acosh_example")
    end

    test "Add" do
      check_onnx_test_case!("node", "test_add")
      # TODO: Update Axon broadcasting semantics
      # check_onnx_test_case!("node", "test_add_bcast")
      check_onnx_test_case!("node", "test_add_uint8")
    end

    test "And" do
      check_onnx_test_case!("node", "test_and2d")
      check_onnx_test_case!("node", "test_and3d")
      check_onnx_test_case!("node", "test_and4d")
      # TODO: Update Axon broadcasting semantics
      # check_onnx_test_case!("node", "test_and_bcast3v1d")
      check_onnx_test_case!("node", "test_and_bcast3v2d")
      check_onnx_test_case!("node", "test_and_bcast4v2d")
      check_onnx_test_case!("node", "test_and_bcast4v3d")
      check_onnx_test_case!("node", "test_and_bcast4v4d")
    end

    test "ArgMax" do
      check_onnx_test_case!("node", "test_argmax_default_axis_example")
      check_onnx_test_case!("node", "test_argmax_default_axis_example_select_last_index")
      check_onnx_test_case!("node", "test_argmax_default_axis_random")
      check_onnx_test_case!("node", "test_argmax_default_axis_random_select_last_index")
      check_onnx_test_case!("node", "test_argmax_keepdims_example")
      check_onnx_test_case!("node", "test_argmax_keepdims_example_select_last_index")
      check_onnx_test_case!("node", "test_argmax_keepdims_random")
      check_onnx_test_case!("node", "test_argmax_keepdims_random_select_last_index")
      check_onnx_test_case!("node", "test_argmax_negative_axis_keepdims_example")

      check_onnx_test_case!(
        "node",
        "test_argmax_negative_axis_keepdims_example_select_last_index"
      )

      check_onnx_test_case!("node", "test_argmax_negative_axis_keepdims_random")
      check_onnx_test_case!("node", "test_argmax_negative_axis_keepdims_random_select_last_index")
      check_onnx_test_case!("node", "test_argmax_no_keepdims_example")
      check_onnx_test_case!("node", "test_argmax_no_keepdims_example_select_last_index")
      check_onnx_test_case!("node", "test_argmax_no_keepdims_random")
      check_onnx_test_case!("node", "test_argmax_no_keepdims_random_select_last_index")
    end

    test "ArgMin" do
      check_onnx_test_case!("node", "test_argmin_default_axis_example")
      check_onnx_test_case!("node", "test_argmin_default_axis_example_select_last_index")
      check_onnx_test_case!("node", "test_argmin_default_axis_random")
      check_onnx_test_case!("node", "test_argmin_default_axis_random_select_last_index")
      check_onnx_test_case!("node", "test_argmin_keepdims_example")
      check_onnx_test_case!("node", "test_argmin_keepdims_example_select_last_index")
      check_onnx_test_case!("node", "test_argmin_keepdims_random")
      check_onnx_test_case!("node", "test_argmin_keepdims_random_select_last_index")
      check_onnx_test_case!("node", "test_argmin_negative_axis_keepdims_example")

      check_onnx_test_case!(
        "node",
        "test_argmin_negative_axis_keepdims_example_select_last_index"
      )

      check_onnx_test_case!("node", "test_argmin_negative_axis_keepdims_random")
      check_onnx_test_case!("node", "test_argmin_negative_axis_keepdims_random_select_last_index")
      check_onnx_test_case!("node", "test_argmin_no_keepdims_example")
      check_onnx_test_case!("node", "test_argmin_no_keepdims_example_select_last_index")
      check_onnx_test_case!("node", "test_argmin_no_keepdims_random")
      check_onnx_test_case!("node", "test_argmin_no_keepdims_random_select_last_index")
    end

    test "Asin" do
      check_onnx_test_case!("node", "test_asin")
      check_onnx_test_case!("node", "test_asin_example")
    end

    test "Asinh" do
      check_onnx_test_case!("node", "test_asinh")
      check_onnx_test_case!("node", "test_asinh_example")
    end

    test "AveragePool" do
      check_onnx_test_case!("node", "test_averagepool_1d_default")
      check_onnx_test_case!("node", "test_averagepool_2d_default")
      # TODO: Count include pad
      # check_onnx_test_case!("node", "test_averagepool_2d_pads")
      check_onnx_test_case!("node", "test_averagepool_2d_pads_count_include_pad")
      # check_onnx_test_case!("node", "test_averagepool_2d_precomputed_pads")
      check_onnx_test_case!("node", "test_averagepool_2d_precomputed_pads_count_include_pad")
      # check_onnx_test_case!("node", "test_averagepool_2d_precomputed_same_upper")
      check_onnx_test_case!("node", "test_averagepool_2d_precomputed_strides")
      # check_onnx_test_case!("node", "test_averagepool_2d_same_lower")
      # check_onnx_test_case!("node", "test_averagepool_2d_same_upper")
      check_onnx_test_case!("node", "test_averagepool_2d_strides")
      check_onnx_test_case!("node", "test_averagepool_3d_default")
      # TODO: Ceil mode
      # check_onnx_test_case!("node", "test_averagepool_2d_ceil")
    end

    test "BitShift" do
      check_onnx_test_case!("node", "test_bitshift_left_uint8")
      check_onnx_test_case!("node", "test_bitshift_left_uint16")
      check_onnx_test_case!("node", "test_bitshift_left_uint32")
      check_onnx_test_case!("node", "test_bitshift_left_uint64")
      check_onnx_test_case!("node", "test_bitshift_right_uint8")
      check_onnx_test_case!("node", "test_bitshift_right_uint16")
      check_onnx_test_case!("node", "test_bitshift_right_uint32")
      check_onnx_test_case!("node", "test_bitshift_right_uint64")
    end

    test "Cast" do
      # check_onnx_test_case!("node", "test_cast_BFLOAT16_to_FLOAT")
      check_onnx_test_case!("node", "test_cast_DOUBLE_to_FLOAT")
      check_onnx_test_case!("node", "test_cast_DOUBLE_to_FLOAT16")
      check_onnx_test_case!("node", "test_cast_FLOAT16_to_FLOAT")
      # check_onnx_test_case!("node", "test_cast_FLOAT_to_BFLOAT16")
      check_onnx_test_case!("node", "test_cast_FLOAT_to_DOUBLE")
      check_onnx_test_case!("node", "test_cast_FLOAT_to_FLOAT16")
      # check_onnx_test_case!("node", "test_cast_FLOAT_to_STRING")
      # check_onnx_test_case!("node", "test_cast_STRING_to_FLOAT")
    end

    test "Ceil" do
      check_onnx_test_case!("node", "test_ceil")
      check_onnx_test_case!("node", "test_ceil_example")
    end

    test "Celu" do
      check_onnx_test_case!("node", "test_celu")
      check_onnx_test_case!("node", "test_celu_expanded")
    end

    test "Concat" do
      check_onnx_test_case!("node", "test_concat_1d_axis_0")
      check_onnx_test_case!("node", "test_concat_1d_axis_negative_1")
      check_onnx_test_case!("node", "test_concat_2d_axis_0")
      check_onnx_test_case!("node", "test_concat_2d_axis_1")
      check_onnx_test_case!("node", "test_concat_2d_axis_negative_1")
      check_onnx_test_case!("node", "test_concat_2d_axis_negative_2")
      check_onnx_test_case!("node", "test_concat_3d_axis_0")
      check_onnx_test_case!("node", "test_concat_3d_axis_1")
      check_onnx_test_case!("node", "test_concat_3d_axis_2")
      check_onnx_test_case!("node", "test_concat_3d_axis_negative_1")
      check_onnx_test_case!("node", "test_concat_3d_axis_negative_2")
      check_onnx_test_case!("node", "test_concat_3d_axis_negative_3")
    end

    test "Constant" do
      check_onnx_test_case!("node", "test_constant")
      # TODO
      # check_onnx_test_case!("node", "test_constant_pad")
    end

    test "Cos" do
      check_onnx_test_case!("node", "test_cos")
      check_onnx_test_case!("node", "test_cos_example")
    end

    test "Cosh" do
      check_onnx_test_case!("node", "test_cosh")
      check_onnx_test_case!("node", "test_cosh_example")
    end

    test "Div" do
      check_onnx_test_case!("node", "test_div")
      # TODO: Update Axon broadcasting semantics
      # check_onnx_test_case!("node", "test_div_bcast")
      # check_onnx_test_case!("node", "test_div_example")
      # TODO: Cast?
      # check_onnx_test_case!("node", "test_div_uint8")
    end

    test "Equal" do
      check_onnx_test_case!("node", "test_equal")
      # TODO: Update Axon broadcasting semantics
      # check_onnx_test_case!("node", "test_equal_bcast")
    end

    test "Erf" do
      check_onnx_test_case!("node", "test_erf")
    end

    test "Exp" do
      check_onnx_test_case!("node", "test_exp")
      check_onnx_test_case!("node", "test_exp_example")
    end

    test "Floor" do
      check_onnx_test_case!("node", "test_floor")
      check_onnx_test_case!("node", "test_floor_example")
    end

    test "Gather" do
      check_onnx_test_case!("node", "test_gather_0")
      check_onnx_test_case!("node", "test_gather_1")
      check_onnx_test_case!("node", "test_gather_2d_indices")
    end

    test "GlobalAveragePool" do
      check_onnx_test_case!("node", "test_globalaveragepool")
      check_onnx_test_case!("node", "test_globalaveragepool_precomputed")
    end

    test "GlobalMaxPool" do
      check_onnx_test_case!("node", "test_globalmaxpool")
      check_onnx_test_case!("node", "test_globalmaxpool_precomputed")
    end

    test "Greater" do
      check_onnx_test_case!("node", "test_greater")
      # TODO: Update Axon broadcasting semantics
      # check_onnx_test_case!("node", "test_greater_bcast")
    end

    test "GreaterOrEqual" do
      check_onnx_test_case!("node", "test_greater_equal")
      check_onnx_test_case!("node", "test_greater_equal_expanded")
      # TODO: Update Axon broadcasting semantics
      # check_onnx_test_case!("node", "test_greater_equal_bcast")
      # check_onnx_test_case!("node", "test_greater_equal_bcast_expanded")
    end

    test "HardSigmoid" do
      check_onnx_test_case!("node", "test_hardsigmoid")
      check_onnx_test_case!("node", "test_hardsigmoid_default")
    end

    test "HardSwish" do
      check_onnx_test_case!("node", "test_hardswish")
      check_onnx_test_case!("node", "test_hardswish_expanded")
    end

    test "Identity" do
      check_onnx_test_case!("node", "test_identity")
      # TODO: Sequence types
      # check_onnx_test_case!("node", "test_identity_sequence")
    end

    test "If" do
      check_onnx_test_case!("node", "test_if")
    end

    test "LeakyRelu" do
      check_onnx_test_case!("node", "test_leakyrelu")
      check_onnx_test_case!("node", "test_leakyrelu_default")
      check_onnx_test_case!("node", "test_leakyrelu_example")
    end

    test "Less" do
      check_onnx_test_case!("node", "test_less")
      # TODO: Update Axon broadcasting semantics
      # check_onnx_test_case!("node", "test_less_bcast")
    end

    test "LessOrEqual" do
      check_onnx_test_case!("node", "test_less_equal")
      check_onnx_test_case!("node", "test_less_equal_expanded")
      # TODO: Update Axon broadcasting semantics
      # check_onnx_test_case!("node", "test_less_equal_bcast")
      # check_onnx_test_case!("node", "test_less_equal_bcast_expanded")
    end

    test "Log" do
      check_onnx_test_case!("node", "test_log")
      check_onnx_test_case!("node", "test_log_example")
    end

    test "LogSoftmax" do
      check_onnx_test_case!("node", "test_logsoftmax_axis_0")
      # TODO: ReduceMax
      # check_onnx_test_case!("node", "test_logsoftmax_axis_0_expanded")
      check_onnx_test_case!("node", "test_logsoftmax_axis_1")
      # TODO: ReduceMax
      # check_onnx_test_case!("node", "test_logsoftmax_axis_1_expanded")
      check_onnx_test_case!("node", "test_logsoftmax_axis_2")
      # TODO: ReduceMax
      # check_onnx_test_case!("node", "test_logsoftmax_axis_2_expanded")
      check_onnx_test_case!("node", "test_logsoftmax_default_axis")
      # TODO: ReduceMax
      # check_onnx_test_case!("node", "test_logsoftmax_default_axis_expanded")
      check_onnx_test_case!("node", "test_logsoftmax_large_number")
      # TODO: ReduceMax
      # check_onnx_test_case!("node", "test_logsoftmax_large_number_expanded")
      check_onnx_test_case!("node", "test_logsoftmax_negative_axis")
      # TODO: ReduceMax
      # check_onnx_test_case!("node", "test_logsoftmax_negative_axis_expanded")
      check_onnx_test_case!("node", "test_logsoftmax_example_1")
      # TODO: ReduceMax
      # check_onnx_test_case!("node", "test_logsoftmax_example_1_expanded")
    end

    test "MaxPool" do
      check_onnx_test_case!("node", "test_maxpool_1d_default")
      check_onnx_test_case!("node", "test_maxpool_2d_default")
      # TODO: https://github.com/elixir-nx/axon/issues/185
      # check_onnx_test_case!("node", "test_maxpool_2d_dilations")
      check_onnx_test_case!("node", "test_maxpool_2d_pads")
      # TODO: Adjust pads for same lower behavior
      # check_onnx_test_case!("node", "test_maxpool_2d_same_lower")
      check_onnx_test_case!("node", "test_maxpool_2d_same_upper")
      check_onnx_test_case!("node", "test_maxpool_2d_strides")
      check_onnx_test_case!("node", "test_maxpool_2d_uint8")
      check_onnx_test_case!("node", "test_maxpool_2d_precomputed_pads")
      check_onnx_test_case!("node", "test_maxpool_2d_precomputed_same_upper")
      check_onnx_test_case!("node", "test_maxpool_2d_precomputed_strides")
      # TODO: Return indices in MaxPool
      # check_onnx_test_case!("node", "test_maxpool_with_argmax_2d_precomputed_pads")
      # check_onnx_test_case!("node", "test_maxpool_with_argmax_2d_precomputed_strides")
      check_onnx_test_case!("node", "test_maxpool_3d_default")

      # TODO: Reevaluate this behavior
      assert_raise ArgumentError, ~r/invalid ceil_mode/, fn ->
        check_onnx_test_case!("node", "test_maxpool_2d_ceil")
      end
    end

    test "Neg" do
      check_onnx_test_case!("node", "test_neg")
      check_onnx_test_case!("node", "test_neg_example")
    end

    test "LRN" do
      check_onnx_test_case!("node", "test_lrn")
      check_onnx_test_case!("node", "test_lrn_default")
    end

    test "Not" do
      check_onnx_test_case!("node", "test_not_2d")
      check_onnx_test_case!("node", "test_not_3d")
      check_onnx_test_case!("node", "test_not_4d")
    end

    test "Or" do
      check_onnx_test_case!("node", "test_or2d")
      check_onnx_test_case!("node", "test_or3d")
      check_onnx_test_case!("node", "test_or4d")
      # TODO: Update Axon broadcasting semantics
      # check_onnx_test_case!("node", "test_or_bcast3v1d")
      check_onnx_test_case!("node", "test_or_bcast3v2d")
      check_onnx_test_case!("node", "test_or_bcast4v2d")
      check_onnx_test_case!("node", "test_or_bcast4v3d")
      check_onnx_test_case!("node", "test_or_bcast4v4d")
    end

    test "Pow" do
      check_onnx_test_case!("node", "test_pow")
      # TODO: Update Axon broadcasting semantics
      # check_onnx_test_case!("node", "test_pow_bcast_array")
      check_onnx_test_case!("node", "test_pow_bcast_scalar")
      check_onnx_test_case!("node", "test_pow_example")
      check_onnx_test_case!("node", "test_pow_types_float")
      check_onnx_test_case!("node", "test_pow_types_float32_int32")
      check_onnx_test_case!("node", "test_pow_types_float32_int64")
      check_onnx_test_case!("node", "test_pow_types_float32_uint32")
      check_onnx_test_case!("node", "test_pow_types_float32_uint64")
      check_onnx_test_case!("node", "test_pow_types_int")
      check_onnx_test_case!("node", "test_pow_types_int32_float32")
      check_onnx_test_case!("node", "test_pow_types_int32_int32")
      check_onnx_test_case!("node", "test_pow_types_int64_float32")
      check_onnx_test_case!("node", "test_pow_types_int64_int64")
    end

    test "ReduceLogSum" do
      check_onnx_test_case!("node", "test_reduce_log_sum")
      check_onnx_test_case!("node", "test_reduce_log_sum_asc_axes")
      check_onnx_test_case!("node", "test_reduce_log_sum_default")
      check_onnx_test_case!("node", "test_reduce_log_sum_desc_axes")
      check_onnx_test_case!("node", "test_reduce_log_sum_negative_axes")
    end

    test "ReduceLogSumExp" do
      check_onnx_test_case!("node", "test_reduce_log_sum_exp_default_axes_keepdims_example")
      check_onnx_test_case!("node", "test_reduce_log_sum_exp_default_axes_keepdims_random")
      check_onnx_test_case!("node", "test_reduce_log_sum_exp_do_not_keepdims_example")
      check_onnx_test_case!("node", "test_reduce_log_sum_exp_do_not_keepdims_random")
      check_onnx_test_case!("node", "test_reduce_log_sum_exp_keepdims_example")
      check_onnx_test_case!("node", "test_reduce_log_sum_exp_keepdims_random")
      check_onnx_test_case!("node", "test_reduce_log_sum_exp_negative_axes_keepdims_example")
      check_onnx_test_case!("node", "test_reduce_log_sum_exp_negative_axes_keepdims_random")
    end

    test "ReduceMax" do
      check_onnx_test_case!("node", "test_reduce_max_default_axes_keepdim_example")
      check_onnx_test_case!("node", "test_reduce_max_default_axes_keepdims_random")
      check_onnx_test_case!("node", "test_reduce_max_do_not_keepdims_example")
      check_onnx_test_case!("node", "test_reduce_max_do_not_keepdims_random")
      check_onnx_test_case!("node", "test_reduce_max_keepdims_example")
      check_onnx_test_case!("node", "test_reduce_max_keepdims_random")
      check_onnx_test_case!("node", "test_reduce_max_negative_axes_keepdims_example")
      check_onnx_test_case!("node", "test_reduce_max_negative_axes_keepdims_random")
    end

    test "ReduceMean" do
      check_onnx_test_case!("node", "test_reduce_mean_default_axes_keepdims_example")
      check_onnx_test_case!("node", "test_reduce_mean_default_axes_keepdims_random")
      check_onnx_test_case!("node", "test_reduce_mean_do_not_keepdims_example")
      check_onnx_test_case!("node", "test_reduce_mean_do_not_keepdims_random")
      check_onnx_test_case!("node", "test_reduce_mean_keepdims_example")
      check_onnx_test_case!("node", "test_reduce_mean_keepdims_random")
      check_onnx_test_case!("node", "test_reduce_mean_negative_axes_keepdims_example")
      check_onnx_test_case!("node", "test_reduce_mean_negative_axes_keepdims_random")
    end

    test "ReduceMin" do
      check_onnx_test_case!("node", "test_reduce_min_default_axes_keepdims_example")
      check_onnx_test_case!("node", "test_reduce_min_default_axes_keepdims_random")
      check_onnx_test_case!("node", "test_reduce_min_do_not_keepdims_example")
      check_onnx_test_case!("node", "test_reduce_min_do_not_keepdims_random")
      check_onnx_test_case!("node", "test_reduce_min_keepdims_example")
      check_onnx_test_case!("node", "test_reduce_min_keepdims_random")
      check_onnx_test_case!("node", "test_reduce_min_negative_axes_keepdims_example")
      check_onnx_test_case!("node", "test_reduce_min_negative_axes_keepdims_random")
    end

    test "ReduceProd" do
      check_onnx_test_case!("node", "test_reduce_prod_default_axes_keepdims_example")
      check_onnx_test_case!("node", "test_reduce_prod_default_axes_keepdims_random")
      check_onnx_test_case!("node", "test_reduce_prod_do_not_keepdims_example")
      check_onnx_test_case!("node", "test_reduce_prod_do_not_keepdims_random")
      check_onnx_test_case!("node", "test_reduce_prod_keepdims_example")
      check_onnx_test_case!("node", "test_reduce_prod_keepdims_random")
      check_onnx_test_case!("node", "test_reduce_prod_negative_axes_keepdims_example")
      check_onnx_test_case!("node", "test_reduce_prod_negative_axes_keepdims_random")
    end

    test "ReduceSumSquare" do
      check_onnx_test_case!("node", "test_reduce_sum_square_default_axes_keepdims_example")
      check_onnx_test_case!("node", "test_reduce_sum_square_default_axes_keepdims_random")
      check_onnx_test_case!("node", "test_reduce_sum_square_do_not_keepdims_example")
      check_onnx_test_case!("node", "test_reduce_sum_square_do_not_keepdims_random")
      check_onnx_test_case!("node", "test_reduce_sum_square_keepdims_example")
      check_onnx_test_case!("node", "test_reduce_sum_square_keepdims_random")
      check_onnx_test_case!("node", "test_reduce_sum_square_negative_axes_keepdims_example")
      check_onnx_test_case!("node", "test_reduce_sum_square_negative_axes_keepdims_random")
    end

    test "Relu" do
      check_onnx_test_case!("node", "test_relu")
    end

    test "Selu" do
      check_onnx_test_case!("node", "test_selu")
      check_onnx_test_case!("node", "test_selu_default")
      check_onnx_test_case!("node", "test_selu_example")
    end

    test "Sigmoid" do
      check_onnx_test_case!("node", "test_sigmoid")
      check_onnx_test_case!("node", "test_sigmoid_example")
    end

    test "Sign" do
      check_onnx_test_case!("node", "test_sign")
    end

    test "Sin" do
      check_onnx_test_case!("node", "test_sin")
      check_onnx_test_case!("node", "test_sin_example")
    end

    test "Sinh" do
      check_onnx_test_case!("node", "test_sinh")
      check_onnx_test_case!("node", "test_sinh_example")
    end

    test "Softmax" do
      check_onnx_test_case!("node", "test_softmax_axis_0")
      # TODO: ReduceMax
      # check_onnx_test_case!("node", "test_softmax_axis_0_expanded")
      check_onnx_test_case!("node", "test_softmax_axis_1")
      # TODO: ReduceMax
      # check_onnx_test_case!("node", "test_softmax_axis_1_expanded")
      check_onnx_test_case!("node", "test_softmax_axis_2")
      # TODO: ReduceMax
      # check_onnx_test_case!("node", "test_softmax_axis_2_expanded")
      check_onnx_test_case!("node", "test_softmax_default_axis")
      # TODO: ReduceMax
      # check_onnx_test_case!("node", "test_softmax_default_axis_expanded")
      check_onnx_test_case!("node", "test_softmax_large_number")
      # TODO: ReduceMax
      # check_onnx_test_case!("node", "test_softmax_large_number_expanded")
      check_onnx_test_case!("node", "test_softmax_negative_axis")
      # TODO: ReduceMax
      # check_onnx_test_case!("node", "test_softmax_negative_axis_expanded")
      check_onnx_test_case!("node", "test_softmax_example")
      # TODO: ReduceMax
      # check_onnx_test_case!("node", "test_softmax_example_expanded")
    end

    test "Softsign" do
      check_onnx_test_case!("node", "test_softsign")
      check_onnx_test_case!("node", "test_softsign_example")
    end

    test "Sqrt" do
      check_onnx_test_case!("node", "test_sqrt")
      check_onnx_test_case!("node", "test_sqrt_example")
    end

    test "Sum" do
      check_onnx_test_case!("node", "test_sum_example")
      check_onnx_test_case!("node", "test_sum_one_input")
      check_onnx_test_case!("node", "test_sum_two_inputs")
    end

    test "Tan" do
      check_onnx_test_case!("node", "test_tan")
      check_onnx_test_case!("node", "test_tan_example")
    end

    test "Tanh" do
      check_onnx_test_case!("node", "test_tanh")
      check_onnx_test_case!("node", "test_tanh_example")
    end

    test "Transpose" do
      check_onnx_test_case!("node", "test_transpose_default")
      check_onnx_test_case!("node", "test_transpose_all_permutations_0")
      check_onnx_test_case!("node", "test_transpose_all_permutations_1")
      check_onnx_test_case!("node", "test_transpose_all_permutations_2")
      check_onnx_test_case!("node", "test_transpose_all_permutations_3")
      check_onnx_test_case!("node", "test_transpose_all_permutations_4")
      check_onnx_test_case!("node", "test_transpose_all_permutations_5")
    end

    test "Xor" do
      check_onnx_test_case!("node", "test_xor2d")
      check_onnx_test_case!("node", "test_xor3d")
      check_onnx_test_case!("node", "test_xor4d")
      # TODO: Update Axon broadcasting semantics
      # check_onnx_test_case!("node", "test_xor_bcast3v1d")
      check_onnx_test_case!("node", "test_xor_bcast3v2d")
      check_onnx_test_case!("node", "test_xor_bcast4v2d")
      check_onnx_test_case!("node", "test_xor_bcast4v3d")
      check_onnx_test_case!("node", "test_xor_bcast4v4d")
    end
  end

  describe "pytorch converted tests" do
    test "ELU" do
      check_onnx_test_case!("pytorch-converted", "test_ELU")
    end

    test "LeakyReLU" do
      check_onnx_test_case!("pytorch-converted", "test_LeakyReLU")
      check_onnx_test_case!("pytorch-converted", "test_LeakyReLU_with_negval")
    end

    test "LogSoftmax" do
      check_onnx_test_case!("pytorch-converted", "test_log_softmax_lastdim")
      check_onnx_test_case!("pytorch-converted", "test_log_softmax_dim3")
      check_onnx_test_case!("pytorch-converted", "test_LogSoftmax")
    end

    test "ReLU" do
      check_onnx_test_case!("pytorch-converted", "test_ReLU")
    end

    test "SELU" do
      check_onnx_test_case!("pytorch-converted", "test_SELU")
    end

    test "Sigmoid" do
      check_onnx_test_case!("pytorch-converted", "test_Sigmoid")
    end

    test "Softmax" do
      check_onnx_test_case!("pytorch-converted", "test_Softmax")
      check_onnx_test_case!("pytorch-converted", "test_softmax_functional_dim3")
      check_onnx_test_case!("pytorch-converted", "test_softmax_lastdim")
    end

    test "Softmin" do
      check_onnx_test_case!("pytorch-converted", "test_Softmin")
    end

    test "Softplus" do
      check_onnx_test_case!("pytorch-converted", "test_Softplus")
    end

    test "Softsign" do
      check_onnx_test_case!("pytorch-converted", "test_Softsign")
    end

    test "Tanh" do
      check_onnx_test_case!("pytorch-converted", "test_Tanh")
    end
  end

  describe "pytorch operator tests" do
    test "basic" do
      check_onnx_test_case!("pytorch-operator", "test_operator_basic")
    end

    test "conv" do
      Nx.Defn.default_options(compiler: EXLA)
      check_onnx_test_case!("pytorch-operator", "test_operator_conv")
    end

    test "exp" do
      check_onnx_test_case!("pytorch-operator", "test_operator_exp")
    end

    test "flatten" do
      check_onnx_test_case!("pytorch-operator", "test_operator_flatten")
    end

    test "maxpool" do
      check_onnx_test_case!("pytorch-operator", "test_operator_maxpool")
    end

    test "permute" do
      check_onnx_test_case!("pytorch-operator", "test_operator_permute2")
    end

    test "reduce_mean" do
      check_onnx_test_case!("pytorch-operator", "test_operator_reduced_mean")
      check_onnx_test_case!("pytorch-operator", "test_operator_reduced_mean_keepdim")
    end

    test "selu" do
      check_onnx_test_case!("pytorch-operator", "test_operator_selu")
    end
  end

  describe "simple tests" do
    test "sign model" do
      check_onnx_test_case!("simple", "test_sign_model")
    end

    test "single relu model" do
      check_onnx_test_case!("simple", "test_single_relu_model")
    end
  end

  describe "real tests" do
    test "bvlc alexnet" do
      Nx.Defn.default_options(compiler: EXLA)
      check_onnx_model!("bvlc_alexnet")
    end

    test "resnet50" do
      Nx.Defn.default_options(compiler: EXLA)
      check_onnx_model!("resnet50")
    end

    test "shufflenet" do
      Nx.Defn.default_options(compiler: EXLA)
      check_onnx_model!("shufflenet")
    end

    test "vgg19" do
      Nx.Defn.default_options(compiler: EXLA)
      check_onnx_model!("vgg19")
    end
  end
end
