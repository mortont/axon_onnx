from onnx import numpy_helper
import onnxruntime as rt
import numpy as np
import shutil
import onnx
import time
import glob
import sys
import os

def _try_load_and_check_model(path_to_onnx):
    """
    Loads and checks the ONNX model, or returns an error code.
    """
    try:
        onnx_model = onnx.load(path_to_onnx)
        onnx.checker.check_model(onnx_model)
    except FileNotFoundError as e:
        print(e)
        return False
    except onnx.checker.ValidationError as e:
        print(e)
        return False
    else:
        return True

def _try_load_and_test_model(path_to_onnx):
    """
    Loads and tests the ONNX model, returning list of test results.
    """
    # Load session
    sess = rt.InferenceSession(path_to_onnx)
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    # Test folders should lie in same directory
    top_level_dir = os.path.dirname(path_to_onnx)
    test_dir_pattern = os.path.join(top_level_dir, "test_data_set_*")
    results = []
    for test_dir in glob.glob(test_dir_pattern):
        input_pattern = os.path.join(test_dir, "input_*.pb")
        output_pattern = os.path.join(test_dir, "output_*.pb")
        # glob ordering is arbitrary, sorting them will ensure correct
        # pairings of inputs and outputs
        inputs = sorted(glob.glob(input_pattern))
        outputs = sorted(glob.glob(output_pattern))
        for inp, expected_out in zip(inputs, outputs):
            # Initialize TensorProto
            inp_tensor = onnx.TensorProto()
            expected_out_tensor = onnx.TensorProto()
            # Parse protobuf
            with open(inp, 'rb') as inp_f, open(expected_out, 'rb') as exp_o_f:
                inp_tensor.ParseFromString(inp_f.read())
                expected_out_tensor.ParseFromString(exp_o_f.read())
            # Retrieve actual value
            actual_out_tensor = sess.run([output_name], {
                input_name: numpy_helper.to_array(inp_tensor)
            })[0]
            # Compare the results and output result
            try:
                np.testing.assert_allclose(
                    actual_out_tensor,
                    numpy_helper.to_array(expected_out_tensor),
                    rtol=5e-4,
                    atol=1e-3
                )
                sys.stdout.write('.')
                # Mutability :(
                results.append(1)
            except AssertionError as e:
                print(f'Input: {numpy_helper.to_array(inp_tensor)}')
                print(e)
                results.append(0)
    # If we've made it this far then everything went well
    return results

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Check ONNX Model')
    parser.add_argument('path_to_onnx', type=str)
    args = parser.parse_args()

    term_size = shutil.get_terminal_size((80, 20)).columns
    title_str = f' {args.path_to_onnx} '
    title_str = title_str.center(term_size, '=')
    print(title_str)
    start = time.time()
    # This must pass in order for the next suite to make
    # sense, so assert here and fail early
    assert _try_load_and_check_model(args.path_to_onnx) == True
    # Now run the actual suite of tests
    results = _try_load_and_test_model(args.path_to_onnx)
    end = time.time()
    total = end - start
    # Print a summary
    print('\n\nFinished in {:.3f} seconds'.format(total))
    print(f'{len(results)} tests, {len(results) - sum(results)} failures')
    print('='*term_size)
    # Exit code is number of failures
    exit(len(results) - sum(results))