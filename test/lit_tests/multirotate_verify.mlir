// RUN: enzymexlamlir-opt --split-input-file --verify-diagnostics %s

// Valid multi_rotate operation (no error expected)
func.func @multi_rotate_valid(%arg0: tensor<20x24x80xf64>) -> tensor<20x24x80xf64> {
    %0, %1, %2, %3, %4 = "enzymexla.multi_rotate"(%arg0) <{
        dimension = 0 : i32,
        left_amount = 2 : i32,
        right_amount = 2 : i32
    }> : (tensor<20x24x80xf64>) -> (tensor<20x24x80xf64>, tensor<20x24x80xf64>, tensor<20x24x80xf64>, tensor<20x24x80xf64>, tensor<20x24x80xf64>)
    return %2 : tensor<20x24x80xf64>
}

// -----

func.func @multi_rotate_negative_left_amount_excess(%arg0: tensor<20x24x80xf64>) -> tensor<20x24x80xf64> {
    // expected-error @+1 {{if left_amount is negative, its absolute value must be less than or equal to right_amount, got -2 (left) and 1 (right)}}
    %0 = "enzymexla.multi_rotate"(%arg0) <{
        dimension = 0 : i32,
        left_amount = -2 : i32,
        right_amount = 1 : i32
    }> : (tensor<20x24x80xf64>) -> tensor<20x24x80xf64>
    return %0 : tensor<20x24x80xf64>
}

// -----

func.func @multi_rotate_negative_right_amount(%arg0: tensor<20x24x80xf64>) -> tensor<20x24x80xf64> {
    // expected-error @+1 {{if right_amount is negative, its absolute value must be less than or equal to left_amount, got 1 (left) and -2 (right)}}
    %0, %1 = "enzymexla.multi_rotate"(%arg0) <{
        dimension = 0 : i32,
        left_amount = 1 : i32,
        right_amount = -2 : i32
    }> : (tensor<20x24x80xf64>) -> (tensor<20x24x80xf64>, tensor<20x24x80xf64>)
    return %0 : tensor<20x24x80xf64>
}

// -----

func.func @multi_rotate_invalid_dimension_negative(%arg0: tensor<20x24x80xf64>) -> tensor<20x24x80xf64> {
    // expected-error @+1 {{dimension -1 is out of range for tensor of rank 3}}
    %0, %1, %2 = "enzymexla.multi_rotate"(%arg0) <{
        dimension = -1 : i32,
        left_amount = 1 : i32,
        right_amount = 1 : i32
    }> : (tensor<20x24x80xf64>) -> (tensor<20x24x80xf64>, tensor<20x24x80xf64>, tensor<20x24x80xf64>)
    return %1 : tensor<20x24x80xf64>
}

// -----

func.func @multi_rotate_invalid_dimension_too_large(%arg0: tensor<20x24x80xf64>) -> tensor<20x24x80xf64> {
    // expected-error @+1 {{dimension 5 is out of range for tensor of rank 3}}
    %0, %1, %2 = "enzymexla.multi_rotate"(%arg0) <{
        dimension = 5 : i32,
        left_amount = 1 : i32,
        right_amount = 1 : i32
    }> : (tensor<20x24x80xf64>) -> (tensor<20x24x80xf64>, tensor<20x24x80xf64>, tensor<20x24x80xf64>)
    return %1 : tensor<20x24x80xf64>
}

// -----

func.func @multi_rotate_wrong_num_results(%arg0: tensor<20x24x80xf64>) -> tensor<20x24x80xf64> {
    // expected-error @+1 {{expected 5 results (left_amount + right_amount + 1), got 3}}
    %0, %1, %2 = "enzymexla.multi_rotate"(%arg0) <{
        dimension = 0 : i32,
        left_amount = 2 : i32,
        right_amount = 2 : i32
    }> : (tensor<20x24x80xf64>) -> (tensor<20x24x80xf64>, tensor<20x24x80xf64>, tensor<20x24x80xf64>)
    return %1 : tensor<20x24x80xf64>
}

// -----

func.func @multi_rotate_result_type_mismatch(%arg0: tensor<20x24x80xf64>) -> tensor<10x24x80xf64> {
    // expected-error @+1 {{all results must have the same type as the operand}}
    %0, %1, %2 = "enzymexla.multi_rotate"(%arg0) <{
        dimension = 0 : i32,
        left_amount = 1 : i32,
        right_amount = 1 : i32
    }> : (tensor<20x24x80xf64>) -> (tensor<20x24x80xf64>, tensor<10x24x80xf64>, tensor<20x24x80xf64>)
    return %1 : tensor<10x24x80xf64>
}

// -----

func.func @multi_rotate_result_element_type_mismatch(%arg0: tensor<20x24x80xf64>) -> tensor<20x24x80xf32> {
    // expected-error @+1 {{all results must have the same type as the operand}}
    %0, %1, %2 = "enzymexla.multi_rotate"(%arg0) <{
        dimension = 0 : i32,
        left_amount = 1 : i32,
        right_amount = 1 : i32
    }> : (tensor<20x24x80xf64>) -> (tensor<20x24x80xf64>, tensor<20x24x80xf64>, tensor<20x24x80xf32>)
    return %2 : tensor<20x24x80xf32>
}
