// RUN: enzymexlamlir-opt --split-input-file --verify-diagnostics %s | FileCheck %s

// Valid multi_slice operation (no error expected)
func.func @multi_slice_valid(%arg0: tensor<20x24x80xf64>) -> tensor<4x24x80xf64> {
    %0, %1, %2, %3, %4 = "enzymexla.multi_slice"(%arg0) <{
        start_indices = array<i64: 8, 0, 0>,
        limit_indices = array<i64: 12, 24, 80>,
        strides = array<i64: 1, 1, 1>,
        dimension = 0 : si32,
        left_amount = 2 : si32,
        right_amount = 2 : si32
    }> : (tensor<20x24x80xf64>) -> (tensor<4x24x80xf64>, tensor<4x24x80xf64>, tensor<4x24x80xf64>, tensor<4x24x80xf64>, tensor<4x24x80xf64>)
    return %2 : tensor<4x24x80xf64>
}

// -----

func.func @multi_slice_negative_left_amount(%arg0: tensor<20x24x80xf64>) -> tensor<4x24x80xf64> {
    // expected-error @+1 {{left_amount must be non-negative, got -1}}
    %0 = "enzymexla.multi_slice"(%arg0) <{
        start_indices = array<i64: 8, 0, 0>,
        limit_indices = array<i64: 12, 24, 80>,
        strides = array<i64: 1, 1, 1>,
        dimension = 0 : si32,
        left_amount = -1 : si32,
        right_amount = 1 : si32
    }> : (tensor<20x24x80xf64>) -> tensor<4x24x80xf64>
    return %0 : tensor<4x24x80xf64>
}

// -----

func.func @multi_slice_negative_right_amount(%arg0: tensor<20x24x80xf64>) -> tensor<4x24x80xf64> {
    // expected-error @+1 {{right_amount must be non-negative, got -3}}
    %0, %1 = "enzymexla.multi_slice"(%arg0) <{
        start_indices = array<i64: 8, 0, 0>,
        limit_indices = array<i64: 12, 24, 80>,
        strides = array<i64: 1, 1, 1>,
        dimension = 0 : si32,
        left_amount = 1 : si32,
        right_amount = -3 : si32
    }> : (tensor<20x24x80xf64>) -> (tensor<4x24x80xf64>, tensor<4x24x80xf64>)
    return %0 : tensor<4x24x80xf64>
}

// -----

func.func @multi_slice_invalid_dimension_negative(%arg0: tensor<20x24x80xf64>) -> tensor<4x24x80xf64> {
    // expected-error @+1 {{dimension -1 is out of range for tensor of rank 3}}
    %0, %1, %2 = "enzymexla.multi_slice"(%arg0) <{
        start_indices = array<i64: 8, 0, 0>,
        limit_indices = array<i64: 12, 24, 80>,
        strides = array<i64: 1, 1, 1>,
        dimension = -1 : si32,
        left_amount = 1 : si32,
        right_amount = 1 : si32
    }> : (tensor<20x24x80xf64>) -> (tensor<4x24x80xf64>, tensor<4x24x80xf64>, tensor<4x24x80xf64>)
    return %1 : tensor<4x24x80xf64>
}

// -----

func.func @multi_slice_invalid_dimension_too_large(%arg0: tensor<20x24x80xf64>) -> tensor<4x24x80xf64> {
    // expected-error @+1 {{dimension 3 is out of range for tensor of rank 3}}
    %0, %1, %2 = "enzymexla.multi_slice"(%arg0) <{
        start_indices = array<i64: 8, 0, 0>,
        limit_indices = array<i64: 12, 24, 80>,
        strides = array<i64: 1, 1, 1>,
        dimension = 3 : si32,
        left_amount = 1 : si32,
        right_amount = 1 : si32
    }> : (tensor<20x24x80xf64>) -> (tensor<4x24x80xf64>, tensor<4x24x80xf64>, tensor<4x24x80xf64>)
    return %1 : tensor<4x24x80xf64>
}

// -----

func.func @multi_slice_start_indices_size_mismatch(%arg0: tensor<20x24x80xf64>) -> tensor<4x24x80xf64> {
    // expected-error @+1 {{start_indices size 2 does not match tensor rank 3}}
    %0, %1, %2 = "enzymexla.multi_slice"(%arg0) <{
        start_indices = array<i64: 8, 0>,
        limit_indices = array<i64: 12, 24, 80>,
        strides = array<i64: 1, 1, 1>,
        dimension = 0 : si32,
        left_amount = 1 : si32,
        right_amount = 1 : si32
    }> : (tensor<20x24x80xf64>) -> (tensor<4x24x80xf64>, tensor<4x24x80xf64>, tensor<4x24x80xf64>)
    return %1 : tensor<4x24x80xf64>
}

// -----

func.func @multi_slice_limit_indices_size_mismatch(%arg0: tensor<20x24x80xf64>) -> tensor<4x24x80xf64> {
    // expected-error @+1 {{limit_indices size 4 does not match tensor rank 3}}
    %0, %1, %2 = "enzymexla.multi_slice"(%arg0) <{
        start_indices = array<i64: 8, 0, 0>,
        limit_indices = array<i64: 12, 24, 80, 5>,
        strides = array<i64: 1, 1, 1>,
        dimension = 0 : si32,
        left_amount = 1 : si32,
        right_amount = 1 : si32
    }> : (tensor<20x24x80xf64>) -> (tensor<4x24x80xf64>, tensor<4x24x80xf64>, tensor<4x24x80xf64>)
    return %1 : tensor<4x24x80xf64>
}

// -----

func.func @multi_slice_strides_size_mismatch(%arg0: tensor<20x24x80xf64>) -> tensor<4x24x80xf64> {
    // expected-error @+1 {{strides size 1 does not match tensor rank 3}}
    %0, %1, %2 = "enzymexla.multi_slice"(%arg0) <{
        start_indices = array<i64: 8, 0, 0>,
        limit_indices = array<i64: 12, 24, 80>,
        strides = array<i64: 1>,
        dimension = 0 : si32,
        left_amount = 1 : si32,
        right_amount = 1 : si32
    }> : (tensor<20x24x80xf64>) -> (tensor<4x24x80xf64>, tensor<4x24x80xf64>, tensor<4x24x80xf64>)
    return %1 : tensor<4x24x80xf64>
}

// -----

func.func @multi_slice_non_positive_stride(%arg0: tensor<20x24x80xf64>) -> tensor<4x24x80xf64> {
    // expected-error @+1 {{strides must be positive, got 0 at index 1}}
    %0, %1, %2 = "enzymexla.multi_slice"(%arg0) <{
        start_indices = array<i64: 8, 0, 0>,
        limit_indices = array<i64: 12, 24, 80>,
        strides = array<i64: 1, 0, 1>,
        dimension = 0 : si32,
        left_amount = 1 : si32,
        right_amount = 1 : si32
    }> : (tensor<20x24x80xf64>) -> (tensor<4x24x80xf64>, tensor<4x24x80xf64>, tensor<4x24x80xf64>)
    return %1 : tensor<4x24x80xf64>
}

// -----

func.func @multi_slice_negative_stride(%arg0: tensor<20x24x80xf64>) -> tensor<4x24x80xf64> {
    // expected-error @+1 {{strides must be positive, got -1 at index 2}}
    %0, %1, %2 = "enzymexla.multi_slice"(%arg0) <{
        start_indices = array<i64: 8, 0, 0>,
        limit_indices = array<i64: 12, 24, 80>,
        strides = array<i64: 1, 1, -1>,
        dimension = 0 : si32,
        left_amount = 1 : si32,
        right_amount = 1 : si32
    }> : (tensor<20x24x80xf64>) -> (tensor<4x24x80xf64>, tensor<4x24x80xf64>, tensor<4x24x80xf64>)
    return %1 : tensor<4x24x80xf64>
}

// -----

func.func @multi_slice_wrong_num_results(%arg0: tensor<20x24x80xf64>) -> tensor<4x24x80xf64> {
    // expected-error @+1 {{expected 5 results (left_amount + right_amount + 1), got 3}}
    %0, %1, %2 = "enzymexla.multi_slice"(%arg0) <{
        start_indices = array<i64: 8, 0, 0>,
        limit_indices = array<i64: 12, 24, 80>,
        strides = array<i64: 1, 1, 1>,
        dimension = 0 : si32,
        left_amount = 2 : si32,
        right_amount = 2 : si32
    }> : (tensor<20x24x80xf64>) -> (tensor<4x24x80xf64>, tensor<4x24x80xf64>, tensor<4x24x80xf64>)
    return %1 : tensor<4x24x80xf64>
}

// -----

func.func @multi_slice_result_shape_mismatch(%arg0: tensor<20x24x80xf64>) -> tensor<8x24x80xf64> {
    // expected-error @+1 {{result #1 has type 'tensor<8x24x80xf64>' but expected 'tensor<4x24x80xf64>' based on slice parameters}}
    %0, %1, %2 = "enzymexla.multi_slice"(%arg0) <{
        start_indices = array<i64: 8, 0, 0>,
        limit_indices = array<i64: 12, 24, 80>,
        strides = array<i64: 1, 1, 1>,
        dimension = 0 : si32,
        left_amount = 1 : si32,
        right_amount = 1 : si32
    }> : (tensor<20x24x80xf64>) -> (tensor<4x24x80xf64>, tensor<8x24x80xf64>, tensor<4x24x80xf64>)
    return %1 : tensor<8x24x80xf64>
}

// -----

func.func @multi_slice_result_element_type_mismatch(%arg0: tensor<20x24x80xf64>) -> tensor<4x24x80xf32> {
    // expected-error @+1 {{result #2 has type 'tensor<4x24x80xf32>' but expected 'tensor<4x24x80xf64>' based on slice parameters}}
    %0, %1, %2 = "enzymexla.multi_slice"(%arg0) <{
        start_indices = array<i64: 8, 0, 0>,
        limit_indices = array<i64: 12, 24, 80>,
        strides = array<i64: 1, 1, 1>,
        dimension = 0 : si32,
        left_amount = 1 : si32,
        right_amount = 1 : si32
    }> : (tensor<20x24x80xf64>) -> (tensor<4x24x80xf64>, tensor<4x24x80xf64>, tensor<4x24x80xf32>)
    return %2 : tensor<4x24x80xf32>
}

// -----

// Valid multi_slice with non-unit strides
func.func @multi_slice_with_strides(%arg0: tensor<20x24x80xf64>) -> tensor<2x12x40xf64> {
    %0, %1, %2 = "enzymexla.multi_slice"(%arg0) <{
        start_indices = array<i64: 8, 0, 0>,
        limit_indices = array<i64: 12, 24, 80>,
        strides = array<i64: 2, 2, 2>,
        dimension = 0 : si32,
        left_amount = 1 : si32,
        right_amount = 1 : si32
    }> : (tensor<20x24x80xf64>) -> (tensor<2x12x40xf64>, tensor<2x12x40xf64>, tensor<2x12x40xf64>)
    return %1 : tensor<2x12x40xf64>
}
