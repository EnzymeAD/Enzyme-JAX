// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=reduce_unused_multislice" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

// Test 1: Only center result used - should become a regular slice
func.func @multi_slice_only_center(%arg0: tensor<20x24x80xf64>) -> tensor<1x8x72xf64> {
    %0, %1, %2, %3, %4, %5 = "enzymexla.multi_slice"(%arg0) <{
        start_indices = array<i64: 1, 0, 3>,
        limit_indices = array<i64: 2, 8, 75>,
        strides = array<i64: 1, 1, 1>,
        dimension = 2 : si32,
        left_amount = 2 : si32,
        right_amount = 3 : si32
    }> : (tensor<20x24x80xf64>) -> (tensor<1x8x72xf64>, tensor<1x8x72xf64>, tensor<1x8x72xf64>, tensor<1x8x72xf64>, tensor<1x8x72xf64>, tensor<1x8x72xf64>)
    return %2 : tensor<1x8x72xf64>
}

// CHECK-LABEL:   func.func @multi_slice_only_center(
// CHECK-SAME:                                       %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: tensor<20x24x80xf64>) -> tensor<1x8x72xf64> {
// CHECK:           %[[VAL_1:.*]] = stablehlo.slice %[[VAL_0]] [1:2, 0:8, 3:75] : (tensor<20x24x80xf64>) -> tensor<1x8x72xf64>
// CHECK:           return %[[VAL_1]] : tensor<1x8x72xf64>
// CHECK:         }


// Test 2: Only left-most result used - should become a regular slice
func.func @multi_slice_only_left(%arg0: tensor<20x24x80xf64>) -> tensor<1x8x72xf64> {
    %0, %1, %2, %3, %4, %5 = "enzymexla.multi_slice"(%arg0) <{
        start_indices = array<i64: 1, 0, 3>,
        limit_indices = array<i64: 2, 8, 75>,
        strides = array<i64: 1, 1, 1>,
        dimension = 2 : si32,
        left_amount = 2 : si32,
        right_amount = 3 : si32
    }> : (tensor<20x24x80xf64>) -> (tensor<1x8x72xf64>, tensor<1x8x72xf64>, tensor<1x8x72xf64>, tensor<1x8x72xf64>, tensor<1x8x72xf64>, tensor<1x8x72xf64>)
    return %0 : tensor<1x8x72xf64>
}

// CHECK-LABEL:   func.func @multi_slice_only_left(
// CHECK-SAME:                                     %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: tensor<20x24x80xf64>) -> tensor<1x8x72xf64> {
// CHECK:           %[[VAL_1:.*]] = stablehlo.slice %[[VAL_0]] [1:2, 0:8, 1:73] : (tensor<20x24x80xf64>) -> tensor<1x8x72xf64>
// CHECK:           return %[[VAL_1]] : tensor<1x8x72xf64>
// CHECK:         }


// Test 3: Only right-most result used - should become a regular slice
func.func @multi_slice_only_right(%arg0: tensor<20x24x80xf64>) -> tensor<1x8x72xf64> {
    %0, %1, %2, %3, %4, %5 = "enzymexla.multi_slice"(%arg0) <{
        start_indices = array<i64: 1, 0, 3>,
        limit_indices = array<i64: 2, 8, 75>,
        strides = array<i64: 1, 1, 1>,
        dimension = 2 : si32,
        left_amount = 2 : si32,
        right_amount = 3 : si32
    }> : (tensor<20x24x80xf64>) -> (tensor<1x8x72xf64>, tensor<1x8x72xf64>, tensor<1x8x72xf64>, tensor<1x8x72xf64>, tensor<1x8x72xf64>, tensor<1x8x72xf64>)
    return %5 : tensor<1x8x72xf64>
}

// CHECK-LABEL:   func.func @multi_slice_only_right(
// CHECK-SAME:                                      %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: tensor<20x24x80xf64>) -> tensor<1x8x72xf64> {
// CHECK:           %[[VAL_1:.*]] = stablehlo.slice %[[VAL_0]] [1:2, 0:8, 6:78] : (tensor<20x24x80xf64>) -> tensor<1x8x72xf64>
// CHECK:           return %[[VAL_1]] : tensor<1x8x72xf64>
// CHECK:         }


// Test 4: Two consecutive results used - should become smaller multi_slice
func.func @multi_slice_consecutive(%arg0: tensor<20x24x80xf64>) -> (tensor<1x8x72xf64>, tensor<1x8x72xf64>) {
    %0, %1, %2, %3, %4, %5 = "enzymexla.multi_slice"(%arg0) <{
        start_indices = array<i64: 1, 0, 3>,
        limit_indices = array<i64: 2, 8, 75>,
        strides = array<i64: 1, 1, 1>,
        dimension = 2 : si32,
        left_amount = 2 : si32,
        right_amount = 3 : si32
    }> : (tensor<20x24x80xf64>) -> (tensor<1x8x72xf64>, tensor<1x8x72xf64>, tensor<1x8x72xf64>, tensor<1x8x72xf64>, tensor<1x8x72xf64>, tensor<1x8x72xf64>)
    return %2, %3 : tensor<1x8x72xf64>, tensor<1x8x72xf64>
}

// CHECK-LABEL:   func.func @multi_slice_consecutive(
// CHECK-SAME:                                       %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: tensor<20x24x80xf64>) -> (tensor<1x8x72xf64>, tensor<1x8x72xf64>) {
// CHECK:           %[[VAL_1:.*]]:2 = "enzymexla.multi_slice"(%[[VAL_0]]) <{dimension = 2 : si32, left_amount = 0 : si32, limit_indices = array<i64: 2, 8, 75>, right_amount = 1 : si32, start_indices = array<i64: 1, 0, 3>, strides = array<i64: 1, 1, 1>}> : (tensor<20x24x80xf64>) -> (tensor<1x8x72xf64>, tensor<1x8x72xf64>)
// CHECK:           return %[[VAL_1]]#0, %[[VAL_1]]#1 : tensor<1x8x72xf64>, tensor<1x8x72xf64>
// CHECK:         }


// Test 5: Non-contiguous results used - should keep range between first and last used
func.func @multi_slice_non_contiguous(%arg0: tensor<20x24x80xf64>) -> (tensor<1x8x72xf64>, tensor<1x8x72xf64>) {
    %0, %1, %2, %3, %4, %5 = "enzymexla.multi_slice"(%arg0) <{
        start_indices = array<i64: 1, 0, 3>,
        limit_indices = array<i64: 2, 8, 75>,
        strides = array<i64: 1, 1, 1>,
        dimension = 2 : si32,
        left_amount = 2 : si32,
        right_amount = 3 : si32
    }> : (tensor<20x24x80xf64>) -> (tensor<1x8x72xf64>, tensor<1x8x72xf64>, tensor<1x8x72xf64>, tensor<1x8x72xf64>, tensor<1x8x72xf64>, tensor<1x8x72xf64>)
    return %2, %5 : tensor<1x8x72xf64>, tensor<1x8x72xf64>
}

// CHECK-LABEL:   func.func @multi_slice_non_contiguous(
// CHECK-SAME:                                          %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: tensor<20x24x80xf64>) -> (tensor<1x8x72xf64>, tensor<1x8x72xf64>) {
// CHECK:           %[[VAL_1:.*]]:4 = "enzymexla.multi_slice"(%[[VAL_0]]) <{dimension = 2 : si32, left_amount = 0 : si32, limit_indices = array<i64: 2, 8, 75>, right_amount = 3 : si32, start_indices = array<i64: 1, 0, 3>, strides = array<i64: 1, 1, 1>}> : (tensor<20x24x80xf64>) -> (tensor<1x8x72xf64>, tensor<1x8x72xf64>, tensor<1x8x72xf64>, tensor<1x8x72xf64>)
// CHECK:           return %[[VAL_1]]#0, %[[VAL_1]]#3 : tensor<1x8x72xf64>, tensor<1x8x72xf64>
// CHECK:         }


// Test 6: All results used - should not change
func.func @multi_slice_all_used(%arg0: tensor<20x24x80xf64>) -> (tensor<1x8x72xf64>, tensor<1x8x72xf64>, tensor<1x8x72xf64>, tensor<1x8x72xf64>, tensor<1x8x72xf64>, tensor<1x8x72xf64>) {
    %0, %1, %2, %3, %4, %5 = "enzymexla.multi_slice"(%arg0) <{
        start_indices = array<i64: 1, 0, 3>,
        limit_indices = array<i64: 2, 8, 75>,
        strides = array<i64: 1, 1, 1>,
        dimension = 2 : si32,
        left_amount = 2 : si32,
        right_amount = 3 : si32
    }> : (tensor<20x24x80xf64>) -> (tensor<1x8x72xf64>, tensor<1x8x72xf64>, tensor<1x8x72xf64>, tensor<1x8x72xf64>, tensor<1x8x72xf64>, tensor<1x8x72xf64>)
    return %0, %1, %2, %3, %4, %5 : tensor<1x8x72xf64>, tensor<1x8x72xf64>, tensor<1x8x72xf64>, tensor<1x8x72xf64>, tensor<1x8x72xf64>, tensor<1x8x72xf64>
}

// CHECK-LABEL:   func.func @multi_slice_all_used(
// CHECK-SAME:                                    %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: tensor<20x24x80xf64>) -> (tensor<1x8x72xf64>, tensor<1x8x72xf64>, tensor<1x8x72xf64>, tensor<1x8x72xf64>, tensor<1x8x72xf64>, tensor<1x8x72xf64>) {
// CHECK:           %[[VAL_1:.*]]:6 = "enzymexla.multi_slice"(%[[VAL_0]]) <{dimension = 2 : si32, left_amount = 2 : si32, limit_indices = array<i64: 2, 8, 75>, right_amount = 3 : si32, start_indices = array<i64: 1, 0, 3>, strides = array<i64: 1, 1, 1>}> : (tensor<20x24x80xf64>) -> (tensor<1x8x72xf64>, tensor<1x8x72xf64>, tensor<1x8x72xf64>, tensor<1x8x72xf64>, tensor<1x8x72xf64>, tensor<1x8x72xf64>)
// CHECK:           return %[[VAL_1]]#0, %[[VAL_1]]#1, %[[VAL_1]]#2, %[[VAL_1]]#3, %[[VAL_1]]#4, %[[VAL_1]]#5 : tensor<1x8x72xf64>, tensor<1x8x72xf64>, tensor<1x8x72xf64>, tensor<1x8x72xf64>, tensor<1x8x72xf64>, tensor<1x8x72xf64>
// CHECK:         }


// Test 7: Different dimension - test on dimension 0
func.func @multi_slice_dim0(%arg0: tensor<20x24x80xf64>) -> tensor<4x24x80xf64> {
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

// CHECK-LABEL:   func.func @multi_slice_dim0(
// CHECK-SAME:                                %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: tensor<20x24x80xf64>) -> tensor<4x24x80xf64> {
// CHECK:           %[[VAL_1:.*]] = stablehlo.slice %[[VAL_0]] [8:12, 0:24, 0:80] : (tensor<20x24x80xf64>) -> tensor<4x24x80xf64>
// CHECK:           return %[[VAL_1]] : tensor<4x24x80xf64>
// CHECK:         }
