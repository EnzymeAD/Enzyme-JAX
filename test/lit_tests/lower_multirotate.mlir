// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=lower_multirotate" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

// Test lowering with no effective rotation
func.func @multirotate_no_amount(%arg0: tensor<10xf32>) -> tensor<10xf32> {
    %0 = "enzymexla.multi_rotate"(%arg0) <{dimension = 0 : i32, left_amount = 0 : i32, right_amount = 0 : i32}> : (tensor<10xf32>) -> tensor<10xf32>
    return %0 : tensor<10xf32>
}

// CHECK-LABEL:   func.func @multirotate_no_amount(
// CHECK-SAME:                                     %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: tensor<10xf32>) -> tensor<10xf32> {
// CHECK-NEXT:           return %[[VAL_0]] : tensor<10xf32>
// CHECK-NEXT:         }


// Test basic lowering of MultiRotateOp with both left and right amounts
func.func @basic_lower_multirotate(%arg0: tensor<10x20xf32>) -> (tensor<10x20xf32>, tensor<10x20xf32>, tensor<10x20xf32>) {
    %0:3 = "enzymexla.multi_rotate"(%arg0) <{dimension = 1 : i32, left_amount = 1 : i32, right_amount = 1 : i32}> : (tensor<10x20xf32>) -> (tensor<10x20xf32>, tensor<10x20xf32>, tensor<10x20xf32>)
    return %0#0, %0#1, %0#2 : tensor<10x20xf32>, tensor<10x20xf32>, tensor<10x20xf32>
}

// CHECK-LABEL:   func.func @basic_lower_multirotate(
// CHECK-SAME:                                       %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: tensor<10x20xf32>) -> (tensor<10x20xf32>, tensor<10x20xf32>, tensor<10x20xf32>) {
// CHECK:           %[[VAL_1:.*]] = "enzymexla.rotate"(%[[VAL_0]]) <{amount = 1 : i32, dimension = 1 : i32}> : (tensor<10x20xf32>) -> tensor<10x20xf32>
// CHECK:           %[[VAL_2:.*]] = "enzymexla.rotate"(%[[VAL_0]]) <{amount = 19 : i32, dimension = 1 : i32}> : (tensor<10x20xf32>) -> tensor<10x20xf32>
// CHECK:           return %[[VAL_1]], %[[VAL_0]], %[[VAL_2]] : tensor<10x20xf32>, tensor<10x20xf32>, tensor<10x20xf32>
// CHECK:         }


// Test lowering with only left amounts (positive rotations)
func.func @left_only_multirotate(%arg0: tensor<8x16xf64>) -> (tensor<8x16xf64>, tensor<8x16xf64>, tensor<8x16xf64>, tensor<8x16xf64>) {
    %0:4 = "enzymexla.multi_rotate"(%arg0) <{dimension = 0 : i32, left_amount = 3 : i32, right_amount = 0 : i32}> : (tensor<8x16xf64>) -> (tensor<8x16xf64>, tensor<8x16xf64>, tensor<8x16xf64>, tensor<8x16xf64>)
    return %0#0, %0#1, %0#2, %0#3 : tensor<8x16xf64>, tensor<8x16xf64>, tensor<8x16xf64>, tensor<8x16xf64>
}

// CHECK-LABEL:   func.func @left_only_multirotate(
// CHECK-SAME:                                     %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: tensor<8x16xf64>) -> (tensor<8x16xf64>, tensor<8x16xf64>, tensor<8x16xf64>, tensor<8x16xf64>) {
// CHECK:           %[[VAL_1:.*]] = "enzymexla.rotate"(%[[VAL_0]]) <{amount = 3 : i32, dimension = 0 : i32}> : (tensor<8x16xf64>) -> tensor<8x16xf64>
// CHECK:           %[[VAL_2:.*]] = "enzymexla.rotate"(%[[VAL_0]]) <{amount = 2 : i32, dimension = 0 : i32}> : (tensor<8x16xf64>) -> tensor<8x16xf64>
// CHECK:           %[[VAL_3:.*]] = "enzymexla.rotate"(%[[VAL_0]]) <{amount = 1 : i32, dimension = 0 : i32}> : (tensor<8x16xf64>) -> tensor<8x16xf64>
// CHECK:           return %[[VAL_1]], %[[VAL_2]], %[[VAL_3]], %[[VAL_0]] : tensor<8x16xf64>, tensor<8x16xf64>, tensor<8x16xf64>, tensor<8x16xf64>
// CHECK:         }


// Test lowering with only right amounts (negative rotations)
func.func @right_only_multirotate(%arg0: tensor<8x16xf64>) -> (tensor<8x16xf64>, tensor<8x16xf64>, tensor<8x16xf64>, tensor<8x16xf64>) {
    %0:4 = "enzymexla.multi_rotate"(%arg0) <{dimension = 1 : i32, left_amount = 0 : i32, right_amount = 3 : i32}> : (tensor<8x16xf64>) -> (tensor<8x16xf64>, tensor<8x16xf64>, tensor<8x16xf64>, tensor<8x16xf64>)
    return %0#0, %0#1, %0#2, %0#3 : tensor<8x16xf64>, tensor<8x16xf64>, tensor<8x16xf64>, tensor<8x16xf64>
}

// CHECK-LABEL:   func.func @right_only_multirotate(
// CHECK-SAME:                                      %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: tensor<8x16xf64>) -> (tensor<8x16xf64>, tensor<8x16xf64>, tensor<8x16xf64>, tensor<8x16xf64>) {
// CHECK:           %[[VAL_1:.*]] = "enzymexla.rotate"(%[[VAL_0]]) <{amount = 15 : i32, dimension = 1 : i32}> : (tensor<8x16xf64>) -> tensor<8x16xf64>
// CHECK:           %[[VAL_2:.*]] = "enzymexla.rotate"(%[[VAL_0]]) <{amount = 14 : i32, dimension = 1 : i32}> : (tensor<8x16xf64>) -> tensor<8x16xf64>
// CHECK:           %[[VAL_3:.*]] = "enzymexla.rotate"(%[[VAL_0]]) <{amount = 13 : i32, dimension = 1 : i32}> : (tensor<8x16xf64>) -> tensor<8x16xf64>
// CHECK:           return %[[VAL_0]], %[[VAL_1]], %[[VAL_2]], %[[VAL_3]] : tensor<8x16xf64>, tensor<8x16xf64>, tensor<8x16xf64>, tensor<8x16xf64>
// CHECK:         }


// Test lowering with larger range
func.func @large_range_multirotate(%arg0: tensor<20x24x80xf64>) -> (tensor<20x24x80xf64>, tensor<20x24x80xf64>, tensor<20x24x80xf64>, tensor<20x24x80xf64>) {
    %0:4 = "enzymexla.multi_rotate"(%arg0) <{dimension = 2 : i32, left_amount = 2 : i32, right_amount = 1 : i32}> : (tensor<20x24x80xf64>) -> (tensor<20x24x80xf64>, tensor<20x24x80xf64>, tensor<20x24x80xf64>, tensor<20x24x80xf64>)
    return %0#0, %0#1, %0#2, %0#3 : tensor<20x24x80xf64>, tensor<20x24x80xf64>, tensor<20x24x80xf64>, tensor<20x24x80xf64>
}

// CHECK-LABEL:   func.func @large_range_multirotate(
// CHECK-SAME:                                       %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: tensor<20x24x80xf64>) -> (tensor<20x24x80xf64>, tensor<20x24x80xf64>, tensor<20x24x80xf64>, tensor<20x24x80xf64>) {
// CHECK:           %[[VAL_1:.*]] = "enzymexla.rotate"(%[[VAL_0]]) <{amount = 2 : i32, dimension = 2 : i32}> : (tensor<20x24x80xf64>) -> tensor<20x24x80xf64>
// CHECK:           %[[VAL_2:.*]] = "enzymexla.rotate"(%[[VAL_0]]) <{amount = 1 : i32, dimension = 2 : i32}> : (tensor<20x24x80xf64>) -> tensor<20x24x80xf64>
// CHECK:           %[[VAL_3:.*]] = "enzymexla.rotate"(%[[VAL_0]]) <{amount = 79 : i32, dimension = 2 : i32}> : (tensor<20x24x80xf64>) -> tensor<20x24x80xf64>
// CHECK:           return %[[VAL_1]], %[[VAL_2]], %[[VAL_0]], %[[VAL_3]] : tensor<20x24x80xf64>, tensor<20x24x80xf64>, tensor<20x24x80xf64>, tensor<20x24x80xf64>
// CHECK:         }


// Test lowering with negative left_amount
func.func @multirotate_left_neg_equal(%arg0: tensor<10xf32>) -> tensor<10xf32> {
    %0 = "enzymexla.multi_rotate"(%arg0) <{dimension = 0 : i32, left_amount = -1 : i32, right_amount = 1 : i32}> : (tensor<10xf32>) -> tensor<10xf32>
    return %0 : tensor<10xf32>
}

// CHECK-LABEL:   func.func @multirotate_left_neg_equal(
// CHECK-SAME:                                         %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: tensor<10xf32>) -> tensor<10xf32> {
// CHECK-NEXT:           %[[VAL_1:.*]] = "enzymexla.rotate"(%[[VAL_0]]) <{amount = 9 : i32, dimension = 0 : i32}> : (tensor<10xf32>) -> tensor<10xf32>
// CHECK-NEXT:           return %[[VAL_1]] : tensor<10xf32>
// CHECK-NEXT:         }


// Test lowering with negative right_amout
func.func @multirotate_right_neg_equal(%arg0: tensor<10xf32>) -> tensor<10xf32> {
    %0 = "enzymexla.multi_rotate"(%arg0) <{dimension = 0 : i32, left_amount = 1 : i32, right_amount = -1 : i32}> : (tensor<10xf32>) -> tensor<10xf32>
    return %0 : tensor<10xf32>
}

// CHECK-LABEL:   func.func @multirotate_right_neg_equal(
// CHECK-SAME:                                         %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: tensor<10xf32>) -> tensor<10xf32> {
// CHECK-NEXT:           %[[VAL_1:.*]] = "enzymexla.rotate"(%[[VAL_0]]) <{amount = 1 : i32, dimension = 0 : i32}> : (tensor<10xf32>) -> tensor<10xf32>
// CHECK-NEXT:           return %[[VAL_1]] : tensor<10xf32>
// CHECK-NEXT:         }
