// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=lower_multislice" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

// Test basic lowering of MultiSliceOp with both left and right amounts
func.func @basic_lower_multislice(%arg0: tensor<10x20xf32>) -> (tensor<10x5xf32>, tensor<10x5xf32>, tensor<10x5xf32>) {
    %0:3 = "enzymexla.multi_slice"(%arg0) <{dimension = 1 : si32, left_amount = 1 : si32, limit_indices = array<i64: 10, 10>, right_amount = 1 : si32, start_indices = array<i64: 0, 5>, strides = array<i64: 1, 1>}> : (tensor<10x20xf32>) -> (tensor<10x5xf32>, tensor<10x5xf32>, tensor<10x5xf32>)
    return %0#0, %0#1, %0#2 : tensor<10x5xf32>, tensor<10x5xf32>, tensor<10x5xf32>
}

// CHECK-LABEL:   func.func @basic_lower_multislice(
// CHECK-SAME:                                      %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: tensor<10x20xf32>) -> (tensor<10x5xf32>, tensor<10x5xf32>, tensor<10x5xf32>) {
// CHECK:           %[[VAL_1:.*]] = stablehlo.slice %[[VAL_0]] [0:10, 4:9] : (tensor<10x20xf32>) -> tensor<10x5xf32>
// CHECK:           %[[VAL_2:.*]] = stablehlo.slice %[[VAL_0]] [0:10, 5:10] : (tensor<10x20xf32>) -> tensor<10x5xf32>
// CHECK:           %[[VAL_3:.*]] = stablehlo.slice %[[VAL_0]] [0:10, 6:11] : (tensor<10x20xf32>) -> tensor<10x5xf32>
// CHECK:           return %[[VAL_1]], %[[VAL_2]], %[[VAL_3]] : tensor<10x5xf32>, tensor<10x5xf32>, tensor<10x5xf32>
// CHECK:         }


// Test lowering with only left amounts
func.func @left_only_multislice(%arg0: tensor<8x16xf64>) -> (tensor<8x4xf64>, tensor<8x4xf64>, tensor<8x4xf64>) {
    %0:3 = "enzymexla.multi_slice"(%arg0) <{dimension = 1 : si32, left_amount = 2 : si32, limit_indices = array<i64: 8, 10>, right_amount = 0 : si32, start_indices = array<i64: 0, 6>, strides = array<i64: 1, 1>}> : (tensor<8x16xf64>) -> (tensor<8x4xf64>, tensor<8x4xf64>, tensor<8x4xf64>)
    return %0#0, %0#1, %0#2 : tensor<8x4xf64>, tensor<8x4xf64>, tensor<8x4xf64>
}

// CHECK-LABEL:   func.func @left_only_multislice(
// CHECK-SAME:                                    %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: tensor<8x16xf64>) -> (tensor<8x4xf64>, tensor<8x4xf64>, tensor<8x4xf64>) {
// CHECK:           %[[VAL_1:.*]] = stablehlo.slice %[[VAL_0]] [0:8, 4:8] : (tensor<8x16xf64>) -> tensor<8x4xf64>
// CHECK:           %[[VAL_2:.*]] = stablehlo.slice %[[VAL_0]] [0:8, 5:9] : (tensor<8x16xf64>) -> tensor<8x4xf64>
// CHECK:           %[[VAL_3:.*]] = stablehlo.slice %[[VAL_0]] [0:8, 6:10] : (tensor<8x16xf64>) -> tensor<8x4xf64>
// CHECK:           return %[[VAL_1]], %[[VAL_2]], %[[VAL_3]] : tensor<8x4xf64>, tensor<8x4xf64>, tensor<8x4xf64>
// CHECK:         }


// Test lowering with only right amounts
func.func @right_only_multislice(%arg0: tensor<8x16xf64>) -> (tensor<8x4xf64>, tensor<8x4xf64>, tensor<8x4xf64>) {
    %0:3 = "enzymexla.multi_slice"(%arg0) <{dimension = 1 : si32, left_amount = 0 : si32, limit_indices = array<i64: 8, 10>, right_amount = 2 : si32, start_indices = array<i64: 0, 6>, strides = array<i64: 1, 1>}> : (tensor<8x16xf64>) -> (tensor<8x4xf64>, tensor<8x4xf64>, tensor<8x4xf64>)
    return %0#0, %0#1, %0#2 : tensor<8x4xf64>, tensor<8x4xf64>, tensor<8x4xf64>
}

// CHECK-LABEL:   func.func @right_only_multislice(
// CHECK-SAME:                                     %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: tensor<8x16xf64>) -> (tensor<8x4xf64>, tensor<8x4xf64>, tensor<8x4xf64>) {
// CHECK:           %[[VAL_1:.*]] = stablehlo.slice %[[VAL_0]] [0:8, 6:10] : (tensor<8x16xf64>) -> tensor<8x4xf64>
// CHECK:           %[[VAL_2:.*]] = stablehlo.slice %[[VAL_0]] [0:8, 7:11] : (tensor<8x16xf64>) -> tensor<8x4xf64>
// CHECK:           %[[VAL_3:.*]] = stablehlo.slice %[[VAL_0]] [0:8, 8:12] : (tensor<8x16xf64>) -> tensor<8x4xf64>
// CHECK:           return %[[VAL_1]], %[[VAL_2]], %[[VAL_3]] : tensor<8x4xf64>, tensor<8x4xf64>, tensor<8x4xf64>
// CHECK:         }


// Test lowering on different dimension
func.func @different_dim_multislice(%arg0: tensor<20x24x80xf64>) -> (tensor<5x24x80xf64>, tensor<5x24x80xf64>, tensor<5x24x80xf64>, tensor<5x24x80xf64>) {
    %0:4 = "enzymexla.multi_slice"(%arg0) <{dimension = 0 : si32, left_amount = 2 : si32, limit_indices = array<i64: 15, 24, 80>, right_amount = 1 : si32, start_indices = array<i64: 10, 0, 0>, strides = array<i64: 1, 1, 1>}> : (tensor<20x24x80xf64>) -> (tensor<5x24x80xf64>, tensor<5x24x80xf64>, tensor<5x24x80xf64>, tensor<5x24x80xf64>)
    return %0#0, %0#1, %0#2, %0#3 : tensor<5x24x80xf64>, tensor<5x24x80xf64>, tensor<5x24x80xf64>, tensor<5x24x80xf64>
}

// CHECK-LABEL:   func.func @different_dim_multislice(
// CHECK-SAME:                                        %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: tensor<20x24x80xf64>) -> (tensor<5x24x80xf64>, tensor<5x24x80xf64>, tensor<5x24x80xf64>, tensor<5x24x80xf64>) {
// CHECK:           %[[VAL_1:.*]] = stablehlo.slice %[[VAL_0]] [8:13, 0:24, 0:80] : (tensor<20x24x80xf64>) -> tensor<5x24x80xf64>
// CHECK:           %[[VAL_2:.*]] = stablehlo.slice %[[VAL_0]] [9:14, 0:24, 0:80] : (tensor<20x24x80xf64>) -> tensor<5x24x80xf64>
// CHECK:           %[[VAL_3:.*]] = stablehlo.slice %[[VAL_0]] [10:15, 0:24, 0:80] : (tensor<20x24x80xf64>) -> tensor<5x24x80xf64>
// CHECK:           %[[VAL_4:.*]] = stablehlo.slice %[[VAL_0]] [11:16, 0:24, 0:80] : (tensor<20x24x80xf64>) -> tensor<5x24x80xf64>
// CHECK:           return %[[VAL_1]], %[[VAL_2]], %[[VAL_3]], %[[VAL_4]] : tensor<5x24x80xf64>, tensor<5x24x80xf64>, tensor<5x24x80xf64>, tensor<5x24x80xf64>
// CHECK:         }


// Test lowering with non-unit strides
func.func @strided_multislice(%arg0: tensor<10x20xf32>) -> (tensor<10x3xf32>, tensor<10x3xf32>) {
    %0:2 = "enzymexla.multi_slice"(%arg0) <{dimension = 1 : si32, left_amount = 1 : si32, limit_indices = array<i64: 10, 12>, right_amount = 0 : si32, start_indices = array<i64: 0, 6>, strides = array<i64: 1, 2>}> : (tensor<10x20xf32>) -> (tensor<10x3xf32>, tensor<10x3xf32>)
    return %0#0, %0#1 : tensor<10x3xf32>, tensor<10x3xf32>
}

// CHECK-LABEL:   func.func @strided_multislice(
// CHECK-SAME:                                  %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: tensor<10x20xf32>) -> (tensor<10x3xf32>, tensor<10x3xf32>) {
// CHECK:           %[[VAL_1:.*]] = stablehlo.slice %[[VAL_0]] [0:10, 5:11:2] : (tensor<10x20xf32>) -> tensor<10x3xf32>
// CHECK:           %[[VAL_2:.*]] = stablehlo.slice %[[VAL_0]] [0:10, 6:12:2] : (tensor<10x20xf32>) -> tensor<10x3xf32>
// CHECK:           return %[[VAL_1]], %[[VAL_2]] : tensor<10x3xf32>, tensor<10x3xf32>
// CHECK:         }
