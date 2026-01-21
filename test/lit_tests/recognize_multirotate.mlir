// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=recognize_multirotate" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

// Test basic combination of multiple rotates on the same input and dimension
func.func @basic_multirotate(%arg0: tensor<20x24x80xf64>) -> (tensor<20x24x80xf64>, tensor<20x24x80xf64>, tensor<20x24x80xf64>) {
    %0 = "enzymexla.rotate"(%arg0) <{amount = 2 : si32, dimension = 2 : si32}> : (tensor<20x24x80xf64>) -> tensor<20x24x80xf64>
    %1 = "enzymexla.rotate"(%arg0) <{amount = 1 : si32, dimension = 2 : si32}> : (tensor<20x24x80xf64>) -> tensor<20x24x80xf64>
    %2 = "enzymexla.rotate"(%arg0) <{amount = -1 : si32, dimension = 2 : si32}> : (tensor<20x24x80xf64>) -> tensor<20x24x80xf64>
    return %0, %1, %2 : tensor<20x24x80xf64>, tensor<20x24x80xf64>, tensor<20x24x80xf64>
}

// CHECK-LABEL:   func.func @basic_multirotate(
// CHECK-SAME:                                 %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: tensor<20x24x80xf64>) -> (tensor<20x24x80xf64>, tensor<20x24x80xf64>, tensor<20x24x80xf64>) {
// CHECK:           %[[VAL_1:.*]]:4 = "enzymexla.multi_rotate"(%[[VAL_0]]) <{dimension = 2 : si32, left_amount = 2 : si32, right_amount = 1 : si32}> : (tensor<20x24x80xf64>) -> (tensor<20x24x80xf64>, tensor<20x24x80xf64>, tensor<20x24x80xf64>, tensor<20x24x80xf64>)
// CHECK:           return %[[VAL_1]]#0, %[[VAL_1]]#1, %[[VAL_1]]#3 : tensor<20x24x80xf64>, tensor<20x24x80xf64>, tensor<20x24x80xf64>
// CHECK:         }


// Test with consecutive rotations (dense case)
func.func @dense_multirotate(%arg0: tensor<10x20xf32>) -> (tensor<10x20xf32>, tensor<10x20xf32>, tensor<10x20xf32>) {
    %0 = "enzymexla.rotate"(%arg0) <{amount = 1 : si32, dimension = 1 : si32}> : (tensor<10x20xf32>) -> tensor<10x20xf32>
    %1 = "enzymexla.rotate"(%arg0) <{amount = 0 : si32, dimension = 1 : si32}> : (tensor<10x20xf32>) -> tensor<10x20xf32>
    %2 = "enzymexla.rotate"(%arg0) <{amount = -1 : si32, dimension = 1 : si32}> : (tensor<10x20xf32>) -> tensor<10x20xf32>
    return %0, %1, %2 : tensor<10x20xf32>, tensor<10x20xf32>, tensor<10x20xf32>
}

// CHECK-LABEL:   func.func @dense_multirotate(
// CHECK-SAME:                                 %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: tensor<10x20xf32>) -> (tensor<10x20xf32>, tensor<10x20xf32>, tensor<10x20xf32>) {
// CHECK:           %[[VAL_1:.*]]:3 = "enzymexla.multi_rotate"(%[[VAL_0]]) <{dimension = 1 : si32, left_amount = 1 : si32, right_amount = 1 : si32}> : (tensor<10x20xf32>) -> (tensor<10x20xf32>, tensor<10x20xf32>, tensor<10x20xf32>)
// CHECK:           return %[[VAL_1]]#0, %[[VAL_1]]#1, %[[VAL_1]]#2 : tensor<10x20xf32>, tensor<10x20xf32>, tensor<10x20xf32>
// CHECK:         }


// Test different dimensions - should NOT combine
func.func @different_dimensions(%arg0: tensor<20x24x80xf64>) -> (tensor<20x24x80xf64>, tensor<20x24x80xf64>) {
    %0 = "enzymexla.rotate"(%arg0) <{amount = 1 : si32, dimension = 1 : si32}> : (tensor<20x24x80xf64>) -> tensor<20x24x80xf64>
    %1 = "enzymexla.rotate"(%arg0) <{amount = 1 : si32, dimension = 2 : si32}> : (tensor<20x24x80xf64>) -> tensor<20x24x80xf64>
    return %0, %1 : tensor<20x24x80xf64>, tensor<20x24x80xf64>
}

// CHECK-LABEL:   func.func @different_dimensions(
// CHECK-SAME:                                    %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: tensor<20x24x80xf64>) -> (tensor<20x24x80xf64>, tensor<20x24x80xf64>) {
// CHECK:           %[[VAL_1:.*]] = "enzymexla.rotate"(%[[VAL_0]]) <{amount = 1 : si32, dimension = 1 : si32}> : (tensor<20x24x80xf64>) -> tensor<20x24x80xf64>
// CHECK:           %[[VAL_2:.*]] = "enzymexla.rotate"(%[[VAL_0]]) <{amount = 1 : si32, dimension = 2 : si32}> : (tensor<20x24x80xf64>) -> tensor<20x24x80xf64>
// CHECK:           return %[[VAL_1]], %[[VAL_2]] : tensor<20x24x80xf64>, tensor<20x24x80xf64>
// CHECK:         }


// Test single rotation - should NOT combine (need at least 2)
func.func @single_rotate(%arg0: tensor<10x20xf32>) -> tensor<10x20xf32> {
    %0 = "enzymexla.rotate"(%arg0) <{amount = 3 : si32, dimension = 0 : si32}> : (tensor<10x20xf32>) -> tensor<10x20xf32>
    return %0 : tensor<10x20xf32>
}

// CHECK-LABEL:   func.func @single_rotate(
// CHECK-SAME:                             %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: tensor<10x20xf32>) -> tensor<10x20xf32> {
// CHECK:           %[[VAL_1:.*]] = "enzymexla.rotate"(%[[VAL_0]]) <{amount = 3 : si32, dimension = 0 : si32}> : (tensor<10x20xf32>) -> tensor<10x20xf32>
// CHECK:           return %[[VAL_1]] : tensor<10x20xf32>
// CHECK:         }


// Test all positive amounts
func.func @all_positive(%arg0: tensor<8x16xf64>) -> (tensor<8x16xf64>, tensor<8x16xf64>) {
    %0 = "enzymexla.rotate"(%arg0) <{amount = 3 : si32, dimension = 0 : si32}> : (tensor<8x16xf64>) -> tensor<8x16xf64>
    %1 = "enzymexla.rotate"(%arg0) <{amount = 1 : si32, dimension = 0 : si32}> : (tensor<8x16xf64>) -> tensor<8x16xf64>
    return %0, %1 : tensor<8x16xf64>, tensor<8x16xf64>
}

// CHECK-LABEL:   func.func @all_positive(
// CHECK-SAME:                            %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: tensor<8x16xf64>) -> (tensor<8x16xf64>, tensor<8x16xf64>) {
// CHECK:           %[[VAL_1:.*]]:4 = "enzymexla.multi_rotate"(%[[VAL_0]]) <{dimension = 0 : si32, left_amount = 3 : si32, right_amount = 0 : si32}> : (tensor<8x16xf64>) -> (tensor<8x16xf64>, tensor<8x16xf64>, tensor<8x16xf64>, tensor<8x16xf64>)
// CHECK:           return %[[VAL_1]]#0, %[[VAL_1]]#2 : tensor<8x16xf64>, tensor<8x16xf64>
// CHECK:         }


// Test all negative amounts
func.func @all_negative(%arg0: tensor<8x16xf64>) -> (tensor<8x16xf64>, tensor<8x16xf64>) {
    %0 = "enzymexla.rotate"(%arg0) <{amount = -1 : si32, dimension = 1 : si32}> : (tensor<8x16xf64>) -> tensor<8x16xf64>
    %1 = "enzymexla.rotate"(%arg0) <{amount = -3 : si32, dimension = 1 : si32}> : (tensor<8x16xf64>) -> tensor<8x16xf64>
    return %0, %1 : tensor<8x16xf64>, tensor<8x16xf64>
}

// CHECK-LABEL:   func.func @all_negative(
// CHECK-SAME:                            %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: tensor<8x16xf64>) -> (tensor<8x16xf64>, tensor<8x16xf64>) {
// CHECK:           %[[VAL_1:.*]]:4 = "enzymexla.multi_rotate"(%[[VAL_0]]) <{dimension = 1 : si32, left_amount = 0 : si32, right_amount = 3 : si32}> : (tensor<8x16xf64>) -> (tensor<8x16xf64>, tensor<8x16xf64>, tensor<8x16xf64>, tensor<8x16xf64>)
// CHECK:           return %[[VAL_1]]#1, %[[VAL_1]]#3 : tensor<8x16xf64>, tensor<8x16xf64>
// CHECK:         }
