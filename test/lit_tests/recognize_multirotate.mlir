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


// Test non-contiguous positive amounts with gap - should NOT combine
// Amounts {1, 3} are not contiguous (gap at 2), only {1} neighbors identity
func.func @non_contiguous_positive(%arg0: tensor<8x16xf64>) -> (tensor<8x16xf64>, tensor<8x16xf64>) {
    %0 = "enzymexla.rotate"(%arg0) <{amount = 3 : si32, dimension = 0 : si32}> : (tensor<8x16xf64>) -> tensor<8x16xf64>
    %1 = "enzymexla.rotate"(%arg0) <{amount = 1 : si32, dimension = 0 : si32}> : (tensor<8x16xf64>) -> tensor<8x16xf64>
    return %0, %1 : tensor<8x16xf64>, tensor<8x16xf64>
}

// CHECK-LABEL:   func.func @non_contiguous_positive(
// CHECK-SAME:                                       %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: tensor<8x16xf64>) -> (tensor<8x16xf64>, tensor<8x16xf64>) {
// CHECK:           %[[VAL_1:.*]] = "enzymexla.rotate"(%[[VAL_0]]) <{amount = 3 : si32, dimension = 0 : si32}> : (tensor<8x16xf64>) -> tensor<8x16xf64>
// CHECK:           %[[VAL_2:.*]] = "enzymexla.rotate"(%[[VAL_0]]) <{amount = 1 : si32, dimension = 0 : si32}> : (tensor<8x16xf64>) -> tensor<8x16xf64>
// CHECK:           return %[[VAL_1]], %[[VAL_2]] : tensor<8x16xf64>, tensor<8x16xf64>
// CHECK:         }


// Test non-contiguous negative amounts with gap - should NOT combine
// Amounts {-1, -3} are not contiguous (gap at -2), only {-1} neighbors identity
func.func @non_contiguous_negative(%arg0: tensor<8x16xf64>) -> (tensor<8x16xf64>, tensor<8x16xf64>) {
    %0 = "enzymexla.rotate"(%arg0) <{amount = -1 : si32, dimension = 1 : si32}> : (tensor<8x16xf64>) -> tensor<8x16xf64>
    %1 = "enzymexla.rotate"(%arg0) <{amount = -3 : si32, dimension = 1 : si32}> : (tensor<8x16xf64>) -> tensor<8x16xf64>
    return %0, %1 : tensor<8x16xf64>, tensor<8x16xf64>
}

// CHECK-LABEL:   func.func @non_contiguous_negative(
// CHECK-SAME:                                       %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: tensor<8x16xf64>) -> (tensor<8x16xf64>, tensor<8x16xf64>) {
// CHECK:           %[[VAL_1:.*]] = "enzymexla.rotate"(%[[VAL_0]]) <{amount = -1 : si32, dimension = 1 : si32}> : (tensor<8x16xf64>) -> tensor<8x16xf64>
// CHECK:           %[[VAL_2:.*]] = "enzymexla.rotate"(%[[VAL_0]]) <{amount = -3 : si32, dimension = 1 : si32}> : (tensor<8x16xf64>) -> tensor<8x16xf64>
// CHECK:           return %[[VAL_1]], %[[VAL_2]] : tensor<8x16xf64>, tensor<8x16xf64>
// CHECK:         }


// Test contiguous positive amounts neighboring identity - should combine
// Amounts {1, 2, 3} are contiguous and {1} neighbors identity
func.func @contiguous_positive(%arg0: tensor<8x16xf64>) -> (tensor<8x16xf64>, tensor<8x16xf64>, tensor<8x16xf64>) {
    %0 = "enzymexla.rotate"(%arg0) <{amount = 3 : si32, dimension = 0 : si32}> : (tensor<8x16xf64>) -> tensor<8x16xf64>
    %1 = "enzymexla.rotate"(%arg0) <{amount = 2 : si32, dimension = 0 : si32}> : (tensor<8x16xf64>) -> tensor<8x16xf64>
    %2 = "enzymexla.rotate"(%arg0) <{amount = 1 : si32, dimension = 0 : si32}> : (tensor<8x16xf64>) -> tensor<8x16xf64>
    return %0, %1, %2 : tensor<8x16xf64>, tensor<8x16xf64>, tensor<8x16xf64>
}

// CHECK-LABEL:   func.func @contiguous_positive(
// CHECK-SAME:                                   %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: tensor<8x16xf64>) -> (tensor<8x16xf64>, tensor<8x16xf64>, tensor<8x16xf64>) {
// CHECK:           %[[VAL_1:.*]]:4 = "enzymexla.multi_rotate"(%[[VAL_0]]) <{dimension = 0 : si32, left_amount = 3 : si32, right_amount = 0 : si32}> : (tensor<8x16xf64>) -> (tensor<8x16xf64>, tensor<8x16xf64>, tensor<8x16xf64>, tensor<8x16xf64>)
// CHECK:           return %[[VAL_1]]#0, %[[VAL_1]]#1, %[[VAL_1]]#2 : tensor<8x16xf64>, tensor<8x16xf64>, tensor<8x16xf64>
// CHECK:         }


// Test contiguous negative amounts neighboring identity - should combine
// Amounts {-1, -2, -3} are contiguous and {-1} neighbors identity
func.func @contiguous_negative(%arg0: tensor<8x16xf64>) -> (tensor<8x16xf64>, tensor<8x16xf64>, tensor<8x16xf64>) {
    %0 = "enzymexla.rotate"(%arg0) <{amount = -1 : si32, dimension = 1 : si32}> : (tensor<8x16xf64>) -> tensor<8x16xf64>
    %1 = "enzymexla.rotate"(%arg0) <{amount = -2 : si32, dimension = 1 : si32}> : (tensor<8x16xf64>) -> tensor<8x16xf64>
    %2 = "enzymexla.rotate"(%arg0) <{amount = -3 : si32, dimension = 1 : si32}> : (tensor<8x16xf64>) -> tensor<8x16xf64>
    return %0, %1, %2 : tensor<8x16xf64>, tensor<8x16xf64>, tensor<8x16xf64>
}

// CHECK-LABEL:   func.func @contiguous_negative(
// CHECK-SAME:                                   %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: tensor<8x16xf64>) -> (tensor<8x16xf64>, tensor<8x16xf64>, tensor<8x16xf64>) {
// CHECK:           %[[VAL_1:.*]]:4 = "enzymexla.multi_rotate"(%[[VAL_0]]) <{dimension = 1 : si32, left_amount = 0 : si32, right_amount = 3 : si32}> : (tensor<8x16xf64>) -> (tensor<8x16xf64>, tensor<8x16xf64>, tensor<8x16xf64>, tensor<8x16xf64>)
// CHECK:           return %[[VAL_1]]#1, %[[VAL_1]]#2, %[[VAL_1]]#3 : tensor<8x16xf64>, tensor<8x16xf64>, tensor<8x16xf64>
// CHECK:         }


// Test amounts far from identity - should NOT combine
// Amounts {2, 3} are contiguous but don't neighbor identity (neither is 1 or -1)
func.func @far_from_identity(%arg0: tensor<8x16xf64>) -> (tensor<8x16xf64>, tensor<8x16xf64>) {
    %0 = "enzymexla.rotate"(%arg0) <{amount = 2 : si32, dimension = 0 : si32}> : (tensor<8x16xf64>) -> tensor<8x16xf64>
    %1 = "enzymexla.rotate"(%arg0) <{amount = 3 : si32, dimension = 0 : si32}> : (tensor<8x16xf64>) -> tensor<8x16xf64>
    return %0, %1 : tensor<8x16xf64>, tensor<8x16xf64>
}

// CHECK-LABEL:   func.func @far_from_identity(
// CHECK-SAME:                                 %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: tensor<8x16xf64>) -> (tensor<8x16xf64>, tensor<8x16xf64>) {
// CHECK:           %[[VAL_1:.*]] = "enzymexla.rotate"(%[[VAL_0]]) <{amount = 2 : si32, dimension = 0 : si32}> : (tensor<8x16xf64>) -> tensor<8x16xf64>
// CHECK:           %[[VAL_2:.*]] = "enzymexla.rotate"(%[[VAL_0]]) <{amount = 3 : si32, dimension = 0 : si32}> : (tensor<8x16xf64>) -> tensor<8x16xf64>
// CHECK:           return %[[VAL_1]], %[[VAL_2]] : tensor<8x16xf64>, tensor<8x16xf64>
// CHECK:         }


// Test partial combination - only qualifying rotations get combined
// Amounts {-2, 1, 2}: {1, 2} neighbors identity and combines, {-2} stays separate
func.func @partial_combination(%arg0: tensor<10x20xf32>) -> (tensor<10x20xf32>, tensor<10x20xf32>, tensor<10x20xf32>) {
    %0 = "enzymexla.rotate"(%arg0) <{amount = -2 : si32, dimension = 0 : si32}> : (tensor<10x20xf32>) -> tensor<10x20xf32>
    %1 = "enzymexla.rotate"(%arg0) <{amount = 1 : si32, dimension = 0 : si32}> : (tensor<10x20xf32>) -> tensor<10x20xf32>
    %2 = "enzymexla.rotate"(%arg0) <{amount = 2 : si32, dimension = 0 : si32}> : (tensor<10x20xf32>) -> tensor<10x20xf32>
    return %0, %1, %2 : tensor<10x20xf32>, tensor<10x20xf32>, tensor<10x20xf32>
}

// CHECK-LABEL:   func.func @partial_combination(
// CHECK-SAME:                                   %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: tensor<10x20xf32>) -> (tensor<10x20xf32>, tensor<10x20xf32>, tensor<10x20xf32>) {
// CHECK:           %[[VAL_2:.*]]:3 = "enzymexla.multi_rotate"(%[[VAL_0]]) <{dimension = 0 : si32, left_amount = 2 : si32, right_amount = 0 : si32}> : (tensor<10x20xf32>) -> (tensor<10x20xf32>, tensor<10x20xf32>, tensor<10x20xf32>)
// CHECK:           %[[VAL_1:.*]] = "enzymexla.rotate"(%[[VAL_0]]) <{amount = -2 : si32, dimension = 0 : si32}> : (tensor<10x20xf32>) -> tensor<10x20xf32>
// CHECK:           return %[[VAL_1]], %[[VAL_2]]#1, %[[VAL_2]]#0 : tensor<10x20xf32>, tensor<10x20xf32>, tensor<10x20xf32>
// CHECK:         }


// Test combining rotations from both sides of identity via neighbor rule
// Amounts {-1, 1}: both neighbor identity, combine into multirotate spanning [-1, 1]
func.func @both_sides_neighbor(%arg0: tensor<10x20xf32>) -> (tensor<10x20xf32>, tensor<10x20xf32>) {
    %0 = "enzymexla.rotate"(%arg0) <{amount = -1 : si32, dimension = 0 : si32}> : (tensor<10x20xf32>) -> tensor<10x20xf32>
    %1 = "enzymexla.rotate"(%arg0) <{amount = 1 : si32, dimension = 0 : si32}> : (tensor<10x20xf32>) -> tensor<10x20xf32>
    return %0, %1 : tensor<10x20xf32>, tensor<10x20xf32>
}

// CHECK-LABEL:   func.func @both_sides_neighbor(
// CHECK-SAME:                                   %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: tensor<10x20xf32>) -> (tensor<10x20xf32>, tensor<10x20xf32>) {
// CHECK:           %[[VAL_1:.*]]:3 = "enzymexla.multi_rotate"(%[[VAL_0]]) <{dimension = 0 : si32, left_amount = 1 : si32, right_amount = 1 : si32}> : (tensor<10x20xf32>) -> (tensor<10x20xf32>, tensor<10x20xf32>, tensor<10x20xf32>)
// CHECK:           return %[[VAL_1]]#2, %[[VAL_1]]#0 : tensor<10x20xf32>, tensor<10x20xf32>
// CHECK:         }


// Test with explicit zero rotation included
func.func @with_zero_rotation(%arg0: tensor<10x20xf32>) -> (tensor<10x20xf32>, tensor<10x20xf32>, tensor<10x20xf32>) {
    %0 = "enzymexla.rotate"(%arg0) <{amount = 0 : si32, dimension = 1 : si32}> : (tensor<10x20xf32>) -> tensor<10x20xf32>
    %1 = "enzymexla.rotate"(%arg0) <{amount = 1 : si32, dimension = 1 : si32}> : (tensor<10x20xf32>) -> tensor<10x20xf32>
    %2 = "enzymexla.rotate"(%arg0) <{amount = 2 : si32, dimension = 1 : si32}> : (tensor<10x20xf32>) -> tensor<10x20xf32>
    return %0, %1, %2 : tensor<10x20xf32>, tensor<10x20xf32>, tensor<10x20xf32>
}

// CHECK-LABEL:   func.func @with_zero_rotation(
// CHECK-SAME:                                  %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: tensor<10x20xf32>) -> (tensor<10x20xf32>, tensor<10x20xf32>, tensor<10x20xf32>) {
// CHECK:           %[[VAL_1:.*]]:3 = "enzymexla.multi_rotate"(%[[VAL_0]]) <{dimension = 1 : si32, left_amount = 2 : si32, right_amount = 0 : si32}> : (tensor<10x20xf32>) -> (tensor<10x20xf32>, tensor<10x20xf32>, tensor<10x20xf32>)
// CHECK:           return %[[VAL_0]], %[[VAL_1]]#1, %[[VAL_1]]#0 : tensor<10x20xf32>, tensor<10x20xf32>, tensor<10x20xf32>
// CHECK:         }

// Test with an unused value (this value should be ignored)
func.func @with_unused_value(%arg0: tensor<10x20xf32>) -> (tensor<10x20xf32>, tensor<10x20xf32>) {
    %0 = "enzymexla.rotate"(%arg0) <{amount = 0 : si32, dimension = 1 : si32}> : (tensor<10x20xf32>) -> tensor<10x20xf32>
    %1 = "enzymexla.rotate"(%arg0) <{amount = 1 : si32, dimension = 1 : si32}> : (tensor<10x20xf32>) -> tensor<10x20xf32> // unused!
    %2 = "enzymexla.rotate"(%arg0) <{amount = 2 : si32, dimension = 1 : si32}> : (tensor<10x20xf32>) -> tensor<10x20xf32>
    return %0, %2 : tensor<10x20xf32>, tensor<10x20xf32>
}

// CHECK-LABEL:   func.func @with_unused_value(
// CHECK-SAME:                                  %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: tensor<10x20xf32>) -> (tensor<10x20xf32>, tensor<10x20xf32>) {
// CHECK:           %[[VAL_1:.*]] = "enzymexla.rotate"(%[[VAL_0]]) <{amount = 2 : si32, dimension = 1 : si32}> : (tensor<10x20xf32>) -> tensor<10x20xf32>
// CHECK:           return %[[VAL_0]], %[[VAL_1]] : tensor<10x20xf32>, tensor<10x20xf32>
// CHECK:         }


sdy.mesh @mesh = <["x"=2, "y"=2]>
func.func @recognize_multirotate_negative(%1604: tensor<4x1520x3056xf32>) -> (tensor<4x1520x3056xf32>, tensor<4x1520x3056xf32>, tensor<4x1520x3056xf32>, tensor<4x1520x3056xf32>, tensor<4x1520x3056xf32>) {
    %2706 = "enzymexla.rotate"(%1604) <{amount = 3054 : si32, dimension = 2 : si32}> {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y"}, {"x"}]>]>} : (tensor<4x1520x3056xf32>) -> tensor<4x1520x3056xf32>
    %2707 = "enzymexla.rotate"(%1604) <{amount = 3053 : si32, dimension = 2 : si32}> {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y"}, {"x"}]>]>} : (tensor<4x1520x3056xf32>) -> tensor<4x1520x3056xf32>
    %2710 = "enzymexla.rotate"(%1604) <{amount = 3055 : si32, dimension = 2 : si32}> {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y"}, {"x"}]>]>} : (tensor<4x1520x3056xf32>) -> tensor<4x1520x3056xf32>
    %2715 = "enzymexla.rotate"(%1604) <{amount = 1 : si32, dimension = 2 : si32}> {enzymexla.non_negative = [#enzymexla<guaranteed NOTGUARANTEED>], sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y"}, {"x"}]>]>} : (tensor<4x1520x3056xf32>) -> tensor<4x1520x3056xf32>
    %2719 = "enzymexla.rotate"(%1604) <{amount = 2 : si32, dimension = 2 : si32}> {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y"}, {"x"}]>]>} : (tensor<4x1520x3056xf32>) -> tensor<4x1520x3056xf32>
    return %2706, %2707, %2710, %2715, %2719 : tensor<4x1520x3056xf32>, tensor<4x1520x3056xf32>, tensor<4x1520x3056xf32>, tensor<4x1520x3056xf32>, tensor<4x1520x3056xf32>
}

// CHECK-LABEL:   func.func @recognize_multirotate_negative
// CHECK-SAME:         (%[[ARG:.+]]: tensor<4x1520x3056xf32>) -> (
// CHECK:          %{{.*}}:6 = "enzymexla.multi_rotate"(%arg0) <{dimension = 2 : si32, left_amount = 2 : si32, right_amount = 3 : si32}> {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y"}, {"x"}]>, <@mesh, [{}, {"y"}, {"x"}]>, <@mesh, [{}, {"y"}, {"x"}]>, <@mesh, [{}, {"y"}, {"x"}]>, <@mesh, [{}, {"y"}, {"x"}]>, <@mesh, [{}, {"y"}, {"x"}]>]>} : (tensor<4x1520x3056xf32>) -> (tensor<4x1520x3056xf32>, tensor<4x1520x3056xf32>, tensor<4x1520x3056xf32>, tensor<4x1520x3056xf32>, tensor<4x1520x3056xf32>, tensor<4x1520x3056xf32>)
// CHECK-NOT:      enzymexla.rotate
