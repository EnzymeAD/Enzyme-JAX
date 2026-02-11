// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=recognize_multislice" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

// Test basic combination of multiple slices on the same input differing by offset
func.func @basic_multislice(%arg0: tensor<20x24x80xf64>) -> (tensor<20x24x10xf64>, tensor<20x24x10xf64>, tensor<20x24x10xf64>) {
    %0 = stablehlo.slice %arg0 [0:20, 0:24, 5:15] : (tensor<20x24x80xf64>) -> tensor<20x24x10xf64>
    %1 = stablehlo.slice %arg0 [0:20, 0:24, 6:16] : (tensor<20x24x80xf64>) -> tensor<20x24x10xf64>
    %2 = stablehlo.slice %arg0 [0:20, 0:24, 7:17] : (tensor<20x24x80xf64>) -> tensor<20x24x10xf64>
    return %0, %1, %2 : tensor<20x24x10xf64>, tensor<20x24x10xf64>, tensor<20x24x10xf64>
}

// CHECK-LABEL:   func.func @basic_multislice(
// CHECK-SAME:                                %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: tensor<20x24x80xf64>) -> (tensor<20x24x10xf64>, tensor<20x24x10xf64>, tensor<20x24x10xf64>) {
// CHECK:           %[[VAL_1:.*]]:3 = "enzymexla.multi_slice"(%[[VAL_0]]) <{amount = 2 : si32, dimension = 2 : si32, limit_indices = array<i64: 20, 24, 15>, start_indices = array<i64: 0, 0, 5>, strides = array<i64: 1, 1, 1>}> : (tensor<20x24x80xf64>) -> (tensor<20x24x10xf64>, tensor<20x24x10xf64>, tensor<20x24x10xf64>)
// CHECK:           return %[[VAL_1]]#0, %[[VAL_1]]#1, %[[VAL_1]]#2 : tensor<20x24x10xf64>, tensor<20x24x10xf64>, tensor<20x24x10xf64>
// CHECK:         }


// Test with negative offsets (slices shifted left from base)
func.func @negative_offsets(%arg0: tensor<20x24x80xf64>) -> (tensor<20x24x10xf64>, tensor<20x24x10xf64>, tensor<20x24x10xf64>) {
    %0 = stablehlo.slice %arg0 [0:20, 0:24, 3:13] : (tensor<20x24x80xf64>) -> tensor<20x24x10xf64>
    %1 = stablehlo.slice %arg0 [0:20, 0:24, 4:14] : (tensor<20x24x80xf64>) -> tensor<20x24x10xf64>
    %2 = stablehlo.slice %arg0 [0:20, 0:24, 5:15] : (tensor<20x24x80xf64>) -> tensor<20x24x10xf64>
    return %0, %1, %2 : tensor<20x24x10xf64>, tensor<20x24x10xf64>, tensor<20x24x10xf64>
}

// CHECK-LABEL:   func.func @negative_offsets(
// CHECK-SAME:                                %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: tensor<20x24x80xf64>) -> (tensor<20x24x10xf64>, tensor<20x24x10xf64>, tensor<20x24x10xf64>) {
// CHECK:           %[[VAL_1:.*]]:3 = "enzymexla.multi_slice"(%[[VAL_0]]) <{amount = 2 : si32, dimension = 2 : si32, limit_indices = array<i64: 20, 24, 13>, start_indices = array<i64: 0, 0, 3>, strides = array<i64: 1, 1, 1>}> : (tensor<20x24x80xf64>) -> (tensor<20x24x10xf64>, tensor<20x24x10xf64>, tensor<20x24x10xf64>)
// CHECK:           return %[[VAL_1]]#0, %[[VAL_1]]#1, %[[VAL_1]]#2 : tensor<20x24x10xf64>, tensor<20x24x10xf64>, tensor<20x24x10xf64>
// CHECK:         }


// Test with both positive and negative offsets from a middle slice
func.func @both_directions(%arg0: tensor<20x24x80xf64>) -> (tensor<20x24x10xf64>, tensor<20x24x10xf64>, tensor<20x24x10xf64>) {
    %0 = stablehlo.slice %arg0 [0:20, 0:24, 4:14] : (tensor<20x24x80xf64>) -> tensor<20x24x10xf64>
    %1 = stablehlo.slice %arg0 [0:20, 0:24, 5:15] : (tensor<20x24x80xf64>) -> tensor<20x24x10xf64>
    %2 = stablehlo.slice %arg0 [0:20, 0:24, 6:16] : (tensor<20x24x80xf64>) -> tensor<20x24x10xf64>
    return %0, %1, %2 : tensor<20x24x10xf64>, tensor<20x24x10xf64>, tensor<20x24x10xf64>
}

// CHECK-LABEL:   func.func @both_directions(
// CHECK-SAME:                               %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: tensor<20x24x80xf64>) -> (tensor<20x24x10xf64>, tensor<20x24x10xf64>, tensor<20x24x10xf64>) {
// CHECK:           %[[VAL_1:.*]]:3 = "enzymexla.multi_slice"(%[[VAL_0]]) <{amount = 2 : si32, dimension = 2 : si32, limit_indices = array<i64: 20, 24, 14>, start_indices = array<i64: 0, 0, 4>, strides = array<i64: 1, 1, 1>}> : (tensor<20x24x80xf64>) -> (tensor<20x24x10xf64>, tensor<20x24x10xf64>, tensor<20x24x10xf64>)
// CHECK:           return %[[VAL_1]]#0, %[[VAL_1]]#1, %[[VAL_1]]#2 : tensor<20x24x10xf64>, tensor<20x24x10xf64>, tensor<20x24x10xf64>
// CHECK:         }


// Test slices along different dimensions - should NOT combine
func.func @different_dimensions(%arg0: tensor<20x24x80xf64>) -> (tensor<20x10x80xf64>, tensor<20x24x10xf64>) {
    %0 = stablehlo.slice %arg0 [0:20, 5:15, 0:80] : (tensor<20x24x80xf64>) -> tensor<20x10x80xf64>
    %1 = stablehlo.slice %arg0 [0:20, 0:24, 5:15] : (tensor<20x24x80xf64>) -> tensor<20x24x10xf64>
    return %0, %1 : tensor<20x10x80xf64>, tensor<20x24x10xf64>
}

// CHECK-LABEL:   func.func @different_dimensions(
// CHECK-SAME:                                    %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: tensor<20x24x80xf64>) -> (tensor<20x10x80xf64>, tensor<20x24x10xf64>) {
// CHECK:           %[[VAL_1:.*]] = stablehlo.slice %[[VAL_0]] [0:20, 5:15, 0:80] : (tensor<20x24x80xf64>) -> tensor<20x10x80xf64>
// CHECK:           %[[VAL_2:.*]] = stablehlo.slice %[[VAL_0]] [0:20, 0:24, 5:15] : (tensor<20x24x80xf64>) -> tensor<20x24x10xf64>
// CHECK:           return %[[VAL_1]], %[[VAL_2]] : tensor<20x10x80xf64>, tensor<20x24x10xf64>
// CHECK:         }


// Test single slice - should NOT combine (need at least 2)
func.func @single_slice(%arg0: tensor<10x20xf32>) -> tensor<10x10xf32> {
    %0 = stablehlo.slice %arg0 [0:10, 5:15] : (tensor<10x20xf32>) -> tensor<10x10xf32>
    return %0 : tensor<10x10xf32>
}

// CHECK-LABEL:   func.func @single_slice(
// CHECK-SAME:                            %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: tensor<10x20xf32>) -> tensor<10x10xf32> {
// CHECK:           %[[VAL_1:.*]] = stablehlo.slice %[[VAL_0]] [0:10, 5:15] : (tensor<10x20xf32>) -> tensor<10x10xf32>
// CHECK:           return %[[VAL_1]] : tensor<10x10xf32>
// CHECK:         }


// Test non-contiguous offsets with gap - should NOT combine
// Offsets {0, 2} are not contiguous (gap at 1)
func.func @non_contiguous_offsets(%arg0: tensor<8x16xf64>) -> (tensor<8x4xf64>, tensor<8x4xf64>) {
    %0 = stablehlo.slice %arg0 [0:8, 5:9] : (tensor<8x16xf64>) -> tensor<8x4xf64>
    %1 = stablehlo.slice %arg0 [0:8, 7:11] : (tensor<8x16xf64>) -> tensor<8x4xf64>
    return %0, %1 : tensor<8x4xf64>, tensor<8x4xf64>
}

// CHECK-LABEL:   func.func @non_contiguous_offsets(
// CHECK-SAME:                                      %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: tensor<8x16xf64>) -> (tensor<8x4xf64>, tensor<8x4xf64>) {
// CHECK:           %[[VAL_1:.*]] = stablehlo.slice %[[VAL_0]] [0:8, 5:9] : (tensor<8x16xf64>) -> tensor<8x4xf64>
// CHECK:           %[[VAL_2:.*]] = stablehlo.slice %[[VAL_0]] [0:8, 7:11] : (tensor<8x16xf64>) -> tensor<8x4xf64>
// CHECK:           return %[[VAL_1]], %[[VAL_2]] : tensor<8x4xf64>, tensor<8x4xf64>
// CHECK:         }


// Test different slice sizes along varying dimension - should NOT combine
func.func @different_slice_sizes(%arg0: tensor<20x24x80xf64>) -> (tensor<20x24x10xf64>, tensor<20x24x15xf64>) {
    %0 = stablehlo.slice %arg0 [0:20, 0:24, 5:15] : (tensor<20x24x80xf64>) -> tensor<20x24x10xf64>
    %1 = stablehlo.slice %arg0 [0:20, 0:24, 6:21] : (tensor<20x24x80xf64>) -> tensor<20x24x15xf64>
    return %0, %1 : tensor<20x24x10xf64>, tensor<20x24x15xf64>
}

// CHECK-LABEL:   func.func @different_slice_sizes(
// CHECK-SAME:                                     %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: tensor<20x24x80xf64>) -> (tensor<20x24x10xf64>, tensor<20x24x15xf64>) {
// CHECK:           %[[VAL_1:.*]] = stablehlo.slice %[[VAL_0]] [0:20, 0:24, 5:15] : (tensor<20x24x80xf64>) -> tensor<20x24x10xf64>
// CHECK:           %[[VAL_2:.*]] = stablehlo.slice %[[VAL_0]] [0:20, 0:24, 6:21] : (tensor<20x24x80xf64>) -> tensor<20x24x15xf64>
// CHECK:           return %[[VAL_1]], %[[VAL_2]] : tensor<20x24x10xf64>, tensor<20x24x15xf64>
// CHECK:         }


// Test different strides - should NOT combine
func.func @different_strides(%arg0: tensor<20x24x80xf64>) -> (tensor<20x24x10xf64>, tensor<20x24x5xf64>) {
    %0 = stablehlo.slice %arg0 [0:20, 0:24, 5:15:1] : (tensor<20x24x80xf64>) -> tensor<20x24x10xf64>
    %1 = stablehlo.slice %arg0 [0:20, 0:24, 6:16:2] : (tensor<20x24x80xf64>) -> tensor<20x24x5xf64>
    return %0, %1 : tensor<20x24x10xf64>, tensor<20x24x5xf64>
}

// CHECK-LABEL:   func.func @different_strides(
// CHECK-SAME:                                 %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: tensor<20x24x80xf64>) -> (tensor<20x24x10xf64>, tensor<20x24x5xf64>) {
// CHECK:           %[[VAL_1:.*]] = stablehlo.slice %[[VAL_0]] [0:20, 0:24, 5:15] : (tensor<20x24x80xf64>) -> tensor<20x24x10xf64>
// CHECK:           %[[VAL_2:.*]] = stablehlo.slice %[[VAL_0]] [0:20, 0:24, 6:16:2] : (tensor<20x24x80xf64>) -> tensor<20x24x5xf64>
// CHECK:           return %[[VAL_1]], %[[VAL_2]] : tensor<20x24x10xf64>, tensor<20x24x5xf64>
// CHECK:         }


// Test slicing along dimension 0
func.func @slice_dim0(%arg0: tensor<20x24x80xf64>) -> (tensor<5x24x80xf64>, tensor<5x24x80xf64>, tensor<5x24x80xf64>) {
    %0 = stablehlo.slice %arg0 [0:5, 0:24, 0:80] : (tensor<20x24x80xf64>) -> tensor<5x24x80xf64>
    %1 = stablehlo.slice %arg0 [1:6, 0:24, 0:80] : (tensor<20x24x80xf64>) -> tensor<5x24x80xf64>
    %2 = stablehlo.slice %arg0 [2:7, 0:24, 0:80] : (tensor<20x24x80xf64>) -> tensor<5x24x80xf64>
    return %0, %1, %2 : tensor<5x24x80xf64>, tensor<5x24x80xf64>, tensor<5x24x80xf64>
}

// CHECK-LABEL:   func.func @slice_dim0(
// CHECK-SAME:                          %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: tensor<20x24x80xf64>) -> (tensor<5x24x80xf64>, tensor<5x24x80xf64>, tensor<5x24x80xf64>) {
// CHECK:           %[[VAL_1:.*]]:3 = "enzymexla.multi_slice"(%[[VAL_0]]) <{amount = 2 : si32, dimension = 0 : si32, limit_indices = array<i64: 5, 24, 80>, start_indices = array<i64: 0, 0, 0>, strides = array<i64: 1, 1, 1>}> : (tensor<20x24x80xf64>) -> (tensor<5x24x80xf64>, tensor<5x24x80xf64>, tensor<5x24x80xf64>)
// CHECK:           return %[[VAL_1]]#0, %[[VAL_1]]#1, %[[VAL_1]]#2 : tensor<5x24x80xf64>, tensor<5x24x80xf64>, tensor<5x24x80xf64>
// CHECK:         }


// Test slicing along dimension 1
func.func @slice_dim1(%arg0: tensor<20x24x80xf64>) -> (tensor<20x8x80xf64>, tensor<20x8x80xf64>) {
    %0 = stablehlo.slice %arg0 [0:20, 3:11, 0:80] : (tensor<20x24x80xf64>) -> tensor<20x8x80xf64>
    %1 = stablehlo.slice %arg0 [0:20, 4:12, 0:80] : (tensor<20x24x80xf64>) -> tensor<20x8x80xf64>
    return %0, %1 : tensor<20x8x80xf64>, tensor<20x8x80xf64>
}

// CHECK-LABEL:   func.func @slice_dim1(
// CHECK-SAME:                          %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: tensor<20x24x80xf64>) -> (tensor<20x8x80xf64>, tensor<20x8x80xf64>) {
// CHECK:           %[[VAL_1:.*]]:2 = "enzymexla.multi_slice"(%[[VAL_0]]) <{amount = 1 : si32, dimension = 1 : si32, limit_indices = array<i64: 20, 11, 80>, start_indices = array<i64: 0, 3, 0>, strides = array<i64: 1, 1, 1>}> : (tensor<20x24x80xf64>) -> (tensor<20x8x80xf64>, tensor<20x8x80xf64>)
// CHECK:           return %[[VAL_1]]#0, %[[VAL_1]]#1 : tensor<20x8x80xf64>, tensor<20x8x80xf64>
// CHECK:         }


// Test with non-unit strides (all same stride)
func.func @non_unit_strides(%arg0: tensor<20x40xf32>) -> (tensor<20x10xf32>, tensor<20x10xf32>) {
    %0 = stablehlo.slice %arg0 [0:20, 0:20:2] : (tensor<20x40xf32>) -> tensor<20x10xf32>
    %1 = stablehlo.slice %arg0 [0:20, 1:21:2] : (tensor<20x40xf32>) -> tensor<20x10xf32>
    return %0, %1 : tensor<20x10xf32>, tensor<20x10xf32>
}

// CHECK-LABEL:   func.func @non_unit_strides(
// CHECK-SAME:                                %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: tensor<20x40xf32>) -> (tensor<20x10xf32>, tensor<20x10xf32>) {
// CHECK:           %[[VAL_1:.*]]:2 = "enzymexla.multi_slice"(%[[VAL_0]]) <{amount = 1 : si32, dimension = 1 : si32, limit_indices = array<i64: 20, 20>, start_indices = array<i64: 0, 0>, strides = array<i64: 1, 2>}> : (tensor<20x40xf32>) -> (tensor<20x10xf32>, tensor<20x10xf32>)
// CHECK:           return %[[VAL_1]]#0, %[[VAL_1]]#1 : tensor<20x10xf32>, tensor<20x10xf32>
// CHECK:         }


// Test partial combination - only contiguous slices neighboring the trigger get combined
// Slices at offsets {0, 1, 3}: {0, 1} are contiguous, {3} has gap
func.func @partial_combination(%arg0: tensor<10x20xf32>) -> (tensor<10x5xf32>, tensor<10x5xf32>, tensor<10x5xf32>) {
    %0 = stablehlo.slice %arg0 [0:10, 2:7] : (tensor<10x20xf32>) -> tensor<10x5xf32>
    %1 = stablehlo.slice %arg0 [0:10, 3:8] : (tensor<10x20xf32>) -> tensor<10x5xf32>
    %2 = stablehlo.slice %arg0 [0:10, 5:10] : (tensor<10x20xf32>) -> tensor<10x5xf32>
    return %0, %1, %2 : tensor<10x5xf32>, tensor<10x5xf32>, tensor<10x5xf32>
}

// CHECK-LABEL:   func.func @partial_combination(
// CHECK-SAME:                                   %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: tensor<10x20xf32>) -> (tensor<10x5xf32>, tensor<10x5xf32>, tensor<10x5xf32>) {
// CHECK:           %[[VAL_1:.*]]:2 = "enzymexla.multi_slice"(%[[VAL_0]]) <{amount = 1 : si32, dimension = 1 : si32, limit_indices = array<i64: 10, 7>, start_indices = array<i64: 0, 2>, strides = array<i64: 1, 1>}> : (tensor<10x20xf32>) -> (tensor<10x5xf32>, tensor<10x5xf32>)
// CHECK:           %[[VAL_2:.*]] = stablehlo.slice %[[VAL_0]] [0:10, 5:10] : (tensor<10x20xf32>) -> tensor<10x5xf32>
// CHECK:           return %[[VAL_1]]#0, %[[VAL_1]]#1, %[[VAL_2]] : tensor<10x5xf32>, tensor<10x5xf32>, tensor<10x5xf32>
// CHECK:         }


// Test with an unused slice creating a gap - should NOT combine
// Slices at offsets {0, 2} after filtering unused offset 1 are not contiguous
func.func @with_unused_slice_gap(%arg0: tensor<10x20xf32>) -> (tensor<10x5xf32>, tensor<10x5xf32>) {
    %0 = stablehlo.slice %arg0 [0:10, 2:7] : (tensor<10x20xf32>) -> tensor<10x5xf32>
    %1 = stablehlo.slice %arg0 [0:10, 3:8] : (tensor<10x20xf32>) -> tensor<10x5xf32>  // unused!
    %2 = stablehlo.slice %arg0 [0:10, 4:9] : (tensor<10x20xf32>) -> tensor<10x5xf32>
    return %0, %2 : tensor<10x5xf32>, tensor<10x5xf32>
}

// CHECK-LABEL:   func.func @with_unused_slice_gap(
// CHECK-SAME:                                     %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: tensor<10x20xf32>) -> (tensor<10x5xf32>, tensor<10x5xf32>) {
// CHECK:           %[[VAL_1:.*]] = stablehlo.slice %[[VAL_0]] [0:10, 2:7] : (tensor<10x20xf32>) -> tensor<10x5xf32>
// CHECK:           %[[VAL_2:.*]] = stablehlo.slice %[[VAL_0]] [0:10, 4:9] : (tensor<10x20xf32>) -> tensor<10x5xf32>
// CHECK:           return %[[VAL_1]], %[[VAL_2]] : tensor<10x5xf32>, tensor<10x5xf32>
// CHECK:         }


// Test with an unused slice that doesn't break contiguity - should combine
// Slices at offsets {0, 1, 2} with offset 2 unused still leaves {0, 1} contiguous
func.func @with_unused_slice_no_gap(%arg0: tensor<10x20xf32>) -> (tensor<10x5xf32>, tensor<10x5xf32>) {
    %0 = stablehlo.slice %arg0 [0:10, 2:7] : (tensor<10x20xf32>) -> tensor<10x5xf32>
    %1 = stablehlo.slice %arg0 [0:10, 3:8] : (tensor<10x20xf32>) -> tensor<10x5xf32>
    %2 = stablehlo.slice %arg0 [0:10, 4:9] : (tensor<10x20xf32>) -> tensor<10x5xf32>  // unused!
    return %0, %1 : tensor<10x5xf32>, tensor<10x5xf32>
}

// CHECK-LABEL:   func.func @with_unused_slice_no_gap(
// CHECK-SAME:                                        %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: tensor<10x20xf32>) -> (tensor<10x5xf32>, tensor<10x5xf32>) {
// CHECK:           %[[VAL_1:.*]]:2 = "enzymexla.multi_slice"(%[[VAL_0]]) <{amount = 1 : si32, dimension = 1 : si32, limit_indices = array<i64: 10, 7>, start_indices = array<i64: 0, 2>, strides = array<i64: 1, 1>}> : (tensor<10x20xf32>) -> (tensor<10x5xf32>, tensor<10x5xf32>)
// CHECK:           return %[[VAL_1]]#0, %[[VAL_1]]#1 : tensor<10x5xf32>, tensor<10x5xf32>
// CHECK:         }


// Test 2D tensor
func.func @slice_2d(%arg0: tensor<100x200xf32>) -> (tensor<100x50xf32>, tensor<100x50xf32>, tensor<100x50xf32>) {
    %0 = stablehlo.slice %arg0 [0:100, 10:60] : (tensor<100x200xf32>) -> tensor<100x50xf32>
    %1 = stablehlo.slice %arg0 [0:100, 11:61] : (tensor<100x200xf32>) -> tensor<100x50xf32>
    %2 = stablehlo.slice %arg0 [0:100, 12:62] : (tensor<100x200xf32>) -> tensor<100x50xf32>
    return %0, %1, %2 : tensor<100x50xf32>, tensor<100x50xf32>, tensor<100x50xf32>
}

// CHECK-LABEL:   func.func @slice_2d(
// CHECK-SAME:                        %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: tensor<100x200xf32>) -> (tensor<100x50xf32>, tensor<100x50xf32>, tensor<100x50xf32>) {
// CHECK:           %[[VAL_1:.*]]:3 = "enzymexla.multi_slice"(%[[VAL_0]]) <{amount = 2 : si32, dimension = 1 : si32, limit_indices = array<i64: 100, 60>, start_indices = array<i64: 0, 10>, strides = array<i64: 1, 1>}> : (tensor<100x200xf32>) -> (tensor<100x50xf32>, tensor<100x50xf32>, tensor<100x50xf32>)
// CHECK:           return %[[VAL_1]]#0, %[[VAL_1]]#1, %[[VAL_1]]#2 : tensor<100x50xf32>, tensor<100x50xf32>, tensor<100x50xf32>
// CHECK:         }


// Test 4D tensor
func.func @slice_4d(%arg0: tensor<8x16x32x64xf32>) -> (tensor<8x16x10x64xf32>, tensor<8x16x10x64xf32>) {
    %0 = stablehlo.slice %arg0 [0:8, 0:16, 5:15, 0:64] : (tensor<8x16x32x64xf32>) -> tensor<8x16x10x64xf32>
    %1 = stablehlo.slice %arg0 [0:8, 0:16, 6:16, 0:64] : (tensor<8x16x32x64xf32>) -> tensor<8x16x10x64xf32>
    return %0, %1 : tensor<8x16x10x64xf32>, tensor<8x16x10x64xf32>
}

// CHECK-LABEL:   func.func @slice_4d(
// CHECK-SAME:                        %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: tensor<8x16x32x64xf32>) -> (tensor<8x16x10x64xf32>, tensor<8x16x10x64xf32>) {
// CHECK:           %[[VAL_1:.*]]:2 = "enzymexla.multi_slice"(%[[VAL_0]]) <{amount = 1 : si32, dimension = 2 : si32, limit_indices = array<i64: 8, 16, 15, 64>, start_indices = array<i64: 0, 0, 5, 0>, strides = array<i64: 1, 1, 1, 1>}> : (tensor<8x16x32x64xf32>) -> (tensor<8x16x10x64xf32>, tensor<8x16x10x64xf32>)
// CHECK:           return %[[VAL_1]]#0, %[[VAL_1]]#1 : tensor<8x16x10x64xf32>, tensor<8x16x10x64xf32>
// CHECK:         }


// Test many contiguous slices
func.func @many_slices(%arg0: tensor<20x100xf64>) -> (tensor<20x10xf64>, tensor<20x10xf64>, tensor<20x10xf64>, tensor<20x10xf64>, tensor<20x10xf64>) {
    %0 = stablehlo.slice %arg0 [0:20, 0:10] : (tensor<20x100xf64>) -> tensor<20x10xf64>
    %1 = stablehlo.slice %arg0 [0:20, 1:11] : (tensor<20x100xf64>) -> tensor<20x10xf64>
    %2 = stablehlo.slice %arg0 [0:20, 2:12] : (tensor<20x100xf64>) -> tensor<20x10xf64>
    %3 = stablehlo.slice %arg0 [0:20, 3:13] : (tensor<20x100xf64>) -> tensor<20x10xf64>
    %4 = stablehlo.slice %arg0 [0:20, 4:14] : (tensor<20x100xf64>) -> tensor<20x10xf64>
    return %0, %1, %2, %3, %4 : tensor<20x10xf64>, tensor<20x10xf64>, tensor<20x10xf64>, tensor<20x10xf64>, tensor<20x10xf64>
}

// CHECK-LABEL:   func.func @many_slices(
// CHECK-SAME:                           %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: tensor<20x100xf64>) -> (tensor<20x10xf64>, tensor<20x10xf64>, tensor<20x10xf64>, tensor<20x10xf64>, tensor<20x10xf64>) {
// CHECK:           %[[VAL_1:.*]]:5 = "enzymexla.multi_slice"(%[[VAL_0]]) <{amount = 4 : si32, dimension = 1 : si32, limit_indices = array<i64: 20, 10>, start_indices = array<i64: 0, 0>, strides = array<i64: 1, 1>}> : (tensor<20x100xf64>) -> (tensor<20x10xf64>, tensor<20x10xf64>, tensor<20x10xf64>, tensor<20x10xf64>, tensor<20x10xf64>)
// CHECK:           return %[[VAL_1]]#0, %[[VAL_1]]#1, %[[VAL_1]]#2, %[[VAL_1]]#3, %[[VAL_1]]#4 : tensor<20x10xf64>, tensor<20x10xf64>, tensor<20x10xf64>, tensor<20x10xf64>, tensor<20x10xf64>
// CHECK:         }


// Test slices from different inputs - should NOT combine
func.func @different_inputs(%arg0: tensor<10x20xf32>, %arg1: tensor<10x20xf32>) -> (tensor<10x5xf32>, tensor<10x5xf32>) {
    %0 = stablehlo.slice %arg0 [0:10, 5:10] : (tensor<10x20xf32>) -> tensor<10x5xf32>
    %1 = stablehlo.slice %arg1 [0:10, 6:11] : (tensor<10x20xf32>) -> tensor<10x5xf32>
    return %0, %1 : tensor<10x5xf32>, tensor<10x5xf32>
}

// CHECK-LABEL:   func.func @different_inputs(
// CHECK-SAME:                                %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: tensor<10x20xf32>,
// CHECK-SAME:                                %[[VAL_1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: tensor<10x20xf32>) -> (tensor<10x5xf32>, tensor<10x5xf32>) {
// CHECK:           %[[VAL_2:.*]] = stablehlo.slice %[[VAL_0]] [0:10, 5:10] : (tensor<10x20xf32>) -> tensor<10x5xf32>
// CHECK:           %[[VAL_3:.*]] = stablehlo.slice %[[VAL_1]] [0:10, 6:11] : (tensor<10x20xf32>) -> tensor<10x5xf32>
// CHECK:           return %[[VAL_2]], %[[VAL_3]] : tensor<10x5xf32>, tensor<10x5xf32>
// CHECK:         }
