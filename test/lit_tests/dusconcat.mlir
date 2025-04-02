// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt="passes=512" --split-input-file | FileCheck %s
// We enable pass 512 explicitly as DUSDUS and DUSDUSConcat are under that flag bit in your example code.
// If DUSConcat is moved outside that flag, you might not need the passes= option.

module {
  func.func @dus_concat_test(
    %arg0: tensor<1x10x8xf32>,   // Input A
    %arg1: tensor<1x10x100xf32>, // Input B (target for DUS)
    %arg2: tensor<1x10x8xf32>,   // Input C
    %arg3: tensor<1x10x50xf32>   // Update tensor
  ) -> tensor<1x10x116xf32> {

    // Concatenate A, B, C along dimension 2. Result shape: 1x10x(8+100+8) = 1x10x116
    %concat = stablehlo.concatenate %arg0, %arg1, %arg2, dimension = 2 : (tensor<1x10x8xf32>, tensor<1x10x100xf32>, tensor<1x10x8xf32>) -> tensor<1x10x116xf32>

    // Constants for start indices: [0, 0, 10]
    // Dimension 0 (batch): start=0, size=1 (matches update shape)
    // Dimension 1 (rows): start=0, size=10 (matches update shape)
    // Dimension 2 (cols/concat dim): start=10, size=50 (matches update shape)
    // Note: Input B starts at offset 8 along dim 2. The update starts at absolute index 10,
    // which is relative index 10-8=2 within B. The update ends at 10+50=60, which is < offset 8+100=108.
    // So, the update falls entirely within B.
    %c0_i64 = stablehlo.constant dense : tensor<i64>
    %c10_i64 = stablehlo.constant dense : tensor<i64>

    // Dynamic update slice on the concatenated tensor
    %dus = stablehlo.dynamic_update_slice %concat, %arg3, %c0_i64, %c0_i64, %c10_i64 : (tensor<1x10x116xf32>, tensor<1x10x50xf32>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x10x116xf32>

    func.return %dus : tensor<1x10x116xf32>
  }
}

// CHECK-LABEL: func.func @dus_concat_test
// CHECK-SAME:    (%[[ARG0:.*]]: tensor<1x10x8xf32>, %[[ARG1:.*]]: tensor<1x10x100xf32>, %[[ARG2:.*]]: tensor<1x10x8xf32>, %[[ARG3:.*]]: tensor<1x10x50xf32>) -> tensor<1x10x116xf32>
// CHECK-NEXT:    // Constants for the new DUS start indices [0, 0, 2]
// CHECK-DAG:     %[[C0_0:.*]] = stablehlo.constant dense : tensor<i64>
// CHECK-DAG:     %[[C0_1:.*]] = stablehlo.constant dense : tensor<i64>
// CHECK-DAG:     %[[C2:.*]] = stablehlo.constant dense : tensor<i64>
// CHECK-NEXT:    // New DUS applied directly to the target input (%arg1)
// CHECK-NEXT:    %[[NEW_DUS:.*]] = stablehlo.dynamic_update_slice %[[ARG1]], %[[ARG3]], %[[C0_0]], %[[C0_1]], %[[C2]] : (tensor<1x10x100xf32>, tensor<1x10x50xf32>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x10x100xf32>
// CHECK-NEXT:    // New concatenate using the result of the new DUS
// CHECK-NEXT:    %[[NEW_CONCAT:.*]] = stablehlo.concatenate %[[ARG0]], %[[NEW_DUS]], %[[ARG2]], dimension = 2 : (tensor<1x10x8xf32>, tensor<1x10x100xf32>, tensor<1x10x8xf32>) -> tensor<1x10x116xf32>
// CHECK-NEXT:    return %[[NEW_CONCAT]] : tensor<1x10x116xf32>
// CHECK-NEXT:  }