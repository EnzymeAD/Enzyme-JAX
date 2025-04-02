// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt --split-input-file | FileCheck %s

module {
  func.func @dus_pad_operand_test(
    %original_data: tensor<10x20xf32>,
    %update_data: tensor<5x8xf32>,
    %pad_val_scalar: tensor<f32>
  ) -> tensor<13x27xf32> { // Padded shape: (2+10+1) x (3+20+4) = 13x27

    // Pad the original data
    // Low padding: [2, 3], High padding: [1, 4], Interior padding: [0, 0]
    %padded_val = stablehlo.pad %original_data, %pad_val_scalar,
        low = [2, 3], high = [1, 4], interior = [0, 0] :
        (tensor<10x20xf32>, tensor<f32>) -> tensor<13x27xf32>

    // Start indices for DUS: [4, 5]
    // Update region starts at [4, 5] in the padded tensor.
    // Update shape is 5x8.
    // The original data region in the padded tensor is [2:12, 3:23).
    // The update region [4:9, 5:13) is fully contained within [2:12, 3:23).
    %c4_i64 = stablehlo.constant dense<4> : tensor<i64>
    %c5_i64 = stablehlo.constant dense<5> : tensor<i64>

    // DUS applies the update to the padded value
    %dus_result = stablehlo.dynamic_update_slice %padded_val, %update_data, %c4_i64, %c5_i64 :
        (tensor<13x27xf32>, tensor<5x8xf32>, tensor<i64>, tensor<i64>) -> tensor<13x27xf32>

    func.return %dus_result : tensor<13x27xf32>
  }
}

// CHECK:  func.func @dus_pad_operand_test(%arg0: tensor<10x20xf32>, %arg1: tensor<5x8xf32>, %arg2: tensor<f32>) -> tensor<13x27xf32> {
// CHECK-NEXT:    %c = stablehlo.constant dense<2> : tensor<i64>
// CHECK-NEXT:    %0 = stablehlo.dynamic_update_slice %arg0, %arg1, %c, %c : (tensor<10x20xf32>, tensor<5x8xf32>, tensor<i64>, tensor<i64>) -> tensor<10x20xf32>
// CHECK-NEXT:    %1 = stablehlo.pad %0, %arg2, low = [2, 3], high = [1, 4], interior = [0, 0] : (tensor<10x20xf32>, tensor<f32>) -> tensor<13x27xf32>
// CHECK-NEXT:    return %1 : tensor<13x27xf32>
// CHECK-NEXT:  }