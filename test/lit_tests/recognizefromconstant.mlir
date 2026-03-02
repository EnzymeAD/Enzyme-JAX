// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt="max_constant_expansion=1" | FileCheck %s

func.func @detect_iota() -> tensor<4x5x6xf64> {
    %cst = stablehlo.constant dense<"0x0000000000001440000000000000144000000000000014400000000000001440000000000000144000000000000014400000000000001840000000000000184000000000000018400000000000001840000000000000184000000000000018400000000000001C400000000000001C400000000000001C400000000000001C400000000000001C400000000000001C400000000000002040000000000000204000000000000020400000000000002040000000000000204000000000000020400000000000002240000000000000224000000000000022400000000000002240000000000000224000000000000022400000000000001440000000000000144000000000000014400000000000001440000000000000144000000000000014400000000000001840000000000000184000000000000018400000000000001840000000000000184000000000000018400000000000001C400000000000001C400000000000001C400000000000001C400000000000001C400000000000001C400000000000002040000000000000204000000000000020400000000000002040000000000000204000000000000020400000000000002240000000000000224000000000000022400000000000002240000000000000224000000000000022400000000000001440000000000000144000000000000014400000000000001440000000000000144000000000000014400000000000001840000000000000184000000000000018400000000000001840000000000000184000000000000018400000000000001C400000000000001C400000000000001C400000000000001C400000000000001C400000000000001C400000000000002040000000000000204000000000000020400000000000002040000000000000204000000000000020400000000000002240000000000000224000000000000022400000000000002240000000000000224000000000000022400000000000001440000000000000144000000000000014400000000000001440000000000000144000000000000014400000000000001840000000000000184000000000000018400000000000001840000000000000184000000000000018400000000000001C400000000000001C400000000000001C400000000000001C400000000000001C400000000000001C40000000000000204000000000000020400000000000002040000000000000204000000000000020400000000000002040000000000000224000000000000022400000000000002240000000000000224000000000000022400000000000002240"> : tensor<4x5x6xf64>
    return %cst : tensor<4x5x6xf64>
}

// CHECK: func.func @detect_iota() -> tensor<4x5x6xf64> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<5.000000e+00> : tensor<4x5x6xf64>
// CHECK-NEXT:     %0 = stablehlo.iota dim = 1 : tensor<4x5x6xf64>
// CHECK-NEXT:     %1 = stablehlo.add %0, %cst : tensor<4x5x6xf64>
// CHECK-NEXT:     return %1 : tensor<4x5x6xf64>
// CHECK-NEXT: }

func.func @detect_splatted_constant() -> tensor<4x5x6xf64> {
    %cst = stablehlo.constant dense<"0x000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F"> : tensor<4x5x6xf64>
    return %cst : tensor<4x5x6xf64>
}

// CHECK: func.func @detect_splatted_constant() -> tensor<4x5x6xf64> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<1.000000e+00> : tensor<4x5x6xf64>
// CHECK-NEXT:     return %cst : tensor<4x5x6xf64>
// CHECK-NEXT: }

// Test: Padded tensor with zeros at the end (high padding)
// Data is [1.0, 2.0], followed by 6 zeros -> tensor<8xf64>
func.func @detect_padded_high() -> tensor<8xf64> {
    // 1.0, 2.0 followed by 6 zeros
    %cst = stablehlo.constant dense<[1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]> : tensor<8xf64>
    return %cst : tensor<8xf64>
}

// CHECK: func.func @detect_padded_high() -> tensor<8xf64> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:     %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<2xf64>
// CHECK-NEXT:     %0 = stablehlo.iota dim = 0 : tensor<2xf64>
// CHECK-NEXT:     %1 = stablehlo.add %0, %cst_0 : tensor<2xf64>
// CHECK-NEXT:     %2 = stablehlo.pad %1, %cst, low = [0], high = [6], interior = [0] : (tensor<2xf64>, tensor<f64>) -> tensor<8xf64>
// CHECK-NEXT:     return %2 : tensor<8xf64>
// CHECK-NEXT: }

// Test: Padded tensor with zeros at the start (low padding)
// 6 zeros followed by data [1.0, 2.0] -> tensor<8xf64>
func.func @detect_padded_low() -> tensor<8xf64> {
    // 6 zeros followed by 1.0, 2.0
    %cst = stablehlo.constant dense<[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0]> : tensor<8xf64>
    return %cst : tensor<8xf64>
}

// CHECK: func.func @detect_padded_low() -> tensor<8xf64> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:     %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<2xf64>
// CHECK-NEXT:     %0 = stablehlo.iota dim = 0 : tensor<2xf64>
// CHECK-NEXT:     %1 = stablehlo.add %0, %cst_0 : tensor<2xf64>
// CHECK-NEXT:     %2 = stablehlo.pad %1, %cst, low = [6], high = [0], interior = [0] : (tensor<2xf64>, tensor<f64>) -> tensor<8xf64>
// CHECK-NEXT:     return %2 : tensor<8xf64>
// CHECK-NEXT: }

// Test: 2D padded tensor - 2x5 with padding in second dimension
// Data is [[1,2],[3,4]] (2x2), padded to 2x5 with zeros on the right
func.func @detect_padded_2d() -> tensor<2x5xf64> {
    // Row 0: 1.0, 2.0, 0, 0, 0
    // Row 1: 3.0, 4.0, 0, 0, 0
    // [[1, 2, 0, 0, 0], [3, 4, 0, 0, 0]]
    %cst = stablehlo.constant dense<[[1.0, 2.0, 0.0, 0.0, 0.0], [3.0, 4.0, 0.0, 0.0, 0.0]]> : tensor<2x5xf64>
    return %cst : tensor<2x5xf64>
}

// CHECK: func.func @detect_padded_2d() -> tensor<2x5xf64> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<{{\[\[}}1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]]> : tensor<2x2xf64>
// CHECK-NEXT:     %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:     %0 = stablehlo.pad %cst, %cst_0, low = [0, 0], high = [0, 3], interior = [0, 0] : (tensor<2x2xf64>, tensor<f64>) -> tensor<2x5xf64>
// CHECK-NEXT:     return %0 : tensor<2x5xf64>
// CHECK-NEXT: }

// Test: 2D padded tensor with low padding - 2x5 with zeros on the left
// Data is [[1,2],[3,4]] (2x2), padded with zeros on the left
func.func @detect_padded_2d_low() -> tensor<2x5xf64> {
    // Row 0: 0, 0, 0, 1.0, 2.0
    // [[0, 0, 0, 1, 2], [0, 0, 0, 3, 4]]
    %cst = stablehlo.constant dense<[[0.0, 0.0, 0.0, 1.0, 2.0], [0.0, 0.0, 0.0, 3.0, 4.0]]> : tensor<2x5xf64>
    return %cst : tensor<2x5xf64>
}

// CHECK: func.func @detect_padded_2d_low() -> tensor<2x5xf64> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<{{\[\[}}1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]]> : tensor<2x2xf64>
// CHECK-NEXT:     %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:     %0 = stablehlo.pad %cst, %cst_0, low = [0, 3], high = [0, 0], interior = [0, 0] : (tensor<2x2xf64>, tensor<f64>) -> tensor<2x5xf64>
// CHECK-NEXT:     return %0 : tensor<2x5xf64>
// CHECK-NEXT: }

// Test: 2D padded in first dimension - 5x2 with zeros at the top
func.func @detect_padded_2d_dim0() -> tensor<5x2xf64> {
    // Row 0: 0, 0
    // Row 1: 0, 0
    // Row 2: 0, 0
    // [[0, 0], [0, 0], [0, 0], [1, 2], [3, 4]]
    %cst = stablehlo.constant dense<[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [1.0, 2.0], [3.0, 4.0]]> : tensor<5x2xf64>
    return %cst : tensor<5x2xf64>
}

// CHECK: func.func @detect_padded_2d_dim0() -> tensor<5x2xf64> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<{{\[\[}}1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]]> : tensor<2x2xf64>
// CHECK-NEXT:     %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:     %0 = stablehlo.pad %cst, %cst_0, low = [3, 0], high = [0, 0], interior = [0, 0] : (tensor<2x2xf64>, tensor<f64>) -> tensor<5x2xf64>
// CHECK-NEXT:     return %0 : tensor<5x2xf64>
// CHECK-NEXT: }

// Test: 2D padded tensor with padding on ALL sides - 5x5 with 2x2 data in the center
// The 2x2 data block [[1,2],[3,4]] is at position [1:3, 1:3] surrounded by zeros
func.func @detect_padded_all_sides() -> tensor<5x5xf64> {
    // Row 0: 0, 0, 0, 0, 0
    // Row 1: 0, 1, 2, 0, 0
    // Row 2: 0, 3, 4, 0, 0
    // Row 3: 0, 0, 0, 0, 0
    // Row 4: 0, 0, 0, 0, 0
    // [[0,0,0,0,0], [0,1,2,0,0], [0,3,4,0,0], [0,0,0,0,0], [0,0,0,0,0]]
    %cst = stablehlo.constant dense<[[0.0, 0.0, 0.0, 0.0, 0.0],
                                     [0.0, 1.0, 2.0, 0.0, 0.0],
                                     [0.0, 3.0, 4.0, 0.0, 0.0],
                                     [0.0, 0.0, 0.0, 0.0, 0.0],
                                     [0.0, 0.0, 0.0, 0.0, 0.0]]> : tensor<5x5xf64>
    return %cst : tensor<5x5xf64>
}

// CHECK: func.func @detect_padded_all_sides() -> tensor<5x5xf64> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<{{\[\[}}1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]]> : tensor<2x2xf64>
// CHECK-NEXT:     %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:     %0 = stablehlo.pad %cst, %cst_0, low = [1, 1], high = [2, 2], interior = [0, 0] : (tensor<2x2xf64>, tensor<f64>) -> tensor<5x5xf64>
// CHECK-NEXT:     return %0 : tensor<5x5xf64>
// CHECK-NEXT: }
