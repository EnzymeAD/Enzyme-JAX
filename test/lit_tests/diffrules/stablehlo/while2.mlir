// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_dup argTys=enzyme_dup,enzyme_dup mode=ForwardMode" | FileCheck %s --check-prefix=FORWARD
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_active argTys=enzyme_active,enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops | FileCheck %s --check-prefix=REVERSE

module {
  func.func public @main(%arg5: tensor<5xf32>, %1: tensor<f32>) -> (tensor<f32>) {
    %c_2 = stablehlo.constant dense<1> : tensor<i64>
    %c_3 = stablehlo.constant dense<0> : tensor<i64>
    %2:3 = stablehlo.while(%iterArg_6 = %arg5, %iterArg_7 = %c_3, %iterArg_8 = %1) : tensor<5xf32>, tensor<i64>, tensor<f32>
     cond {
      %18 = stablehlo.compare  LT, %iterArg_7, %c_2,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %18 : tensor<i1>
    } do {
      // %29 = stablehlo.dynamic_slice %iterArg_6, %c_3, sizes = [1] : (tensor<5xf32>, tensor<i64>) -> tensor<1xf32>
      %29 = "stablehlo.slice"(%iterArg_6) {
        start_indices = array<i64: 0>,
        limit_indices = array<i64: 1>,
        strides = array<i64: 1>
      } : (tensor<5xf32>) -> tensor<1xf32>
      %30 = stablehlo.reshape %29 : (tensor<1xf32>) -> tensor<f32>
      stablehlo.return %iterArg_6, %c_3, %30 : tensor<5xf32>, tensor<i64>, tensor<f32>
    }
    return %2#2 : tensor<f32>
  }
}

// FORWARD:  func.func @main(%arg0: tensor<5xf32>, %arg1: tensor<5xf32>, %arg2: tensor<f32>, %arg3: tensor<f32>) -> (tensor<f32>, tensor<f32>)
// FORWARD-NEXT:    %c = stablehlo.constant dense<1> : tensor<i64>
// FORWARD-NEXT:    %c_0 = stablehlo.constant dense<0> : tensor<i64>
// FORWARD-NEXT:    %0:5 = stablehlo.while(%iterArg = %arg0, %iterArg_1 = %arg1, %iterArg_2 = %c_0, %iterArg_3 = %arg2, %iterArg_4 = %arg3) : tensor<5xf32>, tensor<5xf32>, tensor<i64>, tensor<f32>, tensor<f32>
// FORWARD-NEXT:     cond {
// FORWARD-NEXT:      %1 = stablehlo.compare  LT, %iterArg_2, %c,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
// FORWARD-NEXT:      stablehlo.return %1 : tensor<i1>
// FORWARD-NEXT:    } do {
// FORWARD-NEXT:      %1 = stablehlo.slice %iterArg_1 [0:1] : (tensor<5xf32>) -> tensor<1xf32>
// FORWARD-NEXT:      %2 = stablehlo.slice %iterArg [0:1] : (tensor<5xf32>) -> tensor<1xf32>
// FORWARD-NEXT:      %3 = stablehlo.reshape %1 : (tensor<1xf32>) -> tensor<f32>
// FORWARD-NEXT:      %4 = stablehlo.reshape %2 : (tensor<1xf32>) -> tensor<f32>
// FORWARD-NEXT:      stablehlo.return %iterArg, %iterArg_1, %c_0, %4, %3 : tensor<5xf32>, tensor<5xf32>, tensor<i64>, tensor<f32>, tensor<f32>
// FORWARD-NEXT:    }
// FORWARD-NEXT:    return %0#3, %0#4 : tensor<f32>, tensor<f32>
// FORWARD-NEXT:  }

// REVERSE:  func.func @main(%arg0: tensor<5xf32>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> (tensor<5xf32>, tensor<f32>) {
// REVERSE-NEXT:    %c = stablehlo.constant dense<0> : tensor<i64>
// REVERSE-NEXT:    %c_0 = stablehlo.constant dense<1> : tensor<i64>
// REVERSE-NEXT:    %cst = arith.constant dense<0.000000e+00> : tensor<1xf32>
// REVERSE-NEXT:    %cst_1 = arith.constant dense<0.000000e+00> : tensor<5xf32>
// REVERSE-NEXT:    %cst_2 = arith.constant dense<0.000000e+00> : tensor<f32>
// REVERSE-NEXT:    %0:4 = stablehlo.while(%iterArg = %arg0, %iterArg_3 = %c, %iterArg_4 = %arg1, %iterArg_5 = %c) : tensor<5xf32>, tensor<i64>, tensor<f32>, tensor<i64>
// REVERSE-NEXT:     cond {
// REVERSE-NEXT:      %5 = stablehlo.compare  LT, %iterArg_3, %c_0,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
// REVERSE-NEXT:      stablehlo.return %5 : tensor<i1>
// REVERSE-NEXT:    } do {
// REVERSE-NEXT:      %5 = stablehlo.add %iterArg_5, %c_0 : tensor<i64>
// REVERSE-NEXT:      %6 = stablehlo.slice %iterArg [0:1] : (tensor<5xf32>) -> tensor<1xf32>
// REVERSE-NEXT:      %7 = stablehlo.reshape %6 : (tensor<1xf32>) -> tensor<f32>
// REVERSE-NEXT:      stablehlo.return %iterArg, %c, %7, %5 : tensor<5xf32>, tensor<i64>, tensor<f32>, tensor<i64>
// REVERSE-NEXT:    }
// REVERSE-NEXT:    %1 = arith.addf %arg2, %cst_2 : tensor<f32>
// REVERSE-NEXT:    %2:3 = stablehlo.while(%iterArg = %c, %iterArg_3 = %cst_1, %iterArg_4 = %1) : tensor<i64>, tensor<5xf32>, tensor<f32>
// REVERSE-NEXT:     cond {
// REVERSE-NEXT:      %5 = stablehlo.compare  LT, %iterArg, %0#3 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// REVERSE-NEXT:      stablehlo.return %5 : tensor<i1>
// REVERSE-NEXT:    } do {
// REVERSE-NEXT:      %5 = stablehlo.add %iterArg, %c_0 : tensor<i64>
// REVERSE-NEXT:      %6 = stablehlo.reshape %iterArg_4 : (tensor<f32>) -> tensor<1xf32>
// REVERSE-NEXT:      %7 = arith.addf %6, %cst : tensor<1xf32>
// REVERSE-NEXT:      %8 = stablehlo.pad %7, %cst_2, low = [0], high = [4], interior = [0] : (tensor<1xf32>, tensor<f32>) -> tensor<5xf32>
// REVERSE-NEXT:      %9 = arith.addf %iterArg_3, %8 : tensor<5xf32>
// REVERSE-NEXT:      stablehlo.return %5, %9, %cst_2 : tensor<i64>, tensor<5xf32>, tensor<f32>
// REVERSE-NEXT:    }
// REVERSE-NEXT:    %3 = arith.addf %2#1, %cst_1 : tensor<5xf32>
// REVERSE-NEXT:    %4 = arith.addf %2#2, %cst_2 : tensor<f32>
// REVERSE-NEXT:    return %3, %4 : tensor<5xf32>, tensor<f32>
// REVERSE-NEXT:  }
