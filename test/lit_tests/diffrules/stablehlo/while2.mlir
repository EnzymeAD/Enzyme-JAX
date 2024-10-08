// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_dup argTys=enzyme_dup,enzyme_dup mode=ForwardMode" | FileCheck %s --check-prefix=FORWARD
// TODO: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_active argTys=enzyme_active,enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops | FileCheck %s --check-prefix=REVERSE

module {
  func.func public @main(%arg5: tensor<5xf32>, %1: tensor<f32>) -> (tensor<f32>) {
    %c_2 = stablehlo.constant dense<1> : tensor<i64>
    %c_3 = stablehlo.constant dense<0> : tensor<i64>
    %2:3 = stablehlo.while(%iterArg_6 = %arg5, %iterArg_7 = %c_3, %iterArg_8 = %1) : tensor<5xf32>, tensor<i64>, tensor<f32>
     cond {
      %18 = stablehlo.compare  LT, %iterArg_7, %c_2,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %18 : tensor<i1>
    } do {
      %29 = stablehlo.dynamic_slice %iterArg_6, %c_3, sizes = [1] : (tensor<5xf32>, tensor<i64>) -> tensor<1xf32>
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
// FORWARD-NEXT:      %1 = stablehlo.dynamic_slice %iterArg_1, %c_0, sizes = [1] : (tensor<5xf32>, tensor<i64>) -> tensor<1xf32>
// FORWARD-NEXT:      %2 = stablehlo.dynamic_slice %iterArg, %c_0, sizes = [1] : (tensor<5xf32>, tensor<i64>) -> tensor<1xf32>
// FORWARD-NEXT:      %3 = stablehlo.reshape %1 : (tensor<1xf32>) -> tensor<f32>
// FORWARD-NEXT:      %4 = stablehlo.reshape %2 : (tensor<1xf32>) -> tensor<f32>
// FORWARD-NEXT:      stablehlo.return %iterArg, %iterArg_1, %c_0, %4, %3 : tensor<5xf32>, tensor<5xf32>, tensor<i64>, tensor<f32>, tensor<f32>
// FORWARD-NEXT:    }
// FORWARD-NEXT:    return %0#3, %0#4 : tensor<f32>, tensor<f32>
// FORWARD-NEXT:  }
