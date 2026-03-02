// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= argTys=enzyme_active,enzyme_active retTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --arith-raise --enzyme-hlo-opt | FileCheck %s

module {
  func.func @main(%a: tensor<9xf32>, %init: tensor<f32>) -> tensor<f32> {
    %6 = "stablehlo.reduce"(%a, %init) <{dimensions = array<i64: 0>}> ({
        ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
        %8 = "stablehlo.multiply"(%arg2, %arg3) : (tensor<f32>, tensor<f32>) -> tensor<f32>
        "stablehlo.return"(%8) : (tensor<f32>) -> ()
        }) : (tensor<9xf32>, tensor<f32>) -> tensor<f32>
    return %6 : tensor<f32>
  }
}



// CHECK:  func.func @main(%arg0: tensor<9xf32>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> (tensor<9xf32>, tensor<f32>) {
// CHECK-NEXT:    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.multiply across dimensions = [0] : (tensor<9xf32>, tensor<f32>) -> tensor<f32>
// CHECK-NEXT:    %1 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f32>) -> tensor<9xf32>
// CHECK-NEXT:    %2 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f32>) -> tensor<9xf32>
// CHECK-NEXT:    %3 = stablehlo.divide %2, %arg0 : tensor<9xf32>
// CHECK-NEXT:    %4 = stablehlo.multiply %1, %3 : tensor<9xf32>
// CHECK-NEXT:    %5 = stablehlo.divide %0, %arg1 : tensor<f32>
// CHECK-NEXT:    %6 = stablehlo.multiply %5, %arg2 : tensor<f32>
// CHECK-NEXT:    return %4, %6 : tensor<9xf32>, tensor<f32>
// CHECK-NEXT:  }
