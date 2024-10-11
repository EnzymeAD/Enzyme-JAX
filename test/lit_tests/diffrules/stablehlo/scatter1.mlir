// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_dup argTys=enzyme_dup mode=ForwardMode" | FileCheck %s --check-prefix=FORWARD
// TODO: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_active argTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops | FileCheck %s --check-prefix=REVERSE

func.func @main(%x : tensor<5xf32>) -> tensor<5xf32> {
  %c_30 = stablehlo.constant dense<0> : tensor<1xi32>
  %cst_65 = stablehlo.constant dense<1.920000e+05> : tensor<f32>
  %res = "stablehlo.scatter"(%x, %c_30, %cst_65) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg8: tensor<f32>, %arg9: tensor<f32>):
      %153 = stablehlo.multiply %arg8, %arg9 : tensor<f32>
      stablehlo.return %153 : tensor<f32>
  }) : (tensor<5xf32>, tensor<1xi32>, tensor<f32>) -> tensor<5xf32>
  func.return %res : tensor<5xf32>
}

// FORWARD:  func.func @main(%arg0: tensor<5xf32>, %arg1: tensor<5xf32>) -> (tensor<5xf32>, tensor<5xf32>) {
// FORWARD-NEXT:    %c = stablehlo.constant dense<0> : tensor<1xi32>
// FORWARD-NEXT:    %cst = stablehlo.constant dense<1.920000e+05> : tensor<f32>
// FORWARD-NEXT:    %cst_0 = arith.constant dense<0.000000e+00> : tensor<f32>
// FORWARD-NEXT:    %0:2 = "stablehlo.scatter"(%arg0, %arg1, %c, %cst, %cst_0) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
// FORWARD-NEXT:    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>, %arg4: tensor<f32>, %arg5: tensor<f32>):
// FORWARD-NEXT:      %1 = stablehlo.multiply %arg3, %arg4 : tensor<f32>
// FORWARD-NEXT:      %2 = stablehlo.multiply %arg2, %arg4 : tensor<f32>
// FORWARD-NEXT:      stablehlo.return %2, %1 : tensor<f32>, tensor<f32>
// FORWARD-NEXT:    }) : (tensor<5xf32>, tensor<5xf32>, tensor<1xi32>, tensor<f32>, tensor<f32>) -> (tensor<5xf32>, tensor<5xf32>)
// FORWARD-NEXT:    return %0#0, %0#1 : tensor<5xf32>, tensor<5xf32>
// FORWARD-NEXT:  }

// REVERSE:  func.func @main(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>) -> tensor<2xf32> {
// REVERSE-NEXT:    %cst = arith.constant dense<0.000000e+00> : tensor<2xf32>
// REVERSE-NEXT:    %0 = arith.addf %arg1, %cst : tensor<2xf32>
// REVERSE-NEXT:    %1 = stablehlo.cosine %arg0 : tensor<2xf32>
// REVERSE-NEXT:    %2 = stablehlo.multiply %0, %1 : tensor<2xf32>
// REVERSE-NEXT:    %3 = arith.addf %2, %cst : tensor<2xf32>
// REVERSE-NEXT:    return %3 : tensor<2xf32>
// REVERSE-NEXT:  }
