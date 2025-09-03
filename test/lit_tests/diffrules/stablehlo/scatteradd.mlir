// TODO: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_dup argTys=enzyme_dup,enzyme_dup mode=ForwardMode" | FileCheck %s --check-prefix=FORWARD
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_active argTys=enzyme_active,enzyme_dup mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops | FileCheck %s --check-prefix=REVERSE

module {
  func.func public @main(%arg0: tensor<2x5xf32>, %arg1: tensor<2x5xf32>) -> (tensor<2x5xf32>) {
    %c = stablehlo.constant dense<[[3, 1, 0, 4, 2]]> : tensor<1x5xi32>
    %2 = "stablehlo.scatter"(%arg0, %c, %arg1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>) :
      %5 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %5 : tensor<f32>
    }) : (tensor<2x5xf32>, tensor<1x5xi32>, tensor<2x5xf32>) -> tensor<2x5xf32>
    return %2 : tensor<2x5xf32>
  }
}

// REVERSE: func.func @main(%arg0: tensor<2x5xf32>, %arg1: tensor<2x5xf32>, %arg2: tensor<2x5xf32>, %arg3: tensor<2x5xf32>) -> tensor<2x5xf32> {
// REVERSE-NEXT:    %c = stablehlo.constant dense<{{\[\[}}3, 1, 0, 4, 2{{\]\]}}> : tensor<1x5xi32>
// REVERSE-NEXT:    %cst = arith.constant dense<0.000000e+00> : tensor<2x5xf32>
// REVERSE-NEXT:    %0 = arith.addf %arg3, %cst : tensor<2x5xf32>
// REVERSE-NEXT:    %1 = "stablehlo.gather"(%0, %c) <{dimension_numbers = #stablehlo.gather<offset_dims = [0], collapsed_slice_dims = [1], start_index_map = [1]>, slice_sizes = array<i64: 2, 1>}> : (tensor<2x5xf32>, tensor<1x5xi32>) -> tensor<2x5xf32>
// REVERSE-NEXT:      %2 = arith.addf %1, %cst : tensor<2x5xf32>
// REVERSE-NEXT:      return %2 : tensor<2x5xf32>
// REVERSE-NEXT:    }
