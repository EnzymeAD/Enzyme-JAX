// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_dup argTys=enzyme_dup,enzyme_dup mode=ForwardMode" | FileCheck %s --check-prefix=FORWARD
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_active argTys=enzyme_active,enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops | FileCheck %s --check-prefix=REVERSE
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_active argTys=enzyme_const,enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops | FileCheck %s --check-prefix=REVERSE2

module {
  func.func @main(%prev : tensor<1x1x8192x16x256xbf16>, %update : tensor<1x1x2048x16x256xbf16>) -> tensor<1x1x8192x16x256xbf16> {
    %163 = stablehlo.constant dense<0> : tensor<1xi32>
    %524 = "stablehlo.scatter"(%prev, %163, %update) ({
    ^bb0(%arg113: tensor<bf16>, %arg114: tensor<bf16>):
      stablehlo.return %arg114 : tensor<bf16>
    }) {indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1, 2, 3, 4], scatter_dims_to_operand_dims = [2]>, unique_indices = true} : (tensor<1x1x8192x16x256xbf16>, tensor<1xi32>, tensor<1x1x2048x16x256xbf16>) -> tensor<1x1x8192x16x256xbf16>
    return %524 : tensor<1x1x8192x16x256xbf16>
  }
}

// FORWARD: func.func @main(%arg0: tensor<1x1x8192x16x256xbf16>, %arg1: tensor<1x1x8192x16x256xbf16>, %arg2: tensor<1x1x2048x16x256xbf16>, %arg3: tensor<1x1x2048x16x256xbf16>) -> (tensor<1x1x8192x16x256xbf16>, tensor<1x1x8192x16x256xbf16>) {
// FORWARD-NEXT:    %c = stablehlo.constant dense<0> : tensor<1xi32>
// FORWARD-NEXT:    %0:2 = "stablehlo.scatter"(%arg0, %arg1, %c, %arg2, %arg3) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1, 2, 3, 4], scatter_dims_to_operand_dims = [2]>, unique_indices = true}> ({
// FORWARD-NEXT:    ^bb0(%arg4: tensor<bf16>, %arg5: tensor<bf16>, %arg6: tensor<bf16>, %arg7: tensor<bf16>):
// FORWARD-NEXT:      stablehlo.return %arg6, %arg7 : tensor<bf16>, tensor<bf16>
// FORWARD-NEXT:    }) : (tensor<1x1x8192x16x256xbf16>, tensor<1x1x8192x16x256xbf16>, tensor<1xi32>, tensor<1x1x2048x16x256xbf16>, tensor<1x1x2048x16x256xbf16>) -> (tensor<1x1x8192x16x256xbf16>, tensor<1x1x8192x16x256xbf16>)
// FORWARD-NEXT:    return %0#0, %0#1 : tensor<1x1x8192x16x256xbf16>, tensor<1x1x8192x16x256xbf16>
// FORWARD-NEXT:  }

// REVERSE:  func.func @main(%arg0: tensor<1x1x8192x16x256xbf16>, %arg1: tensor<1x1x2048x16x256xbf16>, %arg2: tensor<1x1x8192x16x256xbf16>) -> (tensor<1x1x8192x16x256xbf16>, tensor<1x1x2048x16x256xbf16>) {
// REVERSE-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
// REVERSE-NEXT:    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<1x1x2048x16x256xbf16>
// REVERSE-NEXT:    %c = stablehlo.constant dense<0> : tensor<1xi32>
// REVERSE-NEXT:    %cst_1 = arith.constant dense<0.000000e+00> : tensor<1x1x8192x16x256xbf16>
// REVERSE-NEXT:    %cst_2 = arith.constant dense<0.000000e+00> : tensor<1x1x2048x16x256xbf16>
// REVERSE-NEXT:    %0 = arith.addf %arg2, %cst_1 : tensor<1x1x8192x16x256xbf16>
// REVERSE-NEXT:    %1 = "stablehlo.scatter"(%0, %c, %cst_0) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1, 2, 3, 4], scatter_dims_to_operand_dims = [2]>, unique_indices = true}> ({
// REVERSE-NEXT:    ^bb0(%arg3: tensor<bf16>, %arg4: tensor<bf16>):
// REVERSE-NEXT:      stablehlo.return %cst : tensor<bf16>
// REVERSE-NEXT:    }) : (tensor<1x1x8192x16x256xbf16>, tensor<1xi32>, tensor<1x1x2048x16x256xbf16>) -> tensor<1x1x8192x16x256xbf16>
// REVERSE-NEXT:    %2 = arith.addf %1, %cst_1 : tensor<1x1x8192x16x256xbf16>
// REVERSE-NEXT:    %3 = "stablehlo.gather"(%0, %c) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3, 4], start_index_map = [2]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 1, 2048, 16, 256>}> : (tensor<1x1x8192x16x256xbf16>, tensor<1xi32>) -> tensor<1x1x2048x16x256xbf16>
// REVERSE-NEXT:    %4 = arith.addf %3, %cst_2 : tensor<1x1x2048x16x256xbf16>
// REVERSE-NEXT:    return %2, %4 : tensor<1x1x8192x16x256xbf16>, tensor<1x1x2048x16x256xbf16>
// REVERSE-NEXT:  }

// REVERSE2: func.func @main(%arg0: tensor<1x1x8192x16x256xbf16>, %arg1: tensor<1x1x2048x16x256xbf16>, %arg2: tensor<1x1x8192x16x256xbf16>) -> tensor<1x1x2048x16x256xbf16> {
// REVERSE2-NEXT:     %c = stablehlo.constant dense<0> : tensor<1xi32>
// REVERSE2-NEXT:     %cst = arith.constant dense<0.000000e+00> : tensor<1x1x8192x16x256xbf16>
// REVERSE2-NEXT:     %cst_0 = arith.constant dense<0.000000e+00> : tensor<1x1x2048x16x256xbf16>
// REVERSE2-NEXT:     %0 = arith.addf %arg2, %cst : tensor<1x1x8192x16x256xbf16>
// REVERSE2-NEXT:     %1 = "stablehlo.gather"(%0, %c) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3, 4], start_index_map = [2]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 1, 2048, 16, 256>}> : (tensor<1x1x8192x16x256xbf16>, tensor<1xi32>) -> tensor<1x1x2048x16x256xbf16>
// REVERSE2-NEXT:     %2 = arith.addf %1, %cst_0 : tensor<1x1x2048x16x256xbf16>
// REVERSE2-NEXT:     return %2 : tensor<1x1x2048x16x256xbf16>
// REVERSE2-NEXT: }
