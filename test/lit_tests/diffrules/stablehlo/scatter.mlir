// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_dup argTys=enzyme_dup,enzyme_dup mode=ForwardMode" | FileCheck %s --check-prefix=FORWARD
// TODO: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_active argTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops | FileCheck %s --check-prefix=REVERSE

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
