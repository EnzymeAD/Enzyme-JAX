// RUN: enzymexlamlir-opt --fuse-jit="dag=true top_down=false" %s | FileCheck %s
// RUN: enzymexlamlir-opt --fuse-jit="dag=true top_down=true" %s | FileCheck %s

// A non-JIT user between the fused calls must still be dominated by the fused
// call, whichever group member the rewrite is matched on.

module {
  llvm.func @k1(%a: !llvm.ptr) {
    llvm.return
  }
  llvm.func @k2(%a: !llvm.ptr, %b: !llvm.ptr) {
    llvm.return
  }
  func.func @main(%x: tensor<2x3xf32>, %y: tensor<2x3xf32>) -> (tensor<3x2xf32>, tensor<2x3xf32>) {
    %0 = enzymexla.jit_call @k1 (%x) <output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>]> : (tensor<2x3xf32>) -> tensor<2x3xf32>
    %t = stablehlo.transpose %0, dims = [1, 0] : (tensor<2x3xf32>) -> tensor<3x2xf32>
    %1 = enzymexla.jit_call @k2 (%y, %0) <output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>]> : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
    return %t, %1 : tensor<3x2xf32>, tensor<2x3xf32>
  }
}

// CHECK: func.func @main(%[[X:.*]]: tensor<2x3xf32>, %[[Y:.*]]: tensor<2x3xf32>)
// CHECK-NEXT: %[[RES:.*]]:2 = enzymexla.jit_call @fused__k1_k2 (%[[X]], %[[Y]]) <output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [1], operand_index = 1, operand_tuple_indices = []>]>
// CHECK-NEXT: %[[T:.*]] = stablehlo.transpose %[[RES]]#0, dims = [1, 0]
// CHECK-NEXT: return %[[T]], %[[RES]]#1
