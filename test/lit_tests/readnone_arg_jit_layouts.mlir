// RUN: enzymexlamlir-opt %s --canonicalize | FileCheck %s

// CHECK: enzymexla.jit_call @callee (%arg0, %arg1) <operand_layouts = [dense<[1, 0]> : tensor<2xindex>, dense<0> : tensor<1xindex>]
// CHECK-SAME: : (tensor<4x4xf32>, tensor<4xf32>) -> tensor<4xf32>

module {
  llvm.func @callee(%a: !llvm.ptr, %dead: !llvm.ptr, %b: !llvm.ptr) {
    llvm.store %b, %a : !llvm.ptr, !llvm.ptr
    llvm.return
  }
  func.func @main(%arg0: tensor<4x4xf32>, %arg1: tensor<4xf32>, %arg2: tensor<4xf32>) -> tensor<4xf32> {
    %0 = enzymexla.jit_call @callee (%arg0, %arg2, %arg1) <{operand_layouts = [dense<[1, 0]> : tensor<2xindex>, dense<[0]> : tensor<1xindex>, dense<[0]> : tensor<1xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 2, operand_tuple_indices = []>]}> : (tensor<4x4xf32>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    return %0 : tensor<4xf32>
  }
}
