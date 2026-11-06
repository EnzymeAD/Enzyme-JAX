// RUN: enzymexlamlir-opt --fuse-jit="dag=true" %s | FileCheck %s

// Fused arguments keep the address space of the wrapper arguments bound to
// them, and wrappers expecting another one cast to it.

module {
  llvm.func private @first(%arg0: !llvm.ptr<1>) {
    %0 = llvm.mlir.constant(1.0 : f32) : f32
    llvm.store %0, %arg0 : f32, !llvm.ptr<1>
    llvm.return
  }
  llvm.func private @second(%arg0: !llvm.ptr) {
    %0 = llvm.mlir.constant(2.0 : f32) : f32
    llvm.store %0, %arg0 : f32, !llvm.ptr
    llvm.return
  }
  func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    %0 = enzymexla.jit_call @first(%arg0) <
      output_operand_aliases = [
        #stablehlo.output_operand_alias<output_tuple_indices = [],
                                        operand_index = 0,
                                        operand_tuple_indices = []>
      ]
    > : (tensor<4xf32>) -> tensor<4xf32>
    %1 = enzymexla.jit_call @second(%0) <
      output_operand_aliases = [
        #stablehlo.output_operand_alias<output_tuple_indices = [],
                                        operand_index = 0,
                                        operand_tuple_indices = []>
      ]
    > : (tensor<4xf32>) -> tensor<4xf32>
    return %1 : tensor<4xf32>
  }
}

// CHECK:       llvm.func private @fused__first_second(%[[ARG:[^:]+]]: !llvm.ptr<1>) attributes {sym_visibility = "private"} {
// CHECK:         llvm.store %{{.+}}, %[[ARG]] : f32, !llvm.ptr<1>
// CHECK:         %[[CAST:.+]] = llvm.addrspacecast %[[ARG]] : !llvm.ptr<1> to !llvm.ptr
// CHECK:         llvm.store %{{.+}}, %[[CAST]] : f32, !llvm.ptr
// CHECK:         llvm.return

// CHECK-LABEL: func.func @main(
// CHECK-SAME:      %[[A:.+]]: tensor<4xf32>)
// CHECK-NEXT:    %[[R:.+]] = enzymexla.jit_call @fused__first_second (%[[A]])
// CHECK-NEXT:    return %[[R]]
