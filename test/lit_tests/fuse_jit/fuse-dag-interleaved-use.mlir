// RUN: enzymexlamlir-opt --fuse-jit="dag=true" %s | FileCheck %s
// RUN: enzymexlamlir-opt --fuse-jit="dag=true top_down=true" %s | FileCheck %s

module {
  llvm.func private @first(%arg0: !llvm.ptr) {
    %0 = llvm.mlir.constant(1.0 : f32) : f32
    llvm.store %0, %arg0 : f32, !llvm.ptr
    llvm.return
  }
  llvm.func private @second(%arg0: !llvm.ptr) {
    %0 = llvm.mlir.constant(2.0 : f32) : f32
    llvm.store %0, %arg0 : f32, !llvm.ptr
    llvm.return
  }
  llvm.func private @copy(%arg0: !llvm.ptr, %arg1: !llvm.ptr) {
    %0 = llvm.load %arg0 : !llvm.ptr -> f32
    llvm.store %0, %arg1 : f32, !llvm.ptr
    llvm.return
  }

  // @second overwrites the result of @first, which the convert still reads:
  // fusing them would make the convert read the value @second wrote.
  func.func @clobbered(%arg0: tensor<4xf32>) -> (tensor<4xf64>, tensor<4xf32>) {
    %0 = enzymexla.jit_call @first(%arg0) <
      output_operand_aliases = [
        #stablehlo.output_operand_alias<output_tuple_indices = [],
                                        operand_index = 0,
                                        operand_tuple_indices = []>
      ]
    > : (tensor<4xf32>) -> tensor<4xf32>
    %1 = stablehlo.convert %0 : (tensor<4xf32>) -> tensor<4xf64>
    %2 = enzymexla.jit_call @second(%0) <
      output_operand_aliases = [
        #stablehlo.output_operand_alias<output_tuple_indices = [],
                                        operand_index = 0,
                                        operand_tuple_indices = []>
      ]
    > : (tensor<4xf32>) -> tensor<4xf32>
    return %1, %2 : tensor<4xf64>, tensor<4xf32>
  }

  // @copy only reads the result of @first, so the calls fuse. The fused call
  // replaces them at the first one, which dominates the convert, whichever
  // call seeds the fusion.
  func.func @interleaved(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> (tensor<4xf64>, tensor<4xf32>) {
    %0 = enzymexla.jit_call @first(%arg0) <
      output_operand_aliases = [
        #stablehlo.output_operand_alias<output_tuple_indices = [],
                                        operand_index = 0,
                                        operand_tuple_indices = []>
      ]
    > : (tensor<4xf32>) -> tensor<4xf32>
    %1 = stablehlo.convert %0 : (tensor<4xf32>) -> tensor<4xf64>
    %2 = enzymexla.jit_call @copy(%0, %arg1) <
      output_operand_aliases = [
        #stablehlo.output_operand_alias<output_tuple_indices = [],
                                        operand_index = 1,
                                        operand_tuple_indices = []>
      ]
    > : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    return %1, %2 : tensor<4xf64>, tensor<4xf32>
  }
}

// CHECK-LABEL: func.func @clobbered(
// CHECK-NEXT:    %[[R0:.+]] = enzymexla.jit_call @first
// CHECK-NEXT:    %[[C:.+]] = stablehlo.convert %[[R0]]
// CHECK-NEXT:    %[[R1:.+]] = enzymexla.jit_call @second (%[[R0]])
// CHECK-NEXT:    return %[[C]], %[[R1]]

// CHECK-LABEL: func.func @interleaved(
// CHECK-SAME:      %[[A:.+]]: tensor<4xf32>, %[[B:.+]]: tensor<4xf32>)
// CHECK-NEXT:    %[[R:.+]]:2 = enzymexla.jit_call @fused__first_copy (%[[A]], %[[B]])
// CHECK-NEXT:    %[[C:.+]] = stablehlo.convert %[[R]]#0
// CHECK-NEXT:    return %[[C]], %[[R]]#1
