// RUN: enzymexlamlir-opt %s --affine-raise --enzyme-hlo-opt | FileCheck %s

module {
  func.func @transpose_kernel(%arg0: memref<100x100xf32, 1>, %arg1: memref<100x100xf32, 1>) {
    affine.parallel (%arg2, %arg3) = (0, 0) to (100, 100) {
      %0 = affine.load %arg0[%arg3, %arg2] : memref<100x100xf32, 1>
      affine.store %0, %arg1[%arg2, %arg3] : memref<100x100xf32, 1>
    }
    return
  }

  func.func @main(%arg0: tensor<100x100xf32>) -> tensor<100x100xf32> {
    %0:2 = enzymexla.jit_call @transpose_kernel(%arg0, %arg0) {output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>]} : (tensor<100x100xf32>, tensor<100x100xf32>) -> (tensor<100x100xf32>, tensor<100x100xf32>)
    return %0#1 : tensor<100x100xf32>
  }
}

// CHECK:  func.func @main(%arg0: tensor<100x100xf32>) -> tensor<100x100xf32> {
// CHECK-NEXT:    %0:2 = call @transpose_kernel_raised(%arg0, %arg0) : (tensor<100x100xf32>, tensor<100x100xf32>) -> (tensor<100x100xf32>, tensor<100x100xf32>)
// CHECK-NEXT:    return %0#1 : tensor<100x100xf32>
// CHECK-NEXT:  }

// CHECK:  func.func private @transpose_kernel_raised(%arg0: tensor<100x100xf32>, %arg1: tensor<100x100xf32>) -> (tensor<100x100xf32>, tensor<100x100xf32>) {
// CHECK-NEXT:    %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<100x100xf32>) -> tensor<100x100xf32>
// CHECK-NEXT:    return %arg0, %0 : tensor<100x100xf32>, tensor<100x100xf32>
// CHECK-NEXT:  }
