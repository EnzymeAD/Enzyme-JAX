// RUN: enzymexlamlir-opt %s --affine-raise --arith-raise --enzyme-hlo-opt | FileCheck %s

module {
  func.func private @myfunc(%arg0: memref<204xf64, 1>, %arg1: memref<100xf64, 1>) {
    affine.parallel (%arg2) = (0) to (100) {
      %0 = affine.load %arg0[%arg2 * 2 + 4] : memref<204xf64, 1>
      %1 = affine.load %arg1[%arg2] : memref<100xf64, 1>
      %2 = arith.addf %0, %1 : f64
      affine.store %2, %arg1[%arg2] : memref<100xf64, 1>
    }
    return
  }

  func.func @main(%84: tensor<204xf64>, %85: tensor<100xf64>) -> tensor<100xf64> {
    %86:2 = enzymexla.jit_call @myfunc(%84, %85) {output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>]} : (tensor<204xf64>, tensor<100xf64>) -> (tensor<204xf64>, tensor<100xf64>)
    return %86#1 : tensor<100xf64>
  }
}

// CHECK:  func.func @main(%arg0: tensor<204xf64>, %arg1: tensor<100xf64>) -> tensor<100xf64> {
// CHECK-NEXT:    %0:2 = call @myfunc_raised(%arg0, %arg1) : (tensor<204xf64>, tensor<100xf64>) -> (tensor<204xf64>, tensor<100xf64>)
// CHECK-NEXT:    return %0#1 : tensor<100xf64>
// CHECK-NEXT:  }

// CHECK:  func.func private @myfunc_raised(%arg0: tensor<204xf64>, %arg1: tensor<100xf64>) -> (tensor<204xf64>, tensor<100xf64>) {
// CHECK-NEXT:    %0 = stablehlo.slice %arg0 [4:204:2] : (tensor<204xf64>) -> tensor<100xf64>
// CHECK-NEXT:    %1 = stablehlo.add %0, %arg1 : tensor<100xf64>
// CHECK-NEXT:    return %arg0, %1 : tensor<204xf64>, tensor<100xf64>
// CHECK-NEXT:  }
