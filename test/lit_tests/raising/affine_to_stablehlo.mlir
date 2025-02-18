// RUN: enzymexlamlir-opt %s --affine-raise --enzyme-hlo-opt | FileCheck %s

module {

  func.func private @myfunc(%arg0: memref<1x135x374xf64, 1>) {
    affine.parallel (%arg1) = (0) to (135) {
      %0 = affine.load %arg0[0, %arg1, 360] : memref<1x135x374xf64, 1>
      affine.store %0, %arg0[0, %arg1, 0] : memref<1x135x374xf64, 1>
      %1 = affine.load %arg0[0, %arg1, 7] : memref<1x135x374xf64, 1>
      affine.store %1, %arg0[0, %arg1, 367] : memref<1x135x374xf64, 1>
      %2 = affine.load %arg0[0, %arg1, 361] : memref<1x135x374xf64, 1>
      affine.store %2, %arg0[0, %arg1, 1] : memref<1x135x374xf64, 1>
      %3 = affine.load %arg0[0, %arg1, 8] : memref<1x135x374xf64, 1>
      affine.store %3, %arg0[0, %arg1, 368] : memref<1x135x374xf64, 1>
      %4 = affine.load %arg0[0, %arg1, 362] : memref<1x135x374xf64, 1>
      affine.store %4, %arg0[0, %arg1, 2] : memref<1x135x374xf64, 1>
      %5 = affine.load %arg0[0, %arg1, 9] : memref<1x135x374xf64, 1>
      affine.store %5, %arg0[0, %arg1, 369] : memref<1x135x374xf64, 1>
      %6 = affine.load %arg0[0, %arg1, 363] : memref<1x135x374xf64, 1>
      affine.store %6, %arg0[0, %arg1, 3] : memref<1x135x374xf64, 1>
      %7 = affine.load %arg0[0, %arg1, 10] : memref<1x135x374xf64, 1>
      affine.store %7, %arg0[0, %arg1, 370] : memref<1x135x374xf64, 1>
      %8 = affine.load %arg0[0, %arg1, 364] : memref<1x135x374xf64, 1>
      affine.store %8, %arg0[0, %arg1, 4] : memref<1x135x374xf64, 1>
      %9 = affine.load %arg0[0, %arg1, 11] : memref<1x135x374xf64, 1>
      affine.store %9, %arg0[0, %arg1, 371] : memref<1x135x374xf64, 1>
      %10 = affine.load %arg0[0, %arg1, 365] : memref<1x135x374xf64, 1>
      affine.store %10, %arg0[0, %arg1, 5] : memref<1x135x374xf64, 1>
      %11 = affine.load %arg0[0, %arg1, 12] : memref<1x135x374xf64, 1>
      affine.store %11, %arg0[0, %arg1, 372] : memref<1x135x374xf64, 1>
      %12 = affine.load %arg0[0, %arg1, 366] : memref<1x135x374xf64, 1>
      affine.store %12, %arg0[0, %arg1, 6] : memref<1x135x374xf64, 1>
      %13 = affine.load %arg0[0, %arg1, 13] : memref<1x135x374xf64, 1>
      affine.store %13, %arg0[0, %arg1, 373] : memref<1x135x374xf64, 1>
    }
    return
  }

  func.func @main(%84: tensor<1x135x374xf64>) -> tensor<1x135x374xf64> {
    %85 = enzymexla.jit_call @myfunc(%84) {output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>]} : (tensor<1x135x374xf64>) -> tensor<1x135x374xf64>
    return %85 : tensor<1x135x374xf64>
  }
}

// CHECK:  func.func @main(%arg0: tensor<1x135x374xf64>) -> tensor<1x135x374xf64> {
// CHECK-NEXT:    %0 = call @myfunc_raised(%arg0) : (tensor<1x135x374xf64>) -> tensor<1x135x374xf64>
// CHECK-NEXT:    return %0 : tensor<1x135x374xf64>
// CHECK-NEXT:  }

// CHECK:  func.func private @myfunc_raised(%arg0: tensor<1x135x374xf64>) -> tensor<1x135x374xf64> {
// CHECK-NEXT:    %0 = stablehlo.slice %arg0 [0:1, 0:135, 7:367] : (tensor<1x135x374xf64>) -> tensor<1x135x360xf64>
// CHECK-NEXT:    %1 = stablehlo.slice %arg0 [0:1, 0:135, 7:14] : (tensor<1x135x374xf64>) -> tensor<1x135x7xf64>
// CHECK-NEXT:    %2 = stablehlo.slice %arg0 [0:1, 0:135, 360:367] : (tensor<1x135x374xf64>) -> tensor<1x135x7xf64>
// CHECK-NEXT:    %3 = stablehlo.concatenate %2, %0, %1, dim = 2 : (tensor<1x135x7xf64>, tensor<1x135x360xf64>, tensor<1x135x7xf64>) -> tensor<1x135x374xf64>
// CHECK-NEXT:    return %3 : tensor<1x135x374xf64>
// CHECK-NEXT:  }
