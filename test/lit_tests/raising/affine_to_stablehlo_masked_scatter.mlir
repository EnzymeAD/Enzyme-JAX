// RUN: enzymexlamlir-opt %s --raise-affine-to-stablehlo --canonicalize --enzyme-hlo-opt=max_constant_expansion=0 | FileCheck %s

// A memref.store guarded by an affine.if raises to a masked scatter: the
// store is conditioned on the if's condition, so masked-out positions must
// keep their original value. This is done by gathering the original input,
// selecting between the update and the original using the broadcasted mask,
// and scattering with unique_indices = false.

module {
  func.func @main(%arg0: memref<100xf32>, %arg1: memref<100xf32>) {
    affine.parallel (%i, %j) = (0, 0) to (10, 10) step (1, 1) {
      affine.if affine_set<(d0, d1) : (d0 - d1 >= 0)>(%i, %j) {
        %0 = affine.load %arg1[%i * 10 + %j] : memref<100xf32>
        affine.store %0, %arg0[%i * 10 + %j] : memref<100xf32>
      }
    }
    return
  }
}

// CHECK:  func.func private @main_raised(%arg0: tensor<100xf32>, %arg1: tensor<100xf32>) -> (tensor<100xf32>, tensor<100xf32>) {
// CHECK-NEXT:    %c = stablehlo.constant dense<0> : tensor<10x10xi64>
// CHECK-NEXT:    %0 = stablehlo.iota dim = 0 {enzymexla.non_negative = [#enzymexla<guaranteed GUARANTEED>]} : tensor<10xi64>
// CHECK-NEXT:    %1 = stablehlo.negate %0 : tensor<10xi64>
// CHECK-NEXT:    %2 = stablehlo.iota dim = 0 : tensor<10x10xi64>
// CHECK-NEXT:    %3 = stablehlo.broadcast_in_dim %1, dims = [1] {enzymexla.non_negative = [#enzymexla<guaranteed NOTGUARANTEED>]} : (tensor<10xi64>) -> tensor<10x10xi64>
// CHECK-NEXT:    %4 = stablehlo.add %2, %3 {enzymexla.non_negative = [#enzymexla<guaranteed NOTGUARANTEED>]} : tensor<10x10xi64>
// CHECK-NEXT:    %5 = stablehlo.compare GE, %4, %c : (tensor<10x10xi64>, tensor<10x10xi64>) -> tensor<10x10xi1>
// CHECK-NEXT:    %6 = stablehlo.reshape %arg1 : (tensor<100xf32>) -> tensor<10x10xf32>
// CHECK-NEXT:    %7 = stablehlo.reshape %arg0 : (tensor<100xf32>) -> tensor<10x10xf32>
// CHECK-NEXT:    %8 = stablehlo.select %5, %6, %7 : tensor<10x10xi1>, tensor<10x10xf32>
// CHECK-NEXT:    %9 = stablehlo.reshape %8 : (tensor<10x10xf32>) -> tensor<100xf32>
// CHECK-NEXT:    return %9, %arg1 : tensor<100xf32>, tensor<100xf32>
// CHECK-NEXT:  }
