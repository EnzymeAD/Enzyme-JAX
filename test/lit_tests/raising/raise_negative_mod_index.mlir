// RUN: enzymexlamlir-opt %s --raise-affine-to-stablehlo --enzyme-hlo-opt | FileCheck %s

// An affine mod of a negative exact multiple is 0, not the modulus:
// (%j + (%i * -8) floordiv 3) mod 3 is %j for %i < 3 (the floordiv gives
// -3 and -6 there) and (%j + 1) mod 3 for %i = 3. The lowering used to add
// the modulus whenever the dividend was negative, so the (1, 0) and (2, 0)
// lanes indexed slot 3 of a 3-slot row instead of slot 0: out of the buffer,
// and the scatter dropped their stores.
func.func @negmod(%out: memref<4x3xf64, 1>, %in: memref<4x3xf64, 1>) {
  affine.parallel (%i, %j) = (0, 0) to (4, 3) {
    %v = affine.load %in[%i, %j] : memref<4x3xf64, 1>
    affine.store %v, %out[%i, (%j + (%i * -8) floordiv 3) mod 3] : memref<4x3xf64, 1>
  }
  return
}

// CHECK-LABEL: func.func private @negmod_raised(
// CHECK: %[[REM:.+]] = stablehlo.remainder %{{.+}}, %{{.+}}{{.*}} : tensor<3x4xi64>
// CHECK-NEXT: %[[NEG:.+]] = stablehlo.compare LT, %[[REM]], %{{.+}} : (tensor<3x4xi64>, tensor<3x4xi64>) -> tensor<3x4xi1>
// CHECK: stablehlo.select %[[NEG]], %{{.+}}, %[[REM]] : tensor<3x4xi1>, tensor<3x4xi64>
