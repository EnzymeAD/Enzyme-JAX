// RUN: enzymexlamlir-opt %s --raise-affine-to-stablehlo --canonicalize --arith-raise --enzyme-hlo-opt | FileCheck %s

module {
  func.func private @kernel_with_for(%arg0: memref<187x194xf64, 1>) {
    affine.parallel (%arg1) = (0) to (180) {
      affine.for %arg2 = 0 to 50 {
        %7 = affine.load %arg0[-%arg2 + 134, -%arg1 + 186] : memref<187x194xf64, 1> // [134:84:-1, 186:7:-1]
        affine.store %7, %arg0[%arg2 + 136, %arg1 + 7] : memref<187x194xf64, 1> // [136:186, 7:186]
      }
    }
    return
  }
}

// CHECK:  func.func private @kernel_with_for_raised(%arg0: tensor<187x194xf64>) -> tensor<187x194xf64> {
// CHECK-NEXT:    %c = stablehlo.constant dense<7> : tensor<i64>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<136> : tensor<i64>
// CHECK-NEXT:    %0 = stablehlo.slice %arg0 [85:135, 7:187] : (tensor<187x194xf64>) -> tensor<50x180xf64>
// CHECK-NEXT:    %1 = stablehlo.reverse %0, dims = [0, 1] : tensor<50x180xf64>
// CHECK-NEXT:    %2 = stablehlo.dynamic_update_slice %arg0, %1, %c_0, %c : (tensor<187x194xf64>, tensor<50x180xf64>, tensor<i64>, tensor<i64>) -> tensor<187x194xf64>
// CHECK-NEXT:    return %2 : tensor<187x194xf64>
// CHECK-NEXT:  }
