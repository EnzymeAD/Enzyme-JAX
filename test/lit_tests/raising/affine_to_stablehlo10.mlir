// RUN: enzymexlamlir-opt %s --raise-affine-to-stablehlo --arith-raise --enzyme-hlo-opt | FileCheck %s

module {
  func.func private @kernel(%arg0: memref<187x194xf64, 1>, %arg1: memref<187x194xf64, 1>) {
    affine.parallel (%arg2, %arg3) = (0, 0) to (194, 187) {
      %0 = affine.load %arg0[186-%arg3, 193-%arg2] : memref<187x194xf64, 1>
      affine.store %0, %arg1[%arg3, %arg2] : memref<187x194xf64, 1>
    }
    return
  }
}

// CHECK:  func.func private @kernel_raised(%[[src:.+]]: tensor<187x194xf64>, %[[dst:.+]]: tensor<187x194xf64>) -> (tensor<187x194xf64>, tensor<187x194xf64>) {
// CHECK-NEXT:    %[[rev:.+]] = stablehlo.reverse %[[src]], dims = [0, 1] : tensor<187x194xf64>
// CHECK-NEXT:    return %[[src]], %[[rev]] : tensor<187x194xf64>, tensor<187x194xf64>
// CHECK-NEXT:  }

module {
  func.func private @kernel(%arg0: memref<187x194xf64, 1>, %arg1: memref<187x194xf64, 1>) {
    affine.parallel (%arg2, %arg3) = (0, 0) to (194, 187) {
      %0 = affine.load %arg0[186-%arg3, %arg2] : memref<187x194xf64, 1>
      affine.store %0, %arg1[%arg3, %arg2] : memref<187x194xf64, 1>
    }
    return
  }
}

// CHECK:  func.func private @kernel_raised(%[[src:.+]]: tensor<187x194xf64>, %[[dst:.+]]: tensor<187x194xf64>) -> (tensor<187x194xf64>, tensor<187x194xf64>) {
// CHECK-NEXT:    %[[rev:.+]] = stablehlo.reverse %[[src]], dims = [0] : tensor<187x194xf64>
// CHECK-NEXT:    return %[[src]], %[[rev]] : tensor<187x194xf64>, tensor<187x194xf64>
// CHECK-NEXT:  }

module {
  func.func private @kernel(%arg0: memref<180xf32, 1>, %arg1: memref<180xf32, 1>) {
    affine.parallel (%i) = (0) to (90) {
      %0 = affine.load %arg0[179 - %i] : memref<180xf32, 1>
      affine.store %0, %arg1[%i] : memref<180xf32, 1>
    }
    return
  }
}

// CHECK:  func.func private @kernel_raised(%[[src:.+]]: tensor<180xf32>, %[[dst:.+]]: tensor<180xf32>) -> (tensor<180xf32>, tensor<180xf32>) {
// CHECK-NEXT:    %[[slice:.+]] = stablehlo.slice %[[src]] [90:180] : (tensor<180xf32>) -> tensor<90xf32>
// CHECK-NEXT:    %[[slice_rev:.+]] = stablehlo.reverse %[[slice]], dims = [0] : tensor<90xf32>
// CHECK-NEXT:    %[[slice_unchanged:.+]] = stablehlo.slice %arg1 [90:180] : (tensor<180xf32>) -> tensor<90xf32>
// CHECK-NEXT:    %[[output:.+]] = stablehlo.concatenate %[[slice_rev]], %[[slice_unchanged]], dim = 0 : (tensor<90xf32>, tensor<90xf32>) -> tensor<180xf32>
// CHECK-NEXT:    return %[[src]], %[[output]] : tensor<180xf32>, tensor<180xf32>
// CHECK-NEXT:  }

module {
  func.func private @kernel(%arg0: memref<180xf32, 1>, %arg1: memref<180xf32, 1>) {
    affine.parallel (%i) = (0) to (90) {
      %0 = affine.load %arg0[%i] : memref<180xf32, 1>
      affine.store %0, %arg1[179 - %i] : memref<180xf32, 1>
    }
    return
  }
}

// CHECK:  func.func private @kernel_raised(%[[src:.+]]: tensor<180xf32>, %[[dst:.+]]: tensor<180xf32>) -> (tensor<180xf32>, tensor<180xf32>) {
// CHECK-NEXT:    %[[slice:.+]] = stablehlo.slice %[[src]] [0:90] : (tensor<180xf32>) -> tensor<90xf32>
// CHECK-NEXT:    %[[slice_rev:.+]] = stablehlo.reverse %[[slice]], dims = [0] : tensor<90xf32>
// CHECK-NEXT:    %[[slice_unchanged:.+]] = stablehlo.slice %arg1 [0:90] : (tensor<180xf32>) -> tensor<90xf32>
// CHECK-NEXT:    %[[output:.+]] = stablehlo.concatenate %[[slice_unchanged]], %[[slice_rev]], dim = 0 : (tensor<90xf32>, tensor<90xf32>) -> tensor<180xf32>
// CHECK-NEXT:    return %[[src]], %[[output]] : tensor<180xf32>, tensor<180xf32>
// CHECK-NEXT:  }
