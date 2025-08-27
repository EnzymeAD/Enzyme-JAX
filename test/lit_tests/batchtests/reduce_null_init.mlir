// RUN: enzymexlamlir-opt %s --enzyme-batch --enzyme-hlo-opt | FileCheck %s
// This test reproduces the segfault issue when batchedInit.getDefiningOp() returns null

func.func private @reduce_with_arg_init(%arg0: tensor<16xf64>, %init: tensor<f64>) -> (tensor<f64>) {
    %1 = stablehlo.reduce(%arg0 init: %init) applies stablehlo.add across dimensions = [0] : (tensor<16xf64>, tensor<f64>) -> tensor<f64>
    return %1 : tensor<f64>
}

func.func @main(%arg0: tensor<4x16xf64>, %init: tensor<4xf64>) -> (tensor<4xf64>) {
    %1 = enzyme.batch @reduce_with_arg_init(%arg0, %init) {batch_shape = array<i64: 4>} : (tensor<4x16xf64>, tensor<4xf64>) -> (tensor<4xf64>)
    return %1 : tensor<4xf64>
}

// CHECK: func.func @main(%arg0: tensor<4x16xf64>, %arg1: tensor<4xf64>) -> tensor<4xf64> {
// CHECK-NEXT:    %0 = call @batched_reduce_with_arg_init(%arg0, %arg1) : (tensor<4x16xf64>, tensor<4xf64>) -> tensor<4xf64>
// CHECK-NEXT:    return %0 : tensor<4xf64>
// CHECK-NEXT:  }
// CHECK-NEXT:  func.func private @batched_reduce_with_arg_init(%arg0: tensor<4x16xf64>, %arg1: tensor<4xf64>) -> tensor<4xf64> {
// CHECK:    stablehlo.reduce
// CHECK:    return
// CHECK-NEXT:  }