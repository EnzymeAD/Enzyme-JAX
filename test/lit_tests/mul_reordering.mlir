// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=associative_common_mul_op_reordering" --transform-interpreter --enzyme-hlo-remove-transform

module {
  func.func @main(%366: tensor<128x1008x1008xf64>, %367: tensor<128x1008x1008xf64>, %369: tensor<128x1008x1008xf64>) -> tensor<128x1008x1008xf64> {
    %368 = stablehlo.multiply %366, %367 : tensor<128x1008x1008xf64>
    %370 = stablehlo.multiply %366, %369 : tensor<128x1008x1008xf64>
    %371 = stablehlo.add %368, %370 : tensor<128x1008x1008xf64>
    return %371 : tensor<128x1008x1008xf64>
  }
}

// CHECK:  func.func @main(%arg0: tensor<128x1008x1008xf64>, %arg1: tensor<128x1008x1008xf64>, %arg2: tensor<128x1008x1008xf64>) -> tensor<128x1008x1008xf64> {
// CHECK-NEXT:    %0 = stablehlo.add %arg1, %arg2 : tensor<128x1008x1008xf64>
// CHECK-NEXT:    %1 = stablehlo.multiply %arg0, %0 : tensor<128x1008x1008xf64>
// CHECK-NEXT:    return %1 : tensor<128x1008x1008xf64>
// CHECK-NEXT:  }
