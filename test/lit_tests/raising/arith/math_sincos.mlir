/// RUN: enzymexlamlir-opt --arith-raise %s | FileCheck %s

func.func @sincos_f32(%x: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
  %s, %c = math.sincos %x : tensor<4xf32>
  return %c, %s : tensor<4xf32>, tensor<4xf32>
}

// CHECK-LABEL: func.func @sincos_f32(%arg0: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
// CHECK-NEXT:    %[[S:.+]] = stablehlo.sine %arg0 : tensor<4xf32>
// CHECK-NEXT:    %[[C:.+]] = stablehlo.cosine %arg0 : tensor<4xf32>
// CHECK-NEXT:    return %[[C]], %[[S]] : tensor<4xf32>, tensor<4xf32>
// CHECK-NEXT:  }

func.func @sincos_f64_2d(%x: tensor<2x3xf64>) -> (tensor<2x3xf64>, tensor<2x3xf64>) {
  %s, %c = math.sincos %x : tensor<2x3xf64>
  return %s, %c : tensor<2x3xf64>, tensor<2x3xf64>
}

// CHECK-LABEL: func.func @sincos_f64_2d(%arg0: tensor<2x3xf64>) -> (tensor<2x3xf64>, tensor<2x3xf64>) {
// CHECK-NEXT:    %[[S:.+]] = stablehlo.sine %arg0 : tensor<2x3xf64>
// CHECK-NEXT:    %[[C:.+]] = stablehlo.cosine %arg0 : tensor<2x3xf64>
// CHECK-NEXT:    return %[[S]], %[[C]] : tensor<2x3xf64>, tensor<2x3xf64>
// CHECK-NEXT:  }
