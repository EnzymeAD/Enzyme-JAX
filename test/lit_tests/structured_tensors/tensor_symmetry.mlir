// RUN: enzymexlamlir-opt %s --tensor-symmetric-simplify | FileCheck %s

func.func @broadcast_scalar(%arg0: tensor<f32>) -> tensor<10x10xf32> {
  %0 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f32>) -> tensor<10x10xf32>
  return %0 : tensor<10x10xf32>
}
// CHECK-LABEL: func.func @broadcast_scalar
// CHECK: enzymexla.symmetric_dims = [{{.*}}"{ {0,1} }"{{.*}}]

func.func @broadcast_tensor(%arg0: tensor<10xf32>) -> tensor<10x10xf32> {
  %0 = stablehlo.broadcast_in_dim %arg0, dims = [0] : (tensor<10xf32>) -> tensor<10x10xf32>
  return %0 : tensor<10x10xf32>
}
// CHECK-LABEL: func.func @broadcast_tensor
// CHECK: enzymexla.symmetric_dims = [{{.*}}"{ {0}, {1} }"{{.*}}]

func.func @constant_symmetric() -> tensor<2x2xf32> {
  %0 = stablehlo.constant dense<[[1.0, 2.0], [2.0, 4.0]]> : tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
}
// CHECK-LABEL: func.func @constant_symmetric
// CHECK: enzymexla.symmetric_dims = [{{.*}}"{ {0,1} }"{{.*}}]

func.func @constant_asymmetric() -> tensor<2x2xf32> {
  %0 = stablehlo.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
}
// CHECK-LABEL: func.func @constant_asymmetric
// CHECK: enzymexla.symmetric_dims = [{{.*}}"{ {0}, {1} }"{{.*}}]

func.func @transpose_symmetric() -> tensor<2x2xf32> {
  %0 = stablehlo.constant dense<[[1.0, 2.0], [2.0, 4.0]]> : tensor<2x2xf32>
  %1 = stablehlo.transpose %0, dims = [1, 0] : (tensor<2x2xf32>) -> tensor<2x2xf32>
  return %1 : tensor<2x2xf32>
}
// CHECK-LABEL: func.func @transpose_symmetric
// CHECK: enzymexla.symmetric_dims = [{{.*}}"{ {0,1} }"{{.*}}]

func.func @add_symmetric() -> tensor<2x2xf32> {
  %0 = stablehlo.constant dense<[[1.0, 2.0], [2.0, 4.0]]> : tensor<2x2xf32>
  %1 = stablehlo.constant dense<[[5.0, 6.0], [6.0, 8.0]]> : tensor<2x2xf32>
  %2 = stablehlo.add %0, %1 : tensor<2x2xf32>
  return %2 : tensor<2x2xf32>
}
// CHECK-LABEL: func.func @add_symmetric
// CHECK: enzymexla.symmetric_dims = [{{.*}}"{ {0,1} }"{{.*}}]

func.func @add_mixed() -> tensor<2x2xf32> {
  %0 = stablehlo.constant dense<[[1.0, 2.0], [2.0, 4.0]]> : tensor<2x2xf32>
  %1 = stablehlo.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>
  %2 = stablehlo.add %0, %1 : tensor<2x2xf32>
  return %2 : tensor<2x2xf32>
}
// CHECK-LABEL: func.func @add_mixed
// CHECK: enzymexla.symmetric_dims = [{{.*}}"{ {0}, {1} }"{{.*}}]
