// RUN: enzymexlamlir-opt --partial-symmetry-annotate --enzyme-hlo-generate-td="patterns=transpose_partial_symmetry_simplify" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

func.func @test_constant() -> tensor<2x2xf32> {
  %cst = stablehlo.constant dense<[[1.0, 2.0], [2.0, 3.0]]> : tensor<2x2xf32>
  %0 = stablehlo.transpose %cst, dims = [1, 0] : (tensor<2x2xf32>) -> tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
}
// CHECK: func.func @test_constant() -> tensor<2x2xf32> {
// CHECK-NEXT: %cst = stablehlo.constant {enzymexla.partial_symmetry = [#enzymexla.partial_symmetry<<[0, 1]>>]} dense<{{.*}}> : tensor<2x2xf32>
// CHECK-NEXT: return %cst : tensor<2x2xf32>
// CHECK-NEXT: }

func.func @test_propagate() -> tensor<2x2x2x3xf32> {
  %cst0 = stablehlo.constant dense<[[[1.0, 2.0], [3.0, 4.0]], [[3.0, 4.0], [5.0, 6.0]]]> : tensor<2x2x2xf32>
  %cst1 = stablehlo.constant dense<[[[1.0, 2.0], [2.0, 3.0]], [[2.0, 3.0], [3.0, 4.0]]]> : tensor<2x2x2xf32>
  %0 = stablehlo.add %cst0, %cst1 : tensor<2x2x2xf32>
  %1 = stablehlo.transpose %0, dims = [0, 2, 1] : (tensor<2x2x2xf32>) -> tensor<2x2x2xf32>
  %2 = stablehlo.broadcast_in_dim %1, dims = [1, 0, 2] : (tensor<2x2x2xf32>) -> tensor<2x2x2x3xf32>
  %3 = stablehlo.transpose %2, dims = [0, 2, 1, 3] : (tensor<2x2x2x3xf32>) -> tensor<2x2x2x3xf32>
  return %3 : tensor<2x2x2x3xf32>
}
// CHECK: func.func @test_propagate() -> tensor<2x2x2x3xf32> {
// CHECK-NEXT: %cst = stablehlo.constant {enzymexla.partial_symmetry = [#enzymexla.partial_symmetry<<[0, 1]>>]} dense<{{.*}}> : tensor<2x2x2xf32>
// CHECK-NEXT: %cst_0 = stablehlo.constant {enzymexla.partial_symmetry = [#enzymexla.partial_symmetry<<[0, 1, 2]>>]} dense<{{.*}}> : tensor<2x2x2xf32>
// CHECK-NEXT: %0 = stablehlo.add %cst, %cst_0 {enzymexla.partial_symmetry = [#enzymexla.partial_symmetry<<[0, 1]>>]} : tensor<2x2x2xf32>
// CHECK-NEXT: %1 = stablehlo.transpose %0, dims = [0, 2, 1] {enzymexla.partial_symmetry = [#enzymexla.partial_symmetry<<[0, 2]>>]} : (tensor<2x2x2xf32>) -> tensor<2x2x2xf32>
// CHECK-NEXT: %2 = stablehlo.broadcast_in_dim %1, dims = [1, 0, 2] {enzymexla.partial_symmetry = [#enzymexla.partial_symmetry<<[1, 2]>>]} : (tensor<2x2x2xf32>) -> tensor<2x2x2x3xf32>
// CHECK-NEXT: return %2 : tensor<2x2x2x3xf32>
// CHECK-NEXT: }

func.func @test_add_generate_symmetry(%arg0: tensor<3x2x3xf32>) -> tensor<3x2x3xf32> {
  %0 = stablehlo.transpose %arg0, dims = [2, 1, 0] : (tensor<3x2x3xf32>) -> tensor<3x2x3xf32>
  %1 = stablehlo.add %0, %arg0 : tensor<3x2x3xf32>
  return %1 : tensor<3x2x3xf32>
}
// CHECK: func.func @test_add_generate_symmetry(%arg0: tensor<3x2x3xf32>) -> tensor<3x2x3xf32> {
// CHECK-NEXT: %0 = stablehlo.transpose %arg0, dims = [2, 1, 0] : (tensor<3x2x3xf32>) -> tensor<3x2x3xf32>
// CHECK-NEXT: %1 = stablehlo.add %0, %arg0 {enzymexla.partial_symmetry = [#enzymexla.partial_symmetry<<[0, 2]>>]} : tensor<3x2x3xf32>
// CHECK-NEXT: return %1 : tensor<3x2x3xf32>
// CHECK-NEXT: }

func.func @test_dot_propagate() -> tensor<2x2xf32> {
  %cst0 = stablehlo.constant dense<[[[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]], [[2.0, 3.0, 4.0], [3.0, 4.0, 5.0]]]> : tensor<2x2x3xf32>
  %cst1 = stablehlo.constant dense<[[[1.0, 2.0], [2.0, 3.0]], [[2.0, 3.0], [3.0, 4.0]], [[2.0, 3.0], [3.0, 4.0]]]> : tensor<3x2x2xf32>
  %0 = stablehlo.dot_general %cst0, %cst1, batching_dims = [0, 1] x [1, 2], contracting_dims = [2] x [0] : (tensor<2x2x3xf32>, tensor<3x2x2xf32>) -> tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
}
// CHECK: func.func @test_dot_propagate() -> tensor<2x2xf32> {
// CHECK-NEXT: %cst = stablehlo.constant {enzymexla.partial_symmetry = [#enzymexla.partial_symmetry<<[0, 1]>>]} dense<{{.*}}> : tensor<2x2x3xf32>
// CHECK-NEXT: %cst_0 = stablehlo.constant {enzymexla.partial_symmetry = [#enzymexla.partial_symmetry<<[1, 2]>>]} dense<{{.*}}> : tensor<3x2x2xf32>
// CHECK-NEXT: %0 = stablehlo.dot_general %cst, %cst_0, batching_dims = [0, 1] x [1, 2], contracting_dims = [2] x [0] {enzymexla.partial_symmetry = [#enzymexla.partial_symmetry<<[0, 1]>>]} : (tensor<2x2x3xf32>, tensor<3x2x2xf32>) -> tensor<2x2xf32>
// CHECK-NEXT: return %0 : tensor<2x2xf32>
// CHECK-NEXT: }

func.func @test_dot_generate_symmetry(%arg0: tensor<3x3x3xf32>) -> tensor<3x3x3xf32> {
  %0 = stablehlo.transpose %arg0, dims = [2, 1, 0] : (tensor<3x3x3xf32>) -> tensor<3x3x3xf32>
  %1 = stablehlo.dot_general %arg0, %0, batching_dims = [1] x [1], contracting_dims = [0] x [2] : (tensor<3x3x3xf32>, tensor<3x3x3xf32>) -> tensor<3x3x3xf32>
  return %1 : tensor<3x3x3xf32>
}
// CHECK: func.func @test_dot_generate_symmetry(%arg0: tensor<3x3x3xf32>) -> tensor<3x3x3xf32> {
// CHECK-NEXT: %0 = stablehlo.transpose %arg0, dims = [2, 1, 0] : (tensor<3x3x3xf32>) -> tensor<3x3x3xf32>
// CHECK-NEXT: %1 = stablehlo.dot_general %arg0, %0, batching_dims = [1] x [1], contracting_dims = [0] x [2] {enzymexla.partial_symmetry = [#enzymexla.partial_symmetry<<[1, 2]>>]} : (tensor<3x3x3xf32>, tensor<3x3x3xf32>) -> tensor<3x3x3xf32>
// CHECK-NEXT: return %1 : tensor<3x3x3xf32>
// CHECK-NEXT: }

