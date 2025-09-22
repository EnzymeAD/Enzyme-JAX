// RUN: enzymexlamlir-opt --auto-batching --inline --enzyme-hlo-opt %s | FileCheck %s

module @reactant_mapped_sub attributes {mhlo.num_partitions = 1 : i64, mhlo.num_replicas = 1 : i64} {
  func.func @main(%arg0: tensor<3x5x10xf32>, %arg1: tensor<3x5x10xf32>) -> tensor<5x3x10xf32> attributes {enzymexla.memory_effects = []} {
    %0 = stablehlo.transpose %arg0, dims = [2, 1, 0] : (tensor<3x5x10xf32>) -> tensor<10x5x3xf32>
    %1 = stablehlo.transpose %arg1, dims = [2, 1, 0] : (tensor<3x5x10xf32>) -> tensor<10x5x3xf32>
    %2 = stablehlo.slice %0 [0:10, 0:1, 0:3] : (tensor<10x5x3xf32>) -> tensor<10x1x3xf32>
    %3 = stablehlo.slice %1 [0:10, 0:1, 0:3] : (tensor<10x5x3xf32>) -> tensor<10x1x3xf32>
    %4 = stablehlo.subtract %2, %3 : tensor<10x1x3xf32>
    %5 = stablehlo.slice %0 [0:10, 1:2, 0:3] : (tensor<10x5x3xf32>) -> tensor<10x1x3xf32>
    %6 = stablehlo.slice %1 [0:10, 1:2, 0:3] : (tensor<10x5x3xf32>) -> tensor<10x1x3xf32>
    %7 = stablehlo.subtract %5, %6 : tensor<10x1x3xf32>
    %8 = stablehlo.slice %0 [0:10, 2:3, 0:3] : (tensor<10x5x3xf32>) -> tensor<10x1x3xf32>
    %9 = stablehlo.slice %1 [0:10, 2:3, 0:3] : (tensor<10x5x3xf32>) -> tensor<10x1x3xf32>
    %10 = stablehlo.subtract %8, %9 : tensor<10x1x3xf32>
    %11 = stablehlo.slice %0 [0:10, 3:4, 0:3] : (tensor<10x5x3xf32>) -> tensor<10x1x3xf32>
    %12 = stablehlo.slice %1 [0:10, 3:4, 0:3] : (tensor<10x5x3xf32>) -> tensor<10x1x3xf32>
    %13 = stablehlo.subtract %11, %12 : tensor<10x1x3xf32>
    %14 = stablehlo.slice %0 [0:10, 4:5, 0:3] : (tensor<10x5x3xf32>) -> tensor<10x1x3xf32>
    %15 = stablehlo.slice %1 [0:10, 4:5, 0:3] : (tensor<10x5x3xf32>) -> tensor<10x1x3xf32>
    %16 = stablehlo.subtract %14, %15 : tensor<10x1x3xf32>
    %17 = stablehlo.transpose %4, dims = [1, 2, 0] : (tensor<10x1x3xf32>) -> tensor<1x3x10xf32>
    %18 = stablehlo.transpose %7, dims = [1, 2, 0] : (tensor<10x1x3xf32>) -> tensor<1x3x10xf32>
    %19 = stablehlo.transpose %10, dims = [1, 2, 0] : (tensor<10x1x3xf32>) -> tensor<1x3x10xf32>
    %20 = stablehlo.transpose %13, dims = [1, 2, 0] : (tensor<10x1x3xf32>) -> tensor<1x3x10xf32>
    %21 = stablehlo.transpose %16, dims = [1, 2, 0] : (tensor<10x1x3xf32>) -> tensor<1x3x10xf32>
    %22 = stablehlo.concatenate %17, %18, %19, %20, %21, dim = 0 : (tensor<1x3x10xf32>, tensor<1x3x10xf32>, tensor<1x3x10xf32>, tensor<1x3x10xf32>, tensor<1x3x10xf32>) -> tensor<5x3x10xf32>
    return %22 : tensor<5x3x10xf32>
  }
}

// CHECK: module @reactant_mapped_sub attributes {mhlo.num_partitions = 1 : i64, mhlo.num_replicas = 1 : i64} {
// CHECK-NEXT:   func.func @main(%arg0: tensor<3x5x10xf32>, %arg1: tensor<3x5x10xf32>) -> tensor<5x3x10xf32> attributes {enzymexla.memory_effects = []} {
// CHECK-NEXT:     %0 = stablehlo.slice %arg0 [0:3, 4:5, 0:10] : (tensor<3x5x10xf32>) -> tensor<3x1x10xf32>
// CHECK-NEXT:     %1 = stablehlo.slice %arg0 [0:3, 3:4, 0:10] : (tensor<3x5x10xf32>) -> tensor<3x1x10xf32>
// CHECK-NEXT:     %2 = stablehlo.slice %arg0 [0:3, 2:3, 0:10] : (tensor<3x5x10xf32>) -> tensor<3x1x10xf32>
// CHECK-NEXT:     %3 = stablehlo.slice %arg0 [0:3, 1:2, 0:10] : (tensor<3x5x10xf32>) -> tensor<3x1x10xf32>
// CHECK-NEXT:     %4 = stablehlo.slice %arg0 [0:3, 0:1, 0:10] : (tensor<3x5x10xf32>) -> tensor<3x1x10xf32>
// CHECK-NEXT:     %5 = stablehlo.broadcast_in_dim %4, dims = [3, 2, 1] : (tensor<3x1x10xf32>) -> tensor<1x10x1x3xf32>
// CHECK-NEXT:     %6 = stablehlo.broadcast_in_dim %3, dims = [3, 2, 1] : (tensor<3x1x10xf32>) -> tensor<1x10x1x3xf32>
// CHECK-NEXT:     %7 = stablehlo.broadcast_in_dim %2, dims = [3, 2, 1] : (tensor<3x1x10xf32>) -> tensor<1x10x1x3xf32>
// CHECK-NEXT:     %8 = stablehlo.broadcast_in_dim %1, dims = [3, 2, 1] : (tensor<3x1x10xf32>) -> tensor<1x10x1x3xf32>
// CHECK-NEXT:     %9 = stablehlo.broadcast_in_dim %0, dims = [3, 2, 1] : (tensor<3x1x10xf32>) -> tensor<1x10x1x3xf32>
// CHECK-NEXT:     %10 = stablehlo.concatenate %5, %6, %7, %8, %9, dim = 0 : (tensor<1x10x1x3xf32>, tensor<1x10x1x3xf32>, tensor<1x10x1x3xf32>, tensor<1x10x1x3xf32>, tensor<1x10x1x3xf32>) -> tensor<5x10x1x3xf32>
// CHECK-NEXT:     %11 = stablehlo.broadcast_in_dim %arg1, dims = [3, 0, 1] : (tensor<3x5x10xf32>) -> tensor<5x10x1x3xf32>
// CHECK-NEXT:     %12 = stablehlo.subtract %10, %11 : tensor<5x10x1x3xf32>
// CHECK-NEXT:     %13 = stablehlo.transpose %12, dims = [0, 2, 3, 1] : (tensor<5x10x1x3xf32>) -> tensor<5x1x3x10xf32>
// CHECK-NEXT:     %14 = stablehlo.slice %13 [0:1, 0:1, 0:3, 0:10] : (tensor<5x1x3x10xf32>) -> tensor<1x1x3x10xf32>
// CHECK-NEXT:     %15 = stablehlo.reshape %14 : (tensor<1x1x3x10xf32>) -> tensor<1x3x10xf32>
// CHECK-NEXT:     %16 = stablehlo.slice %13 [1:2, 0:1, 0:3, 0:10] : (tensor<5x1x3x10xf32>) -> tensor<1x1x3x10xf32>
// CHECK-NEXT:     %17 = stablehlo.reshape %16 : (tensor<1x1x3x10xf32>) -> tensor<1x3x10xf32>
// CHECK-NEXT:     %18 = stablehlo.slice %13 [2:3, 0:1, 0:3, 0:10] : (tensor<5x1x3x10xf32>) -> tensor<1x1x3x10xf32>
// CHECK-NEXT:     %19 = stablehlo.reshape %18 : (tensor<1x1x3x10xf32>) -> tensor<1x3x10xf32>
// CHECK-NEXT:     %20 = stablehlo.slice %13 [3:4, 0:1, 0:3, 0:10] : (tensor<5x1x3x10xf32>) -> tensor<1x1x3x10xf32>
// CHECK-NEXT:     %21 = stablehlo.reshape %20 : (tensor<1x1x3x10xf32>) -> tensor<1x3x10xf32>
// CHECK-NEXT:     %22 = stablehlo.slice %13 [4:5, 0:1, 0:3, 0:10] : (tensor<5x1x3x10xf32>) -> tensor<1x1x3x10xf32>
// CHECK-NEXT:     %23 = stablehlo.reshape %22 : (tensor<1x1x3x10xf32>) -> tensor<1x3x10xf32>
// CHECK-NEXT:     %24 = stablehlo.concatenate %15, %17, %19, %21, %23, dim = 0 : (tensor<1x3x10xf32>, tensor<1x3x10xf32>, tensor<1x3x10xf32>, tensor<1x3x10xf32>, tensor<1x3x10xf32>) -> tensor<5x3x10xf32>
// CHECK-NEXT:     return %24 : tensor<5x3x10xf32>
// CHECK-NEXT:   }
// CHECK-NEXT: }
