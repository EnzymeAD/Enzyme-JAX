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
// CHECK-NEXT:     %0 = stablehlo.transpose %arg0, dims = [2, 1, 0] : (tensor<3x5x10xf32>) -> tensor<10x5x3xf32>
// CHECK-NEXT:     %1 = stablehlo.slice %0 [0:10, 0:1, 0:3] : (tensor<10x5x3xf32>) -> tensor<10x1x3xf32>
// CHECK-NEXT:     %2 = stablehlo.slice %0 [0:10, 4:5, 0:3] : (tensor<10x5x3xf32>) -> tensor<10x1x3xf32>
// CHECK-NEXT:     %3 = stablehlo.slice %0 [0:10, 3:4, 0:3] : (tensor<10x5x3xf32>) -> tensor<10x1x3xf32>
// CHECK-NEXT:     %4 = stablehlo.slice %0 [0:10, 2:3, 0:3] : (tensor<10x5x3xf32>) -> tensor<10x1x3xf32>
// CHECK-NEXT:     %5 = stablehlo.slice %0 [0:10, 1:2, 0:3] : (tensor<10x5x3xf32>) -> tensor<10x1x3xf32>
// CHECK-NEXT:     %6 = stablehlo.reshape %1 : (tensor<10x1x3xf32>) -> tensor<1x10x1x3xf32>
// CHECK-NEXT:     %7 = stablehlo.reshape %5 : (tensor<10x1x3xf32>) -> tensor<1x10x1x3xf32>
// CHECK-NEXT:     %8 = stablehlo.reshape %4 : (tensor<10x1x3xf32>) -> tensor<1x10x1x3xf32>
// CHECK-NEXT:     %9 = stablehlo.reshape %3 : (tensor<10x1x3xf32>) -> tensor<1x10x1x3xf32>
// CHECK-NEXT:     %10 = stablehlo.reshape %2 : (tensor<10x1x3xf32>) -> tensor<1x10x1x3xf32>
// CHECK-NEXT:     %11 = stablehlo.concatenate %6, %7, %8, %9, %10, dim = 0 : (tensor<1x10x1x3xf32>, tensor<1x10x1x3xf32>, tensor<1x10x1x3xf32>, tensor<1x10x1x3xf32>, tensor<1x10x1x3xf32>) -> tensor<5x10x1x3xf32>
// CHECK-NEXT:     %12 = stablehlo.broadcast_in_dim %arg1, dims = [3, 0, 1] : (tensor<3x5x10xf32>) -> tensor<5x10x1x3xf32>
// CHECK-NEXT:     %13 = stablehlo.subtract %11, %12 : tensor<5x10x1x3xf32>
// CHECK-NEXT:     %14 = stablehlo.transpose %13, dims = [0, 2, 3, 1] : (tensor<5x10x1x3xf32>) -> tensor<5x1x3x10xf32>
// CHECK-NEXT:     %15 = stablehlo.slice %14 [0:1, 0:1, 0:3, 0:10] : (tensor<5x1x3x10xf32>) -> tensor<1x1x3x10xf32>
// CHECK-NEXT:     %16 = stablehlo.reshape %15 : (tensor<1x1x3x10xf32>) -> tensor<1x3x10xf32>
// CHECK-NEXT:     %17 = stablehlo.slice %14 [1:2, 0:1, 0:3, 0:10] : (tensor<5x1x3x10xf32>) -> tensor<1x1x3x10xf32>
// CHECK-NEXT:     %18 = stablehlo.reshape %17 : (tensor<1x1x3x10xf32>) -> tensor<1x3x10xf32>
// CHECK-NEXT:     %19 = stablehlo.slice %14 [2:3, 0:1, 0:3, 0:10] : (tensor<5x1x3x10xf32>) -> tensor<1x1x3x10xf32>
// CHECK-NEXT:     %20 = stablehlo.reshape %19 : (tensor<1x1x3x10xf32>) -> tensor<1x3x10xf32>
// CHECK-NEXT:     %21 = stablehlo.slice %14 [3:4, 0:1, 0:3, 0:10] : (tensor<5x1x3x10xf32>) -> tensor<1x1x3x10xf32>
// CHECK-NEXT:     %22 = stablehlo.reshape %21 : (tensor<1x1x3x10xf32>) -> tensor<1x3x10xf32>
// CHECK-NEXT:     %23 = stablehlo.slice %14 [4:5, 0:1, 0:3, 0:10] : (tensor<5x1x3x10xf32>) -> tensor<1x1x3x10xf32>
// CHECK-NEXT:     %24 = stablehlo.reshape %23 : (tensor<1x1x3x10xf32>) -> tensor<1x3x10xf32>
// CHECK-NEXT:     %25 = stablehlo.concatenate %16, %18, %20, %22, %24, dim = 0 : (tensor<1x3x10xf32>, tensor<1x3x10xf32>, tensor<1x3x10xf32>, tensor<1x3x10xf32>, tensor<1x3x10xf32>) -> tensor<5x3x10xf32>
// CHECK-NEXT:     return %25 : tensor<5x3x10xf32>
// CHECK-NEXT:   }
// CHECK-NEXT: }
