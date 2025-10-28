// RUN: enzymexlamlir-opt --enzyme-batch --cse %s | FileCheck %s

module @reactant_onehotb... attributes {mhlo.num_partitions = 1 : i64, mhlo.num_replicas = 1 : i64} {
  func.func private @"==_broadcast_scalar"(%arg0: tensor<i64> {enzymexla.memory_effects = []}, %arg1: tensor<i64> {enzymexla.memory_effects = []}) -> (tensor<i1>, tensor<i64>, tensor<i64>) attributes {enzymexla.memory_effects = []} {
    %0 = stablehlo.compare  EQ, %arg0, %arg1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    return %0, %arg0, %arg1 : tensor<i1>, tensor<i64>, tensor<i64>
  }
  func.func private @identity_broadcast_scalar(%arg0: tensor<i1> {enzymexla.memory_effects = []}) -> tensor<i1> attributes {enzymexla.memory_effects = []} {
    return %arg0 : tensor<i1>
  }
  func.func private @unbatched_findfirst(%arg0: tensor<2x3x5xi1> {enzymexla.memory_effects = ["read", "write", "allocate", "free"]}) -> tensor<i64> attributes {enzymexla.memory_effects = ["read", "write", "allocate", "free"]} {
    %c = stablehlo.constant dense<1> : tensor<i64>
    %c_0 = stablehlo.constant dense<9223372036854775807> : tensor<i64>
    %c_1 = stablehlo.constant dense<false> : tensor<i1>
    %c_2 = stablehlo.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]> : tensor<30xi64>
    %0 = enzyme.batch @identity_broadcast_scalar(%arg0) {batch_shape = array<i64: 2, 3, 5>} : (tensor<2x3x5xi1>) -> tensor<2x3x5xi1>
    %1 = stablehlo.transpose %0, dims = [2, 1, 0] : (tensor<2x3x5xi1>) -> tensor<5x3x2xi1>
    %2 = stablehlo.reshape %1 : (tensor<5x3x2xi1>) -> tensor<30xi1>
    %3:2 = stablehlo.reduce(%2 init: %c_1), (%c_2 init: %c_0) across dimensions = [0] : (tensor<30xi1>, tensor<30xi64>, tensor<i1>, tensor<i64>) -> (tensor<i1>, tensor<i64>)
     reducer(%arg1: tensor<i1>, %arg3: tensor<i1>) (%arg2: tensor<i64>, %arg4: tensor<i64>)  {
      %5 = stablehlo.or %arg1, %arg3 : tensor<i1>
      %6 = stablehlo.minimum %arg2, %arg4 : tensor<i64>
      %7 = stablehlo.select %arg3, %arg4, %c_0 : tensor<i1>, tensor<i64>
      %8 = stablehlo.select %arg1, %6, %7 : tensor<i1>, tensor<i64>
      stablehlo.return %5, %8 : tensor<i1>, tensor<i64>
    }
    %4 = stablehlo.add %3#1, %c : tensor<i64>
    return %4 : tensor<i64>
  }
  func.func @main(%arg0: tensor<5x3x2xi64> {enzymexla.memory_effects = ["read", "write", "allocate", "free"]}) -> tensor<1x1x1x4xi64> attributes {enzymexla.memory_effects = ["read", "write", "allocate", "free"]} {
    %c = stablehlo.constant dense<"0x0A000000000000000A000000000000000A000000000000000A000000000000000A000000000000000A000000000000000A000000000000000A000000000000000A000000000000000A000000000000000A000000000000000A000000000000000A000000000000000A000000000000000A000000000000000A000000000000000A000000000000000A000000000000000A000000000000000A000000000000000A000000000000000A000000000000000A000000000000000A000000000000000A000000000000000A000000000000000A000000000000000A000000000000000A000000000000000A000000000000001400000000000000140000000000000014000000000000001400000000000000140000000000000014000000000000001400000000000000140000000000000014000000000000001400000000000000140000000000000014000000000000001400000000000000140000000000000014000000000000001400000000000000140000000000000014000000000000001400000000000000140000000000000014000000000000001400000000000000140000000000000014000000000000001400000000000000140000000000000014000000000000001400000000000000140000000000000014000000000000001E000000000000001E000000000000001E000000000000001E000000000000001E000000000000001E000000000000001E000000000000001E000000000000001E000000000000001E000000000000001E000000000000001E000000000000001E000000000000001E000000000000001E000000000000001E000000000000001E000000000000001E000000000000001E000000000000001E000000000000001E000000000000001E000000000000001E000000000000001E000000000000001E000000000000001E000000000000001E000000000000001E000000000000001E000000000000001E00000000000000280000000000000028000000000000002800000000000000280000000000000028000000000000002800000000000000280000000000000028000000000000002800000000000000280000000000000028000000000000002800000000000000280000000000000028000000000000002800000000000000280000000000000028000000000000002800000000000000280000000000000028000000000000002800000000000000280000000000000028000000000000002800000000000000280000000000000028000000000000002800000000000000280000000000000028000000000000002800000000000000"> : tensor<4x2x3x5xi64>
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [3, 2, 1] : (tensor<5x3x2xi64>) -> tensor<4x2x3x5xi64>
    %1:3 = enzyme.batch @"==_broadcast_scalar"(%0, %c) {batch_shape = array<i64: 4, 2, 3, 5>} : (tensor<4x2x3x5xi64>, tensor<4x2x3x5xi64>) -> (tensor<4x2x3x5xi1>, tensor<4x2x3x5xi64>, tensor<4x2x3x5xi64>)
    %2 = enzyme.batch @unbatched_findfirst(%1#0) {batch_shape = array<i64: 4>} : (tensor<4x2x3x5xi1>) -> tensor<4xi64>
    %3 = stablehlo.reshape %2 : (tensor<4xi64>) -> tensor<1x1x1x4xi64>
    return %3 : tensor<1x1x1x4xi64>
  }
}

// CHECK: func.func private @batched_unbatched_findfirst(%arg0: tensor<4x2x3x5xi1> {enzymexla.memory_effects = ["read", "write", "allocate", "free"]}) -> tensor<4xi64> attributes {enzymexla.memory_effects = ["read", "write", "allocate", "free"]} {
// CHECK-NEXT:     %c = stablehlo.constant dense<1> : tensor<4xi64>
// CHECK-NEXT:     %c_0 = stablehlo.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]> : tensor<30xi64>
// CHECK-NEXT:     %0 = stablehlo.broadcast_in_dim %c_0, dims = [1] : (tensor<30xi64>) -> tensor<4x30xi64>
// CHECK-NEXT:     %1 = call @batched_batched_identity_broadcast_scalar(%arg0) : (tensor<4x2x3x5xi1>) -> tensor<4x2x3x5xi1>
// CHECK-NEXT:     %2 = stablehlo.transpose %1, dims = [0, 3, 2, 1] : (tensor<4x2x3x5xi1>) -> tensor<4x5x3x2xi1>
// CHECK-NEXT:     %3 = stablehlo.reshape %2 : (tensor<4x5x3x2xi1>) -> tensor<4x30xi1>
// CHECK-NEXT:     %c_1 = stablehlo.constant dense<false> : tensor<i1>
// CHECK-NEXT:     %c_2 = stablehlo.constant dense<9223372036854775807> : tensor<i64>
// CHECK-NEXT:     %4:2 = stablehlo.reduce(%3 init: %c_1), (%0 init: %c_2) across dimensions = [1] : (tensor<4x30xi1>, tensor<4x30xi64>, tensor<i1>, tensor<i64>) -> (tensor<4xi1>, tensor<4xi64>)
// CHECK-NEXT:      reducer(%arg1: tensor<i1>, %arg3: tensor<i1>) (%arg2: tensor<i64>, %arg4: tensor<i64>)  {
// CHECK-NEXT:       %6 = stablehlo.or %arg1, %arg3 : tensor<i1>
// CHECK-NEXT:       %7 = stablehlo.minimum %arg2, %arg4 : tensor<i64>
// CHECK-NEXT:       %8 = stablehlo.select %arg3, %arg4, %c_2 : tensor<i1>, tensor<i64>
// CHECK-NEXT:       %9 = stablehlo.select %arg1, %7, %8 : tensor<i1>, tensor<i64>
// CHECK-NEXT:       stablehlo.return %6, %9 : tensor<i1>, tensor<i64>
// CHECK-NEXT:     }
// CHECK-NEXT:     %5 = stablehlo.add %4#1, %c : tensor<4xi64>
// CHECK-NEXT:     return %5 : tensor<4xi64>
// CHECK-NEXT: }
