// RUN: enzymexlamlir-opt %s --raise-affine-to-stablehlo --canonicalize | FileCheck %s 

module {
  func.func @main(%arg0: tensor<3x4xf32> {enzymexla.memory_effects = []}, %arg1: tensor<3x4xi1> {enzymexla.memory_effects = []}, %arg2: tensor<3x4xf32> {enzymexla.memory_effects = []}) -> (tensor<3x4xf32>, tensor<3x4xi1>, tensor<3x4xf32>) attributes {enzymexla.memory_effects = ["read", "write", "allocate", "free"]} {
    %0 = enzymexla.jit_call @masked_kernel (%arg0, %arg1, %arg2) <operand_layouts = [dense<[1, 0]> : tensor<2xindex>, dense<[1, 0]> : tensor<2xindex>, dense<[1, 0]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>], xla_side_effect_free> : (tensor<3x4xf32>, tensor<3x4xi1>, tensor<3x4xf32>) -> tensor<3x4xf32>
    return %0, %arg1, %arg2 : tensor<3x4xf32>, tensor<3x4xi1>, tensor<3x4xf32>
  }
  func.func private @masked_kernel(%arg0: memref<3x4xf32, 1>, %arg1: memref<3x4xi1, 1>, %arg2: memref<3x4xf32, 1>) attributes {enzymexla.memory_effects = ["read", "write", "allocate", "free"]} {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = "enzymexla.memref2pointer"(%arg1) : (memref<3x4xi1, 1>) -> !llvm.ptr<1>

    %1 = "enzymexla.pointer2memref"(%0) : (!llvm.ptr<1>) -> memref<?xi8, 1>
    affine.parallel (%arg3, %arg4) = (0, 0) to (3, 4) {
      %2 = affine.load %1[%arg3 * 4 + %arg4] {alignment = 1 : i64} : memref<?xi8, 1>
      %3 = affine.load %arg2[%arg3, %arg4] {alignment = 4 : i64} : memref<3x4xf32, 1>
      %4 = arith.trunci %2 : i8 to i1
      %5 = arith.select %4, %cst, %3 : f32
      affine.store %5, %arg0[%arg3, %arg4] {alignment = 4 : i64} : memref<3x4xf32, 1>
    }
    return
  }
}

// CHECK:  func.func private @masked_kernel_raised(%arg0: tensor<3x4xf32>, %arg1: tensor<3x4xi1>, %arg2: tensor<3x4xf32>) -> (tensor<3x4xf32>, tensor<3x4xi1>, tensor<3x4xf32>) {
// CHECK-NEXT:    %c = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:    %0 = stablehlo.convert %arg1 : (tensor<3x4xi1>) -> tensor<3x4xi8>
// CHECK-NEXT:    %1 = stablehlo.reshape %0 : (tensor<3x4xi8>) -> tensor<12xi8>
// CHECK-NEXT:    %2 = arith.trunci %1 : tensor<12xi8> to tensor<12xi1>
// CHECK-NEXT:    %3 = stablehlo.reshape %2 : (tensor<12xi1>) -> tensor<3x4xi1>
// CHECK-NEXT:    %4 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3x4xf32>
// CHECK-NEXT:    %5 = arith.select %3, %4, %arg2 : tensor<3x4xi1>, tensor<3x4xf32>
// CHECK-NEXT:    %6 = stablehlo.dynamic_update_slice %arg0, %5, %c, %c : (tensor<3x4xf32>, tensor<3x4xf32>, tensor<i64>, tensor<i64>) -> tensor<3x4xf32>
// CHECK-NEXT:    return %6, %arg1, %arg2 : tensor<3x4xf32>, tensor<3x4xi1>, tensor<3x4xf32>
// CHECK-NEXT:  }
