// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt | FileCheck %s

module {
  func.func @dusdus_to_extend(%base: tensor<20x1536x3056xf32>, %source: tensor<4x1520x3056xf32>) -> tensor<20x1536x3056xf32> {
    %c0 = stablehlo.constant dense<0> : tensor<i32>
    %c16 = stablehlo.constant dense<16> : tensor<i32>
    %c1519 = stablehlo.constant dense<1519> : tensor<i32>
    
    // First DUS with a slice from source at position [0:4, 0:1, 0:3056]
    %slice1 = stablehlo.slice %source [0:4, 0:1, 0:3056] : (tensor<4x1520x3056xf32>) -> tensor<4x1x3056xf32>
    %dus1 = stablehlo.dynamic_update_slice %base, %slice1, %c16, %c0, %c0 : (tensor<20x1536x3056xf32>, tensor<4x1x3056xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<20x1536x3056xf32>
    
    // Second DUS with a slice from source at position [0:4, 1519:1520, 0:3056]
    %slice2 = stablehlo.slice %source [0:4, 1519:1520, 0:3056] : (tensor<4x1520x3056xf32>) -> tensor<4x1x3056xf32>
    %dus2 = stablehlo.dynamic_update_slice %dus1, %slice2, %c16, %c1519, %c0 : (tensor<20x1536x3056xf32>, tensor<4x1x3056xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<20x1536x3056xf32>
    
    func.return %dus2 : tensor<20x1536x3056xf32>
  }
}

// CHECK-LABEL: func.func @dusdus_to_extend
// CHECK-DAG: %[[SLICE:.*]] = stablehlo.slice %arg1 [0:4, 1:1519, 0:3056]
// CHECK-DAG: %[[EXTEND:.*]] = enzymexla.extend(%[[SLICE]]) <{dimension = 1 : i64, lhs = 1 : i64, rhs = 1 : i64}>
// CHECK: stablehlo.dynamic_update_slice %arg0, %[[EXTEND]]
