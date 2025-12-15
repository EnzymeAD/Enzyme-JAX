// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt | FileCheck %s

module {
  // Simple test: two DUS with slices from the same tensor at start and end
  func.func @simple_dusdus_to_extend(%base: tensor<10x10xf32>, %source: tensor<5x10xf32>) -> tensor<10x10xf32> {
    %c0 = stablehlo.constant dense<0> : tensor<i32>
    %c2 = stablehlo.constant dense<2> : tensor<i32>
    %c4 = stablehlo.constant dense<4> : tensor<i32>
    
    // First DUS: insert slice from start of source (row 0)
    %slice_start = stablehlo.slice %source [0:1, 0:10] : (tensor<5x10xf32>) -> tensor<1x10xf32>
    %dus1 = stablehlo.dynamic_update_slice %base, %slice_start, %c2, %c0 : (tensor<10x10xf32>, tensor<1x10xf32>, tensor<i32>, tensor<i32>) -> tensor<10x10xf32>
    
    // Second DUS: insert slice from end of source (row 4)
    %slice_end = stablehlo.slice %source [4:5, 0:10] : (tensor<5x10xf32>) -> tensor<1x10xf32>
    %dus2 = stablehlo.dynamic_update_slice %dus1, %slice_end, %c4, %c0 : (tensor<10x10xf32>, tensor<1x10xf32>, tensor<i32>, tensor<i32>) -> tensor<10x10xf32>
    
    func.return %dus2 : tensor<10x10xf32>
  }
}

// CHECK-LABEL: func.func @simple_dusdus_to_extend
// CHECK-DAG: %[[SLICE:.*]] = stablehlo.slice %arg1 [1:4, 0:10]
// CHECK-DAG: %[[EXTEND:.*]] = enzymexla.extend(%[[SLICE]]) <{dimension = 0 : i64, lhs = 1 : i64, rhs = 1 : i64}>
// CHECK: stablehlo.dynamic_update_slice %arg0, %[[EXTEND]]
