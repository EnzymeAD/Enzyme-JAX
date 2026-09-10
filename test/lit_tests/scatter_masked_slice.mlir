// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt | FileCheck %s

// A store masked by `lane < 6` over a 1-D lane grid: the live box is lanes
// 0..5, so the scatter is sliced to them and becomes an update of that range.
func.func @prefix(%input: tensor<16xf32>, %update: tensor<10xf32>) -> tensor<16xf32> {
  %c6 = stablehlo.constant dense<6> : tensor<10x1xi64>
  %off = stablehlo.constant dense<-1> : tensor<10x1xi64>
  %iota = stablehlo.iota dim = 0 : tensor<10x1xi64>
  %pred = stablehlo.compare LT, %iota, %c6 : (tensor<10x1xi64>, tensor<10x1xi64>) -> tensor<10x1xi1>
  %idx = stablehlo.select %pred, %iota, %off : tensor<10x1xi1>, tensor<10x1xi64>
  %0 = "stablehlo.scatter"(%input, %idx, %update) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
  ^bb0(%a: tensor<f32>, %b: tensor<f32>):
    stablehlo.return %b : tensor<f32>
  }) : (tensor<16xf32>, tensor<10x1xi64>, tensor<10xf32>) -> tensor<16xf32>
  return %0 : tensor<16xf32>
}

// A box in a 2-D lane grid: rows below 5 and columns from 2 on.
func.func @box(%input: tensor<100xf32>, %update: tensor<10x10xf32>) -> tensor<100xf32> {
  %c5 = stablehlo.constant dense<5> : tensor<10x10x1xi64>
  %c2 = stablehlo.constant dense<2> : tensor<10x10x1xi64>
  %c10 = stablehlo.constant dense<10> : tensor<10xi64>
  %off = stablehlo.constant dense<-1> : tensor<10x10x1xi64>
  %i = stablehlo.iota dim = 0 : tensor<10xi64>
  %row = stablehlo.multiply %i, %c10 : tensor<10xi64>
  %rowb = stablehlo.broadcast_in_dim %row, dims = [0] : (tensor<10xi64>) -> tensor<10x10x1xi64>
  %rows = stablehlo.iota dim = 0 : tensor<10x10x1xi64>
  %cols = stablehlo.iota dim = 1 : tensor<10x10x1xi64>
  %lin = stablehlo.add %rowb, %cols : tensor<10x10x1xi64>
  %p0 = stablehlo.compare LT, %rows, %c5 : (tensor<10x10x1xi64>, tensor<10x10x1xi64>) -> tensor<10x10x1xi1>
  %p1 = stablehlo.compare GE, %cols, %c2 : (tensor<10x10x1xi64>, tensor<10x10x1xi64>) -> tensor<10x10x1xi1>
  %pred = stablehlo.and %p0, %p1 : tensor<10x10x1xi1>
  %idx = stablehlo.select %pred, %lin, %off : tensor<10x10x1xi1>, tensor<10x10x1xi64>
  %0 = "stablehlo.scatter"(%input, %idx, %update) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 2>, unique_indices = false}> ({
  ^bb0(%a: tensor<f32>, %b: tensor<f32>):
    stablehlo.return %b : tensor<f32>
  }) : (tensor<100xf32>, tensor<10x10x1xi64>, tensor<10x10xf32>) -> tensor<100xf32>
  return %0 : tensor<100xf32>
}

// The bound is a runtime value: no static box, the scatter stays.
func.func @dynamic_bound(%input: tensor<16xf32>, %update: tensor<10xf32>, %n: tensor<i64>) -> tensor<16xf32> {
  %off = stablehlo.constant dense<-1> : tensor<10x1xi64>
  %iota = stablehlo.iota dim = 0 : tensor<10x1xi64>
  %nb = stablehlo.broadcast_in_dim %n, dims = [] : (tensor<i64>) -> tensor<10x1xi64>
  %pred = stablehlo.compare LT, %iota, %nb : (tensor<10x1xi64>, tensor<10x1xi64>) -> tensor<10x1xi1>
  %idx = stablehlo.select %pred, %iota, %off : tensor<10x1xi1>, tensor<10x1xi64>
  %0 = "stablehlo.scatter"(%input, %idx, %update) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
  ^bb0(%a: tensor<f32>, %b: tensor<f32>):
    stablehlo.return %b : tensor<f32>
  }) : (tensor<16xf32>, tensor<10x1xi64>, tensor<10xf32>) -> tensor<16xf32>
  return %0 : tensor<16xf32>
}

// A mask that is not one interval along the lane: the scatter stays.
func.func @two_intervals(%input: tensor<16xf32>, %update: tensor<10xf32>) -> tensor<16xf32> {
  %c6 = stablehlo.constant dense<6> : tensor<10x1xi64>
  %off = stablehlo.constant dense<-1> : tensor<10x1xi64>
  %iota = stablehlo.iota dim = 0 : tensor<10x1xi64>
  %pred = stablehlo.compare NE, %iota, %c6 : (tensor<10x1xi64>, tensor<10x1xi64>) -> tensor<10x1xi1>
  %idx = stablehlo.select %pred, %iota, %off : tensor<10x1xi1>, tensor<10x1xi64>
  %0 = "stablehlo.scatter"(%input, %idx, %update) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
  ^bb0(%a: tensor<f32>, %b: tensor<f32>):
    stablehlo.return %b : tensor<f32>
  }) : (tensor<16xf32>, tensor<10x1xi64>, tensor<10xf32>) -> tensor<16xf32>
  return %0 : tensor<16xf32>
}

// CHECK:      func.func @prefix(%arg0: tensor<16xf32>, %arg1: tensor<10xf32>) -> tensor<16xf32> {
// CHECK-NEXT:    %0 = stablehlo.slice %arg1 [0:6] : (tensor<10xf32>) -> tensor<6xf32>
// CHECK-NEXT:    %1 = stablehlo.slice %arg0 [6:16] : (tensor<16xf32>) -> tensor<10xf32>
// CHECK-NEXT:    %2 = stablehlo.concatenate %0, %1, dim = 0 : (tensor<6xf32>, tensor<10xf32>) -> tensor<16xf32>
// CHECK-NEXT:    return %2 : tensor<16xf32>
// CHECK-NEXT:  }
// CHECK-NEXT:  func.func @box(%arg0: tensor<100xf32>, %arg1: tensor<10x10xf32>) -> tensor<100xf32> {
// CHECK-NEXT:    %c = stablehlo.constant dense<2> : tensor<i32>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<0> : tensor<i32>
// CHECK-NEXT:    %0 = stablehlo.slice %arg1 [0:5, 2:10] : (tensor<10x10xf32>) -> tensor<5x8xf32>
// CHECK-NEXT:    %1 = stablehlo.reshape %arg0 : (tensor<100xf32>) -> tensor<10x10xf32>
// CHECK-NEXT:    %2 = stablehlo.dynamic_update_slice %1, %0, %c_0, %c : (tensor<10x10xf32>, tensor<5x8xf32>, tensor<i32>, tensor<i32>) -> tensor<10x10xf32>
// CHECK-NEXT:    %3 = stablehlo.reshape %2 : (tensor<10x10xf32>) -> tensor<100xf32>
// CHECK-NEXT:    return %3 : tensor<100xf32>
// CHECK-NEXT:  }
// CHECK-NEXT:  func.func @dynamic_bound(%arg0: tensor<16xf32>, %arg1: tensor<10xf32>, %arg2: tensor<i64>) -> tensor<16xf32> {
// CHECK-NEXT:    %c = stablehlo.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]> : tensor<10xi64>
// CHECK-NEXT:    %0 = stablehlo.broadcast_in_dim %arg2, dims = [] {enzymexla.non_negative = [#enzymexla<guaranteed NOTGUARANTEED>]} : (tensor<i64>) -> tensor<10xi64>
// CHECK-NEXT:    %1 = stablehlo.compare LT, %c, %0 : (tensor<10xi64>, tensor<10xi64>) -> tensor<10xi1>
// CHECK-NEXT:    %2 = stablehlo.slice %arg0 [0:10] : (tensor<16xf32>) -> tensor<10xf32>
// CHECK-NEXT:    %3 = stablehlo.select %1, %arg1, %2 : tensor<10xi1>, tensor<10xf32>
// CHECK-NEXT:    %4 = stablehlo.slice %arg0 [10:16] : (tensor<16xf32>) -> tensor<6xf32>
// CHECK-NEXT:    %5 = stablehlo.concatenate %3, %4, dim = 0 : (tensor<10xf32>, tensor<6xf32>) -> tensor<16xf32>
// CHECK-NEXT:    return %5 : tensor<16xf32>
// CHECK-NEXT:  }
// CHECK-NEXT:  func.func @two_intervals(%arg0: tensor<16xf32>, %arg1: tensor<10xf32>) -> tensor<16xf32> {
// CHECK-NEXT:    %c = stablehlo.constant dense<[true, true, true, true, true, true, false, true, true, true]> : tensor<10xi1>
// CHECK-NEXT:    %0 = stablehlo.slice %arg0 [0:10] : (tensor<16xf32>) -> tensor<10xf32>
// CHECK-NEXT:    %1 = stablehlo.select %c, %arg1, %0 : tensor<10xi1>, tensor<10xf32>
// CHECK-NEXT:    %2 = stablehlo.slice %arg0 [10:16] : (tensor<16xf32>) -> tensor<6xf32>
// CHECK-NEXT:    %3 = stablehlo.concatenate %1, %2, dim = 0 : (tensor<10xf32>, tensor<6xf32>) -> tensor<16xf32>
// CHECK-NEXT:    return %3 : tensor<16xf32>
// CHECK-NEXT:  }
