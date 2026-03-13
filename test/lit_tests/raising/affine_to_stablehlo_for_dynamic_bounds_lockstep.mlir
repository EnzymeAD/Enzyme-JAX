// RUN: enzymexlamlir-opt %s --raise-affine-to-stablehlo | FileCheck %s

#map_lb = affine_map<(d0) -> (d0, 0)>
#map_ub = affine_map<(d0) -> (100, d0)>

module {
  func.func private @dynamic_for_lockstep(%bounds: memref<2xi64>, %tensor: memref<100xf32>) {
    %lb_i64 = affine.load %bounds[0] : memref<2xi64>
    %ub_i64 = affine.load %bounds[1] : memref<2xi64>
    %lb = arith.index_cast %lb_i64 : i64 to index
    %ub = arith.index_cast %ub_i64 : i64 to index
    affine.for %i = max #map_lb(%lb) to min #map_ub(%ub) {
      %val = affine.load %tensor[%i] : memref<100xf32>
      %res = arith.addf %val, %val : f32
      affine.store %res, %tensor[%i] : memref<100xf32>
    }
    return
  }

  func.func @main(%arg0: tensor<2xi64>, %arg1: tensor<100xf32>) -> tensor<100xf32> {
    %0 = enzymexla.jit_call @dynamic_for_lockstep(%arg0, %arg1) {output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 1, operand_tuple_indices = []>]} : (tensor<2xi64>, tensor<100xf32>) -> tensor<100xf32>
    return %0 : tensor<100xf32>
  }
}





// CHECK:  func.func @main(%arg0: tensor<2xi64>, %arg1: tensor<100xf32>) -> tensor<100xf32> {
// CHECK:    %0:2 = call @dynamic_for_lockstep_raised(%arg0, %arg1) : (tensor<2xi64>, tensor<100xf32>) -> (tensor<2xi64>, tensor<100xf32>)
// CHECK:    return %0#1 : tensor<100xf32>
// CHECK:  func.func private @dynamic_for_lockstep_raised(%arg0: tensor<2xi64>, %arg1: tensor<100xf32>) -> (tensor<2xi64>, tensor<100xf32>) {
// CHECK:    %0 = stablehlo.slice %arg0 [0:1] : (tensor<2xi64>) -> tensor<1xi64>
// CHECK:    %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
// CHECK:    %2 = stablehlo.slice %arg0 [1:2] : (tensor<2xi64>) -> tensor<1xi64>
// CHECK:    %3 = stablehlo.reshape %2 : (tensor<1xi64>) -> tensor<i64>
// CHECK:    %c = stablehlo.constant dense<0> : tensor<i64>
// CHECK:    %4 = stablehlo.maximum %1, %c : tensor<i64>
// CHECK:    %c_0 = stablehlo.constant dense<100> : tensor<i64>
// CHECK:    %5 = stablehlo.minimum %c_0, %3 : tensor<i64>
// CHECK:    %c_1 = stablehlo.constant dense<1> : tensor<i64>
// CHECK:    %c_2 = stablehlo.constant dense<-1> : tensor<i64>
// CHECK:    %6 = stablehlo.subtract %5, %4 : tensor<i64>
// CHECK:    %7 = stablehlo.add %c_1, %c_2 : tensor<i64>
// CHECK:    %8 = stablehlo.add %6, %7 : tensor<i64>
// CHECK:    %9 = stablehlo.divide %8, %c_1 : tensor<i64>
// CHECK:    %c_3 = stablehlo.constant dense<0> : tensor<i64>
// CHECK:    %10 = stablehlo.maximum %9, %c_3 : tensor<i64>
// CHECK:    %11 = stablehlo.reshape %10 : (tensor<i64>) -> tensor<1xi64>
// CHECK:    %12 = stablehlo.dynamic_iota %11, dim = 0 : (tensor<1xi64>) -> tensor<?xi64>
// CHECK:    %13 = stablehlo.dynamic_broadcast_in_dim %4, %11, dims = [] : (tensor<i64>, tensor<1xi64>) -> tensor<?xi64>
// CHECK:    %14 = stablehlo.dynamic_broadcast_in_dim %c_1, %11, dims = [] : (tensor<i64>, tensor<1xi64>) -> tensor<?xi64>
// CHECK:    %15 = stablehlo.multiply %12, %14 : tensor<?xi64>
// CHECK:    %16 = stablehlo.add %15, %13 : tensor<?xi64>
// CHECK:    %17 = stablehlo.get_dimension_size %16, dim = 0 : (tensor<?xi64>) -> tensor<i32>
// CHECK:    %18 = stablehlo.convert %17 : (tensor<i32>) -> tensor<i64>
// CHECK:    %19 = stablehlo.reshape %18 : (tensor<i64>) -> tensor<1xi64>
// CHECK:    %c_4 = stablehlo.constant dense<1> : tensor<1xi64>
// CHECK:    %20 = stablehlo.concatenate %19, %c_4, dim = 0 : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK:    %21 = stablehlo.dynamic_reshape %16, %20 : (tensor<?xi64>, tensor<2xi64>) -> tensor<?x1xi64>
// CHECK:    %c_5 = stablehlo.constant dense<1> : tensor<i64>
// CHECK:    %22 = stablehlo.get_dimension_size %21, dim = 0 : (tensor<?x1xi64>) -> tensor<i32>
// CHECK:    %23 = stablehlo.convert %22 : (tensor<i32>) -> tensor<i64>
// CHECK:    %24 = stablehlo.multiply %c_5, %23 : tensor<i64>
// CHECK:    %25 = stablehlo.reshape %24 : (tensor<i64>) -> tensor<1xi64>
// CHECK:    %c_6 = stablehlo.constant dense<1> : tensor<1xi64>
// CHECK:    %26 = stablehlo.concatenate %25, %c_6, dim = 0 : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK:    %27 = stablehlo.dynamic_reshape %21, %26 : (tensor<?x1xi64>, tensor<2xi64>) -> tensor<?x1xi64>
// CHECK:    %28 = "stablehlo.gather"(%arg1, %27) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<100xf32>, tensor<?x1xi64>) -> tensor<?xf32>
// CHECK:    %29 = arith.addf %28, %28 : tensor<?xf32>
// CHECK:    %c_7 = stablehlo.constant dense<1> : tensor<i64>
// CHECK:    %30 = stablehlo.get_dimension_size %16, dim = 0 : (tensor<?xi64>) -> tensor<i32>
// CHECK:    %31 = stablehlo.convert %30 : (tensor<i32>) -> tensor<i64>
// CHECK:    %32 = stablehlo.multiply %c_7, %31 : tensor<i64>
// CHECK:    %33 = stablehlo.reshape %32 : (tensor<i64>) -> tensor<1xi64>
// CHECK:    %c_8 = stablehlo.constant dense<1> : tensor<1xi64>
// CHECK:    %34 = stablehlo.concatenate %33, %c_8, dim = 0 : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK:    %35 = stablehlo.dynamic_reshape %16, %34 : (tensor<?xi64>, tensor<2xi64>) -> tensor<?x1xi64>
// CHECK:    %36 = "stablehlo.scatter"(%arg1, %35, %29) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = true}> ({
// CHECK:    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
// CHECK:      stablehlo.return %arg3 : tensor<f32>
// CHECK:    }) : (tensor<100xf32>, tensor<?x1xi64>, tensor<?xf32>) -> tensor<100xf32>
// CHECK:    return %arg0, %36 : tensor<2xi64>, tensor<100xf32>
