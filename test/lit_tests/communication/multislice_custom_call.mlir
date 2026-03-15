// RUN: enzymexlamlir-opt %s --optimize-communication="multislice_custom_call=1" | FileCheck %s

module  {
  sdy.mesh @mesh = <["a"=2]>
  func.func public @main(%arg0: tensor<10xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", ?}]>}) -> (tensor<7xf32>, tensor<7xf32>, tensor<7xf32>) {
    %1, %2, %3 = "enzymexla.multi_slice"(%arg0) {
      dimension = 0 : i32, 
      amount = 2 : i32, 
      start_indices = array<i64: 0>,
      limit_indices = array<i64: 7>,
      strides = array<i64: 1>,
      sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", ?}]>, <@mesh, [{"a", ?}]>, <@mesh, [{"a", ?}]>]>
    } : (tensor<10xf32>) -> (tensor<7xf32>, tensor<7xf32>, tensor<7xf32>)
    return %1, %2, %3 : tensor<7xf32>, tensor<7xf32>, tensor<7xf32>
  }
}

// CHECK:  func.func public @main(%arg0: tensor<10xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", ?}]>}) -> (tensor<7xf32>, tensor<7xf32>, tensor<7xf32>) {
// CHECK-NEXT:    %0:3 = stablehlo.custom_call @_SPMDInternalOp_MultiSlice(%arg0) {backend_config = "dimension=0,amount=2,start_indices=[0],limit_indices=[7],strides=[1],bufferize=0", sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", ?}]>, <@mesh, [{"a", ?}]>, <@mesh, [{"a", ?}]>]>} : (tensor<10xf32>) -> (tensor<7xf32>, tensor<7xf32>, tensor<7xf32>)
// CHECK-NEXT:    return %0#0, %0#1, %0#2 : tensor<7xf32>, tensor<7xf32>, tensor<7xf32>
// CHECK-NEXT:  }


module {
  sdy.mesh @mesh = <["x"=2, "y"=2]>
  func.func public @main(%arg0: tensor<20x1536x3056xf64> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}, {"x"}]>}) -> (tensor<4x1520x3056xf64>, tensor<4x1520x3056xf64>) {
    %0:2 = "enzymexla.multi_slice"(%arg0) {
      amount = 1 : i32,
      dimension = 1 : i32,
      limit_indices = array<i64: 12, 1529, 3056>,
      start_indices = array<i64: 8, 9, 0>,
      strides = array<i64: 1, 1, 1>,
      sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y"}, {"x"}]>, <@mesh, [{}, {"y"}, {"x"}]>]>
    } : (tensor<20x1536x3056xf64>) -> (tensor<4x1520x3056xf64>, tensor<4x1520x3056xf64>)
    return %0#0, %0#1 : tensor<4x1520x3056xf64>, tensor<4x1520x3056xf64>
  }
}

// CHECK:  func.func public @main(%arg0: tensor<20x1536x3056xf64> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}, {"x"}]>}) -> (tensor<4x1520x3056xf64>, tensor<4x1520x3056xf64>) {
// CHECK-NEXT:    %[[SLICE:.*]] = stablehlo.slice %arg0 [8:12, 0:1536, 0:3056] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y"}, {"x"}]>]>} : (tensor<20x1536x3056xf64>) -> tensor<4x1536x3056xf64>
// CHECK-NEXT:    %[[CC:.*]]:2 = stablehlo.custom_call @_SPMDInternalOp_MultiSlice(%[[SLICE]]) {backend_config = "dimension=1,amount=1,start_indices=[0, 9, 0],limit_indices=[4, 1529, 3056],strides=[1, 1, 1],bufferize=0", sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y"}, {"x"}]>, <@mesh, [{}, {"y"}, {"x"}]>]>} : (tensor<4x1536x3056xf64>) -> (tensor<4x1520x3056xf64>, tensor<4x1520x3056xf64>)
// CHECK-NEXT:    return %[[CC]]#0, %[[CC]]#1 : tensor<4x1520x3056xf64>, tensor<4x1520x3056xf64>
// CHECK-NEXT:  }
