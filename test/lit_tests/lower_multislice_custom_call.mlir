// RUN: enzymexlamlir-opt %s --optimize-communication="multislice_custom_call=1" | FileCheck %s

module  {
  sdy.mesh @mesh = <["a"=2]>
  func.func public @main(%arg0: tensor<10xf32>) -> (tensor<3xf32>, tensor<3xf32>, tensor<3xf32>) {
//    CHECK: %[[RES:.*]]:3 = stablehlo.custom_call @[[SYMA:.*]](%arg0) {backend_config = "dimension=0,amount=2,start_indices=[0],limit_indices=[3],strides=[1]", sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", ?}]>, <@mesh, [{"a", ?}]>, <@mesh, [{"a", ?}]>]>} : (tensor<10xf32>) -> (tensor<3xf32>, tensor<3xf32>, tensor<3xf32>)
//    CHECK: return %[[RES]]#0, %[[RES]]#1, %[[RES]]#2 : tensor<3xf32>, tensor<3xf32>, tensor<3xf32>
    %1, %2, %3 = "enzymexla.multi_slice"(%arg0) {
      dimension = 0 : si32, 
      amount = 2 : si32, 
      start_indices = array<i64: 0>,
      limit_indices = array<i64: 3>,
      strides = array<i64: 1>,
      sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", ?}]>, <@mesh, [{"a", ?}]>, <@mesh, [{"a", ?}]>]>
    } : (tensor<10xf32>) -> (tensor<3xf32>, tensor<3xf32>, tensor<3xf32>)
    return %1, %2, %3 : tensor<3xf32>, tensor<3xf32>, tensor<3xf32>
  }
}
