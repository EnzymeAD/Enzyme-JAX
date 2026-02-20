// RUN: enzymexlamlir-opt %s --optimize-communication="multirotate_custom_call=1" | FileCheck %s

module  {
  sdy.mesh @mesh = <["a"=2]>
  func.func public @main(%arg0: tensor<10xf32>) -> (tensor<10xf32>, tensor<10xf32>, tensor<10xf32>) {
//    CHECK: %[[RES:.*]]:3 = stablehlo.custom_call @[[SYMA:.*]](%arg0) {backend_config = "dimension=0,left_amount=1,right_amount=1", sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", ?}]>, <@mesh, [{"a", ?}]>, <@mesh, [{"a", ?}]>]>} : (tensor<10xf32>) -> (tensor<10xf32>, tensor<10xf32>, tensor<10xf32>)
//    CHECK: return %[[RES]]#0, %[[RES]]#1, %[[RES]]#2 : tensor<10xf32>, tensor<10xf32>, tensor<10xf32>
    %1, %2, %3 = "enzymexla.multi_rotate"(%arg0) {dimension = 0 : i32, left_amount = 1 : i32, right_amount = 1 : i32, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", ?}]>, <@mesh, [{"a", ?}]>, <@mesh, [{"a", ?}]>]>} : (tensor<10xf32>) -> (tensor<10xf32>, tensor<10xf32>, tensor<10xf32>)
    return %1, %2, %3 : tensor<10xf32>, tensor<10xf32>, tensor<10xf32>
  }
}
