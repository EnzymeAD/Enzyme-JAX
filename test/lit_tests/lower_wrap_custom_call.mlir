// RUN: enzymexlamlir-opt %s --optimize-communication="wrap_custom_call=1" | FileCheck %s

module  {
  sdy.mesh @mesh = <["a"=2]>
  func.func public @main(%arg0: tensor<16xf32>) -> (tensor<18xf32>) {
//    CHECK: %[[RES:.*]] = stablehlo.custom_call @[[SYMA:.*]](%arg0) {backend_config = "dimension=0,left_amount=1,right_amount=1", sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", ?}]>]>} : (tensor<16xf32>) -> tensor<18xf32>
//    CHECK: return %[[RES]] : tensor<18xf32>
    %1 = "enzymexla.wrap"(%arg0) {dimension = 0 : i64, lhs = 1 : i64, rhs = 1 : i64, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", ?}]>]>} : (tensor<16xf32>) -> (tensor<18xf32>)
    return %1 : tensor<18xf32>
  }
}
