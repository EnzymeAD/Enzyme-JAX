// RUN: enzymexlamlir-opt %s --optimize-communication="multirotate_custom_call=1" | FileCheck %s

module  {
  sdy.mesh @mesh = <["a"=2]>
  func.func public @main(%arg0: tensor<10xf32>) -> (tensor<10xf32>, tensor<10xf32>, tensor<10xf32>) {
//    CHECK: %[[RES:.*]]:3 = stablehlo.custom_call @[[SYMA:.*]](%arg0) {backend_config = "dimension=0,left_amount=1,right_amount=1,bufferize=0", sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", ?}]>, <@mesh, [{"a", ?}]>, <@mesh, [{"a", ?}]>]>} : (tensor<10xf32>) -> (tensor<10xf32>, tensor<10xf32>, tensor<10xf32>)
//    CHECK: return %[[RES]]#0, %[[RES]]#1, %[[RES]]#2 : tensor<10xf32>, tensor<10xf32>, tensor<10xf32>
    %1, %2, %3 = "enzymexla.multi_rotate"(%arg0) {dimension = 0 : i32, left_amount = 1 : i32, right_amount = 1 : i32, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", ?}]>, <@mesh, [{"a", ?}]>, <@mesh, [{"a", ?}]>]>} : (tensor<10xf32>) -> (tensor<10xf32>, tensor<10xf32>, tensor<10xf32>)
    return %1, %2, %3 : tensor<10xf32>, tensor<10xf32>, tensor<10xf32>
  }

  // CHECK-LABEL: func.func public @merge_sibling_multirotate
  // CHECK-SAME: (%[[ARG:.+]]: tensor<10xf32>)
  func.func public @merge_sibling_multirotate(%arg0: tensor<10xf32>) -> (tensor<10xf32>, tensor<10xf32>) {
    // CHECK: %[[MERGED:.+]]:3 = stablehlo.custom_call @_SPMDInternalOp_MultiRotate(%[[ARG]]) {backend_config = "dimension=0,left_amount=1,right_amount=1,bufferize=0"
    // CHECK-NOT: @_SPMDInternalOp_MultiRotate
    // CHECK: return %[[MERGED]]#0, %[[MERGED]]#2
    %l:2 = stablehlo.custom_call @_SPMDInternalOp_MultiRotate(%arg0) {backend_config = "dimension=0,left_amount=1,right_amount=0,bufferize=0", sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", ?}]>, <@mesh, [{"a", ?}]>]>} : (tensor<10xf32>) -> (tensor<10xf32>, tensor<10xf32>)
    %r:2 = stablehlo.custom_call @_SPMDInternalOp_MultiRotate(%arg0) {backend_config = "dimension=0,left_amount=0,right_amount=1,bufferize=0", sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", ?}]>, <@mesh, [{"a", ?}]>]>} : (tensor<10xf32>) -> (tensor<10xf32>, tensor<10xf32>)
    return %l#0, %r#1 : tensor<10xf32>, tensor<10xf32>
  }

  // CHECK-LABEL: func.func public @merge_chained_multirotate
  // CHECK-SAME: (%[[ARG:.+]]: tensor<10xf32>)
  func.func public @merge_chained_multirotate(%arg0: tensor<10xf32>) -> (tensor<10xf32>, tensor<10xf32>) {
    // CHECK: %[[MERGED:.+]]:3 = stablehlo.custom_call @_SPMDInternalOp_MultiRotate(%[[ARG]]) {backend_config = "dimension=0,left_amount=2,right_amount=0,bufferize=0"
    // CHECK-NOT: @_SPMDInternalOp_MultiRotate
    // CHECK: return %[[MERGED]]#1, %[[MERGED]]#0
    %first:2 = stablehlo.custom_call @_SPMDInternalOp_MultiRotate(%arg0) {backend_config = "dimension=0,left_amount=1,right_amount=0,bufferize=0", sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", ?}]>, <@mesh, [{"a", ?}]>]>} : (tensor<10xf32>) -> (tensor<10xf32>, tensor<10xf32>)
    %second:2 = stablehlo.custom_call @_SPMDInternalOp_MultiRotate(%first#0) {backend_config = "dimension=0,left_amount=1,right_amount=0,bufferize=0", sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", ?}]>, <@mesh, [{"a", ?}]>]>} : (tensor<10xf32>) -> (tensor<10xf32>, tensor<10xf32>)
    return %first#0, %second#0 : tensor<10xf32>, tensor<10xf32>
  }

  // CHECK-LABEL: func.func public @no_merge_exceeding_last_shard
  // CHECK-SAME: (%[[ARG:.+]]: tensor<9xf32>)
  func.func public @no_merge_exceeding_last_shard(%arg0: tensor<9xf32>) -> tensor<9xf32> {
    // CHECK: %[[FIRST:.+]]:3 = stablehlo.custom_call @_SPMDInternalOp_MultiRotate(%[[ARG]]) {backend_config = "dimension=0,left_amount=2,right_amount=0,bufferize=0"
    // CHECK: %[[SECOND:.+]]:4 = stablehlo.custom_call @_SPMDInternalOp_MultiRotate(%[[FIRST]]#0) {backend_config = "dimension=0,left_amount=3,right_amount=0,bufferize=0"
    // CHECK: return %[[SECOND]]#0
    %first:3 = stablehlo.custom_call @_SPMDInternalOp_MultiRotate(%arg0) {backend_config = "dimension=0,left_amount=2,right_amount=0,bufferize=0", sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", ?}]>, <@mesh, [{"a", ?}]>, <@mesh, [{"a", ?}]>]>} : (tensor<9xf32>) -> (tensor<9xf32>, tensor<9xf32>, tensor<9xf32>)
    %second:4 = stablehlo.custom_call @_SPMDInternalOp_MultiRotate(%first#0) {backend_config = "dimension=0,left_amount=3,right_amount=0,bufferize=0", sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", ?}]>, <@mesh, [{"a", ?}]>, <@mesh, [{"a", ?}]>, <@mesh, [{"a", ?}]>]>} : (tensor<9xf32>) -> (tensor<9xf32>, tensor<9xf32>, tensor<9xf32>, tensor<9xf32>)
    return %second#0 : tensor<9xf32>
  }
}
