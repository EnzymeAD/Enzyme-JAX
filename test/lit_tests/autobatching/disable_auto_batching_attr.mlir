// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt=enable_auto_batching_passes=true | FileCheck %s
// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt=enable_auto_batching_passes=true | stablehlo-translate - -allow-unregistered-dialect --interpret

module {
  // 1. UnitAttr {enzymexla.disable_auto_batching} keeps GreedyWhileLoopBatchFission from fissioning/lifting.
  // CHECK-LABEL: func.func private @disabled_unit_attr
  // CHECK: stablehlo.while
  // CHECK: stablehlo.sine
  func.func private @disabled_unit_attr(%x: tensor<4xf32>) -> tensor<4xf32> {
    %c0 = stablehlo.constant dense<0> : tensor<i64>
    %c1 = stablehlo.constant dense<1> : tensor<i64>
    %c4 = stablehlo.constant dense<4> : tensor<i64>
    %z = stablehlo.constant dense<0.0> : tensor<4xf32>
    %r:2 = stablehlo.while(%i = %c0, %acc = %z) : tensor<i64>, tensor<4xf32> attributes {enzymexla.disable_auto_batching}
    cond {
      %cmp = stablehlo.compare LT, %i, %c4 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %cmp : tensor<i1>
    } do {
      %s = stablehlo.dynamic_slice %x, %i, sizes = [1] : (tensor<4xf32>, tensor<i64>) -> tensor<1xf32>
      %y = stablehlo.sine %s : tensor<1xf32>
      %u = stablehlo.dynamic_update_slice %acc, %y, %i : (tensor<4xf32>, tensor<1xf32>, tensor<i64>) -> tensor<4xf32>
      %ni = stablehlo.add %i, %c1 : tensor<i64>
      stablehlo.return %ni, %u : tensor<i64>, tensor<4xf32>
    }
    return %r#1 : tensor<4xf32>
  }

  // 2. BoolAttr {enzymexla.disable_auto_batching = true} also prevents reduction lifting & parallel_while batching.
  // CHECK-LABEL: func.func private @disabled_reduction_and_parallel
  // CHECK: stablehlo.while
  func.func private @disabled_reduction_and_parallel(%x: tensor<4xf32>) -> tensor<1xf32> {
    %c0 = stablehlo.constant dense<0> : tensor<i64>
    %c1 = stablehlo.constant dense<1> : tensor<i64>
    %c4 = stablehlo.constant dense<4> : tensor<i64>
    %z = stablehlo.constant dense<0.0> : tensor<1xf32>
    %r:2 = stablehlo.while(%i = %c0, %sum = %z) : tensor<i64>, tensor<1xf32> attributes {enzymexla.disable_auto_batching = true, enzymexla.parallel}
    cond {
      %cmp = stablehlo.compare LT, %i, %c4 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %cmp : tensor<i1>
    } do {
      %s = stablehlo.dynamic_slice %x, %i, sizes = [1] : (tensor<4xf32>, tensor<i64>) -> tensor<1xf32>
      %ns = stablehlo.add %sum, %s : tensor<1xf32>
      %ni = stablehlo.add %i, %c1 : tensor<i64>
      stablehlo.return %ni, %ns : tensor<i64>, tensor<1xf32>
    }
    return %r#1 : tensor<1xf32>
  }

  // 3. BoolAttr {enzymexla.disable_auto_batching = false} allows normal auto-batching!
  // CHECK-LABEL: func.func private @explicitly_enabled
  // CHECK-NOT: stablehlo.while
  // CHECK: stablehlo.sine
  func.func private @explicitly_enabled(%x: tensor<4xf32>) -> tensor<4xf32> {
    %c0 = stablehlo.constant dense<0> : tensor<i64>
    %c1 = stablehlo.constant dense<1> : tensor<i64>
    %c4 = stablehlo.constant dense<4> : tensor<i64>
    %z = stablehlo.constant dense<0.0> : tensor<4xf32>
    %r:2 = stablehlo.while(%i = %c0, %acc = %z) : tensor<i64>, tensor<4xf32> attributes {enzymexla.disable_auto_batching = false}
    cond {
      %cmp = stablehlo.compare LT, %i, %c4 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %cmp : tensor<i1>
    } do {
      %s = stablehlo.dynamic_slice %x, %i, sizes = [1] : (tensor<4xf32>, tensor<i64>) -> tensor<1xf32>
      %y = stablehlo.sine %s : tensor<1xf32>
      %u = stablehlo.dynamic_update_slice %acc, %y, %i : (tensor<4xf32>, tensor<1xf32>, tensor<i64>) -> tensor<4xf32>
      %ni = stablehlo.add %i, %c1 : tensor<i64>
      stablehlo.return %ni, %u : tensor<i64>, tensor<4xf32>
    }
    return %r#1 : tensor<4xf32>
  }

  func.func @main() {
    %x = stablehlo.constant dense<[0.0, 1.0, 2.0, 3.0]> : tensor<4xf32>
    %a = func.call @disabled_unit_attr(%x) : (tensor<4xf32>) -> tensor<4xf32>
    %b = func.call @explicitly_enabled(%x) : (tensor<4xf32>) -> tensor<4xf32>
    check.expect_almost_eq %a, %b : tensor<4xf32>

    %s = func.call @disabled_reduction_and_parallel(%x) : (tensor<4xf32>) -> tensor<1xf32>
    check.expect_almost_eq_const %s, dense<[6.0]> : tensor<1xf32>
    return
  }
}
