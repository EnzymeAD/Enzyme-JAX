// RUN: enzymexlamlir-opt %s --enzyme --canonicalize --remove-unnecessary-enzyme-ops --enzyme-simplify-math --arith-raise --canonicalize | FileCheck %s

// Reverse-mode batch_norm_training inside a while must not use forward results
// from the other loop body.

module {
  func.func @in_loop(%x: tensor<2x3xf32>, %scale: tensor<3xf32>, %offset: tensor<3xf32>) -> tensor<2x3xf32> {
    %c0 = stablehlo.constant dense<0> : tensor<i64>
    %c1 = stablehlo.constant dense<1> : tensor<i64>
    %c3 = stablehlo.constant dense<3> : tensor<i64>
    %r:2 = stablehlo.while(%i = %c0, %acc = %x) : tensor<i64>, tensor<2x3xf32>
     cond {
      %cmp = stablehlo.compare  LT, %i, %c3 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %cmp : tensor<i1>
    } do {
      %out, %mean, %var = "stablehlo.batch_norm_training"(%acc, %scale, %offset) <{epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64}> : (tensor<2x3xf32>, tensor<3xf32>, tensor<3xf32>) -> (tensor<2x3xf32>, tensor<3xf32>, tensor<3xf32>)
      %inext = stablehlo.add %i, %c1 : tensor<i64>
      stablehlo.return %inext, %out : tensor<i64>, tensor<2x3xf32>
    }
    return %r#1 : tensor<2x3xf32>
  }

  func.func @main(%x: tensor<2x3xf32>, %scale: tensor<3xf32>, %offset: tensor<3xf32>, %seed: tensor<2x3xf32>) -> (tensor<2x3xf32>, tensor<2x3xf32>, tensor<3xf32>, tensor<3xf32>) {
    %loop:4 = enzyme.autodiff @in_loop(%x, %scale, %offset, %seed) {
      activity = [#enzyme<activity enzyme_active>, #enzyme<activity enzyme_active>, #enzyme<activity enzyme_active>],
      ret_activity = [#enzyme<activity enzyme_active>]
    } : (tensor<2x3xf32>, tensor<3xf32>, tensor<3xf32>, tensor<2x3xf32>) -> (tensor<2x3xf32>, tensor<2x3xf32>, tensor<3xf32>, tensor<3xf32>)
    return %loop#0, %loop#1, %loop#2, %loop#3 : tensor<2x3xf32>, tensor<2x3xf32>, tensor<3xf32>, tensor<3xf32>
  }
}

// CHECK-LABEL: func.func private @diffein_loop(
// CHECK:         stablehlo.while
// CHECK:           "stablehlo.batch_norm_training"
// CHECK:         stablehlo.while
// CHECK:           stablehlo.dynamic_slice {{.*}}sizes
// CHECK:           %[[X:.+]] = stablehlo.reshape
// CHECK:           %{{.+}}, %[[MEAN:.+]], %[[VAR:.+]] = "stablehlo.batch_norm_training"(%[[X]],
// CHECK:           "stablehlo.batch_norm_grad"(%[[X]], %{{.+}}, %[[MEAN]], %[[VAR]],
