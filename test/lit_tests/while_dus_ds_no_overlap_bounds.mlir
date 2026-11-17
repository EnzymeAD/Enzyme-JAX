// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s
// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=while_dus_ds_simplify;dus_dynamic_slice_simplify" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s --check-prefix=TD

// 1. Constant DUS start vs loop-varying DS start with disjoint bounds.
// CHECK-LABEL: func.func @const_vs_iv(
// CHECK:         stablehlo.while
// CHECK:         do {
// CHECK-NOT:       dynamic_update_slice
// CHECK:           stablehlo.dynamic_slice %arg0,
// TD-LABEL: func.func @const_vs_iv(
// TD:         do {
// TD:           %[[UPD:.+]] = stablehlo.dynamic_update_slice %[[ARR:[^ ,]+]], %arg1, %c
// TD-NEXT:      %[[DS:.+]] = stablehlo.dynamic_slice %[[ARR]], %{{.+}}, sizes = [2]
func.func @const_vs_iv(%arr: tensor<10xf32>, %u: tensor<2xf32>) -> (tensor<10xf32>, tensor<2xf32>) {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %c2 = stablehlo.constant dense<2> : tensor<i64>
  %c5 = stablehlo.constant dense<5> : tensor<i64>
  %z = stablehlo.constant dense<0.0> : tensor<2xf32>
  %0:3 = stablehlo.while(%i = %c0, %a = %arr, %acc = %z) : tensor<i64>, tensor<10xf32>, tensor<2xf32>
  cond {
    %cmp = stablehlo.compare LT, %i, %c5 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %cmp : tensor<i1>
  } do {
    %idx = stablehlo.add %i, %c2 : tensor<i64>
    %a1 = stablehlo.dynamic_update_slice %a, %u, %c0 : (tensor<10xf32>, tensor<2xf32>, tensor<i64>) -> tensor<10xf32>
    %s = stablehlo.dynamic_slice %a1, %idx, sizes = [2] : (tensor<10xf32>, tensor<i64>) -> tensor<2xf32>
    %acc1 = stablehlo.add %acc, %s : tensor<2xf32>
    %inext = stablehlo.add %i, %c1 : tensor<i64>
    stablehlo.return %inext, %a1, %acc1 : tensor<i64>, tensor<10xf32>, tensor<2xf32>
  }
  return %0#1, %0#2 : tensor<10xf32>, tensor<2xf32>
}

// 2. Opposite-direction loop indices: DUS at i in [0, 3], DS at 9 - i in [6, 9].
// CHECK-LABEL: func.func @opposite_iv(
// CHECK:         do {
// CHECK:           %[[REV:.+]] = stablehlo.subtract %c_2, %iterArg
// CHECK-NEXT:      %[[DUS:.+]] = stablehlo.dynamic_update_slice %iterArg_3, %arg1, %iterArg
// CHECK-NEXT:      %[[DS:.+]] = stablehlo.dynamic_slice %iterArg_3, %[[REV]], sizes = [1]
func.func @opposite_iv(%arr: tensor<10xf32>, %u: tensor<1xf32>) -> (tensor<10xf32>, tensor<1xf32>) {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %c4 = stablehlo.constant dense<4> : tensor<i64>
  %c9 = stablehlo.constant dense<9> : tensor<i64>
  %z = stablehlo.constant dense<0.0> : tensor<1xf32>
  %0:3 = stablehlo.while(%i = %c0, %a = %arr, %acc = %z) : tensor<i64>, tensor<10xf32>, tensor<1xf32>
  cond {
    %cmp = stablehlo.compare LT, %i, %c4 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %cmp : tensor<i1>
  } do {
    %rev = stablehlo.subtract %c9, %i : tensor<i64>
    %a1 = stablehlo.dynamic_update_slice %a, %u, %i : (tensor<10xf32>, tensor<1xf32>, tensor<i64>) -> tensor<10xf32>
    %s = stablehlo.dynamic_slice %a1, %rev, sizes = [1] : (tensor<10xf32>, tensor<i64>) -> tensor<1xf32>
    %acc1 = stablehlo.add %acc, %s : tensor<1xf32>
    %inext = stablehlo.add %i, %c1 : tensor<i64>
    stablehlo.return %inext, %a1, %acc1 : tensor<i64>, tensor<10xf32>, tensor<1xf32>
  }
  return %0#1, %0#2 : tensor<10xf32>, tensor<1xf32>
}

// 3. Non-unit affine scale: DUS at 2*i, DS at 2*i + 1.
// CHECK-LABEL: func.func @scale2_iv(
// CHECK:         do {
// CHECK:           %[[EVEN:.+]] = stablehlo.multiply %iterArg, %c_1
// CHECK-NEXT:      %[[ODD:.+]] = stablehlo.add %[[EVEN]], %c_0
// CHECK-NEXT:      %[[DUS:.+]] = stablehlo.dynamic_update_slice %iterArg_3, %arg1, %[[EVEN]]
// CHECK-NEXT:      %[[DS:.+]] = stablehlo.dynamic_slice %iterArg_3, %[[ODD]], sizes = [1]
func.func @scale2_iv(%arr: tensor<10xf32>, %u: tensor<1xf32>) -> (tensor<10xf32>, tensor<1xf32>) {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %c2 = stablehlo.constant dense<2> : tensor<i64>
  %c5 = stablehlo.constant dense<5> : tensor<i64>
  %z = stablehlo.constant dense<0.0> : tensor<1xf32>
  %0:3 = stablehlo.while(%i = %c0, %a = %arr, %acc = %z) : tensor<i64>, tensor<10xf32>, tensor<1xf32>
  cond {
    %cmp = stablehlo.compare LT, %i, %c5 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %cmp : tensor<i1>
  } do {
    %even = stablehlo.multiply %i, %c2 : tensor<i64>
    %odd = stablehlo.add %even, %c1 : tensor<i64>
    %a1 = stablehlo.dynamic_update_slice %a, %u, %even : (tensor<10xf32>, tensor<1xf32>, tensor<i64>) -> tensor<10xf32>
    %s = stablehlo.dynamic_slice %a1, %odd, sizes = [1] : (tensor<10xf32>, tensor<i64>) -> tensor<1xf32>
    %acc1 = stablehlo.add %acc, %s : tensor<1xf32>
    %inext = stablehlo.add %i, %c1 : tensor<i64>
    stablehlo.return %inext, %a1, %acc1 : tensor<i64>, tensor<10xf32>, tensor<1xf32>
  }
  return %0#1, %0#2 : tensor<10xf32>, tensor<1xf32>
}

// 4. Shared SSA base outside loop with in-bounds enzymexla.bounds.
// CHECK-LABEL: func.func @bounded_outside_loop(
// CHECK-NOT:     dynamic_update_slice
// CHECK:         %[[DS:.+]] = stablehlo.dynamic_slice %arg0, %0, sizes = [2]
// CHECK-NEXT:    return %[[DS]]
// TD-LABEL: func.func @bounded_outside_loop(
// TD:         %[[DS:.+]] = stablehlo.dynamic_slice %arg0, %0, sizes = [2]
func.func @bounded_outside_loop(%arr: tensor<10xf32>, %u: tensor<2xf32>, %idx: tensor<i64> {enzymexla.bounds = [0, 6]}) -> tensor<2xf32> {
  %c2 = stablehlo.constant dense<2> : tensor<i64>
  %idx2 = stablehlo.add %idx, %c2 : tensor<i64>
  %a1 = stablehlo.dynamic_update_slice %arr, %u, %idx : (tensor<10xf32>, tensor<2xf32>, tensor<i64>) -> tensor<10xf32>
  %s = stablehlo.dynamic_slice %a1, %idx2, sizes = [2] : (tensor<10xf32>, tensor<i64>) -> tensor<2xf32>
  return %s : tensor<2xf32>
}

// 5. Strided parity disjointness with independent bounded variables: 2*i vs 2*j + 1.
// CHECK-LABEL: func.func @strided_parity_disjoint(
// CHECK-NOT:     dynamic_update_slice
// CHECK:         %[[DS:.+]] = stablehlo.dynamic_slice %arg0, %{{.+}}, sizes = [1]
func.func @strided_parity_disjoint(%arr: tensor<10xf32>, %u: tensor<1xf32>, %i: tensor<i64> {enzymexla.bounds = [0, 4]}, %j: tensor<i64> {enzymexla.bounds = [0, 4]}) -> tensor<1xf32> {
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %c2 = stablehlo.constant dense<2> : tensor<i64>
  %even = stablehlo.multiply %i, %c2 : tensor<i64>
  %j2 = stablehlo.multiply %j, %c2 : tensor<i64>
  %odd = stablehlo.add %j2, %c1 : tensor<i64>
  %a1 = stablehlo.dynamic_update_slice %arr, %u, %even : (tensor<10xf32>, tensor<1xf32>, tensor<i64>) -> tensor<10xf32>
  %s = stablehlo.dynamic_slice %a1, %odd, sizes = [1] : (tensor<10xf32>, tensor<i64>) -> tensor<1xf32>
  return %s : tensor<1xf32>
}

// 6. Negative test: when idx == 8, idx + 2 == 10 clamps to 8 and overlaps DUS at 8!
// CHECK-LABEL: func.func @clamping_can_overlap(
// CHECK:         %[[DUS:.+]] = stablehlo.dynamic_update_slice %arg0, %arg1, %arg2
// CHECK-NEXT:    %[[DS:.+]] = stablehlo.dynamic_slice %[[DUS]], %0, sizes = [2]
func.func @clamping_can_overlap(%arr: tensor<10xf32>, %u: tensor<2xf32>, %idx: tensor<i64> {enzymexla.bounds = [0, 8]}) -> tensor<2xf32> {
  %c2 = stablehlo.constant dense<2> : tensor<i64>
  %idx2 = stablehlo.add %idx, %c2 : tensor<i64>
  %a1 = stablehlo.dynamic_update_slice %arr, %u, %idx : (tensor<10xf32>, tensor<2xf32>, tensor<i64>) -> tensor<10xf32>
  %s = stablehlo.dynamic_slice %a1, %idx2, sizes = [2] : (tensor<10xf32>, tensor<i64>) -> tensor<2xf32>
  return %s : tensor<2xf32>
}

// 7. Negative test: constant -1 clamps to 0, overlapping DUS at 0!
// TD-LABEL: func.func @const_neg_clamp_overlap(
// TD:         %[[DUS:.+]] = stablehlo.dynamic_update_slice %arg0, %arg1, %c
// TD-NEXT:    %[[DS:.+]] = stablehlo.dynamic_slice %[[DUS]], %c_0, sizes = [1]
func.func @const_neg_clamp_overlap(%arr: tensor<4xf32>, %u: tensor<1xf32>) -> tensor<1xf32> {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %cm1 = stablehlo.constant dense<-1> : tensor<i64>
  %a1 = stablehlo.dynamic_update_slice %arr, %u, %c0 : (tensor<4xf32>, tensor<1xf32>, tensor<i64>) -> tensor<4xf32>
  %s = stablehlo.dynamic_slice %a1, %cm1, sizes = [1] : (tensor<4xf32>, tensor<i64>) -> tensor<1xf32>
  return %s : tensor<1xf32>
}
