// RUN: enzymexlamlir-opt %s --enzyme --canonicalize --remove-unnecessary-enzyme-ops --enzyme-simplify-math --arith-raise --canonicalize | FileCheck %s
// RUN: enzymexlamlir-opt %s --enzyme --canonicalize --remove-unnecessary-enzyme-ops --enzyme-simplify-math --arith-raise --canonicalize | stablehlo-translate --interpret

module {
  func.func @without_checkpointing(%arg0: tensor<f64>) -> tensor<f64> {
    %c = stablehlo.constant dense<1> : tensor<i64>
    %c_0 = stablehlo.constant dense<9> : tensor<i64>
    %c_1 = stablehlo.constant dense<0> : tensor<i64>
    %0:2 = stablehlo.while(%iterArg = %c_1, %iterArg_2 = %arg0) : tensor<i64>, tensor<f64> attributes {enzymexla.disable_min_cut}
     cond {
      %1 = stablehlo.compare  LT, %iterArg, %c_0 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %1 : tensor<i1>
    } do {
      %1 = stablehlo.add %iterArg, %c : tensor<i64>
      %2 = stablehlo.convert %1 : (tensor<i64>) -> tensor<f64>
      %3 = stablehlo.multiply %iterArg_2, %2 : tensor<f64>
      stablehlo.return %1, %3 : tensor<i64>, tensor<f64>
    }
    return %0#1 : tensor<f64>
  }

  func.func @with_checkpointing(%arg0: tensor<f64>) -> tensor<f64> {
    %c = stablehlo.constant dense<1> : tensor<i64>
    %c_0 = stablehlo.constant dense<9> : tensor<i64>
    %c_1 = stablehlo.constant dense<0> : tensor<i64>
    %0:2 = stablehlo.while(%iterArg = %c_1, %iterArg_2 = %arg0) : tensor<i64>, tensor<f64> attributes {enzymexla.disable_min_cut, enzymexla.enable_checkpointing = true }
     cond {
      %1 = stablehlo.compare  LT, %iterArg, %c_0 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %1 : tensor<i1>
    } do {
      %1 = stablehlo.add %iterArg, %c : tensor<i64>
      %2 = stablehlo.convert %1 : (tensor<i64>) -> tensor<f64>
      %3 = stablehlo.multiply %iterArg_2, %2 : tensor<f64>
      stablehlo.return %1, %3 : tensor<i64>, tensor<f64>
    }
    return %0#1 : tensor<f64>
  }

  func.func @with_checkpointing_diff(%arg0: tensor<f64>, %arg1:  tensor<f64>) -> (tensor<f64>, tensor<f64>) {
    %diffe_checkpointing:2 = enzyme.autodiff @with_checkpointing(%arg0, %arg1) {
      activity=[#enzyme<activity enzyme_active>],
      ret_activity=[#enzyme<activity enzyme_active>]
    } : (tensor<f64>, tensor<f64>) -> (tensor<f64>, tensor<f64>)
    return %diffe_checkpointing#0, %diffe_checkpointing#1 : tensor<f64>, tensor<f64>
  }

  func.func @without_checkpointing_diff(%arg0: tensor<f64>, %arg1:  tensor<f64>) -> (tensor<f64>, tensor<f64>) {
    %diffe_checkpointing:2 = enzyme.autodiff @without_checkpointing(%arg0, %arg1) {
      activity=[#enzyme<activity enzyme_active>],
      ret_activity=[#enzyme<activity enzyme_active>]
    } : (tensor<f64>, tensor<f64>) -> (tensor<f64>, tensor<f64>)
    return %diffe_checkpointing#0, %diffe_checkpointing#1 : tensor<f64>, tensor<f64>
  }

  func.func @main() {
    %input = stablehlo.constant dense<1.0000001> : tensor<f64>
    %diffe = stablehlo.constant dense<1.0> : tensor<f64>

    %diffe_checkpointing:2 = enzyme.autodiff @with_checkpointing(%input, %diffe) {
      activity=[#enzyme<activity enzyme_active>],
      ret_activity=[#enzyme<activity enzyme_active>]
    } : (tensor<f64>, tensor<f64>) -> (tensor<f64>, tensor<f64>)

    %diffe_no_checkpointing:2 = enzyme.autodiff @without_checkpointing(%input, %diffe) {
      activity=[#enzyme<activity enzyme_active>],
      ret_activity=[#enzyme<activity enzyme_active>]
    } : (tensor<f64>, tensor<f64>) -> (tensor<f64>, tensor<f64>)

    check.expect_almost_eq %diffe_checkpointing#0, %diffe_no_checkpointing#0 : tensor<f64>
    check.expect_almost_eq %diffe_checkpointing#1, %diffe_no_checkpointing#1 : tensor<f64>

    return
  }
}

// CHECK:  func.func private @diffewith_checkpointing(%arg0: tensor<f64>, %arg1: tensor<f64>) -> (tensor<f64>, tensor<f64>) {
// CHECK-NEXT:    %[[zero3:.+]] = stablehlo.constant dense<0.000000e+00> : tensor<3xf64>
// CHECK-NEXT:    %[[c2:.+]] = stablehlo.constant dense<2> : tensor<i64>
// CHECK-NEXT:    %[[c3:.+]] = stablehlo.constant dense<3> : tensor<i64>
// CHECK-NEXT:    %[[c0:.+]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %[[c1:.+]] = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:    %[[zero:.+]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:    %[[fwdOuter:.+]]:2 = stablehlo.while(%iterArg = %[[c0]], %iterArg_4 = %arg0) : tensor<i64>, tensor<f64>
// CHECK-NEXT:     cond {
// CHECK-NEXT:      %2 = stablehlo.compare  LT, %iterArg, %[[c3]] : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:      stablehlo.return %2 : tensor<i1>
// CHECK-NEXT:    } do {
// CHECK-NEXT:      %2 = stablehlo.multiply %iterArg, %[[c3]] : tensor<i64>
// CHECK-NEXT:      %[[fwdInner:.+]]:2 = stablehlo.while(%iterArg_5 = %[[c0]], %iterArg_6 = %iterArg_4) : tensor<i64>, tensor<f64>
// CHECK-NEXT:       cond {
// CHECK-NEXT:        %5 = stablehlo.compare  LT, %iterArg_5, %[[c3]] : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:        stablehlo.return %5 : tensor<i1>
// CHECK-NEXT:      } do {
// CHECK-NEXT:        %5 = stablehlo.add %iterArg_5, %2 : tensor<i64>
// CHECK-NEXT:        %6 = stablehlo.add %5, %[[c1]] : tensor<i64>
// CHECK-NEXT:        %7 = stablehlo.convert %6 : (tensor<i64>) -> tensor<f64>
// CHECK-NEXT:        %8 = stablehlo.multiply %iterArg_6, %7 : tensor<f64>
// CHECK-NEXT:        %9 = stablehlo.add %iterArg_5, %[[c1]] : tensor<i64>
// CHECK-NEXT:        stablehlo.return %9, %8 : tensor<i64>, tensor<f64>
// CHECK-NEXT:      }
// CHECK-NEXT:      %4 = stablehlo.add %iterArg, %[[c1]] : tensor<i64>
// CHECK-NEXT:      stablehlo.return %4, %[[fwdInner]]#1 : tensor<i64>, tensor<f64>
// CHECK-NEXT:    }
// CHECK-NEXT:    %[[bwdOuter:.+]]:3 = stablehlo.while(%iterArg = %[[c0]], %iterArg_4 = %arg1, %iterArg_5 = %[[zero]]) : tensor<i64>, tensor<f64>, tensor<f64>
// CHECK-NEXT:     cond {
// CHECK-NEXT:      %[[cmp:.+]] = stablehlo.compare  LT, %iterArg, %[[c3]] : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:      stablehlo.return %[[cmp]] : tensor<i1>
// CHECK-NEXT:    } do {
// CHECK-NEXT:      %[[v2:.+]] = stablehlo.subtract %[[c2]], %iterArg : tensor<i64>
// CHECK-NEXT:      %[[v3:.+]] = stablehlo.multiply %[[c3]], %[[v2]] : tensor<i64>
// CHECK-NEXT:      %[[revInner:.+]]:3 = stablehlo.while(%iterArg_6 = %[[c0]], %iterArg_7 = %iterArg_4, %iterArg_8 = %[[zero3]]) : tensor<i64>, tensor<f64>, tensor<3xf64>
// CHECK-NEXT:       cond {
// CHECK-NEXT:        %[[cmp:.+]] = stablehlo.compare  LT, %iterArg_6, %[[c3]] : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:        stablehlo.return %[[cmp]] : tensor<i1>
// CHECK-NEXT:      } do {
// CHECK-NEXT:        %7 = stablehlo.subtract %[[c2]], %iterArg_6 : tensor<i64>
// CHECK-NEXT:        %8 = stablehlo.add %[[v3]], %7 : tensor<i64>
// CHECK-NEXT:        %9 = stablehlo.multiply %[[c1]], %8 : tensor<i64>
// CHECK-NEXT:        %10 = stablehlo.add %9, %[[c1]] : tensor<i64>
// CHECK-NEXT:        %11 = stablehlo.convert %10 : (tensor<i64>) -> tensor<f64>
// CHECK-NEXT:        %12 = stablehlo.reshape %11 : (tensor<f64>) -> tensor<1xf64>
// CHECK-NEXT:        %13 = stablehlo.dynamic_update_slice %iterArg_8, %12, %iterArg_6 : (tensor<3xf64>, tensor<1xf64>, tensor<i64>) -> tensor<3xf64>
// CHECK-NEXT:        %14 = stablehlo.multiply %iterArg_7, %11 : tensor<f64>
// CHECK-NEXT:        %15 = stablehlo.add %iterArg_6, %[[c1]] : tensor<i64>
// CHECK-NEXT:        stablehlo.return %15, %14, %13 : tensor<i64>, tensor<f64>, tensor<3xf64>
// CHECK-NEXT:      }
// CHECK-NEXT:      %[[revLoop:.+]]:4 = stablehlo.while(%iterArg_6 = %[[c0]], %iterArg_7 = %iterArg_4, %iterArg_8 = %[[c2]], %iterArg_9 = %iterArg_5) : tensor<i64>, tensor<f64>, tensor<i64>, tensor<f64>
// CHECK-NEXT:       cond {
// CHECK-NEXT:        %7 = stablehlo.compare  LT, %iterArg_6, %[[c3]] : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:        stablehlo.return %7 : tensor<i1>
// CHECK-NEXT:      } do {
// CHECK-NEXT:        %7 = stablehlo.dynamic_slice %4#2, %iterArg_8, sizes = [1] : (tensor<3xf64>, tensor<i64>) -> tensor<1xf64>
// CHECK-NEXT:        %8 = stablehlo.reshape %7 : (tensor<1xf64>) -> tensor<f64>
// CHECK-NEXT:        %9 = stablehlo.multiply %iterArg_7, %8 : tensor<f64>
// CHECK-NEXT:        %10 = stablehlo.add %iterArg_9, %9 : tensor<f64>
// CHECK-NEXT:        %11 = stablehlo.add %iterArg_6, %[[c1]] : tensor<i64>
// CHECK-NEXT:        %12 = stablehlo.subtract %iterArg_8, %[[c1]] : tensor<i64>
// CHECK-NEXT:        stablehlo.return %11, %10, %12, %[[zero]] : tensor<i64>, tensor<f64>, tensor<i64>, tensor<f64>
// CHECK-NEXT:      }
// CHECK-NEXT:      %[[newIter:.+]] = stablehlo.add %iterArg, %[[c1]] : tensor<i64>
// CHECK-NEXT:      stablehlo.return %[[newIter]], %[[revLoop]]#1, %[[revLoop]]#3 : tensor<i64>, tensor<f64>, tensor<f64>
// CHECK-NEXT:    }
// CHECK-NEXT:    return %[[fwdOuter]]#1, %[[bwdOuter]]#1 : tensor<f64>, tensor<f64>
// CHECK-NEXT:  }
