// RUN: enzymexlamlir-opt %s --enzyme --canonicalize --remove-unnecessary-enzyme-ops --enzyme-simplify-math --arith-raise --canonicalize | FileCheck %s
// RUN: enzymexlamlir-opt %s --enzyme --canonicalize --remove-unnecessary-enzyme-ops --enzyme-simplify-math --arith-raise --canonicalize | stablehlo-translate --interpret

module {
  func.func @without_checkpointing(%arg0: tensor<f64>) -> tensor<f64> {
    %c = stablehlo.constant dense<1> : tensor<i64>
    %c_0 = stablehlo.constant dense<9> : tensor<i64>
    %c_1 = stablehlo.constant dense<0> : tensor<i64>
    %0:2 = stablehlo.while(%iterArg = %c_1, %iterArg_2 = %arg0) : tensor<i64>, tensor<f64> attributes {enzyme.disable_mincut}
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
    %0:2 = stablehlo.while(%iterArg = %c_1, %iterArg_2 = %arg0) : tensor<i64>, tensor<f64> attributes {enzyme.disable_mincut, enzymexla.enable_checkpointing = true }
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
// CHECK-NEXT:    %[[fwdOuter:.+]]:3 = stablehlo.while(%iterArg = %[[c0]], %iterArg_4 = %arg0, %iterArg_5 = %[[zero3]]) : tensor<i64>, tensor<f64>, tensor<3xf64>
// CHECK-NEXT:     cond {
// CHECK-NEXT:      %2 = stablehlo.compare  LT, %iterArg, %[[c3]] : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:      stablehlo.return %2 : tensor<i1>
// CHECK-NEXT:    } do {
// CHECK-NEXT:      %2 = stablehlo.reshape %iterArg_4 : (tensor<f64>) -> tensor<1xf64>
// CHECK-NEXT:      %3 = stablehlo.dynamic_update_slice %iterArg_5, %2, %iterArg : (tensor<3xf64>, tensor<1xf64>, tensor<i64>) -> tensor<3xf64>
// CHECK-NEXT:      %4 = stablehlo.multiply %iterArg, %c_0 : tensor<i64>
// CHECK-NEXT:      %[[fwdInner:.+]]:2 = stablehlo.while(%iterArg_6 = %c_1, %iterArg_7 = %iterArg_4) : tensor<i64>, tensor<f64>
// CHECK-NEXT:       cond {
// CHECK-NEXT:        %7 = stablehlo.compare  LT, %iterArg_6, %c_0 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:        stablehlo.return %7 : tensor<i1>
// CHECK-NEXT:      } do {
// CHECK-NEXT:        %7 = stablehlo.add %iterArg_6, %4 : tensor<i64>
// CHECK-NEXT:        %8 = stablehlo.add %7, %c_2 : tensor<i64>
// CHECK-NEXT:        %9 = stablehlo.convert %8 : (tensor<i64>) -> tensor<f64>
// CHECK-NEXT:        %10 = stablehlo.multiply %iterArg_7, %9 : tensor<f64>
// CHECK-NEXT:        %11 = stablehlo.add %iterArg_6, %c_2 : tensor<i64>
// CHECK-NEXT:        stablehlo.return %11, %10 : tensor<i64>, tensor<f64>
// CHECK-NEXT:      }
// CHECK-NEXT:      %6 = stablehlo.add %iterArg, %c_2 : tensor<i64>
// CHECK-NEXT:      stablehlo.return %6, %[[fwdInner]]#1, %3 : tensor<i64>, tensor<f64>, tensor<3xf64>
// CHECK-NEXT:    }
// CHECK-NEXT:    %[[bwdOuter:.+]]:5 = stablehlo.while(%iterArg = %[[c0]], %iterArg_4 = %arg1, %iterArg_5 = %[[zero]], %iterArg_6 = %[[zero]], %iterArg_7 = %c) : tensor<i64>, tensor<f64>, tensor<f64>, tensor<f64>
// CHECK-NEXT:     cond {
// CHECK-NEXT:      %[[cmp:.+]] = stablehlo.compare  LT, %iterArg, %[[c3]] : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:      stablehlo.return %[[cmp]] : tensor<i1>
// CHECK-NEXT:    } do {
// CHECK-NEXT:      %2 = stablehlo.dynamic_slice %0#2, %iterArg_7, sizes = [1] : (tensor<3xf64>, tensor<i64>) -> tensor<1xf64>
// CHECK-NEXT:      %3 = stablehlo.reshape %2 : (tensor<1xf64>) -> tensor<f64>
// CHECK-NEXT:      %4 = stablehlo.subtract %c, %iterArg : tensor<i64>
// CHECK-NEXT:      %5 = stablehlo.multiply %c_0, %4 : tensor<i64>
// CHECK-NEXT:      %6:3 = stablehlo.while(%iterArg_8 = %c_1, %iterArg_9 = %3, %iterArg_10 = %cst) : tensor<i64>, tensor<f64>, tensor<3xf64>
// CHECK-NEXT:       cond {
// CHECK-NEXT:        %10 = stablehlo.compare  LT, %iterArg_8, %c_0 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:        stablehlo.return %10 : tensor<i1>
// CHECK-NEXT:      } do {
// CHECK-NEXT:        %[[subc:.+]] = stablehlo.subtract %c_0, %c_2 : tensor<i64>
// CHECK-NEXT:        %[[subi:.+]] = stablehlo.subtract %[[subc]], %iterArg_8 : tensor<i64>
// CHECK-NEXT:        %[[add:.+]] = stablehlo.add %5, %[[subi]] : tensor<i64>
// CHECK-NEXT:        %[[mul:.+]] = stablehlo.multiply %c_2, %[[add]] : tensor<i64>
// CHECK-NEXT:        %[[add2:.+]] = stablehlo.add %[[mul]], %c_2 : tensor<i64>
// CHECK-NEXT:        %[[conv:.+]] = stablehlo.convert %[[add2]] : (tensor<i64>) -> tensor<f64>
// CHECK-NEXT:        %[[reshape:.+]] = stablehlo.reshape %[[conv]] : (tensor<f64>) -> tensor<1xf64>
// CHECK-NEXT:        %[[dus:.+]] = stablehlo.dynamic_update_slice %iterArg_10, %[[reshape]], %iterArg_8 : (tensor<3xf64>, tensor<1xf64>, tensor<i64>) -> tensor<3xf64>
// CHECK-NEXT:        %[[mul2:.+]] = stablehlo.multiply %iterArg_9, %[[conv]] : tensor<f64>
// CHECK-NEXT:        %[[ivnext:.+]] = stablehlo.add %iterArg_8, %c_2 : tensor<i64>
// CHECK-NEXT:        stablehlo.return %[[ivnext]], %[[mul2]], %[[dus]] : tensor<i64>, tensor<f64>, tensor<3xf64>
// CHECK-NEXT:      }
// CHECK-NEXT:      %7:5 = stablehlo.while(%iterArg_8 = %c_1, %iterArg_9 = %iterArg_4, %iterArg_10 = %iterArg_5, %iterArg_11 = %iterArg_6, %iterArg_12 = %c) : tensor<i64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<i64>
// CHECK-NEXT:       cond {
// CHECK-NEXT:        %10 = stablehlo.compare  LT, %iterArg_8, %c_0 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:        stablehlo.return %10 : tensor<i1>
// CHECK-NEXT:      } do {
// CHECK-NEXT:        %10 = stablehlo.add %iterArg_10, %iterArg_9 : tensor<f64>
// CHECK-NEXT:        %11 = stablehlo.dynamic_slice %6#2, %iterArg_12, sizes = [1] : (tensor<3xf64>, tensor<i64>) -> tensor<1xf64>
// CHECK-NEXT:        %12 = stablehlo.reshape %11 : (tensor<1xf64>) -> tensor<f64>
// CHECK-NEXT:        %13 = stablehlo.multiply %10, %12 : tensor<f64>
// CHECK-NEXT:        %14 = stablehlo.add %iterArg_11, %13 : tensor<f64>
// CHECK-NEXT:        %15 = stablehlo.add %iterArg_8, %c_2 : tensor<i64>
// CHECK-NEXT:        %16 = stablehlo.subtract %iterArg_12, %c_2 : tensor<i64>
// CHECK-NEXT:        stablehlo.return %15, %14, %cst_3, %cst_3, %16 : tensor<i64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<i64>
// CHECK-NEXT:      }
// CHECK-NEXT:      %8 = stablehlo.add %iterArg, %c_2 : tensor<i64>
// CHECK-NEXT:      %9 = stablehlo.subtract %iterArg_7, %c_2 : tensor<i64>
// CHECK-NEXT:      stablehlo.return %8, %7#1, %7#2, %7#3, %9 : tensor<i64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<i64>
// CHECK-NEXT:    }
// CHECK-NEXT:    return %[[fwdOuter]]#1, %[[bwdOuter]]#1 : tensor<f64>, tensor<f64>
// CHECK-NEXT:  }
