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

// CHECK: module {
// CHECK-NEXT:   func.func @without_checkpointing(%arg0: tensor<f64>) -> tensor<f64> {
// CHECK-NEXT:     %c = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:     %c_0 = stablehlo.constant dense<9> : tensor<i64>
// CHECK-NEXT:     %c_1 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:     %0:2 = stablehlo.while(%iterArg = %c_1, %iterArg_2 = %arg0) : tensor<i64>, tensor<f64> attributes {enzyme.disable_mincut}
// CHECK-NEXT:     cond {
// CHECK-NEXT:       %1 = stablehlo.compare LT, %iterArg, %c_0 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:       stablehlo.return %1 : tensor<i1>
// CHECK-NEXT:     } do {
// CHECK-NEXT:       %1 = stablehlo.add %iterArg, %c : tensor<i64>
// CHECK-NEXT:       %2 = stablehlo.convert %1 : (tensor<i64>) -> tensor<f64>
// CHECK-NEXT:       %3 = stablehlo.multiply %iterArg_2, %2 : tensor<f64>
// CHECK-NEXT:       stablehlo.return %1, %3 : tensor<i64>, tensor<f64>
// CHECK-NEXT:     }
// CHECK-NEXT:     return %0#1 : tensor<f64>
// CHECK-NEXT:   }
// CHECK-NEXT:   func.func @with_checkpointing(%arg0: tensor<f64>) -> tensor<f64> {
// CHECK-NEXT:     %c = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:     %c_0 = stablehlo.constant dense<9> : tensor<i64>
// CHECK-NEXT:     %c_1 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:     %0:2 = stablehlo.while(%iterArg = %c_1, %iterArg_2 = %arg0) : tensor<i64>, tensor<f64> attributes {enzyme.disable_mincut, enzymexla.enable_checkpointing = true}
// CHECK-NEXT:     cond {
// CHECK-NEXT:       %1 = stablehlo.compare LT, %iterArg, %c_0 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:       stablehlo.return %1 : tensor<i1>
// CHECK-NEXT:     } do {
// CHECK-NEXT:       %1 = stablehlo.add %iterArg, %c : tensor<i64>
// CHECK-NEXT:       %2 = stablehlo.convert %1 : (tensor<i64>) -> tensor<f64>
// CHECK-NEXT:       %3 = stablehlo.multiply %iterArg_2, %2 : tensor<f64>
// CHECK-NEXT:       stablehlo.return %1, %3 : tensor<i64>, tensor<f64>
// CHECK-NEXT:     }
// CHECK-NEXT:     return %0#1 : tensor<f64>
// CHECK-NEXT:   }
// CHECK-NEXT:   func.func @with_checkpointing_diff(%arg0: tensor<f64>, %arg1: tensor<f64>) -> (tensor<f64>, tensor<f64>) {
// CHECK-NEXT:     %0:2 = call @diffewith_checkpointing(%arg0, %arg1) : (tensor<f64>, tensor<f64>) -> (tensor<f64>, tensor<f64>)
// CHECK-NEXT:     return %0#0, %0#1 : tensor<f64>, tensor<f64>
// CHECK-NEXT:   }
// CHECK-NEXT:   func.func @without_checkpointing_diff(%arg0: tensor<f64>, %arg1: tensor<f64>) -> (tensor<f64>, tensor<f64>) {
// CHECK-NEXT:     %0:2 = call @diffewithout_checkpointing(%arg0, %arg1) : (tensor<f64>, tensor<f64>) -> (tensor<f64>, tensor<f64>)
// CHECK-NEXT:     return %0#0, %0#1 : tensor<f64>, tensor<f64>
// CHECK-NEXT:   }
// CHECK-NEXT:   func.func @main() {
// CHECK-NEXT:     %cst = stablehlo.constant dense<1.0000001000000001> : tensor<f64>
// CHECK-NEXT:     %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
// CHECK-NEXT:     %0:2 = call @diffewith_checkpointing(%cst, %cst_0) : (tensor<f64>, tensor<f64>) -> (tensor<f64>, tensor<f64>)
// CHECK-NEXT:     %1:2 = call @diffewithout_checkpointing(%cst, %cst_0) : (tensor<f64>, tensor<f64>) -> (tensor<f64>, tensor<f64>)
// CHECK-NEXT:     check.expect_almost_eq %0#0, %1#0 : tensor<f64>
// CHECK-NEXT:     check.expect_almost_eq %0#1, %1#1 : tensor<f64>
// CHECK-NEXT:     return
// CHECK-NEXT:   }
// CHECK-NEXT:   func.func private @diffewith_checkpointing(%arg0: tensor<f64>, %arg1: tensor<f64>) -> (tensor<f64>, tensor<f64>) {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<3xf64>
// CHECK-NEXT:     %c = stablehlo.constant dense<2> : tensor<i64>
// CHECK-NEXT:     %c_0 = stablehlo.constant dense<3> : tensor<i64>
// CHECK-NEXT:     %c_1 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:     %c_2 = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:     %cst_3 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:     %0:3 = stablehlo.while(%iterArg = %c_1, %iterArg_4 = %arg0, %iterArg_5 = %cst) : tensor<i64>, tensor<f64>, tensor<3xf64>
// CHECK-NEXT:     cond {
// CHECK-NEXT:       %2 = stablehlo.compare LT, %iterArg, %c_0 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:       stablehlo.return %2 : tensor<i1>
// CHECK-NEXT:     } do {
// CHECK-NEXT:       %2 = stablehlo.multiply %iterArg, %c_0 : tensor<i64>
// CHECK-NEXT:       %3:2 = stablehlo.while(%iterArg_6 = %c_1, %iterArg_7 = %iterArg_4) : tensor<i64>, tensor<f64> attributes {enzyme.disable_mincut}
// CHECK-NEXT:       cond {
// CHECK-NEXT:         %7 = stablehlo.compare LT, %iterArg_6, %c_0 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:         stablehlo.return %7 : tensor<i1>
// CHECK-NEXT:       } do {
// CHECK-NEXT:         %7 = stablehlo.add %iterArg_6, %2 : tensor<i64>
// CHECK-NEXT:         %8 = stablehlo.add %7, %c_2 : tensor<i64>
// CHECK-NEXT:         %9 = stablehlo.convert %8 : (tensor<i64>) -> tensor<f64>
// CHECK-NEXT:         %10 = stablehlo.multiply %iterArg_7, %9 : tensor<f64>
// CHECK-NEXT:         %11 = stablehlo.add %iterArg_6, %c_2 : tensor<i64>
// CHECK-NEXT:         stablehlo.return %11, %10 : tensor<i64>, tensor<f64>
// CHECK-NEXT:       }
// CHECK-NEXT:       %4 = stablehlo.reshape %iterArg_4 : (tensor<f64>) -> tensor<1xf64>
// CHECK-NEXT:       %5 = stablehlo.dynamic_update_slice %iterArg_5, %4, %iterArg : (tensor<3xf64>, tensor<1xf64>, tensor<i64>) -> tensor<3xf64>
// CHECK-NEXT:       %6 = stablehlo.add %iterArg, %c_2 : tensor<i64>
// CHECK-NEXT:       stablehlo.return %6, %3#1, %5 : tensor<i64>, tensor<f64>, tensor<3xf64>
// CHECK-NEXT:     }
// CHECK-NEXT:     %1:5 = stablehlo.while(%iterArg = %c_1, %iterArg_4 = %arg1, %iterArg_5 = %cst_3, %iterArg_6 = %cst_3, %iterArg_7 = %c) : tensor<i64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<i64>
// CHECK-NEXT:     cond {
// CHECK-NEXT:       %2 = stablehlo.compare LT, %iterArg, %c_0 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:       stablehlo.return %2 : tensor<i1>
// CHECK-NEXT:     } do {
// CHECK-NEXT:       %2 = stablehlo.subtract %c, %iterArg : tensor<i64>
// CHECK-NEXT:       %3 = stablehlo.multiply %c_0, %2 : tensor<i64>
// CHECK-NEXT:       %4 = stablehlo.dynamic_slice %0#2, %iterArg_7, sizes = [1] : (tensor<3xf64>, tensor<i64>) -> tensor<1xf64>
// CHECK-NEXT:       %5 = stablehlo.reshape %4 : (tensor<1xf64>) -> tensor<f64>
// CHECK-NEXT:       %6:3 = stablehlo.while(%iterArg_8 = %c_1, %iterArg_9 = %5, %iterArg_10 = %cst) : tensor<i64>, tensor<f64>, tensor<3xf64>
// CHECK-NEXT:       cond {
// CHECK-NEXT:         %10 = stablehlo.compare LT, %iterArg_8, %c_0 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:         stablehlo.return %10 : tensor<i1>
// CHECK-NEXT:       } do {
// CHECK-NEXT:         %10 = stablehlo.add %3, %iterArg_8 : tensor<i64>
// CHECK-NEXT:         %11 = stablehlo.multiply %c_2, %10 : tensor<i64>
// CHECK-NEXT:         %12 = stablehlo.add %11, %c_2 : tensor<i64>
// CHECK-NEXT:         %13 = stablehlo.convert %12 : (tensor<i64>) -> tensor<f64>
// CHECK-NEXT:         %14 = stablehlo.reshape %13 : (tensor<f64>) -> tensor<1xf64>
// CHECK-NEXT:         %15 = stablehlo.dynamic_update_slice %iterArg_10, %14, %iterArg_8 : (tensor<3xf64>, tensor<1xf64>, tensor<i64>) -> tensor<3xf64>
// CHECK-NEXT:         %16 = stablehlo.multiply %iterArg_9, %13 : tensor<f64>
// CHECK-NEXT:         %17 = stablehlo.add %iterArg_8, %c_2 : tensor<i64>
// CHECK-NEXT:         stablehlo.return %17, %16, %15 : tensor<i64>, tensor<f64>, tensor<3xf64>
// CHECK-NEXT:       }
// CHECK-NEXT:       %7:5 = stablehlo.while(%iterArg_8 = %c_1, %iterArg_9 = %iterArg_4, %iterArg_10 = %iterArg_5, %iterArg_11 = %iterArg_6, %iterArg_12 = %c) : tensor<i64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<i64>
// CHECK-NEXT:       cond {
// CHECK-NEXT:         %10 = stablehlo.compare LT, %iterArg_8, %c_0 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:         stablehlo.return %10 : tensor<i1>
// CHECK-NEXT:       } do {
// CHECK-NEXT:         %10 = stablehlo.add %iterArg_10, %iterArg_9 : tensor<f64>
// CHECK-NEXT:         %11 = stablehlo.dynamic_slice %6#2, %iterArg_12, sizes = [1] : (tensor<3xf64>, tensor<i64>) -> tensor<1xf64>
// CHECK-NEXT:         %12 = stablehlo.reshape %11 : (tensor<1xf64>) -> tensor<f64>
// CHECK-NEXT:         %13 = stablehlo.multiply %10, %12 : tensor<f64>
// CHECK-NEXT:         %14 = stablehlo.add %iterArg_11, %13 : tensor<f64>
// CHECK-NEXT:         %15 = stablehlo.add %iterArg_8, %c_2 : tensor<i64>
// CHECK-NEXT:         %16 = stablehlo.subtract %iterArg_12, %c_2 : tensor<i64>
// CHECK-NEXT:         stablehlo.return %15, %14, %cst_3, %cst_3, %16 : tensor<i64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<i64>
// CHECK-NEXT:       }
// CHECK-NEXT:       %8 = stablehlo.add %iterArg, %c_2 : tensor<i64>
// CHECK-NEXT:       %9 = stablehlo.subtract %iterArg_7, %c_2 : tensor<i64>
// CHECK-NEXT:       stablehlo.return %8, %7#1, %7#2, %7#3, %9 : tensor<i64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<i64>
// CHECK-NEXT:     }
// CHECK-NEXT:     return %0#1, %1#1 : tensor<f64>, tensor<f64>
// CHECK-NEXT:   }
// CHECK-NEXT:   func.func private @diffewithout_checkpointing(%arg0: tensor<f64>, %arg1: tensor<f64>) -> (tensor<f64>, tensor<f64>) {
// CHECK-NEXT:     %c = stablehlo.constant dense<8> : tensor<i64>
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<9xf64>
// CHECK-NEXT:     %c_0 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:     %c_1 = stablehlo.constant dense<9> : tensor<i64>
// CHECK-NEXT:     %c_2 = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:     %0:3 = stablehlo.while(%iterArg = %c_0, %iterArg_3 = %arg0, %iterArg_4 = %cst) : tensor<i64>, tensor<f64>, tensor<9xf64>
// CHECK-NEXT:     cond {
// CHECK-NEXT:       %2 = stablehlo.compare LT, %iterArg, %c_1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:       stablehlo.return %2 : tensor<i1>
// CHECK-NEXT:     } do {
// CHECK-NEXT:       %2 = stablehlo.add %iterArg, %c_2 : tensor<i64>
// CHECK-NEXT:       %3 = stablehlo.convert %2 : (tensor<i64>) -> tensor<f64>
// CHECK-NEXT:       %4 = stablehlo.reshape %3 : (tensor<f64>) -> tensor<1xf64>
// CHECK-NEXT:       %5 = stablehlo.dynamic_update_slice %iterArg_4, %4, %iterArg : (tensor<9xf64>, tensor<1xf64>, tensor<i64>) -> tensor<9xf64>
// CHECK-NEXT:       %6 = stablehlo.multiply %iterArg_3, %3 : tensor<f64>
// CHECK-NEXT:       stablehlo.return %2, %6, %5 : tensor<i64>, tensor<f64>, tensor<9xf64>
// CHECK-NEXT:     }
// CHECK-NEXT:     %1:3 = stablehlo.while(%iterArg = %c_0, %iterArg_3 = %arg1, %iterArg_4 = %c) : tensor<i64>, tensor<f64>, tensor<i64>
// CHECK-NEXT:     cond {
// CHECK-NEXT:       %2 = stablehlo.compare LT, %iterArg, %c_1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:       stablehlo.return %2 : tensor<i1>
// CHECK-NEXT:     } do {
// CHECK-NEXT:       %2 = stablehlo.add %iterArg, %c_2 : tensor<i64>
// CHECK-NEXT:       %3 = stablehlo.dynamic_slice %0#2, %iterArg_4, sizes = [1] : (tensor<9xf64>, tensor<i64>) -> tensor<1xf64>
// CHECK-NEXT:       %4 = stablehlo.reshape %3 : (tensor<1xf64>) -> tensor<f64>
// CHECK-NEXT:       %5 = stablehlo.multiply %iterArg_3, %4 : tensor<f64>
// CHECK-NEXT:       %6 = stablehlo.subtract %iterArg_4, %c_2 : tensor<i64>
// CHECK-NEXT:       stablehlo.return %2, %5, %6 : tensor<i64>, tensor<f64>, tensor<i64>
// CHECK-NEXT:     }
// CHECK-NEXT:     return %0#1, %1#1 : tensor<f64>, tensor<f64>
// CHECK-NEXT:   }
// CHECK-NEXT: }
