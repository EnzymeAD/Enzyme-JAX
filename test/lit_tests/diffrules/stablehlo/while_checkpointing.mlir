// RUN: enzymexlamlir-opt %s --enzyme --canonicalize --remove-unnecessary-enzyme-ops --enzyme-simplify-math --arith-raise --enzyme-hlo-opt --canonicalize | FileCheck %s
// RUN: enzymexlamlir-opt %s --enzyme --canonicalize --remove-unnecessary-enzyme-ops --enzyme-simplify-math --enzyme-hlo-unroll --arith-raise --enzyme-hlo-opt --canonicalize | stablehlo-translate --interpret

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
// CHECK-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<3xf64>
// CHECK-NEXT:    %c = stablehlo.constant dense<-1> : tensor<i64>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<2> : tensor<i64>
// CHECK-NEXT:    %c_1 = stablehlo.constant dense<3> : tensor<i64>
// CHECK-NEXT:    %c_2 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %c_3 = stablehlo.constant dense<9> : tensor<i64>
// CHECK-NEXT:    %c_4 = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:    %0:3 = stablehlo.while(%iterArg = %c_2, %iterArg_5 = %arg0, %iterArg_6 = %cst) : tensor<i64>, tensor<f64>, tensor<3xf64>
// CHECK-NEXT:     cond {
// CHECK-NEXT:      %2 = stablehlo.compare  LT, %iterArg, %c_1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:      stablehlo.return %2 : tensor<i1>
// CHECK-NEXT:    } do {
// CHECK-NEXT:      %2 = stablehlo.reshape %iterArg_5 : (tensor<f64>) -> tensor<1xf64>
// CHECK-NEXT:      %3 = stablehlo.dynamic_update_slice %iterArg_6, %2, %iterArg : (tensor<3xf64>, tensor<1xf64>, tensor<i64>) -> tensor<3xf64>
// CHECK-NEXT:      %4 = stablehlo.multiply %iterArg, %c_1 : tensor<i64>
// CHECK-NEXT:      %5:2 = stablehlo.while(%iterArg_7 = %c_2, %iterArg_8 = %iterArg_5) : tensor<i64>, tensor<f64>
// CHECK-NEXT:       cond {
// CHECK-NEXT:        %7 = stablehlo.compare  LT, %iterArg_7, %c_1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:        stablehlo.return %7 : tensor<i1>
// CHECK-NEXT:      } do {
// CHECK-NEXT:        %7 = stablehlo.add %iterArg_7, %4 : tensor<i64>
// CHECK-NEXT:        %8:2 = func.call @while_checkpoint(%7, %iterArg_8, %c_3, %c_4) : (tensor<i64>, tensor<f64>, tensor<i64>, tensor<i64>) -> (tensor<i64>, tensor<f64>)
// CHECK-NEXT:        %9 = stablehlo.add %iterArg_7, %c_4 : tensor<i64>
// CHECK-NEXT:        stablehlo.return %9, %8#1 : tensor<i64>, tensor<f64>
// CHECK-NEXT:      }
// CHECK-NEXT:      %6 = stablehlo.add %iterArg, %c_4 : tensor<i64>
// CHECK-NEXT:      stablehlo.return %6, %5#1, %3 : tensor<i64>, tensor<f64>, tensor<3xf64>
// CHECK-NEXT:    }
// CHECK-NEXT:    %1:3 = stablehlo.while(%iterArg = %c_0, %iterArg_5 = %arg1, %iterArg_6 = %c_0) : tensor<i64>, tensor<f64>, tensor<i64>
// CHECK-NEXT:     cond {
// CHECK-NEXT:      %2 = stablehlo.compare  LT, %iterArg, %c : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:      stablehlo.return %2 : tensor<i1>
// CHECK-NEXT:    } do {
// CHECK-NEXT:      %2 = stablehlo.dynamic_slice %0#2, %iterArg_6, sizes = [1] : (tensor<3xf64>, tensor<i64>) -> tensor<1xf64>
// CHECK-NEXT:      %3 = stablehlo.reshape %2 : (tensor<1xf64>) -> tensor<f64>
// CHECK-NEXT:      %4:2 = stablehlo.while(%iterArg_7 = %c_0, %iterArg_8 = %iterArg_5) : tensor<i64>, tensor<f64>
// CHECK-NEXT:       cond {
// CHECK-NEXT:        %7 = stablehlo.compare  LT, %iterArg_7, %c : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:        stablehlo.return %7 : tensor<i1>
// CHECK-NEXT:      } do {
// CHECK-NEXT:        %7 = stablehlo.multiply %iterArg, %c_1 : tensor<i64>
// CHECK-NEXT:        %8 = stablehlo.add %7, %iterArg_7 : tensor<i64>
// CHECK-NEXT:        %9:2 = stablehlo.while(%iterArg_9 = %c_2, %iterArg_10 = %3) : tensor<i64>, tensor<f64>
// CHECK-NEXT:         cond {
// CHECK-NEXT:          %12 = stablehlo.compare  LT, %iterArg_9, %iterArg_7 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:          stablehlo.return %12 : tensor<i1>
// CHECK-NEXT:        } do {
// CHECK-NEXT:          %12:2 = func.call @while_checkpoint(%8, %iterArg_10, %c_3, %c_4) : (tensor<i64>, tensor<f64>, tensor<i64>, tensor<i64>) -> (tensor<i64>, tensor<f64>)
// CHECK-NEXT:          %13 = stablehlo.add %iterArg_9, %c_4 : tensor<i64>
// CHECK-NEXT:          stablehlo.return %13, %12#1 : tensor<i64>, tensor<f64>
// CHECK-NEXT:        }
// CHECK-NEXT:        %10 = func.call @diffewhile_checkpoint(%8, %9#1, %c_3, %c_4, %iterArg_8) : (tensor<i64>, tensor<f64>, tensor<i64>, tensor<i64>, tensor<f64>) -> tensor<f64>
// CHECK-NEXT:        %11 = stablehlo.add %iterArg_7, %c : tensor<i64>
// CHECK-NEXT:        stablehlo.return %11, %10 : tensor<i64>, tensor<f64>
// CHECK-NEXT:      }
// CHECK-NEXT:      %5 = stablehlo.add %iterArg, %c : tensor<i64>
// CHECK-NEXT:      %6 = stablehlo.subtract %iterArg_6, %c_4 : tensor<i64>
// CHECK-NEXT:      stablehlo.return %5, %4#1, %6 : tensor<i64>, tensor<f64>, tensor<i64>
// CHECK-NEXT:    }
// CHECK-NEXT:    return %0#1, %1#1 : tensor<f64>, tensor<f64>
// CHECK-NEXT:  }
