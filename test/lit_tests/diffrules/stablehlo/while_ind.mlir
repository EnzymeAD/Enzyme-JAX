// RUN: enzymexlamlir-opt %s --enzyme --canonicalize --remove-unnecessary-enzyme-ops --enzyme-simplify-math --arith-raise --canonicalize | FileCheck %s
// RUN: enzymexlamlir-opt %s --enzyme --canonicalize --remove-unnecessary-enzyme-ops --enzyme-simplify-math --arith-raise --canonicalize | stablehlo-translate --interpret

module @reactant_differe... attributes {mhlo.num_partitions = 1 : i64, mhlo.num_replicas = 1 : i64} {
  func.func private @f(%arg0: tensor<f64>) -> (tensor<f64>) {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %c = stablehlo.constant dense<0> : tensor<i64>
    %c_0 = stablehlo.constant dense<16> : tensor<i64>
    %c_1 = stablehlo.constant dense<1> : tensor<i64>
    %0:3 = stablehlo.while(%iterArg = %c, %iterArg_2 = %cst, %iterArg_3 = %cst) : tensor<i64>, tensor<f64>, tensor<f64> attributes {enzyme.disable_mincut}
     cond {
      %1 = stablehlo.compare  LT, %iterArg, %c_0 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %1 : tensor<i1>
    } do {
      %1 = stablehlo.add %iterArg, %c_1 : tensor<i64>
      %2 = stablehlo.add %iterArg_2, %iterArg_3 : tensor<f64>
      stablehlo.return %1, %2, %arg0 : tensor<i64>, tensor<f64>, tensor<f64>
    }
    return %0#1 : tensor<f64>
  }
  func.func private @f_mc(%arg0: tensor<f64>) -> (tensor<f64>) {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %c = stablehlo.constant dense<0> : tensor<i64>
    %c_0 = stablehlo.constant dense<16> : tensor<i64>
    %c_1 = stablehlo.constant dense<1> : tensor<i64>
    %0:3 = stablehlo.while(%iterArg = %c, %iterArg_2 = %cst, %iterArg_3 = %cst) : tensor<i64>, tensor<f64>, tensor<f64>
     cond {
      %1 = stablehlo.compare  LT, %iterArg, %c_0 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %1 : tensor<i1>
    } do {
      %1 = stablehlo.add %iterArg, %c_1 : tensor<i64>
      %2 = stablehlo.add %iterArg_2, %iterArg_3 : tensor<f64>
      stablehlo.return %1, %2, %arg0 : tensor<i64>, tensor<f64>, tensor<f64>
    }
    return %0#1 : tensor<f64>
  }
  func.func private @f_cp(%arg0: tensor<f64>) -> (tensor<f64>) {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %c = stablehlo.constant dense<0> : tensor<i64>
    %c_0 = stablehlo.constant dense<16> : tensor<i64>
    %c_1 = stablehlo.constant dense<1> : tensor<i64>
    %0:3 = stablehlo.while(%iterArg = %c, %iterArg_2 = %cst, %iterArg_3 = %cst) : tensor<i64>, tensor<f64>, tensor<f64> attributes {enzyme.disable_mincut, enzymexla.enable_checkpointing = true}
     cond {
      %1 = stablehlo.compare  LT, %iterArg, %c_0 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %1 : tensor<i1>
    } do {
      %1 = stablehlo.add %iterArg, %c_1 : tensor<i64>
      %2 = stablehlo.add %iterArg_2, %iterArg_3 : tensor<f64>
      stablehlo.return %1, %2, %arg0 : tensor<i64>, tensor<f64>, tensor<f64>
    }
    return %0#1 : tensor<f64>
  }
  func.func @main() {
    %cst0 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %cst2 = stablehlo.constant dense<15.000000e+00> : tensor<f64>
    
    %0 = enzyme.autodiff @f(%cst0, %cst) {activity = [#enzyme<activity enzyme_active>], ret_activity = [#enzyme<activity enzyme_activenoneed>]} : (tensor<f64>, tensor<f64>) -> (tensor<f64>)
    check.expect_almost_eq %0, %cst2 : tensor<f64>
    
    %1 = enzyme.autodiff @f_mc(%cst0, %cst) {activity = [#enzyme<activity enzyme_active>], ret_activity = [#enzyme<activity enzyme_activenoneed>]} : (tensor<f64>, tensor<f64>) -> (tensor<f64>)
    check.expect_almost_eq %1, %cst2 : tensor<f64>
    
    %2 = enzyme.autodiff @f_cp(%cst0, %cst) {activity = [#enzyme<activity enzyme_active>], ret_activity = [#enzyme<activity enzyme_activenoneed>]} : (tensor<f64>, tensor<f64>) -> (tensor<f64>)
    check.expect_almost_eq %2, %cst2 : tensor<f64>

    return
  }
}


// CHECK:  func.func private @diffef(%arg0: tensor<f64>, %arg1: tensor<f64>) -> tensor<f64> {
// CHECK-NEXT:    %c = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<16> : tensor<i64>
// CHECK-NEXT:    %c_1 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:    %0:4 = stablehlo.while(%iterArg = %c_1, %iterArg_2 = %arg1, %iterArg_3 = %cst, %iterArg_4 = %cst) : tensor<i64>, tensor<f64>, tensor<f64>, tensor<f64>
// CHECK-NEXT:     cond {
// CHECK-NEXT:      %1 = stablehlo.compare  LT, %iterArg, %c_0 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:      stablehlo.return %1 : tensor<i1>
// CHECK-NEXT:    } do {
// CHECK-NEXT:      %1 = stablehlo.add %iterArg, %c : tensor<i64>
// CHECK-NEXT:      %2 = stablehlo.add %iterArg_4, %iterArg_3 : tensor<f64>
// CHECK-NEXT:      stablehlo.return %1, %iterArg_2, %iterArg_2, %2 : tensor<i64>, tensor<f64>, tensor<f64>, tensor<f64>
// CHECK-NEXT:    }
// CHECK-NEXT:    return %0#3 : tensor<f64>
// CHECK-NEXT:  }
