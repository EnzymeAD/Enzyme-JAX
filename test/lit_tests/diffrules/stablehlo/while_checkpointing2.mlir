// RUN: enzymexlamlir-opt %s --enzyme --canonicalize --remove-unnecessary-enzyme-ops --enzyme-simplify-math --arith-raise --canonicalize | FileCheck %s

module {
  func.func private @f(%arg0: tensor<31xf64>) -> (tensor<f64>) {

    %cst_121 = stablehlo.constant dense<0.000000e+00> : tensor<15xf64>

    %c_101 = stablehlo.constant dense<16> : tensor<i64>
    %c_102 = stablehlo.constant dense<0> : tensor<i64>
    %c_103 = stablehlo.constant dense<1> : tensor<i64>

    %7697:3 = stablehlo.while(%iterArg = %c_102, %iterArg_202 = %cst_121, %iterArg_203 = %arg0) : tensor<i64>, tensor<15xf64>, tensor<31xf64> attributes {enzyme.disable_mincut, enzymexla.enable_checkpointing = true}
     cond {
      %23100 = stablehlo.compare  LT, %iterArg, %c_101 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %23100 : tensor<i1>
    } do {
      %23132 = stablehlo.add %iterArg, %c_103 : tensor<i64>
      stablehlo.return %23132, %cst_121, %arg0 : tensor<i64>, tensor<15xf64>, tensor<31xf64>
    }
    %22962 = stablehlo.slice %7697#2 [8:9] : (tensor<31xf64>) -> tensor<1xf64>
    %22963 = stablehlo.reshape %22962 : (tensor<1xf64>) -> tensor<f64>
    return %22963 : tensor<f64>
  }

  func.func @df(%arg0: tensor<31xf64>) -> tensor<31xf64> {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %0 = enzyme.autodiff @f(%arg0, %cst) {activity = [#enzyme<activity enzyme_active>], ret_activity = [#enzyme<activity enzyme_activenoneed>]} : (tensor<31xf64>, tensor<f64>) -> tensor<31xf64>
    return %0 : tensor<31xf64>
  }
}

// CHECK:  func.func private @diffef(%arg0: tensor<31xf64>, %arg1: tensor<f64>) -> tensor<31xf64> {
// CHECK-NEXT:    %c = stablehlo.constant dense<4> : tensor<i64>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:    %c_1 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:    %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<31xf64>
// CHECK-NEXT:    %0 = stablehlo.reshape %arg1 : (tensor<f64>) -> tensor<1xf64>
// CHECK-NEXT:    %1 = stablehlo.pad %0, %cst, low = [8], high = [22], interior = [0] : (tensor<1xf64>, tensor<f64>) -> tensor<31xf64>
// CHECK-NEXT:    %2:3 = stablehlo.while(%iterArg = %c_1, %iterArg_3 = %1, %iterArg_4 = %cst_2) : tensor<i64>, tensor<31xf64>, tensor<31xf64>
// CHECK-NEXT:     cond {
// CHECK-NEXT:      %4 = stablehlo.compare  LT, %iterArg, %c : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:      stablehlo.return %4 : tensor<i1>
// CHECK-NEXT:    } do {
// CHECK-NEXT:      %4:3 = stablehlo.while(%iterArg_5 = %c_1, %iterArg_6 = %iterArg_3, %iterArg_7 = %iterArg_4) : tensor<i64>, tensor<31xf64>, tensor<31xf64>
// CHECK-NEXT:       cond {
// CHECK-NEXT:        %6 = stablehlo.compare  LT, %iterArg_5, %c : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:        stablehlo.return %6 : tensor<i1>
// CHECK-NEXT:      } do {
// CHECK-NEXT:        %6 = stablehlo.add %iterArg_7, %iterArg_6 : tensor<31xf64>
// CHECK-NEXT:        %7 = stablehlo.add %iterArg_5, %c_0 : tensor<i64>
// CHECK-NEXT:        stablehlo.return %7, %cst_2, %6 : tensor<i64>, tensor<31xf64>, tensor<31xf64>
// CHECK-NEXT:      }
// CHECK-NEXT:      %5 = stablehlo.add %iterArg, %c_0 : tensor<i64>
// CHECK-NEXT:      stablehlo.return %5, %4#1, %4#2 : tensor<i64>, tensor<31xf64>, tensor<31xf64>
// CHECK-NEXT:    }
// CHECK-NEXT:    %3 = stablehlo.add %2#2, %2#1 : tensor<31xf64>
// CHECK-NEXT:    return %3 : tensor<31xf64>
// CHECK-NEXT:  }
