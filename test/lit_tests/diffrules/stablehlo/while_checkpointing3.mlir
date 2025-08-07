// RUN: enzymexlamlir-opt %s --enzyme --canonicalize --remove-unnecessary-enzyme-ops --enzyme-simplify-math --arith-raise --canonicalize | FileCheck %s

module {
  func.func private @f(%arg0: tensor<31xf64>) -> (tensor<f64>) {

    %cst_121 = stablehlo.constant dense<2> : tensor<31xi64>

    %c_101 = stablehlo.constant dense<16> : tensor<i64>
    %c_102 = stablehlo.constant dense<0> : tensor<i64>
    %c_103 = stablehlo.constant dense<1> : tensor<i64>

    %7697:3 = stablehlo.while(%iterArg = %c_102, %iterArg_202 = %cst_121, %iterArg_203 = %arg0) : tensor<i64>, tensor<31xi64>, tensor<31xf64> attributes {enzymexla.disable_min_cut, enzymexla.enable_checkpointing = true}
     cond {
      %23100 = stablehlo.compare  LT, %iterArg, %c_101 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %23100 : tensor<i1>
    } do {
      %23132 = stablehlo.add %iterArg, %c_103 : tensor<i64>
      %b = stablehlo.multiply %iterArg_202, %iterArg_202 : tensor<31xi64>
      %conv = stablehlo.convert %b : (tensor<31xi64>) -> tensor<31xf64>
      %m2 = stablehlo.multiply %iterArg_203, %conv : tensor<31xf64>
      stablehlo.return %23132, %b, %m2 : tensor<i64>, tensor<31xi64>, tensor<31xf64>
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
// CHECK-NEXT:    %c = stablehlo.constant dense<3> : tensor<i64>
// CHECK-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<4x31xf64>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<0> : tensor<4x31xi64>
// CHECK-NEXT:    %c_1 = stablehlo.constant dense<4> : tensor<i64>
// CHECK-NEXT:    %c_2 = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:    %c_3 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %c_4 = stablehlo.constant dense<2> : tensor<31xi64>
// CHECK-NEXT:    %cst_5 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:    %cst_6 = stablehlo.constant dense<0.000000e+00> : tensor<31xf64>
// CHECK-NEXT:    %0:5 = stablehlo.while(%iterArg = %c_3, %iterArg_7 = %c_4, %iterArg_8 = %arg0, %iterArg_9 = %c_0, %iterArg_10 = %cst) : tensor<i64>, tensor<31xi64>, tensor<31xf64>, tensor<4x31xi64>, tensor<4x31xf64>
// CHECK-NEXT:     cond {
// CHECK-NEXT:      %4 = stablehlo.compare  LT, %iterArg, %c_1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:      stablehlo.return %4 : tensor<i1>
// CHECK-NEXT:    } do {
// CHECK-NEXT:      %4 = stablehlo.reshape %iterArg_8 : (tensor<31xf64>) -> tensor<1x31xf64>
// CHECK-NEXT:      %5 = stablehlo.dynamic_update_slice %iterArg_10, %4, %iterArg, %c_3 : (tensor<4x31xf64>, tensor<1x31xf64>, tensor<i64>, tensor<i64>) -> tensor<4x31xf64>
// CHECK-NEXT:      %6 = stablehlo.reshape %iterArg_7 : (tensor<31xi64>) -> tensor<1x31xi64>
// CHECK-NEXT:      %7 = stablehlo.dynamic_update_slice %iterArg_9, %6, %iterArg, %c_3 : (tensor<4x31xi64>, tensor<1x31xi64>, tensor<i64>, tensor<i64>) -> tensor<4x31xi64>
// CHECK-NEXT:      %8:3 = stablehlo.while(%iterArg_11 = %c_3, %iterArg_12 = %iterArg_7, %iterArg_13 = %iterArg_8) : tensor<i64>, tensor<31xi64>, tensor<31xf64>
// CHECK-NEXT:       cond {
// CHECK-NEXT:        %10 = stablehlo.compare  LT, %iterArg_11, %c_1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:        stablehlo.return %10 : tensor<i1>
// CHECK-NEXT:      } do {
// CHECK-NEXT:        %10 = stablehlo.multiply %iterArg_12, %iterArg_12 : tensor<31xi64>
// CHECK-NEXT:        %11 = stablehlo.convert %10 : (tensor<31xi64>) -> tensor<31xf64>
// CHECK-NEXT:        %12 = stablehlo.multiply %iterArg_13, %11 : tensor<31xf64>
// CHECK-NEXT:        %13 = stablehlo.add %iterArg_11, %c_2 : tensor<i64>
// CHECK-NEXT:        stablehlo.return %13, %10, %12 : tensor<i64>, tensor<31xi64>, tensor<31xf64>
// CHECK-NEXT:      }
// CHECK-NEXT:      %9 = stablehlo.add %iterArg, %c_2 : tensor<i64>
// CHECK-NEXT:      stablehlo.return %9, %8#1, %8#2, %7, %5 : tensor<i64>, tensor<31xi64>, tensor<31xf64>, tensor<4x31xi64>, tensor<4x31xf64>
// CHECK-NEXT:    }
// CHECK-NEXT:    %1 = stablehlo.reshape %arg1 : (tensor<f64>) -> tensor<1xf64>
// CHECK-NEXT:    %2 = stablehlo.pad %1, %cst_5, low = [8], high = [22], interior = [0] : (tensor<1xf64>, tensor<f64>) -> tensor<31xf64>
// CHECK-NEXT:    %3:5 = stablehlo.while(%iterArg = %c_3, %iterArg_7 = %2, %iterArg_8 = %cst_6, %iterArg_9 = %cst_6, %iterArg_10 = %c) : tensor<i64>, tensor<31xf64>, tensor<31xf64>, tensor<31xf64>, tensor<i64>
// CHECK-NEXT:     cond {
// CHECK-NEXT:      %4 = stablehlo.compare  LT, %iterArg, %c_1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:      stablehlo.return %4 : tensor<i1>
// CHECK-NEXT:    } do {
// CHECK-NEXT:      %4 = stablehlo.dynamic_slice %0#4, %iterArg_10, %c_3, sizes = [1, 31] : (tensor<4x31xf64>, tensor<i64>, tensor<i64>) -> tensor<1x31xf64>
// CHECK-NEXT:      %5 = stablehlo.reshape %4 : (tensor<1x31xf64>) -> tensor<31xf64>
// CHECK-NEXT:      %6 = stablehlo.dynamic_slice %0#3, %iterArg_10, %c_3, sizes = [1, 31] : (tensor<4x31xi64>, tensor<i64>, tensor<i64>) -> tensor<1x31xi64>
// CHECK-NEXT:      %7 = stablehlo.reshape %6 : (tensor<1x31xi64>) -> tensor<31xi64>
// CHECK-NEXT:      %8:4 = stablehlo.while(%iterArg_11 = %c_3, %iterArg_12 = %7, %iterArg_13 = %5, %iterArg_14 = %cst) : tensor<i64>, tensor<31xi64>, tensor<31xf64>, tensor<4x31xf64>
// CHECK-NEXT:       cond {
// CHECK-NEXT:        %12 = stablehlo.compare  LT, %iterArg_11, %c_1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:        stablehlo.return %12 : tensor<i1>
// CHECK-NEXT:      } do {
// CHECK-NEXT:        %12 = stablehlo.multiply %iterArg_12, %iterArg_12 : tensor<31xi64>
// CHECK-NEXT:        %13 = stablehlo.convert %12 : (tensor<31xi64>) -> tensor<31xf64>
// CHECK-NEXT:        %14 = stablehlo.reshape %13 : (tensor<31xf64>) -> tensor<1x31xf64>
// CHECK-NEXT:        %15 = stablehlo.dynamic_update_slice %iterArg_14, %14, %iterArg_11, %c_3 : (tensor<4x31xf64>, tensor<1x31xf64>, tensor<i64>, tensor<i64>) -> tensor<4x31xf64>
// CHECK-NEXT:        %16 = stablehlo.multiply %iterArg_13, %13 : tensor<31xf64>
// CHECK-NEXT:        %17 = stablehlo.add %iterArg_11, %c_2 : tensor<i64>
// CHECK-NEXT:        stablehlo.return %17, %12, %16, %15 : tensor<i64>, tensor<31xi64>, tensor<31xf64>, tensor<4x31xf64>
// CHECK-NEXT:      }
// CHECK-NEXT:      %9:5 = stablehlo.while(%iterArg_11 = %c_3, %iterArg_12 = %iterArg_7, %iterArg_13 = %iterArg_8, %iterArg_14 = %iterArg_9, %iterArg_15 = %c) : tensor<i64>, tensor<31xf64>, tensor<31xf64>, tensor<31xf64>, tensor<i64>
// CHECK-NEXT:       cond {
// CHECK-NEXT:        %12 = stablehlo.compare  LT, %iterArg_11, %c_1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:        stablehlo.return %12 : tensor<i1>
// CHECK-NEXT:      } do {
// CHECK-NEXT:        %12 = stablehlo.add %iterArg_13, %iterArg_12 : tensor<31xf64>
// CHECK-NEXT:        %13 = stablehlo.dynamic_slice %8#3, %iterArg_15, %c_3, sizes = [1, 31] : (tensor<4x31xf64>, tensor<i64>, tensor<i64>) -> tensor<1x31xf64>
// CHECK-NEXT:        %14 = stablehlo.reshape %13 : (tensor<1x31xf64>) -> tensor<31xf64>
// CHECK-NEXT:        %15 = stablehlo.multiply %12, %14 : tensor<31xf64>
// CHECK-NEXT:        %16 = stablehlo.add %iterArg_14, %15 : tensor<31xf64>
// CHECK-NEXT:        %17 = stablehlo.add %iterArg_11, %c_2 : tensor<i64>
// CHECK-NEXT:        %18 = stablehlo.subtract %iterArg_15, %c_2 : tensor<i64>
// CHECK-NEXT:        stablehlo.return %17, %16, %cst_6, %cst_6, %18 : tensor<i64>, tensor<31xf64>, tensor<31xf64>, tensor<31xf64>, tensor<i64>
// CHECK-NEXT:      }
// CHECK-NEXT:      %10 = stablehlo.add %iterArg, %c_2 : tensor<i64>
// CHECK-NEXT:      %11 = stablehlo.subtract %iterArg_10, %c_2 : tensor<i64>
// CHECK-NEXT:      stablehlo.return %10, %9#1, %9#2, %9#3, %11 : tensor<i64>, tensor<31xf64>, tensor<31xf64>, tensor<31xf64>, tensor<i64>
// CHECK-NEXT:    }
// CHECK-NEXT:    return %3#1 : tensor<31xf64>
// CHECK-NEXT:  }
