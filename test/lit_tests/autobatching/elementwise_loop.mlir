// RUN: enzymexlamlir-opt --enzyme-hlo-opt --auto-batching --enzyme-hlo-opt %s | FileCheck %s

func.func @main(%arg0: tensor<10xf64>) -> tensor<10xf64> {
    %c = stablehlo.constant dense<1> : tensor<i32>
    %c_0 = stablehlo.constant dense<0> : tensor<i64>
    %c_1 = stablehlo.constant dense<10> : tensor<i64>
    %c_2 = stablehlo.constant dense<1> : tensor<i64>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<10xf64>
    %0:2 = stablehlo.while(%iterArg = %c_0, %iterArg_3 = %cst) : tensor<i64>, tensor<10xf64> attributes {enzymexla.disable_min_cut}
    cond {
      %1 = stablehlo.compare  LT, %iterArg, %c_1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %1 : tensor<i1>
    } do {
      %1 = stablehlo.add %c_2, %iterArg : tensor<i64>
      %2 = stablehlo.convert %1 : (tensor<i64>) -> tensor<i32>
      %3 = stablehlo.subtract %2, %c : tensor<i32>
      %4 = stablehlo.dynamic_slice %arg0, %3, sizes = [1] : (tensor<10xf64>, tensor<i32>) -> tensor<1xf64>

      %sin_res = stablehlo.sine %4 : tensor<1xf64>
      %neg_res = stablehlo.negate %sin_res : tensor<1xf64>
      %cos_res = stablehlo.cosine %4 : tensor<1xf64>
      %5 = stablehlo.add %neg_res, %cos_res : tensor<1xf64>

      %6 = stablehlo.remainder %iterArg, %c_1 : tensor<i64>
      %7 = stablehlo.add %6, %c_2 : tensor<i64>
      %8 = stablehlo.convert %7 : (tensor<i64>) -> tensor<i32>
      %9 = stablehlo.subtract %8, %c : tensor<i32>
      %10 = stablehlo.dynamic_update_slice %iterArg_3, %5, %9 : (tensor<10xf64>, tensor<1xf64>, tensor<i32>) -> tensor<10xf64>

      stablehlo.return %1, %10 : tensor<i64>, tensor<10xf64>
    }
    return %0#1 : tensor<10xf64>
}

// CHECK:  func.func @main(%arg0: tensor<10xf64>) -> tensor<10xf64> {
// CHECK-DAG:    [[COS:%.*]] = stablehlo.cosine %arg0 : tensor<10xf64>
// CHECK-DAG:    [[SIN:%.*]] = stablehlo.sine %arg0 : tensor<10xf64>
// CHECK:    [[RES:%.*]] = stablehlo.subtract [[COS]], [[SIN]] : tensor<10xf64>
// CHECK:    return [[RES]] : tensor<10xf64>
// CHECK:  }
