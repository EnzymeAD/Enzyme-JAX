// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=reshape_dynamic_slice(1);while_is_copy_simplify;greedy_while_loop_batch_fission;broadcast_to_reshape;merge_consecutive_reshapes;reshape_licm(0)" --transform-interpreter --enzyme-hlo-remove-transform --inline --enzyme-hlo-opt --enzyme-hlo-generate-td="patterns=reshape_dynamic_slice(1);while_is_copy_simplify;greedy_while_loop_batch_fission;broadcast_to_reshape;merge_consecutive_reshapes;reshape_licm(0);reshape_elementwise(0)" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

module {
  func.func @main(%arg0: tensor<10xf64>, %arg1: tensor<10xf64>) -> tensor<10xf32> {
    %c = stablehlo.constant dense<1> : tensor<i32>
    %c_0 = stablehlo.constant dense<1> : tensor<i64>
    %c_1 = stablehlo.constant dense<10> : tensor<i64>
    %c_2 = stablehlo.constant dense<0> : tensor<i64>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<10xf32>
    %c_3 = stablehlo.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]> : tensor<10xi32>
    %0:2 = stablehlo.while(%iterArg = %c_2, %iterArg_4 = %cst) : tensor<i64>, tensor<10xf32>
    cond {
      %1 = stablehlo.compare  LT, %iterArg, %c_1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %1 : tensor<i1>
    } do {
      %1 = stablehlo.add %iterArg, %c_0 : tensor<i64>
      %2 = stablehlo.dynamic_slice %c_3, %iterArg, sizes = [1] : (tensor<10xi32>, tensor<i64>) -> tensor<1xi32>
      %3 = stablehlo.reshape %2 : (tensor<1xi32>) -> tensor<i32>
      %4 = stablehlo.dynamic_slice %arg0, %3, sizes = [1] : (tensor<10xf64>, tensor<i32>) -> tensor<1xf64>
      %5 = stablehlo.dynamic_slice %arg1, %3, sizes = [1] : (tensor<10xf64>, tensor<i32>) -> tensor<1xf64>
      %6 = stablehlo.add %4, %5 : tensor<1xf64>
      %7 = stablehlo.maximum %4, %5 : tensor<1xf64>
      %8 = stablehlo.add %6, %7 : tensor<1xf64>
      %9 = stablehlo.convert %8 : (tensor<1xf64>) -> tensor<1xf32>
      %10 = stablehlo.convert %1 : (tensor<i64>) -> tensor<i32>
      %11 = stablehlo.subtract %10, %c : tensor<i32>
      %12 = stablehlo.dynamic_update_slice %iterArg_4, %9, %11 : (tensor<10xf32>, tensor<1xf32>, tensor<i32>) -> tensor<10xf32>
      stablehlo.return %1, %12 : tensor<i64>, tensor<10xf32>
    }
    return %0#1 : tensor<10xf32>
  }
}

// CHECK: func.func @main(%arg0: tensor<10xf64>, %arg1: tensor<10xf64>) -> tensor<10xf32> {
// CHECK-NEXT:     %0 = stablehlo.add %arg0, %arg1 : tensor<10xf64>
// CHECK-NEXT:     %1 = stablehlo.maximum %arg0, %arg1 : tensor<10xf64>
// CHECK-NEXT:     %2 = stablehlo.add %0, %1 : tensor<10xf64>
// CHECK-NEXT:     %3 = stablehlo.convert %2 : (tensor<10xf64>) -> tensor<10xf32>
// CHECK-NEXT:     return %3 : tensor<10xf32>
// CHECK-NEXT: }
