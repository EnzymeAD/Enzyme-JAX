// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=while_body_boundary_transpose_simplify(1)" --transform-interpreter --enzyme-hlo-remove-transform | FileCheck %s --check-prefix=UP
// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=while_body_boundary_transpose_simplify(0)" --transform-interpreter --enzyme-hlo-remove-transform | FileCheck %s --check-prefix=DOWN

func.func @main(%arg0: tensor<16384x16384xf64> {tf.aliasing_output = 0 : i32}) -> tensor<16384x16384xf64> {
  %c = stablehlo.constant dense<1> : tensor<i32>
  %cst = stablehlo.constant dense<9.0860738986875954E-8> : tensor<16382x16382xf64>
  %cst_0 = stablehlo.constant dense<2684354.5600000001> : tensor<16382x16382xf64>
  %cst_1 = stablehlo.constant dense<-2.000000e+00> : tensor<16382x16382xf64>
  %c_2 = stablehlo.constant dense<0> : tensor<i64>
  %c_3 = stablehlo.constant dense<10> : tensor<i64>
  %c_4 = stablehlo.constant dense<1> : tensor<i64>
  %cst_5 = stablehlo.constant dense<0.000000e+00> : tensor<16384x16384xf64>
  // UP: %[[OPERAND_TRANS:.*]] = stablehlo.transpose %cst_5, dims = [0, 1]
  // UP-NEXT: stablehlo.while(%iterArg = %c_2, %iterArg_6 = %arg0, %iterArg_7 = %[[OPERAND_TRANS]])
  // DOWN: %[[OPERAND_TRANS:.*]] = stablehlo.transpose %cst_5, dims = [1, 0]
  // DOWN-NEXT: stablehlo.while(%iterArg = %c_2, %iterArg_6 = %arg0, %iterArg_7 = %[[OPERAND_TRANS]])
  %0:3 = stablehlo.while(%iterArg = %c_2, %iterArg_6 = %arg0, %iterArg_7 = %cst_5) : tensor<i64>, tensor<16384x16384xf64>, tensor<16384x16384xf64>
  cond {
    %1 = stablehlo.compare LT, %iterArg, %c_3 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %1 : tensor<i1>
  } do {
    // UP: %[[NEW_TRANS:.*]] = stablehlo.transpose %iterArg_7, dims = [1, 0]
    // UP-NEXT: %[[I:.*]] = stablehlo.add %iterArg, %c_4
    // UP-NEXT: %[[OLD_TRANS:.*]] = stablehlo.transpose %[[NEW_TRANS]], dims = [1, 0]
    // DOWN-NOT: stablehlo.transpose %iterArg_7, dims = [1, 0]
    %1 = stablehlo.add %iterArg, %c_4 {enzymexla.bounds = [[1, 10]]} : tensor<i64>
    %2 = stablehlo.transpose %iterArg_7, dims = [1, 0] : (tensor<16384x16384xf64>) -> tensor<16384x16384xf64>
    %3 = stablehlo.slice %iterArg_6 [1:16383, 1:16383] : (tensor<16384x16384xf64>) -> tensor<16382x16382xf64>
    %4 = stablehlo.slice %iterArg_6 [1:16383, 0:16382] : (tensor<16384x16384xf64>) -> tensor<16382x16382xf64>
    %5 = stablehlo.slice %iterArg_6 [1:16383, 2:16384] : (tensor<16384x16384xf64>) -> tensor<16382x16382xf64>
    %6 = stablehlo.slice %iterArg_6 [2:16384, 1:16383] : (tensor<16384x16384xf64>) -> tensor<16382x16382xf64>
    %7 = stablehlo.slice %iterArg_6 [0:16382, 1:16383] : (tensor<16384x16384xf64>) -> tensor<16382x16382xf64>
    %8 = stablehlo.multiply %3, %cst_1 : tensor<16382x16382xf64>
    %9 = stablehlo.add %5, %8 : tensor<16382x16382xf64>
    %10 = stablehlo.add %4, %9 : tensor<16382x16382xf64>
    %11 = stablehlo.add %6, %8 : tensor<16382x16382xf64>
    %12 = stablehlo.add %7, %11 : tensor<16382x16382xf64>
    %13 = stablehlo.add %10, %12 : tensor<16382x16382xf64>
    %14 = stablehlo.multiply %cst_0, %13 : tensor<16382x16382xf64>
    %15 = stablehlo.multiply %cst, %14 : tensor<16382x16382xf64>
    %16 = stablehlo.add %3, %15 : tensor<16382x16382xf64>
    // UP: %[[DUS:.*]] = stablehlo.dynamic_update_slice
    // UP-NOT: stablehlo.transpose %[[DUS]]
    // UP: stablehlo.return %[[I]], %[[DUS]], %[[DUS]]
    // DOWN: %[[DUS:.*]] = stablehlo.dynamic_update_slice
    // DOWN-NEXT: %[[OLD_TRANS:.*]] = stablehlo.transpose %[[DUS]], dims = [1, 0]
    // DOWN-NEXT: %[[NEW_TRANS:.*]] = stablehlo.transpose %[[OLD_TRANS]], dims = [1, 0]
    // DOWN-NEXT stablehlo.return %[[I:.*]], %[[DUS]], %[[NEW_TRANS]]
    %17 = stablehlo.dynamic_update_slice %2, %16, %c, %c : (tensor<16384x16384xf64>, tensor<16382x16382xf64>, tensor<i32>, tensor<i32>) -> tensor<16384x16384xf64>
    %18 = stablehlo.transpose %17, dims = [1, 0] : (tensor<16384x16384xf64>) -> tensor<16384x16384xf64>
    stablehlo.return %1, %17, %18 : tensor<i64>, tensor<16384x16384xf64>, tensor<16384x16384xf64>
  }
  return %0#1 : tensor<16384x16384xf64>
}
