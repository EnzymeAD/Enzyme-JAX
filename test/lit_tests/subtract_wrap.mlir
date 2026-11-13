// RUN: enzymexlamlir-opt --enzyme-hlo-opt="enable_convert_to_convolution=true" %s | FileCheck %s
// RUN: enzymexlamlir-opt --enzyme-hlo-opt="enable_convert_to_convolution=false" %s | FileCheck %s --check-prefix=NOCONV

// Issue #1206: subtract of shifted wraps -> merged wrap + convolution
func.func @sub_wrap(%arg0: tensor<4x1534x3056xf64>) -> tensor<4x1534x3070xf64> {
  %0 = "enzymexla.wrap"(%arg0) <{dimension = 2 : i64, lhs = 6 : i64, rhs = 8 : i64}> : (tensor<4x1534x3056xf64>) -> tensor<4x1534x3070xf64>
  %1 = "enzymexla.wrap"(%arg0) <{dimension = 2 : i64, lhs = 7 : i64, rhs = 7 : i64}> : (tensor<4x1534x3056xf64>) -> tensor<4x1534x3070xf64>
  %2 = stablehlo.subtract %0, %1 : tensor<4x1534x3070xf64>
  return %2 : tensor<4x1534x3070xf64>
}

// CHECK-LABEL: func.func @sub_wrap(%arg0: tensor<4x1534x3056xf64>) -> tensor<4x1534x3070xf64>
// CHECK-DAG{LITERAL}: %cst = stablehlo.constant dense<[[[[[-1.000000e+00, 1.000000e+00]]]]]> : tensor<1x1x1x1x2xf64>
// CHECK: %[[W:.+]] = "enzymexla.wrap"({{.*}}) <{dimension = {{2|4}} : i64, lhs = 7 : i64, rhs = 8 : i64}>
// CHECK: stablehlo.convolution({{.*}}, %cst)
// CHECK-NOT: stablehlo.subtract

func.func @sub_wrap_rev(%arg0: tensor<4x1534x3056xf64>) -> tensor<4x1534x3070xf64> {
  %0 = "enzymexla.wrap"(%arg0) <{dimension = 2 : i64, lhs = 7 : i64, rhs = 7 : i64}> : (tensor<4x1534x3056xf64>) -> tensor<4x1534x3070xf64>
  %1 = "enzymexla.wrap"(%arg0) <{dimension = 2 : i64, lhs = 6 : i64, rhs = 8 : i64}> : (tensor<4x1534x3056xf64>) -> tensor<4x1534x3070xf64>
  %2 = stablehlo.subtract %0, %1 : tensor<4x1534x3070xf64>
  return %2 : tensor<4x1534x3070xf64>
}

// CHECK-LABEL: func.func @sub_wrap_rev(%arg0: tensor<4x1534x3056xf64>) -> tensor<4x1534x3070xf64>
// CHECK-DAG{LITERAL}: %cst = stablehlo.constant dense<[[[[[1.000000e+00, -1.000000e+00]]]]]> : tensor<1x1x1x1x2xf64>
// CHECK: "enzymexla.wrap"({{.*}}) <{dimension = {{2|4}} : i64, lhs = 7 : i64, rhs = 8 : i64}>
// CHECK: stablehlo.convolution({{.*}}, %cst)

func.func @add_wrap(%arg0: tensor<4x1534x3056xf64>) -> tensor<4x1534x3070xf64> {
  %0 = "enzymexla.wrap"(%arg0) <{dimension = 2 : i64, lhs = 6 : i64, rhs = 8 : i64}> : (tensor<4x1534x3056xf64>) -> tensor<4x1534x3070xf64>
  %1 = "enzymexla.wrap"(%arg0) <{dimension = 2 : i64, lhs = 7 : i64, rhs = 7 : i64}> : (tensor<4x1534x3056xf64>) -> tensor<4x1534x3070xf64>
  %2 = stablehlo.add %0, %1 : tensor<4x1534x3070xf64>
  return %2 : tensor<4x1534x3070xf64>
}

// CHECK-LABEL: func.func @add_wrap(%arg0: tensor<4x1534x3056xf64>) -> tensor<4x1534x3070xf64>
// CHECK: %[[W2:.+]] = "enzymexla.wrap"(%arg0) <{dimension = 2 : i64, lhs = 7 : i64, rhs = 8 : i64}> : (tensor<4x1534x3056xf64>) -> tensor<4x1534x3071xf64>
// CHECK: "stablehlo.reduce_window"(%[[W2]], {{.*}}) <{window_dimensions = array<i64: 1, 1, 2>}>

// NOCONV-LABEL: func.func @add_wrap(%arg0: tensor<4x1534x3056xf64>) -> tensor<4x1534x3070xf64>
// NOCONV: %[[W3:.+]] = "enzymexla.wrap"(%arg0) <{dimension = 2 : i64, lhs = 7 : i64, rhs = 8 : i64}> : (tensor<4x1534x3056xf64>) -> tensor<4x1534x3071xf64>
// NOCONV: "stablehlo.reduce_window"(%[[W3]], {{.*}}) <{window_dimensions = array<i64: 1, 1, 2>}>

func.func @add_three_wraps(%arg0: tensor<10xf64>) -> tensor<24xf64> {
  %0 = "enzymexla.wrap"(%arg0) <{dimension = 0 : i64, lhs = 8 : i64, rhs = 6 : i64}> : (tensor<10xf64>) -> tensor<24xf64>
  %1 = "enzymexla.wrap"(%arg0) <{dimension = 0 : i64, lhs = 7 : i64, rhs = 7 : i64}> : (tensor<10xf64>) -> tensor<24xf64>
  %2 = "enzymexla.wrap"(%arg0) <{dimension = 0 : i64, lhs = 6 : i64, rhs = 8 : i64}> : (tensor<10xf64>) -> tensor<24xf64>
  %3 = stablehlo.add %0, %1 : tensor<24xf64>
  %4 = stablehlo.add %3, %2 : tensor<24xf64>
  return %4 : tensor<24xf64>
}

// NOCONV-LABEL: func.func @add_three_wraps(%arg0: tensor<10xf64>) -> tensor<24xf64>
// NOCONV: %[[W4:.+]] = "enzymexla.wrap"(%arg0) <{dimension = 0 : i64, lhs = 8 : i64, rhs = 8 : i64}> : (tensor<10xf64>) -> tensor<26xf64>
// NOCONV: "stablehlo.reduce_window"(%[[W4]], {{.*}}) <{window_dimensions = array<i64: 3>}>
