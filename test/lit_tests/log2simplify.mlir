// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(enzyme-hlo-generate-td{patterns=log_simplify;log2_simplify},transform-interpreter,enzyme-hlo-remove-transform)" | FileCheck %s
// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt | FileCheck %s

// CHECK-LABEL: @log2_exp2
// CHECK-NOT: stablehlo.log2
// CHECK: return %arg0 : tensor<4xf32>
func.func @log2_exp2(%x : tensor<4xf32>) -> tensor<4xf32> {
  %e = stablehlo.exp2 %x : tensor<4xf32>
  %l = stablehlo.log2 %e : tensor<4xf32>
  return %l : tensor<4xf32>
}

// log2(exp(x)) -> x * log2(e)
// CHECK-LABEL: @log2_exp
// CHECK-NOT: stablehlo.log2
// CHECK: %[[C:.+]] = stablehlo.constant dense<1.44269502> : tensor<4xf32>
// CHECK: %[[R:.+]] = stablehlo.multiply %arg0, %[[C]] : tensor<4xf32>
// CHECK: return %[[R]] : tensor<4xf32>
func.func @log2_exp(%x : tensor<4xf32>) -> tensor<4xf32> {
  %e = stablehlo.exponential %x : tensor<4xf32>
  %l = stablehlo.log2 %e : tensor<4xf32>
  return %l : tensor<4xf32>
}

// log(exp2(x)) -> x * ln(2)
// CHECK-LABEL: @log_exp2
// CHECK-NOT: stablehlo.log
// CHECK: %[[C:.+]] = stablehlo.constant dense<0.693147182> : tensor<4xf32>
// CHECK: %[[R:.+]] = stablehlo.multiply %arg0, %[[C]] : tensor<4xf32>
// CHECK: return %[[R]] : tensor<4xf32>
func.func @log_exp2(%x : tensor<4xf32>) -> tensor<4xf32> {
  %e = stablehlo.exp2 %x : tensor<4xf32>
  %l = stablehlo.log %e : tensor<4xf32>
  return %l : tensor<4xf32>
}

// The full pipeline folds the divide by 2 into a multiply by 0.5.
// CHECK-LABEL: @log2_sqrt
// CHECK-NOT: stablehlo.sqrt
// CHECK: %[[C:.+]] = stablehlo.constant dense<{{2.000000e\+00|5.000000e-01}}> : tensor<4xf32>
// CHECK: %[[L:.+]] = stablehlo.log2 %arg0 : tensor<4xf32>
// CHECK: %[[R:.+]] = stablehlo.{{divide|multiply}} %[[L]], %[[C]] : tensor<4xf32>
// CHECK: return %[[R]] : tensor<4xf32>
func.func @log2_sqrt(%x : tensor<4xf32>) -> tensor<4xf32> {
  %s = stablehlo.sqrt %x : tensor<4xf32>
  %l = stablehlo.log2 %s : tensor<4xf32>
  return %l : tensor<4xf32>
}

// CHECK-LABEL: @log2_rsqrt
// CHECK-NOT: stablehlo.rsqrt
// CHECK: %[[C:.+]] = stablehlo.constant dense<{{-2.000000e\+00|-5.000000e-01}}> : tensor<4xf32>
// CHECK: %[[L:.+]] = stablehlo.log2 %arg0 : tensor<4xf32>
// CHECK: %[[R:.+]] = stablehlo.{{divide|multiply}} %[[L]], %[[C]] : tensor<4xf32>
// CHECK: return %[[R]] : tensor<4xf32>
func.func @log2_rsqrt(%x : tensor<4xf32>) -> tensor<4xf32> {
  %s = stablehlo.rsqrt %x : tensor<4xf32>
  %l = stablehlo.log2 %s : tensor<4xf32>
  return %l : tensor<4xf32>
}

// The sqrt is still needed elsewhere: leave the log2 alone.
// CHECK-LABEL: @log2_sqrt_multi_use
// CHECK: %[[S:.+]] = stablehlo.sqrt %arg0 : tensor<4xf32>
// CHECK: %[[L:.+]] = stablehlo.log2 %[[S]] : tensor<4xf32>
// CHECK: return %[[S]], %[[L]]
func.func @log2_sqrt_multi_use(%x : tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
  %s = stablehlo.sqrt %x : tensor<4xf32>
  %l = stablehlo.log2 %s : tensor<4xf32>
  return %s, %l : tensor<4xf32>, tensor<4xf32>
}
