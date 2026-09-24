// RUN: enzymexlamlir-opt --arith-raise %s | FileCheck %s

module {
  // stablehlo has no base-2 or base-10 log/exp, so they are rescaled onto the
  // natural ones.

  // log10(x) -> log(x) * (1 / ln(10))
  // CHECK-LABEL: @log10_f32
  // CHECK: %[[SCALE:.+]] = stablehlo.constant dense<{{.*}}> : tensor<4xf32>
  // CHECK: %[[LOG:.+]] = stablehlo.log %arg0 : tensor<4xf32>
  // CHECK: %[[MUL:.+]] = stablehlo.multiply %[[LOG]], %[[SCALE]] : tensor<4xf32>
  // CHECK: return %[[MUL]] : tensor<4xf32>
  // CHECK-NOT: math.log10
  func.func @log10_f32(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    %0 = math.log10 %arg0 : tensor<4xf32>
    return %0 : tensor<4xf32>
  }

  // CHECK-LABEL: @log10_f64
  // CHECK: %[[SCALE:.+]] = stablehlo.constant dense<{{.*}}> : tensor<4xf64>
  // CHECK: %[[LOG:.+]] = stablehlo.log %arg0 : tensor<4xf64>
  // CHECK: %[[MUL:.+]] = stablehlo.multiply %[[LOG]], %[[SCALE]] : tensor<4xf64>
  // CHECK: return %[[MUL]] : tensor<4xf64>
  // CHECK-NOT: math.log10
  func.func @log10_f64(%arg0: tensor<4xf64>) -> tensor<4xf64> {
    %0 = math.log10 %arg0 : tensor<4xf64>
    return %0 : tensor<4xf64>
  }

  // log2(x) -> log(x) * (1 / ln(2))
  // CHECK-LABEL: @log2_f64
  // CHECK: %[[SCALE:.+]] = stablehlo.constant dense<{{.*}}> : tensor<4xf64>
  // CHECK: %[[LOG:.+]] = stablehlo.log %arg0 : tensor<4xf64>
  // CHECK: %[[MUL:.+]] = stablehlo.multiply %[[LOG]], %[[SCALE]] : tensor<4xf64>
  // CHECK: return %[[MUL]] : tensor<4xf64>
  // CHECK-NOT: math.log2
  func.func @log2_f64(%arg0: tensor<4xf64>) -> tensor<4xf64> {
    %0 = math.log2 %arg0 : tensor<4xf64>
    return %0 : tensor<4xf64>
  }

  // exp2(x) -> exp(x * ln(2))
  // CHECK-LABEL: @exp2_f64
  // CHECK: %[[SCALE:.+]] = stablehlo.constant dense<{{.*}}> : tensor<4xf64>
  // CHECK: %[[MUL:.+]] = stablehlo.multiply %arg0, %[[SCALE]] : tensor<4xf64>
  // CHECK: %[[EXP:.+]] = stablehlo.exponential %[[MUL]] : tensor<4xf64>
  // CHECK: return %[[EXP]] : tensor<4xf64>
  // CHECK-NOT: math.exp2
  func.func @exp2_f64(%arg0: tensor<4xf64>) -> tensor<4xf64> {
    %0 = math.exp2 %arg0 : tensor<4xf64>
    return %0 : tensor<4xf64>
  }

  // Scalars are left alone by the raising pass.
  // CHECK-LABEL: @log10_scalar
  // CHECK: math.log10 %arg0 : f32
  func.func @log10_scalar(%arg0: f32) -> f32 {
    %0 = math.log10 %arg0 : f32
    return %0 : f32
  }
}
