// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td=patterns=parity_simplify --transform-interpreter --enzyme-hlo-remove-transform | FileCheck %s

func.func @cos_neg(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  %0 = stablehlo.negate %arg0 : tensor<4xf32>
  %1 = stablehlo.cosine %0 : tensor<4xf32>
  return %1 : tensor<4xf32>
}

// CHECK-LABEL: func.func @cos_neg
// CHECK-NEXT:    %[[COS:.+]] = stablehlo.cosine %arg0 : tensor<4xf32>
// CHECK-NEXT:    return %[[COS]] : tensor<4xf32>

func.func @abs_neg_int(%arg0: tensor<4xi32>) -> tensor<4xi32> {
  %0 = stablehlo.negate %arg0 : tensor<4xi32>
  %1 = stablehlo.abs %0 : tensor<4xi32>
  return %1 : tensor<4xi32>
}

// CHECK-LABEL: func.func @abs_neg_int
// CHECK-NEXT:    %[[ABS:.+]] = stablehlo.abs %arg0 : tensor<4xi32>
// CHECK-NEXT:    return %[[ABS]] : tensor<4xi32>

// The even rewrite applies even if the negate has other users.
func.func @cos_neg_multi_use(%arg0: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
  %0 = stablehlo.negate %arg0 : tensor<4xf32>
  %1 = stablehlo.cosine %0 : tensor<4xf32>
  return %0, %1 : tensor<4xf32>, tensor<4xf32>
}

// CHECK-LABEL: func.func @cos_neg_multi_use
// CHECK-NEXT:    %[[NEG:.+]] = stablehlo.negate %arg0 : tensor<4xf32>
// CHECK-NEXT:    %[[COS:.+]] = stablehlo.cosine %arg0 : tensor<4xf32>
// CHECK-NEXT:    return %[[NEG]], %[[COS]] : tensor<4xf32>, tensor<4xf32>

func.func @cos_abs(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  %0 = stablehlo.abs %arg0 : tensor<4xf32>
  %1 = stablehlo.cosine %0 : tensor<4xf32>
  return %1 : tensor<4xf32>
}

// CHECK-LABEL: func.func @cos_abs
// CHECK-NEXT:    %[[COS:.+]] = stablehlo.cosine %arg0 : tensor<4xf32>
// CHECK-NEXT:    return %[[COS]] : tensor<4xf32>

func.func @abs_abs_int(%arg0: tensor<4xi32>) -> tensor<4xi32> {
  %0 = stablehlo.abs %arg0 : tensor<4xi32>
  %1 = stablehlo.abs %0 : tensor<4xi32>
  return %1 : tensor<4xi32>
}

// CHECK-LABEL: func.func @abs_abs_int
// CHECK-NEXT:    %[[ABS:.+]] = stablehlo.abs %arg0 : tensor<4xi32>
// CHECK-NEXT:    return %[[ABS]] : tensor<4xi32>

// For complex x, abs(x) is the real modulus rather than x or -x.
func.func @abs_abs_complex(%arg0: tensor<4xcomplex<f32>>) -> tensor<4xf32> {
  %0 = stablehlo.abs %arg0 : (tensor<4xcomplex<f32>>) -> tensor<4xf32>
  %1 = stablehlo.abs %0 : tensor<4xf32>
  return %1 : tensor<4xf32>
}

// CHECK-LABEL: func.func @abs_abs_complex
// CHECK-NEXT:    %[[ABS0:.+]] = stablehlo.abs %arg0 : (tensor<4xcomplex<f32>>) -> tensor<4xf32>
// CHECK-NEXT:    %[[ABS1:.+]] = stablehlo.abs %[[ABS0]] : tensor<4xf32>
// CHECK-NEXT:    return %[[ABS1]] : tensor<4xf32>

// For complex x, abs(x) is the real modulus rather than x or -x, so cos(abs(x))
// cannot be rewritten to cos(x).
func.func @cos_abs_complex(%arg0: tensor<4xcomplex<f32>>) -> tensor<4xf32> {
  %0 = stablehlo.abs %arg0 : (tensor<4xcomplex<f32>>) -> tensor<4xf32>
  %1 = stablehlo.cosine %0 : tensor<4xf32>
  return %1 : tensor<4xf32>
}

// CHECK-LABEL: func.func @cos_abs_complex
// CHECK-NEXT:    %[[ABS:.+]] = stablehlo.abs %arg0 : (tensor<4xcomplex<f32>>) -> tensor<4xf32>
// CHECK-NEXT:    %[[COS:.+]] = stablehlo.cosine %[[ABS]] : tensor<4xf32>
// CHECK-NEXT:    return %[[COS]] : tensor<4xf32>

// cos is also even over the complex numbers.
func.func @cos_neg_complex(%arg0: tensor<4xcomplex<f32>>) -> tensor<4xcomplex<f32>> {
  %0 = stablehlo.negate %arg0 : tensor<4xcomplex<f32>>
  %1 = stablehlo.cosine %0 : tensor<4xcomplex<f32>>
  return %1 : tensor<4xcomplex<f32>>
}

// CHECK-LABEL: func.func @cos_neg_complex
// CHECK-NEXT:    %[[COS:.+]] = stablehlo.cosine %arg0 : tensor<4xcomplex<f32>>
// CHECK-NEXT:    return %[[COS]] : tensor<4xcomplex<f32>>
