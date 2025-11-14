// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=transpose_symmetric_simplify" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

func.func @pass1(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<2x2xf32>) -> tensor<2x2xf32>
  %1 = stablehlo.add %arg0, %0 : tensor<2x2xf32>
  %2 = stablehlo.transpose %1, dims = [1, 0] : (tensor<2x2xf32>) -> tensor<2x2xf32>
  return %2 : tensor<2x2xf32>
}

// CHECK: func.func @pass1(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
// CHECK-NEXT:   %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<2x2xf32>) -> tensor<2x2xf32>
// CHECK-NEXT:   %1 = stablehlo.add %arg0, %0 {enzymexla.guaranteed_symmetric = true} : tensor<2x2xf32>
// CHECK-NEXT:   return %1 : tensor<2x2xf32>
// CHECK-NEXT: }

func.func @pass2(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<2x2xf32>) -> tensor<2x2xf32>
  %1 = stablehlo.add %0, %arg0 : tensor<2x2xf32>
  %2 = stablehlo.transpose %1, dims = [1, 0] : (tensor<2x2xf32>) -> tensor<2x2xf32>
  return %2 : tensor<2x2xf32>
}

// CHECK: func.func @pass2(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
// CHECK-NEXT:   %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<2x2xf32>) -> tensor<2x2xf32>
// CHECK-NEXT:   %1 = stablehlo.add %0, %arg0 {enzymexla.guaranteed_symmetric = true} : tensor<2x2xf32>
// CHECK-NEXT:   return %1 : tensor<2x2xf32>
// CHECK-NEXT: }

func.func @fail1(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<2x2xf32>) -> tensor<2x2xf32>
  %1 = stablehlo.subtract %arg0, %0 : tensor<2x2xf32>
  %2 = stablehlo.transpose %1, dims = [1, 0] : (tensor<2x2xf32>) -> tensor<2x2xf32>
  return %2 : tensor<2x2xf32>
}

// CHECK: func.func @fail1(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
// CHECK-NEXT:   %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<2x2xf32>) -> tensor<2x2xf32>
// CHECK-NEXT:   %1 = stablehlo.subtract %arg0, %0 {enzymexla.guaranteed_symmetric = false} : tensor<2x2xf32>
// CHECK-NEXT:   %2 = stablehlo.transpose %1, dims = [1, 0] : (tensor<2x2xf32>) -> tensor<2x2xf32>
// CHECK-NEXT:   return %2 : tensor<2x2xf32>
// CHECK-NEXT: }
