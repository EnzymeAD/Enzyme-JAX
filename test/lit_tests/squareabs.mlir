// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt | FileCheck %s

func.func @squareabs(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
    %0 = stablehlo.abs %arg0 : tensor<2x2xf32>
    %1 = stablehlo.multiply %0, %0 : tensor<2x2xf32>
    return %1 : tensor<2x2xf32>
}

// CHECK: func.func @squareabs(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32>
// CHECK-NEXT:   %0 = stablehlo.multiply %arg0, %arg0 : tensor<2x2xf32>
// CHECK-NEXT:   return %0 : tensor<2x2xf32>
// CHECK-NEXT: }

func.func @squareabscomplex(%arg0: tensor<2x2xcomplex<f32>>) -> tensor<2x2xf32> {
    %0 = stablehlo.abs %arg0 : (tensor<2x2xcomplex<f32>>) -> tensor<2x2xf32>
    %1 = stablehlo.multiply %0, %0 : tensor<2x2xf32>
    return %1 : tensor<2x2xf32>
}

// CHECK: func.func @squareabscomplex(%arg0: tensor<2x2xcomplex<f32>>) -> tensor<2x2xf32>
// CHECK-NEXT:   %0 = chlo.conj %arg0 : tensor<2x2xcomplex<f32>> -> tensor<2x2xcomplex<f32>>
// CHECK-NEXT:   %1 = stablehlo.multiply %arg0, %0 : tensor<2x2xcomplex<f32>>
// CHECK-NEXT:   %2 = stablehlo.real %1 : (tensor<2x2xcomplex<f32>>) -> tensor<2x2xf32>
// CHECK-NEXT:   return %2 : tensor<2x2xf32>
// CHECK-NEXT: }

// doesn't apply here
func.func @squareabscomplex2(%arg0: tensor<2x2xcomplex<f32>>) -> (tensor<2x2xf32>, tensor<2x2xf32>) {
    // CHECK: %0 = stablehlo.abs %arg0 : (tensor<2x2xcomplex<f32>>) -> tensor<2x2xf32>
    %0 = stablehlo.abs %arg0 : (tensor<2x2xcomplex<f32>>) -> tensor<2x2xf32>
    %1 = stablehlo.multiply %0, %0 : tensor<2x2xf32>
    return %1, %0 : tensor<2x2xf32>, tensor<2x2xf32>
}
