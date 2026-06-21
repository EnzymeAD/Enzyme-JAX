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
// CHECK-DAG:   [[REAL:%[0-9]+]] = stablehlo.real %arg0 : (tensor<2x2xcomplex<f32>>) -> tensor<2x2xf32>
// CHECK-DAG:   [[IMAG:%[0-9]+]] = stablehlo.imag %arg0 : (tensor<2x2xcomplex<f32>>) -> tensor<2x2xf32>
// CHECK-DAG:   [[REAL2:%[0-9]+]] = stablehlo.multiply [[REAL]], [[REAL]] : tensor<2x2xf32>
// CHECK-DAG:   [[IMAG2:%[0-9]+]] = stablehlo.multiply [[IMAG]], [[IMAG]] : tensor<2x2xf32>
// CHECK:       [[ADD:%[0-9]+]] = stablehlo.add [[REAL2]], [[IMAG2]] : tensor<2x2xf32>
// CHECK-NEXT:   return [[ADD]] : tensor<2x2xf32>
// CHECK-NEXT: }

// doesn't apply here
func.func @squareabscomplex2(%arg0: tensor<2x2xcomplex<f32>>) -> (tensor<2x2xf32>, tensor<2x2xf32>) {
    // CHECK: %0 = stablehlo.abs %arg0 : (tensor<2x2xcomplex<f32>>) -> tensor<2x2xf32>
    %0 = stablehlo.abs %arg0 : (tensor<2x2xcomplex<f32>>) -> tensor<2x2xf32>
    %1 = stablehlo.multiply %0, %0 : tensor<2x2xf32>
    return %1, %0 : tensor<2x2xf32>, tensor<2x2xf32>
}
