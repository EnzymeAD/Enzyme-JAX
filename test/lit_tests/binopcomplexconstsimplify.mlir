// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

module {
  func.func @main(%arg0: tensor<2x2xf64>, %arg1: tensor<2x2xf64>, %arg2: tensor<2x2xf64>, %arg3: tensor<2x2xf64>) -> (tensor<2x2xcomplex<f64>>) {
    %0 = stablehlo.complex %arg0, %arg1 : tensor<2x2xcomplex<f64>>
    %1 = stablehlo.complex %arg2, %arg3 : tensor<2x2xcomplex<f64>>
    %2 = stablehlo.add %0, %1 : tensor<2x2xcomplex<f64>>
    return %2 : tensor<2x2xcomplex<f64>>
  }
}

// CHECK: func.func @main(%arg0: tensor<2x2xf64>, %arg1: tensor<2x2xf64>, %arg2: tensor<2x2xf64>, %arg3: tensor<2x2xf64>) -> tensor<2x2xcomplex<f64>> {
// CHECK-NEXT:   %0 = stablehlo.add %arg0, %arg2 : tensor<2x2xf64>
// CHECK-NEXT:   %1 = stablehlo.add %arg1, %arg3 : tensor<2x2xf64>
// CHECK-NEXT:   %2 = stablehlo.complex %0, %1 : tensor<2x2xcomplex<f64>>
// CHECK-NEXT:   return %2 : tensor<2x2xcomplex<f64>>
// CHECK-NEXT: }

module {
  func.func @main(%arg0: tensor<2x2xf64>, %arg1: tensor<2x2xf64>, %arg2: tensor<2x2xf64>, %arg3: tensor<2x2xf64>) -> (tensor<2x2xcomplex<f64>>) {
    %0 = stablehlo.complex %arg0, %arg1 : tensor<2x2xcomplex<f64>>
    %1 = stablehlo.complex %arg2, %arg3 : tensor<2x2xcomplex<f64>>
    %2 = stablehlo.subtract %0, %1 : tensor<2x2xcomplex<f64>>
    return %2 : tensor<2x2xcomplex<f64>>
  }
}

// CHECK: func.func @main(%arg0: tensor<2x2xf64>, %arg1: tensor<2x2xf64>, %arg2: tensor<2x2xf64>, %arg3: tensor<2x2xf64>) -> tensor<2x2xcomplex<f64>> {
// CHECK-NEXT:   %0 = stablehlo.subtract %arg0, %arg2 : tensor<2x2xf64>
// CHECK-NEXT:   %1 = stablehlo.subtract %arg1, %arg3 : tensor<2x2xf64>
// CHECK-NEXT:   %2 = stablehlo.complex %0, %1 : tensor<2x2xcomplex<f64>>
// CHECK-NEXT:   return %2 : tensor<2x2xcomplex<f64>>
// CHECK-NEXT: }
