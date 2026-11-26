// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt | FileCheck %s --check-prefixes=CHECK,NAN
// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(enzyme-hlo-opt{no_nan=true})" | FileCheck %s --check-prefixes=CHECK,NONAN
// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=square_abs_simplify(0)" --transform-interpreter --enzyme-hlo-remove-transform | FileCheck %s --check-prefixes=CHECK,NAN
// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=square_abs_simplify(1)" --transform-interpreter --enzyme-hlo-remove-transform | FileCheck %s --check-prefixes=CHECK,NONAN

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
// NAN-NEXT:     %0 = stablehlo.abs %arg0 : (tensor<2x2xcomplex<f32>>) -> tensor<2x2xf32>
// NAN-NEXT:     %1 = stablehlo.multiply %0, %0 : tensor<2x2xf32>
// NAN-NEXT:     return %1 : tensor<2x2xf32>
// NONAN-NEXT:   %0 = stablehlo.real %arg0 : (tensor<2x2xcomplex<f32>>) -> tensor<2x2xf32>
// NONAN-NEXT:   %1 = stablehlo.imag %arg0 : (tensor<2x2xcomplex<f32>>) -> tensor<2x2xf32>
// NONAN-NEXT:   %2 = stablehlo.multiply %0, %0 : tensor<2x2xf32>
// NONAN-NEXT:   %3 = stablehlo.multiply %1, %1 : tensor<2x2xf32>
// NONAN-NEXT:   %4 = stablehlo.add %2, %3 : tensor<2x2xf32>
// NONAN-NEXT:   return %4 : tensor<2x2xf32>
// CHECK-NEXT: }

// doesn't apply here
func.func @squareabscomplex2(%arg0: tensor<2x2xcomplex<f32>>) -> (tensor<2x2xf32>, tensor<2x2xf32>) {
    // CHECK: %0 = stablehlo.abs %arg0 : (tensor<2x2xcomplex<f32>>) -> tensor<2x2xf32>
    %0 = stablehlo.abs %arg0 : (tensor<2x2xcomplex<f32>>) -> tensor<2x2xf32>
    %1 = stablehlo.multiply %0, %0 : tensor<2x2xf32>
    return %1, %0 : tensor<2x2xf32>, tensor<2x2xf32>
}
