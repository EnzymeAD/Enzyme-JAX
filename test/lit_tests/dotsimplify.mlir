// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s --check-prefixes=CHECK,NAN
// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(enzyme-hlo-opt{no_nan=true})" %s | FileCheck %s --check-prefixes=CHECK,NONAN
// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=dot_general_simplify(0)" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s --check-prefixes=CHECK,NANTD
// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=dot_general_simplify(1)" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s --check-prefixes=CHECK,NONAN

func.func @transpose(%928 : tensor<1x4x1x20x160xbf16>) -> tensor<1x1x48x160xbf16> {
  %cst_141 = stablehlo.constant dense<0.000000e+00> : tensor<1x1x48x4x20xbf16>
  %1097 = stablehlo.dot_general %cst_141, %928, batching_dims = [0, 1] x [0, 2], contracting_dims = [3, 4] x [1, 3], precision = [DEFAULT, DEFAULT] : (tensor<1x1x48x4x20xbf16>, tensor<1x4x1x20x160xbf16>) -> tensor<1x1x48x160xbf16>
  return %1097 : tensor<1x1x48x160xbf16>
}

// CHECK:  func.func @transpose(%arg0: tensor<1x4x1x20x160xbf16>) -> tensor<1x1x48x160xbf16> {
// NAN-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<1x1x160xbf16>
// NAN-NEXT:     %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
// NAN-NEXT:     %0 = stablehlo.reduce(%arg0 init: %cst_0) applies stablehlo.add across dimensions = [1, 3] {enzymexla.finite = [#enzymexla.guaranteed<NOTGUARANTEED>]} : (tensor<1x4x1x20x160xbf16>, tensor<bf16>) -> tensor<1x1x160xbf16>
// NAN-NEXT:     %1 = stablehlo.multiply %0, %cst {enzymexla.no_nan = [#enzymexla.guaranteed<NOTGUARANTEED>]} : tensor<1x1x160xbf16>
// NAN-NEXT:     %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1, 3] : (tensor<1x1x160xbf16>) -> tensor<1x1x48x160xbf16>
// NAN-NEXT:     return %2 : tensor<1x1x48x160xbf16>
// NANTD-NEXT:   %cst = stablehlo.constant dense<0.000000e+00> : tensor<1x1x48x160xbf16>
// NANTD-NEXT:   %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
// NANTD-NEXT:   %0 = stablehlo.reduce(%arg0 init: %cst_0) applies stablehlo.add across dimensions = [1, 3] : (tensor<1x4x1x20x160xbf16>, tensor<bf16>) -> tensor<1x1x160xbf16>
// NANTD-NEXT:   %1 = stablehlo.broadcast_in_dim %0, dims = [0, 1, 3] : (tensor<1x1x160xbf16>) -> tensor<1x1x48x160xbf16>
// NANTD-NEXT:   %2 = stablehlo.multiply %1, %cst : tensor<1x1x48x160xbf16>
// NANTD-NEXT:   return %2 : tensor<1x1x48x160xbf16>
// NONAN-NEXT:   %[[i0:.+]] = stablehlo.constant dense<0.000000e+00> : tensor<1x1x48x160xbf16>
// NONAN-NEXT:   return %[[i0]] : tensor<1x1x48x160xbf16>
// CHECK-NEXT:  }

func.func @complex(%arg0 : tensor<2x3xcomplex<f64>>) -> tensor<4x3xcomplex<f64>> {
  %cst = stablehlo.constant dense<(0.000000e+00,0.000000e+00)> : tensor<4x2xcomplex<f64>>
  %0 = stablehlo.dot_general %cst, %arg0, contracting_dims = [1] x [0] : (tensor<4x2xcomplex<f64>>, tensor<2x3xcomplex<f64>>) -> tensor<4x3xcomplex<f64>>
  return %0 : tensor<4x3xcomplex<f64>>
}

// CHECK:  func.func @complex(%arg0: tensor<2x3xcomplex<f64>>) -> tensor<4x3xcomplex<f64>> {
// NAN-NEXT:     %cst = stablehlo.constant dense<(0.000000e+00,0.000000e+00)> : tensor<3xcomplex<f64>>
// NAN-NEXT:     %cst_0 = stablehlo.constant dense<(0.000000e+00,0.000000e+00)> : tensor<complex<f64>>
// NAN-NEXT:     %0 = stablehlo.reduce(%arg0 init: %cst_0) applies stablehlo.add across dimensions = [0] {enzymexla.complex_is_purely_real = [#enzymexla.guaranteed<NOTGUARANTEED>], enzymexla.finite = [#enzymexla.guaranteed<NOTGUARANTEED>]} : (tensor<2x3xcomplex<f64>>, tensor<complex<f64>>) -> tensor<3xcomplex<f64>>
// NAN-NEXT:     %1 = stablehlo.multiply %0, %cst {enzymexla.no_nan = [#enzymexla.guaranteed<NOTGUARANTEED>]} : tensor<3xcomplex<f64>>
// NAN-NEXT:     %2 = stablehlo.broadcast_in_dim %1, dims = [1] : (tensor<3xcomplex<f64>>) -> tensor<4x3xcomplex<f64>>
// NAN-NEXT:     return %2 : tensor<4x3xcomplex<f64>>
// NANTD-NEXT:   %cst = stablehlo.constant dense<(0.000000e+00,0.000000e+00)> : tensor<4x3xcomplex<f64>>
// NANTD-NEXT:   %cst_0 = stablehlo.constant dense<(0.000000e+00,0.000000e+00)> : tensor<complex<f64>>
// NANTD-NEXT:   %0 = stablehlo.reduce(%arg0 init: %cst_0) applies stablehlo.add across dimensions = [0] : (tensor<2x3xcomplex<f64>>, tensor<complex<f64>>) -> tensor<3xcomplex<f64>>
// NANTD-NEXT:   %1 = stablehlo.broadcast_in_dim %0, dims = [1] : (tensor<3xcomplex<f64>>) -> tensor<4x3xcomplex<f64>>
// NANTD-NEXT:   %2 = stablehlo.multiply %1, %cst : tensor<4x3xcomplex<f64>>
// NANTD-NEXT:   return %2 : tensor<4x3xcomplex<f64>>
// NONAN-NEXT:   %[[i0:.+]] = stablehlo.constant dense<(0.000000e+00,0.000000e+00)> : tensor<4x3xcomplex<f64>>
// NONAN-NEXT:   return %[[i0]] : tensor<4x3xcomplex<f64>>
// CHECK-NEXT:  }
