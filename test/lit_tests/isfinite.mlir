// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(enzyme-hlo-opt{all_finite=true})" %s | FileCheck %s --check-prefix=REMOVED
// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(enzyme-hlo-opt{all_finite=false})" %s | FileCheck %s --check-prefix=SAME
// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(enzyme-hlo-opt)" %s | FileCheck %s --check-prefix=SAME
// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(enzyme-hlo-opt{no_nan=true})" %s | FileCheck %s --check-prefix=SAME
// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(enzyme-hlo-opt{no_nan=false})" %s | FileCheck %s --check-prefix=SAME

module {

  func.func @main(%a : tensor<2x2xf32>) -> tensor<2x2xi1> {
    %r = stablehlo.is_finite %a : (tensor<2x2xf32>) -> tensor<2x2xi1>
    return %r : tensor<2x2xi1>
  }
}

// REMOVED:  func.func @main(%arg0: tensor<2x2xf32>) -> tensor<2x2xi1> {
// REMOVED-NEXT:    %0 = stablehlo.constant dense<true> : tensor<2x2xi1>
// REMOVED-NEXT:    return %0 : tensor<2x2xi1>
// REMOVED-NEXT:  }

// SAME:  func.func @main(%arg0: tensor<2x2xf32>) -> tensor<2x2xi1> {
// SAME-NEXT:    %0 = stablehlo.is_finite %arg0 : (tensor<2x2xf32>) -> tensor<2x2xi1>
// SAME-NEXT:    return %0 : tensor<2x2xi1>
// SAME-NEXT:  }
