// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(enzyme-hlo-opt{all_finite=true})" %s | FileCheck %s --check-prefix=REMOVED
// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(enzyme-hlo-opt{all_finite=false})" %s | FileCheck %s --check-prefix=SAME
// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(enzyme-hlo-opt)" %s | FileCheck %s --check-prefix=SAME
// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(enzyme-hlo-opt{no_nan=true})" %s | FileCheck %s --check-prefix=REMOVED
// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(enzyme-hlo-opt{no_nan=false})" %s | FileCheck %s --check-prefix=SAME

module {

  func.func @main(%a : tensor<2x2xf32>) -> (tensor<2x2xi1>, tensor<2x2xi1>) {
    %eq = stablehlo.compare  EQ, %a, %a,  FLOAT : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xi1>
    %ne = stablehlo.compare  NE, %a, %a,  FLOAT : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xi1>
    return %eq, %ne : tensor<2x2xi1>, tensor<2x2xi1>
  }
}

// REMOVED:  func.func @main(%arg0: tensor<2x2xf32>) -> (tensor<2x2xi1>, tensor<2x2xi1>) {
// REMOVED-NEXT:    %[[i0:.+]] = stablehlo.constant dense<true> : tensor<2x2xi1>
// REMOVED-NEXT:    %[[i1:.+]] = stablehlo.constant dense<false> : tensor<2x2xi1>
// REMOVED-NEXT:    return %[[i0]], %[[i1]] : tensor<2x2xi1>, tensor<2x2xi1>
// REMOVED-NEXT:  }

// SAME:  func.func @main(%arg0: tensor<2x2xf32>) -> (tensor<2x2xi1>, tensor<2x2xi1>) {
// SAME-NEXT:    %[[i0:.+]] = stablehlo.compare  EQ, %arg0, %arg0,  FLOAT : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xi1>
// SAME-NEXT:    %[[i1:.+]] = stablehlo.not  %[[i0]] : tensor<2x2xi1>
// SAME-NEXT:    return %[[i0]], %[[i1]] : tensor<2x2xi1>, tensor<2x2xi1>
// SAME-NEXT:  }
