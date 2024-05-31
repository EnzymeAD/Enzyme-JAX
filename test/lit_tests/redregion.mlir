// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(enzyme-hlo-opt)" %s | FileCheck %s 

module {

  func.func @main(%c_131 : tensor<1x1xi1>, %c_132 : tensor<i1>, %c_138 : tensor<i1>) -> tensor<1xi1> {
    %38063 = stablehlo.reduce(%c_131 init: %c_132) across dimensions = [1] : (tensor<1x1xi1>, tensor<i1>) -> tensor<1xi1>
     reducer(%arg292: tensor<i1>, %arg293: tensor<i1>)  {
      stablehlo.return %c_138 : tensor<i1>
    }
    return %38063 : tensor<1xi1>
  }
}

// CHECK:  func.func @main(%arg0: tensor<1x1xi1>, %arg1: tensor<i1>, %arg2: tensor<i1>) -> tensor<1xi1> {
// CHECK-NEXT:    %0 = stablehlo.reshape %arg2 : (tensor<i1>) -> tensor<1xi1>
// CHECK-NEXT:    return %0 : tensor<1xi1>
// CHECK-NEXT:  }
