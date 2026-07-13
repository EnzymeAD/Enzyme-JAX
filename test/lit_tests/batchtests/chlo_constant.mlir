// RUN: enzymexlamlir-opt --enzyme-batch %s | FileCheck %s

module {
  func.func @f(%x : tensor<3xf64>) -> tensor<3xf64> {
    %cst = chlo.constant dense<2.1> : tensor<3xf64>
    %y = stablehlo.add %x, %cst : tensor<3xf64>
    return %y : tensor<3xf64>
  }
  func.func @df(%x : tensor<10x3xf64>) -> tensor<10x3xf64> {
    %r = enzyme.batch @f(%x) { batch_shape = array<i64: 10> } : (tensor<10x3xf64>) -> (tensor<10x3xf64>)
    return %r : tensor<10x3xf64>
  }
}

// CHECK:    %[[CONST:.+]] = chlo.constant dense<2.100000e+00> : tensor<10x3xf64>
// CHECK-NEXT:    stablehlo.add %{{.+}}, %[[CONST]] : tensor<10x3xf64>
