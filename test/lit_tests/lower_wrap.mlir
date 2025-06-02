// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=lower_wrap" --transform-interpreter --enzyme-hlo-remove-transform -allow-unregistered-dialect --enzyme-hlo-opt %s | FileCheck %s

module @"reactant_loop!" {
  func.func @main(%arg0: tensor<1x24x96xf64>) -> tensor<1x8x96xf64> {
    %0 = stablehlo.slice %arg0 [0:1, 0:8, 8:88] : (tensor<1x24x96xf64>) -> tensor<1x8x80xf64>
    %1 = "enzymexla.wrap"(%0) <{dimension = 2 : i64, lhs = 8 : i64, rhs = 8 : i64}> : (tensor<1x8x80xf64>) -> tensor<1x8x96xf64>
    %2 = stablehlo.concatenate %1, dim = 2 : (tensor<1x8x96xf64>) -> tensor<1x8x96xf64>
    stablehlo.return %2 : tensor<1x8x96xf64>
  }
}

// CHECK:  func.func @main(%arg0: tensor<1x24x96xf64>) -> tensor<1x8x96xf64> {
// CHECK-NEXT:    %0 = stablehlo.slice %arg0 [0:1, 0:8, 8:88] : (tensor<1x24x96xf64>) -> tensor<1x8x80xf64>
// CHECK-NEXT:    %1 = stablehlo.slice %arg0 [0:1, 0:8, 80:88] : (tensor<1x24x96xf64>) -> tensor<1x8x8xf64>
// CHECK-NEXT:    %2 = stablehlo.slice %arg0 [0:1, 0:8, 8:16] : (tensor<1x24x96xf64>) -> tensor<1x8x8xf64>
// CHECK-NEXT:    %3 = stablehlo.concatenate %1, %0, %2, dim = 2 : (tensor<1x8x8xf64>, tensor<1x8x80xf64>, tensor<1x8x8xf64>) -> tensor<1x8x96xf64>
// CHECK-NEXT:    stablehlo.return %3 : tensor<1x8x96xf64>
// CHECK-NEXT:  }

func.func @main2(%arg0: tensor<1x24x96xf64>) -> tensor<1x8x88xf64> {
    %0 = stablehlo.slice %arg0 [0:1, 0:8, 8:88] : (tensor<1x24x96xf64>) -> tensor<1x8x80xf64>
    %1 = "enzymexla.wrap"(%0) <{dimension = 2 : i64, lhs = 0 : i64, rhs = 8 : i64}> : (tensor<1x8x80xf64>) -> tensor<1x8x88xf64>
    stablehlo.return %1 : tensor<1x8x88xf64>
}

// CHECK:  func.func @main2(%arg0: tensor<1x24x96xf64>) -> tensor<1x8x88xf64> {
// CHECK-NEXT:    %0 = stablehlo.slice %arg0 [0:1, 0:8, 8:88] : (tensor<1x24x96xf64>) -> tensor<1x8x80xf64>
// CHECK-NEXT:    %1 = stablehlo.slice %arg0 [0:1, 0:8, 8:16] : (tensor<1x24x96xf64>) -> tensor<1x8x8xf64>
// CHECK-NEXT:    %2 = stablehlo.concatenate %0, %1, dim = 2 : (tensor<1x8x80xf64>, tensor<1x8x8xf64>) -> tensor<1x8x88xf64>
// CHECK-NEXT:    stablehlo.return %2 : tensor<1x8x88xf64>
// CHECK-NEXT:  }

func.func @main3(%arg0: tensor<1x24x96xf64>) -> tensor<1x8x80xf64> {
    %0 = stablehlo.slice %arg0 [0:1, 0:8, 8:88] : (tensor<1x24x96xf64>) -> tensor<1x8x80xf64>
    %1 = "enzymexla.wrap"(%0) <{dimension = 2 : i64, lhs = 0 : i64, rhs = 0 : i64}> : (tensor<1x8x80xf64>) -> tensor<1x8x80xf64>
    stablehlo.return %1 : tensor<1x8x80xf64>
}

// CHECK:  func.func @main3(%arg0: tensor<1x24x96xf64>) -> tensor<1x8x80xf64> {
// CHECK-NEXT:    %0 = stablehlo.slice %arg0 [0:1, 0:8, 8:88] : (tensor<1x24x96xf64>) -> tensor<1x8x80xf64>
// CHECK-NEXT:    stablehlo.return %0 : tensor<1x8x80xf64>
// CHECK-NEXT:  }
