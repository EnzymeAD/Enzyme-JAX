// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

module @"reactant_loop!" {
  func.func @main(%208 : tensor<4x1520x3056xf64>) -> tensor<6x1520x3056xf64> {

      %1220 = "enzymexla.extend"(%208) <{dimension = 0 : i64, lhs = 1 : i64, rhs = 0 : i64}> : (tensor<4x1520x3056xf64>) -> tensor<5x1520x3056xf64>

      %1215 = stablehlo.slice %208 [3:4, 0:1520, 0:3056] : (tensor<4x1520x3056xf64>) -> tensor<1x1520x3056xf64>

      %1238 = stablehlo.concatenate %1220, %1215, dim = 0 : (tensor<5x1520x3056xf64>, tensor<1x1520x3056xf64>) -> tensor<6x1520x3056xf64>

    stablehlo.return %1238 : tensor<6x1520x3056xf64>
  }
}

// CHECK:  func.func @main(%arg0: tensor<4x1520x3056xf64>) -> tensor<6x1520x3056xf64> {
// CHECK-NEXT:    %0 = "enzymexla.extend"(%arg0) <{dimension = 0 : i64, lhs = 1 : i64, rhs = 1 : i64}> : (tensor<4x1520x3056xf64>) -> tensor<6x1520x3056xf64>
// CHECK-NEXT:    stablehlo.return %0 : tensor<6x1520x3056xf64>
// CHECK-NEXT:  }


func.func @main2(%arg0: tensor<4x1520x3056xf64>) -> tensor<8x1520x3056xf64> {
    %0 = "enzymexla.extend"(%arg0) <{dimension = 0 : i64, lhs = 1 : i64, rhs = 2 : i64}> : (tensor<4x1520x3056xf64>) -> tensor<7x1520x3056xf64>
    %1 = stablehlo.slice %arg0 [0:1, 0:1520, 0:3056] : (tensor<4x1520x3056xf64>) -> tensor<1x1520x3056xf64>
    %2 = stablehlo.concatenate %1, %0, dim = 0 : (tensor<1x1520x3056xf64>, tensor<7x1520x3056xf64>) -> tensor<8x1520x3056xf64>
    return %2 : tensor<8x1520x3056xf64>
}

// CHECK:  func.func @main2(%arg0: tensor<4x1520x3056xf64>) -> tensor<8x1520x3056xf64> {
// CHECK-NEXT:    %0 = "enzymexla.extend"(%arg0) <{dimension = 0 : i64, lhs = 2 : i64, rhs = 2 : i64}> : (tensor<4x1520x3056xf64>) -> tensor<8x1520x3056xf64>
// CHECK-NEXT:    return %0 : tensor<8x1520x3056xf64>
// CHECK-NEXT:  }

func.func @main3(%arg0: tensor<4x1520x3056xf64>) -> tensor<9x1520x3056xf64> {
    %0 = "enzymexla.extend"(%arg0) <{dimension = 0 : i64, lhs = 1 : i64, rhs = 2 : i64}> : (tensor<4x1520x3056xf64>) -> tensor<7x1520x3056xf64>
    %1 = stablehlo.slice %arg0 [0:1, 0:1520, 0:3056] : (tensor<4x1520x3056xf64>) -> tensor<1x1520x3056xf64>
    %2 = stablehlo.slice %arg0 [3:4, 0:1520, 0:3056] : (tensor<4x1520x3056xf64>) -> tensor<1x1520x3056xf64>
    %3 = stablehlo.concatenate %1, %0, %2, dim = 0 : (tensor<1x1520x3056xf64>, tensor<7x1520x3056xf64>, tensor<1x1520x3056xf64>) -> tensor<9x1520x3056xf64>
    return %3 : tensor<9x1520x3056xf64>
}

// CHECK:  func.func @main3(%arg0: tensor<4x1520x3056xf64>) -> tensor<9x1520x3056xf64> {
// CHECK-NEXT:    %0 = "enzymexla.extend"(%arg0) <{dimension = 0 : i64, lhs = 2 : i64, rhs = 3 : i64}> : (tensor<4x1520x3056xf64>) -> tensor<9x1520x3056xf64>
// CHECK-NEXT:    return %0 : tensor<9x1520x3056xf64>
// CHECK-NEXT:  }
