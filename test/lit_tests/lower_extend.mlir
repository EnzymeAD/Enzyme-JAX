// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=lower_extend" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

module {
  func.func @main(%arg0: tensor<1x10x80xf64>) -> tensor<1x10x87xf64> {
    %0 = stablehlo.slice %arg0 [0:0, 0:0, 0:0] : (tensor<1x10x80xf64>) -> tensor<0x0x0xf64>
    %1 = "enzymexla.extend"(%arg0) <{dimension = 2 : i64, lhs = 2 : i64, rhs = 5 : i64}> : (tensor<1x10x80xf64>) -> tensor<1x10x87xf64>
    return %1 : tensor<1x10x87xf64>
  }
}

// CHECK-LABEL:   func.func @main(
// CHECK-SAME:                    %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: tensor<1x10x80xf64>) -> tensor<1x10x87xf64> {
// CHECK:           %[[VAL_1:.*]] = stablehlo.slice %[[VAL_0]] [0:1, 0:10, 0:2] : (tensor<1x10x80xf64>) -> tensor<1x10x2xf64>
// CHECK:           %[[VAL_2:.*]] = stablehlo.slice %[[VAL_0]] [0:1, 0:10, 75:80] : (tensor<1x10x80xf64>) -> tensor<1x10x5xf64>
// CHECK:           %[[VAL_3:.*]] = stablehlo.concatenate %[[VAL_1]], %[[VAL_0]], %[[VAL_2]], dim = 2 : (tensor<1x10x2xf64>, tensor<1x10x80xf64>, tensor<1x10x5xf64>) -> tensor<1x10x87xf64>
// CHECK:           return %[[VAL_3]] : tensor<1x10x87xf64>
// CHECK:         }

module {
  func.func @main1(%arg0: tensor<1x10x80xf64>) -> tensor<1x10x82xf64> {
    %0 = stablehlo.slice %arg0 [0:0, 0:0, 0:0] : (tensor<1x10x80xf64>) -> tensor<0x0x0xf64>
    %1 = "enzymexla.extend"(%arg0) <{dimension = 2 : i64, lhs = 2 : i64, rhs = 0 : i64}> : (tensor<1x10x80xf64>) -> tensor<1x10x82xf64>
    return %1 : tensor<1x10x82xf64>
  }
}

// CHECK: func.func @main1(%arg0: tensor<1x10x80xf64>) -> tensor<1x10x82xf64> {
// CHECK-NEXT:       %0 = stablehlo.slice %arg0 [0:1, 0:10, 0:2] : (tensor<1x10x80xf64>) -> tensor<1x10x2xf64>
// CHECK-NEXT:       %1 = stablehlo.concatenate %0, %arg0, dim = 2 : (tensor<1x10x2xf64>, tensor<1x10x80xf64>) -> tensor<1x10x82xf64>
// CHECK-NEXT:       return %1 : tensor<1x10x82xf64>
// CHECK-NEXT: }

module {
  func.func @main2(%arg0: tensor<1x10x80xf64>) -> tensor<1x10x80xf64> {
    %0 = stablehlo.slice %arg0 [0:0, 0:0, 0:0] : (tensor<1x10x80xf64>) -> tensor<0x0x0xf64>
    %1 = "enzymexla.extend"(%arg0) <{dimension = 2 : i64, lhs = 0 : i64, rhs = 0 : i64}> : (tensor<1x10x80xf64>) -> tensor<1x10x80xf64>
    return %1 : tensor<1x10x80xf64>
  }
}

// CHECK:   func.func @main2(
// CHECK-SAME:                    %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: tensor<1x10x80xf64>) -> tensor<1x10x80xf64> {
// CHECK:           return %[[VAL_0]] : tensor<1x10x80xf64>
// CHECK:         }
