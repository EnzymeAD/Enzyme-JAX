// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=recognize_wrap" --transform-interpreter --enzyme-hlo-remove-transform -allow-unregistered-dialect %s | FileCheck %s

module @"reactant_loop!" {

  func.func @main(%arg29: tensor<1x24x96xf64>) -> (tensor<1x8x96xf64>) {
      %a = stablehlo.slice %arg29 [0:1, 0:8, 80:88] : (tensor<1x24x96xf64>) -> tensor<1x8x8xf64>
      %b = stablehlo.slice %arg29 [0:1, 0:8, 8:88] : (tensor<1x24x96xf64>) -> tensor<1x8x80xf64>
      %c = stablehlo.slice %arg29 [0:1, 0:8, 8:16] : (tensor<1x24x96xf64>) -> tensor<1x8x8xf64>
      %11835 = stablehlo.concatenate %a, %b, %c, dim = 2 : (tensor<1x8x8xf64>, tensor<1x8x80xf64>, tensor<1x8x8xf64>) -> tensor<1x8x96xf64>
      stablehlo.return %11835 :  tensor<1x8x96xf64>
  }
}

// CHECK-LABEL:   module @"reactant_loop!" {
// CHECK:           func.func @main(%[[VAL_0:.*]]: tensor<1x24x96xf64>) -> tensor<1x8x96xf64> {
// CHECK:             %[[VAL_1:.*]] = stablehlo.slice %[[VAL_0]] [0:1, 0:8, 8:88] : (tensor<1x24x96xf64>) -> tensor<1x8x80xf64>
// CHECK:             %[[VAL_2:.*]] = "enzymexla.wrap"(%[[VAL_1]]) <{dimension = 2 : i64, lhs = 8 : i64, rhs = 8 : i64}> : (tensor<1x8x80xf64>) -> tensor<1x8x96xf64>
// CHECK:             stablehlo.return %[[VAL_2]] : tensor<1x8x96xf64>
// CHECK:           }
// CHECK:         }

module {
  func.func @main(%in: tensor<20x24x80xf64>) -> (tensor<10x82xf64>) {
      %a = stablehlo.slice %in [11:12, 7:17, 79:80] : (tensor<20x24x80xf64>) -> tensor<1x10x1xf64>
      %b = stablehlo.slice %in [11:12, 7:17, 0:80] : (tensor<20x24x80xf64>) -> tensor<1x10x80xf64>
      %c = stablehlo.slice %in [11:12, 7:17, 0:1] : (tensor<20x24x80xf64>) -> tensor<1x10x1xf64>
      %ar = stablehlo.reshape %a : (tensor<1x10x1xf64>) -> tensor<10x1xf64>
      %br = stablehlo.reshape %b : (tensor<1x10x80xf64>) -> tensor<10x80xf64>
      %cr = stablehlo.reshape %c : (tensor<1x10x1xf64>) -> tensor<10x1xf64>
      %res = stablehlo.concatenate %ar, %br, %cr, dim = 1 : (tensor<10x1xf64>, tensor<10x80xf64>, tensor<10x1xf64>) -> tensor<10x82xf64>
      func.return %res : tensor<10x82xf64>
    }
}

// CHECK-LABEL:   module {
// CHECK:           func.func @main(%[[VAL_0:.*]]: tensor<20x24x80xf64>) -> tensor<10x82xf64> {
// CHECK:             %[[VAL_1:.*]] = stablehlo.slice %[[VAL_0]] [11:12, 7:17, 0:80] : (tensor<20x24x80xf64>) -> tensor<1x10x80xf64>
// CHECK:             %[[VAL_2:.*]] = "enzymexla.wrap"(%[[VAL_1]]) <{dimension = 2 : i64, lhs = 1 : i64, rhs = 1 : i64}> : (tensor<1x10x80xf64>) -> tensor<1x10x82xf64>
// CHECK:             %[[VAL_3:.*]] = stablehlo.reshape %[[VAL_2]] : (tensor<1x10x82xf64>) -> tensor<10x82xf64>
// CHECK:             return %[[VAL_3]] : tensor<10x82xf64>
// CHECK:           }
// CHECK:         }
