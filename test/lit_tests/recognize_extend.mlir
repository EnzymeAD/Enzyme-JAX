// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=recognize_extend" --transform-interpreter --enzyme-hlo-remove-transform %s

module {
  func.func @main(%6595: tensor<20x24x80xf64>) -> (tensor<1x10x82xf64>) {
      %A = stablehlo.slice %6595 [11:12, 7:17, 0:1] : (tensor<20x24x80xf64>) -> tensor<1x10x1xf64>
      %B = stablehlo.slice %6595 [11:12, 7:17, 0:80] : (tensor<20x24x80xf64>) -> tensor<1x10x80xf64>
      %C = stablehlo.slice %6595 [11:12, 7:17, 79:80] : (tensor<20x24x80xf64>) -> tensor<1x10x1xf64>
      %RES = stablehlo.concatenate %A, %B, %C, dim = 2 : (tensor<1x10x1xf64>, tensor<1x10x80xf64>, tensor<1x10x1xf64>) -> tensor<1x10x82xf64>
      func.return %RES :  tensor<1x10x82xf64>
  }
}

// CHECK-LABEL:   module {
// CHECK:           func.func @main(%[[VAL_0:.*]]: tensor<20x24x80xf64>) -> tensor<1x10x82xf64> {
// CHECK:             %[[VAL_1:.*]] = stablehlo.slice %[[VAL_0]] [11:12, 7:17, 0:80] : (tensor<20x24x80xf64>) -> tensor<1x10x80xf64>
// CHECK:             %[[VAL_2:.*]] = "enzymexla.extend"(%[[VAL_1]]) <{dimension = 2 : i64, lhs = 1 : i64, rhs = 1 : i64}> : (tensor<1x10x80xf64>) -> tensor<1x10x82xf64>
// CHECK:             return %[[VAL_2]] : tensor<1x10x82xf64>
// CHECK:           }
// CHECK:         }

module {
  func.func @main(%in1: tensor<1x8x80xf64>, %in2: tensor<1x24x96xf64>) -> (tensor<1x24x8xf64>) {

      %a = stablehlo.slice %in2 [0:1, 0:7, 80:88] : (tensor<1x24x96xf64>) -> tensor<1x7x8xf64>

      %b = stablehlo.slice %in1 [0:1, 0:1, 72:80] : (tensor<1x8x80xf64>) -> tensor<1x1x8xf64>
      %c = stablehlo.slice %in1 [0:1, 0:8, 72:80] : (tensor<1x8x80xf64>) -> tensor<1x8x8xf64>
      %d = stablehlo.slice %in1 [0:1, 7:8, 72:80] : (tensor<1x8x80xf64>) -> tensor<1x1x8xf64>

      %e = stablehlo.slice %in2 [0:1, 17:24, 80:88] : (tensor<1x24x96xf64>) -> tensor<1x7x8xf64>

      %res = stablehlo.concatenate %a, %b, %c, %d, %e, dim = 1 : (tensor<1x7x8xf64>, tensor<1x1x8xf64>, tensor<1x8x8xf64>, tensor<1x1x8xf64>, tensor<1x7x8xf64>) -> tensor<1x24x8xf64>

      func.return %res : tensor<1x24x8xf64>
    }
}
// CHECK-LABEL:   module {
// CHECK:           func.func @main(%[[VAL_0:.*]]: tensor<1x8x80xf64>, %[[VAL_1:.*]]: tensor<1x24x96xf64>) -> tensor<1x24x8xf64> {
// CHECK:             %[[VAL_2:.*]] = stablehlo.slice %[[VAL_1]] [0:1, 0:7, 80:88] : (tensor<1x24x96xf64>) -> tensor<1x7x8xf64>
// CHECK:             %[[VAL_3:.*]] = stablehlo.slice %[[VAL_0]] [0:1, 0:8, 72:80] : (tensor<1x8x80xf64>) -> tensor<1x8x8xf64>
// CHECK:             %[[VAL_4:.*]] = stablehlo.slice %[[VAL_1]] [0:1, 17:24, 80:88] : (tensor<1x24x96xf64>) -> tensor<1x7x8xf64>
// CHECK:             %[[VAL_5:.*]] = "enzymexla.extend"(%[[VAL_3]]) <{dimension = 1 : i64, lhs = 1 : i64, rhs = 1 : i64}> : (tensor<1x8x8xf64>) -> tensor<1x10x8xf64>
// CHECK:             %[[VAL_6:.*]] = stablehlo.concatenate %[[VAL_2]], %[[VAL_5]], %[[VAL_4]], dim = 1 : (tensor<1x7x8xf64>, tensor<1x10x8xf64>, tensor<1x7x8xf64>) -> tensor<1x24x8xf64>
// CHECK:             return %[[VAL_6]] : tensor<1x24x8xf64>
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
      %res = stablehlo.concatenate %cr, %br, %ar, dim = 1 : (tensor<10x1xf64>, tensor<10x80xf64>, tensor<10x1xf64>) -> tensor<10x82xf64>
      // %res = stablehlo.concatenate %ar, %br, %cr, dim = 1 : (tensor<10x1xf64>, tensor<10x80xf64>, tensor<10x1xf64>) -> tensor<10x82xf64>
      func.return %res : tensor<10x82xf64>
    }
}

// CHECK-LABEL:   module {
// CHECK:           func.func @main(%[[VAL_0:.*]]: tensor<20x24x80xf64>) -> tensor<10x82xf64> {
// CHECK:             %[[VAL_1:.*]] = stablehlo.slice %[[VAL_0]] [11:12, 7:17, 0:80] : (tensor<20x24x80xf64>) -> tensor<1x10x80xf64>
// CHECK:             %[[VAL_2:.*]] = "enzymexla.extend"(%[[VAL_1]]) <{dimension = 2 : i64, lhs = 1 : i64, rhs = 1 : i64}> : (tensor<1x10x80xf64>) -> tensor<1x10x82xf64>
// CHECK:             %[[VAL_3:.*]] = stablehlo.reshape %[[VAL_2]] : (tensor<1x10x82xf64>) -> tensor<10x82xf64>
// CHECK:             return %[[VAL_3]] : tensor<10x82xf64>
// CHECK:           }
// CHECK:         }

