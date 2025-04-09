// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=recognize_wrap" --transform-interpreter --enzyme-hlo-remove-transform -allow-unregistered-dialect %s

module @"reactant_loop!" {

  func.func @main(%arg29: tensor<1x24x96xf64>) -> (tensor<1x8x96xf64>) {
      %a = stablehlo.slice %arg29 [0:1, 0:8, 80:88] : (tensor<1x24x96xf64>) -> tensor<1x8x8xf64>
      %b = stablehlo.slice %arg29 [0:1, 0:8, 8:88] : (tensor<1x24x96xf64>) -> tensor<1x8x80xf64>
      %c = stablehlo.slice %arg29 [0:1, 0:8, 8:16] : (tensor<1x24x96xf64>) -> tensor<1x8x8xf64>
      %11835 = stablehlo.concatenate %a, %b, %c, dim = 2 : (tensor<1x8x8xf64>, tensor<1x8x80xf64>, tensor<1x8x8xf64>) -> tensor<1x8x96xf64>
      stablehlo.return %11835 :  tensor<1x8x96xf64>
  }
}

// CHECK:  func.func @main(%arg0: tensor<1x24x96xf64>) -> tensor<1x8x96xf64> {
// CHECK-NEXT:    %0 = stablehlo.slice %arg0 [0:1, 0:8, 8:88] : (tensor<1x24x96xf64>) -> tensor<1x8x80xf64>
// CHECK-NEXT:    %1 = "enzymexla.wrap"(%0) <{dimension = 2 : i64, lhs = 8 : i64, rhs = 8 : i64}> : (tensor<1x8x80xf64>) -> tensor<1x8x96xf64>
// CHECK-NEXT:    %2 = stablehlo.concatenate %1, dim = 2 : (tensor<1x8x96xf64>) -> tensor<1x8x96xf64>
// CHECK-NEXT:    stablehlo.return %2 : tensor<1x8x96xf64>
// CHECK-NEXT:  }

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

// CHECK:    func.func @main(%arg0: tensor<20x24x80xf64>) -> tensor<10x82xf64> {
// CHECK-NEXT:      %0 = stablehlo.slice %arg0 [11:12, 7:17, 0:80] : (tensor<20x24x80xf64>) -> tensor<1x10x80xf64>
// CHECK-NEXT:      %1 = "enzymexla.wrap"(%0) <{dimension = 2 : i64, lhs = 1 : i64, rhs = 1 : i64}> : (tensor<1x10x80xf64>) -> tensor<1x10x82xf64>
// CHECK-NEXT:      %2 = stablehlo.reshape %1 : (tensor<1x10x82xf64>) -> tensor<10x82xf64>
// CHECK-NEXT:      %3 = stablehlo.concatenate %2, dim = 1 : (tensor<10x82xf64>) -> tensor<10x82xf64>
// CHECK-NEXT:      return %3 : tensor<10x82xf64>
// CHECK-NEXT:    }