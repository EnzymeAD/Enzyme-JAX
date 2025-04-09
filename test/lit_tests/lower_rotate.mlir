// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=recognize_rotate" --transform-interpreter --enzyme-hlo-remove-transform -allow-unregistered-dialect %s --enzyme-hlo-opt

module @"reactant_loop!" {
  func.func @main(%arg0: tensor<20x24x96xf64>) -> tensor<4x8x80xf64> {
    %0 = stablehlo.slice %arg0 [8:12, 8:16, 8:88] : (tensor<20x24x96xf64>) -> tensor<4x8x80xf64>
    %1 = "enzymexla.rotate"(%0) <{amount = 2 : si32, dimension = 2 : si32}> : (tensor<4x8x80xf64>) -> tensor<4x8x80xf64>
    %2 = stablehlo.concatenate %1, dim = 2 : (tensor<4x8x80xf64>) -> tensor<4x8x80xf64>
    stablehlo.return %2 : tensor<4x8x80xf64>
  }
}

// CHECK:  func.func @main(%arg0: tensor<20x24x96xf64>) -> tensor<4x8x80xf64> {
// CHECK-NEXT:    %0 = stablehlo.slice %arg0 [8:12, 8:16, 10:88] : (tensor<20x24x96xf64>) -> tensor<4x8x78xf64>
// CHECK-NEXT:    %1 = stablehlo.slice %arg0 [8:12, 8:16, 8:10] : (tensor<20x24x96xf64>) -> tensor<4x8x2xf64>
// CHECK-NEXT:    %2 = stablehlo.concatenate %0, %1, dim = 2 : (tensor<4x8x78xf64>, tensor<4x8x2xf64>) -> tensor<4x8x80xf64>
// CHECK-NEXT:    stablehlo.return %2 : tensor<4x8x80xf64>
// CHECK-NEXT:  }