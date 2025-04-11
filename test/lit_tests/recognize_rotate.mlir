// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=recognize_rotate" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

module @"reactant_loop!" {
  func.func @main(%arg23: tensor<20x24x96xf64>) -> (tensor<4x8x80xf64>) {
      %11781 = stablehlo.slice %arg23 [8:12, 8:16, 10:88] : (tensor<20x24x96xf64>) -> tensor<4x8x78xf64>
      %11782 = stablehlo.slice %arg23 [8:12, 8:16, 8:10] : (tensor<20x24x96xf64>) -> tensor<4x8x2xf64>
      %11783 = stablehlo.concatenate %11781, %11782, dim = 2 : (tensor<4x8x78xf64>, tensor<4x8x2xf64>) -> tensor<4x8x80xf64>
      stablehlo.return %11783 :  tensor<4x8x80xf64>
  }

// CHECK-LABEL:   func.func @main(
// CHECK-SAME:                    %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: tensor<20x24x96xf64>) -> tensor<4x8x80xf64> {
// CHECK:           %[[VAL_1:.*]] = stablehlo.slice %[[VAL_0]] [8:12, 8:16, 8:88] : (tensor<20x24x96xf64>) -> tensor<4x8x80xf64>
// CHECK:           %[[VAL_2:.*]] = "enzymexla.rotate"(%[[VAL_1]]) <{amount = 2 : si32, dimension = 2 : si32}> : (tensor<4x8x80xf64>) -> tensor<4x8x80xf64>
// CHECK:           stablehlo.return %[[VAL_2]] : tensor<4x8x80xf64>
// CHECK:         }

  
  func.func @rotate(%arg0: tensor<12x1024xi64> ) -> (tensor<12x1024xi64> ) {
    %0 = stablehlo.slice %arg0 [0:12, 100:1024]  : (tensor<12x1024xi64>) -> tensor<12x924xi64>
    %1 = stablehlo.slice %arg0 [0:12, 0:100]  : (tensor<12x1024xi64>) -> tensor<12x100xi64>
    %2 = stablehlo.concatenate %0, %1, dim = 1  : (tensor<12x924xi64>, tensor<12x100xi64>) -> tensor<12x1024xi64>
    return %2 : tensor<12x1024xi64>
  }
}

// CHECK-LABEL:   func.func @rotate(
// CHECK-SAME:                      %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: tensor<12x1024xi64>) -> tensor<12x1024xi64> {
// CHECK:           %[[VAL_1:.*]] = "enzymexla.rotate"(%[[VAL_0]]) <{amount = 100 : si32, dimension = 1 : si32}> : (tensor<12x1024xi64>) -> tensor<12x1024xi64>
// CHECK:           return %[[VAL_1]] : tensor<12x1024xi64>
// CHECK:         }

