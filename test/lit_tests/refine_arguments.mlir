// RUN: enzymexlamlir-opt %s --enzyme-refine-arguments="refined-types=tensor<8xi8>" | FileCheck %s

module {
  func.func @main(%arg0: tensor<?xf64>) -> tensor<?xf64> {
    return %arg0 : tensor<?xf64>
  }
}

// CHECK-LABEL: func.func @main(%arg0: tensor<8xi8>) -> tensor<8xi8> {
// CHECK-NEXT:    [[RESHAPE:%.+]] = "stablehlo.reshape"(%arg0) : (tensor<8xi8>) -> tensor<1x8xi8>
// CHECK-NEXT:    [[BC:%.+]] = "stablehlo.bitcast_convert"([[RESHAPE]]) : (tensor<1x8xi8>) -> tensor<1xf64>
// CHECK-NEXT:    [[FINAL:%.+]] = tensor.cast [[BC]] : tensor<1xf64> to tensor<?xf64>
// CHECK-NEXT:    [[R_STATIC:%.+]] = "stablehlo.reshape"([[FINAL]]) : (tensor<?xf64>) -> tensor<1xf64>
// CHECK-NEXT:    [[R_BC:%.+]] = "stablehlo.bitcast_convert"([[R_STATIC]]) : (tensor<1xf64>) -> tensor<1x8xi8>
// CHECK-NEXT:    [[R_FINAL:%.+]] = "stablehlo.reshape"([[R_BC]]) : (tensor<1x8xi8>) -> tensor<8xi8>
// CHECK-NEXT:    return [[R_FINAL]] : tensor<8xi8>
