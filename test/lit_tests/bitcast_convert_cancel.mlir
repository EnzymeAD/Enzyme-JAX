// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt | FileCheck %s

module {
  func.func @main(%arg0: tensor<369664x8xi8>) -> tensor<369664x8xi8> {
    %0 = stablehlo.bitcast_convert %arg0 : (tensor<369664x8xi8>) -> tensor<369664xf64>
    %1 = stablehlo.bitcast_convert %0 : (tensor<369664xf64>) -> tensor<369664x8xi8>
    return %1 : tensor<369664x8xi8>
  }
}
// CHECK-LABEL: func.func @main
// CHECK-SAME: (%[[ARG0:.*]]: tensor<369664x8xi8>)
// CHECK-NOT: stablehlo.bitcast_convert
// CHECK: return %[[ARG0]]
