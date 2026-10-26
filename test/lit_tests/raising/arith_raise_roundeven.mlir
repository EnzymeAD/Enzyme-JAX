// RUN: enzymexlamlir-opt --arith-raise %s | FileCheck %s

module {
  // CHECK-LABEL: @func_roundeven
  // CHECK: stablehlo.round_nearest_even %arg0 : tensor<3xf64>
  // CHECK-NOT: math.roundeven
  func.func @func_roundeven(%arg0: tensor<3xf64>) -> tensor<3xf64> {
      %res = math.roundeven %arg0 : tensor<3xf64>
      func.return %res : tensor<3xf64>
  }

  // CHECK-LABEL: @func_round
  // CHECK: stablehlo.round_nearest_afz %arg0 : tensor<3xf64>
  // CHECK-NOT: math.round
  func.func @func_round(%arg0: tensor<3xf64>) -> tensor<3xf64> {
      %res = math.round %arg0 : tensor<3xf64>
      func.return %res : tensor<3xf64>
  }
}
