// RUN: enzymexlamlir-opt --arith-raise %s | FileCheck %s

module {
  // CHECK-LABEL: @func_cmpf_uno
  // CHECK: %[[LFIN:.+]] = stablehlo.is_finite %arg0 : (tensor<4xf32>) -> tensor<4xi1>
  // CHECK: %[[LNOTFIN:.+]] = stablehlo.not %[[LFIN]] : tensor<4xi1>
  // CHECK: %[[LINF:.+]] = chlo.is_inf %arg0 : tensor<4xf32> -> tensor<4xi1>
  // CHECK: %[[LNOTINF:.+]] = stablehlo.not %[[LINF]] : tensor<4xi1>
  // CHECK: %[[LNAN:.+]] = stablehlo.and %[[LNOTFIN]], %[[LNOTINF]] : tensor<4xi1>
  // CHECK: %[[RFIN:.+]] = stablehlo.is_finite %arg1 : (tensor<4xf32>) -> tensor<4xi1>
  // CHECK: %[[RNOTFIN:.+]] = stablehlo.not %[[RFIN]] : tensor<4xi1>
  // CHECK: %[[RINF:.+]] = chlo.is_inf %arg1 : tensor<4xf32> -> tensor<4xi1>
  // CHECK: %[[RNOTINF:.+]] = stablehlo.not %[[RINF]] : tensor<4xi1>
  // CHECK: %[[RNAN:.+]] = stablehlo.and %[[RNOTFIN]], %[[RNOTINF]] : tensor<4xi1>
  // CHECK: %[[RES:.+]] = stablehlo.or %[[LNAN]], %[[RNAN]] : tensor<4xi1>
  // CHECK: return %[[RES]] : tensor<4xi1>
  func.func @func_cmpf_uno(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xi1> {
    %res = arith.cmpf uno, %arg0, %arg1 : tensor<4xf32>
    func.return %res : tensor<4xi1>
  }

  // CHECK-LABEL: @func_cmpf_uno_same_operand
  // CHECK: %[[FIN:.+]] = stablehlo.is_finite %arg0 : (tensor<4xf32>) -> tensor<4xi1>
  // CHECK: %[[NOTFIN:.+]] = stablehlo.not %[[FIN]] : tensor<4xi1>
  // CHECK: %[[INF:.+]] = chlo.is_inf %arg0 : tensor<4xf32> -> tensor<4xi1>
  // CHECK: %[[NOTINF:.+]] = stablehlo.not %[[INF]] : tensor<4xi1>
  // CHECK: %[[NAN:.+]] = stablehlo.and %[[NOTFIN]], %[[NOTINF]] : tensor<4xi1>
  // CHECK: return %[[NAN]] : tensor<4xi1>
  func.func @func_cmpf_uno_same_operand(%arg0: tensor<4xf32>) -> tensor<4xi1> {
    %res = arith.cmpf uno, %arg0, %arg0 : tensor<4xf32>
    func.return %res : tensor<4xi1>
  }
}
