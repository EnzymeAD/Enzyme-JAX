// RUN: enzymexlamlir-opt --arith-raise %s | FileCheck %s

module {
  // CHECK-LABEL: @func_cmpf_uno
  // CHECK: %[[LNAN:.+]] = stablehlo.compare NE, %arg0, %arg0, FLOAT : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
  // CHECK: %[[RNAN:.+]] = stablehlo.compare NE, %arg1, %arg1, FLOAT : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
  // CHECK: %[[RES:.+]] = stablehlo.or %[[LNAN]], %[[RNAN]] : tensor<4xi1>
  // CHECK: return %[[RES]] : tensor<4xi1>
  func.func @func_cmpf_uno(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xi1> {
    %res = arith.cmpf uno, %arg0, %arg1 : tensor<4xf32>
    func.return %res : tensor<4xi1>
  }

  // CHECK-LABEL: @func_cmpf_uno_same_operand
  // CHECK: %[[NAN:.+]] = stablehlo.compare NE, %arg0, %arg0, FLOAT : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
  // CHECK: return %[[NAN]] : tensor<4xi1>
  func.func @func_cmpf_uno_same_operand(%arg0: tensor<4xf32>) -> tensor<4xi1> {
    %res = arith.cmpf uno, %arg0, %arg0 : tensor<4xf32>
    func.return %res : tensor<4xi1>
  }
}
