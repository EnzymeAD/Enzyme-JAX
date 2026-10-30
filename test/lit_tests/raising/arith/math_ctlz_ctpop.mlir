// RUN: enzymexlamlir-opt --arith-raise %s | FileCheck %s

module {
  // CHECK-LABEL: @func_ctlz
  // CHECK: stablehlo.count_leading_zeros %arg0 : tensor<20x20xi64>
  // CHECK-NOT: math.ctlz
  func.func @func_ctlz(%arg0: tensor<20x20xi64>) -> tensor<20x20xi64> {
      %res = math.ctlz %arg0 : tensor<20x20xi64>
      func.return %res : tensor<20x20xi64>
  }

  // CHECK-LABEL: @func_ctpop
  // CHECK: stablehlo.popcnt %arg0 : tensor<20x20xi64>
  // CHECK-NOT: math.ctpop
  func.func @func_ctpop(%arg0: tensor<20x20xi64>) -> tensor<20x20xi64> {
      %res = math.ctpop %arg0 : tensor<20x20xi64>
      func.return %res : tensor<20x20xi64>
  }
}
