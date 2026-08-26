// RUN: enzymexlamlir-opt --arith-raise %s | FileCheck %s

module {
  // CHECK-LABEL: @func_icmp_ugt
  // CHECK: %[[LHS:.+]] = stablehlo.convert %arg0 : (tensor<20x20xi64>) -> tensor<20x20xui64>
  // CHECK: %[[RHS:.+]] = stablehlo.convert %arg1 : (tensor<20x20xi64>) -> tensor<20x20xui64>
  // CHECK: stablehlo.compare GT, %[[LHS]], %[[RHS]], UNSIGNED : (tensor<20x20xui64>, tensor<20x20xui64>) -> tensor<20x20xi1>
  func.func @func_icmp_ugt(%arg0: tensor<20x20xi64>, %arg1: tensor<20x20xi64>) -> tensor<20x20xi1> {
      %res = arith.cmpi ugt, %arg0, %arg1 : tensor<20x20xi64>
      func.return %res : tensor<20x20xi1>
  }

  // CHECK-LABEL: @func_icmp_sgt
  // CHECK-NOT: stablehlo.convert
  // CHECK: stablehlo.compare GT, %arg0, %arg1, SIGNED : (tensor<20x20xi64>, tensor<20x20xi64>) -> tensor<20x20xi1>
  func.func @func_icmp_sgt(%arg0: tensor<20x20xi64>, %arg1: tensor<20x20xi64>) -> tensor<20x20xi1> {
      %res = arith.cmpi sgt, %arg0, %arg1 : tensor<20x20xi64>
      func.return %res : tensor<20x20xi1>
  }

  // CHECK-LABEL: @func_icmp_sgt_i1
  // CHECK-NOT: stablehlo.convert
  // CHECK: stablehlo.compare GT, %arg0, %arg1, UNSIGNED : (tensor<20x20xi1>, tensor<20x20xi1>) -> tensor<20x20xi1>
  func.func @func_icmp_sgt_i1(%arg0: tensor<20x20xi1>, %arg1: tensor<20x20xi1>) -> tensor<20x20xi1> {
      %res = arith.cmpi sgt, %arg0, %arg1 : tensor<20x20xi1>
      func.return %res : tensor<20x20xi1>
  }
}
