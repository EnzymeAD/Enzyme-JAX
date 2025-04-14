// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=broadcast_iota_simplify" --transform-interpreter --enzyme-hlo-remove-transform -allow-unregistered-dialect --split-input-file %s | FileCheck %s

  func.func @iszero() -> (tensor<32x800xi1>)  {
      %c_33 = stablehlo.constant dense<[true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false]> : tensor<32xi1>
      %21 = stablehlo.broadcast_in_dim %c_33, dims = [0] : (tensor<32xi1>) -> tensor<32x800xi1>
      func.return %21 : tensor<32x800xi1>
  }

// CHECK:  func.func @iszero() -> tensor<32x800xi1> {
// CHECK-NEXT:    %c = stablehlo.constant dense<0> : tensor<32x800xui32>
// CHECK-NEXT:    %0 = stablehlo.iota dim = 0 : tensor<32x800xui32>
// CHECK-NEXT:    %1 = stablehlo.compare  EQ, %0, %c : (tensor<32x800xui32>, tensor<32x800xui32>) -> tensor<32x800xi1>
// CHECK-NEXT:    return %1 : tensor<32x800xi1>
// CHECK-NEXT:  }

  func.func @isend() -> (tensor<32x800xi1>)  {
      %c_32 = stablehlo.constant dense<[false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, true]> : tensor<32xi1>
      %21 = stablehlo.broadcast_in_dim %c_32, dims = [0] : (tensor<32xi1>) -> tensor<32x800xi1>
      func.return %21 : tensor<32x800xi1>
  }

// CHECK:    func.func @isend() -> tensor<32x800xi1> {
// CHECK-NEXT:    %c = stablehlo.constant dense<31> : tensor<32x800xui32>
// CHECK-NEXT:    %0 = stablehlo.iota dim = 0 : tensor<32x800xui32>
// CHECK-NEXT:    %1 = stablehlo.compare  EQ, %0, %c : (tensor<32x800xui32>, tensor<32x800xui32>) -> tensor<32x800xi1>
// CHECK-NEXT:    return %1 : tensor<32x800xi1>
// CHECK-NEXT:  }