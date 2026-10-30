// RUN: enzymexlamlir-opt --arith-raise %s | FileCheck %s

func.func @isinf_tensor(%x: tensor<4xf32>) -> tensor<4xi1> {
  %r = math.isinf %x : tensor<4xf32>
  return %r : tensor<4xi1>
}

// CHECK-LABEL: func.func @isinf_tensor(%arg0: tensor<4xf32>) -> tensor<4xi1> {
// CHECK-NEXT:    %[[R:.+]] = chlo.is_inf %arg0 : tensor<4xf32> -> tensor<4xi1>
// CHECK-NEXT:    return %[[R]] : tensor<4xi1>
// CHECK-NEXT:  }
