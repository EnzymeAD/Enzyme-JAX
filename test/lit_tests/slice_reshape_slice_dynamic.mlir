// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt | FileCheck %s

// A slice of a reshape of a slice whose operand has a dynamic extent: the
// inner slice's bounds are static, so the merged slice's are too. Clamping
// the merged bounds to the operand's extent read the dynamic extent
// (INT64_MIN) as a size and produced a negative start, which aborted the
// pass in result type inference.
func.func @dyn(%arg0: tensor<?xf64>) -> tensor<18xf64> {
  %0 = stablehlo.slice %arg0 [0:216] : (tensor<?xf64>) -> tensor<216xf64>
  %1 = stablehlo.reshape %0 : (tensor<216xf64>) -> tensor<216xf64>
  %2 = stablehlo.slice %1 [0:18] : (tensor<216xf64>) -> tensor<18xf64>
  return %2 : tensor<18xf64>
}

// CHECK:    func.func @dyn(%arg0: tensor<?xf64>) -> tensor<18xf64> {
// CHECK-NEXT:    %0 = stablehlo.slice %arg0 [0:18] : (tensor<?xf64>) -> tensor<18xf64>
// CHECK-NEXT:    return %0 : tensor<18xf64>
// CHECK-NEXT:  }
