// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt="max_iterations=2" | FileCheck %s

// A slice sinks through the whole elementwise chain within one iteration of
// the driver, so two iterations (the second finding nothing to do) suffice.
func.func @chain(%x: tensor<8x8xf64>, %y: tensor<8x8xf64>, %z: tensor<8x8xf64>, %w: tensor<8x8xf64>) -> tensor<2x8xf64> {
  %0 = stablehlo.multiply %x, %y : tensor<8x8xf64>
  %1 = stablehlo.add %0, %z : tensor<8x8xf64>
  %2 = stablehlo.subtract %1, %w : tensor<8x8xf64>
  %3 = stablehlo.negate %2 : tensor<8x8xf64>
  %4 = stablehlo.slice %3 [3:5, 0:8] : (tensor<8x8xf64>) -> tensor<2x8xf64>
  return %4 : tensor<2x8xf64>
}

// CHECK:    func.func @chain(%arg0: tensor<8x8xf64>, %arg1: tensor<8x8xf64>, %arg2: tensor<8x8xf64>, %arg3: tensor<8x8xf64>) -> tensor<2x8xf64> {
// CHECK-NEXT:    %0 = stablehlo.slice %arg0 [3:5, 0:8] : (tensor<8x8xf64>) -> tensor<2x8xf64>
// CHECK-NEXT:    %1 = stablehlo.slice %arg1 [3:5, 0:8] : (tensor<8x8xf64>) -> tensor<2x8xf64>
// CHECK-NEXT:    %2 = stablehlo.multiply %0, %1 : tensor<2x8xf64>
// CHECK-NEXT:    %3 = stablehlo.slice %arg2 [3:5, 0:8] : (tensor<8x8xf64>) -> tensor<2x8xf64>
// CHECK-NEXT:    %4 = stablehlo.add %2, %3 : tensor<2x8xf64>
// CHECK-NEXT:    %5 = stablehlo.slice %arg3 [3:5, 0:8] : (tensor<8x8xf64>) -> tensor<2x8xf64>
// CHECK-NEXT:    %6 = stablehlo.subtract %4, %5 : tensor<2x8xf64>
// CHECK-NEXT:    %7 = stablehlo.negate %6 : tensor<2x8xf64>
// CHECK-NEXT:    return %7 : tensor<2x8xf64>
// CHECK-NEXT:  }
