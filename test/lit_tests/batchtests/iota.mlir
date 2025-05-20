// RUN: enzymexlamlir-opt --enzyme-batch %s | FileCheck %s

func.func @main() -> (tensor<2x5x3x8xf64>) {
    %0 = enzyme.batch @iota() {batch_shape = array<i64: 2, 5>} : () -> (tensor<2x5x3x8xf64>)
    return %0 : tensor<2x5x3x8xf64>
}

func.func @iota() -> (tensor<3x8xf64>) {
    %0 = stablehlo.iota dim = 1 : tensor<3x8xf64>
    return %0 : tensor<3x8xf64>
}

// CHECK: func.func private @batched_iota() -> tensor<2x5x3x8xf64> {
// CHECK-NEXT:    %0 = stablehlo.iota dim = 3 : tensor<2x5x3x8xf64>
// CHECK-NEXT:    return %0 : tensor<2x5x3x8xf64>
// CHECK-NEXT:  }
