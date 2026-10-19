// RUN: enzymexlamlir-opt %s -tessera-to-func -split-input-file | FileCheck %s

// Calls precede the define on purpose: tessera.call must still be converted
// after the tessera.define it calls has been.
func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  %0 = tessera.call @tessera.inv(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

tessera.define @tessera.inv(%arg0: tensor<4xf32>) -> tensor<4xf32> attributes {argModes = [unit], pure = true, tessera.original_name = "inv"} {
  tessera.return %arg0 : tensor<4xf32>
}

// CHECK: func.func @main(%[[X:.*]]: tensor<4xf32>) -> tensor<4xf32>
// CHECK-NEXT: %[[R:.*]] = call @inv(%[[X]]) : (tensor<4xf32>) -> tensor<4xf32>
// CHECK-NEXT: return %[[R]] : tensor<4xf32>

// CHECK: func.func @inv(%[[ARG:.*]]: tensor<4xf32>) -> tensor<4xf32>
// CHECK-NOT: tessera
// CHECK-NEXT: return %[[ARG]] : tensor<4xf32>

// -----

tessera.define private @tessera.ext(f32) -> f32 attributes {argModes = [unit], pure = true, tessera.original_name = "ext"}

// CHECK: func.func private @ext(f32) -> f32

