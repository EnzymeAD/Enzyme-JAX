// RUN: enzymexlamlir-opt %s -llvm-to-tessera -split-input-file | FileCheck %s

func.func @inv(%arg0: tensor<4xf32>) -> tensor<4xf32> attributes {pure_tessera_op = "tessera.inv(x)"} {
  return %arg0 : tensor<4xf32>
}

func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  %0 = call @inv(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// CHECK: tessera.define @tessera.inv(%[[ARG:.*]]: tensor<4xf32>) -> tensor<4xf32>
// CHECK-SAME: tessera.original_name = "inv"
// CHECK-SAME: argModes = [unit]
// CHECK-SAME: pure = true
// CHECK-NEXT: tessera.return %[[ARG]] : tensor<4xf32>

// CHECK: func.func @main(%[[X:.*]]: tensor<4xf32>) -> tensor<4xf32>
// CHECK-NEXT: %[[R:.*]] = tessera.call @tessera.inv(%[[X]]) : (tensor<4xf32>) -> tensor<4xf32>
// CHECK-NEXT: return %[[R]] : tensor<4xf32>

// -----

// Multiple results and a nested region whose terminator must stay as is.
func.func private @two(%arg0: tensor<f32>, %arg1: tensor<i1>) -> (tensor<f32>, tensor<f32>) attributes {tessera_op = "tessera.two(x, c)"} {
  %0 = "stablehlo.if"(%arg1) ({
    stablehlo.return %arg0 : tensor<f32>
  }, {
    stablehlo.return %arg0 : tensor<f32>
  }) : (tensor<i1>) -> tensor<f32>
  return %0, %arg0 : tensor<f32>, tensor<f32>
}

// CHECK: tessera.define private @tessera.two(%{{.*}}: tensor<f32>, %{{.*}}: tensor<i1>) -> (tensor<f32>, tensor<f32>)
// CHECK-SAME: pure = false
// CHECK: stablehlo.return
// CHECK: stablehlo.return
// CHECK: tessera.return %{{.*}}, %{{.*}} : tensor<f32>, tensor<f32>

// -----

// External declarations are lifted too.
func.func private @ext(%arg0: f32) -> f32 attributes {pure_tessera_op = "tessera.ext(x)"}

func.func @use_ext(%arg0: f32) -> f32 {
  %0 = call @ext(%arg0) : (f32) -> f32
  return %0 : f32
}

// CHECK: tessera.define private @tessera.ext(f32) -> f32
// CHECK: tessera.call @tessera.ext(%{{.*}}) : (f32) -> f32
