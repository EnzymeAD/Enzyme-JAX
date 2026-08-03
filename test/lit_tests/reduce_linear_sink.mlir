// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(enzyme-hlo-generate-td{patterns=reduce_linear_sink},transform-interpreter,enzyme-hlo-remove-transform)" %s | FileCheck %s

// A sum reduction distributes over add: the wide add is replaced by an add at
// the reduced shape.
func.func @sink_add(%a: tensor<256x1024xf32>, %b: tensor<256x1024xf32>) -> tensor<256xf32> {
  %init = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %0 = stablehlo.add %a, %b : tensor<256x1024xf32>
  %1 = stablehlo.reduce(%0 init: %init) applies stablehlo.add across dimensions = [1] : (tensor<256x1024xf32>, tensor<f32>) -> tensor<256xf32>
  return %1 : tensor<256xf32>
}

// CHECK-LABEL: func.func @sink_add
// CHECK-SAME: (%[[A:.+]]: tensor<256x1024xf32>, %[[B:.+]]: tensor<256x1024xf32>)
// CHECK-NOT: stablehlo.add %{{.*}} : tensor<256x1024xf32>
// CHECK-DAG: %[[RA:.+]] = stablehlo.reduce(%[[A]] init: %{{.+}}) applies stablehlo.add across dimensions = [1] : (tensor<256x1024xf32>, tensor<f32>) -> tensor<256xf32>
// CHECK-DAG: %[[RB:.+]] = stablehlo.reduce(%[[B]] init: %{{.+}}) applies stablehlo.add across dimensions = [1] : (tensor<256x1024xf32>, tensor<f32>) -> tensor<256xf32>
// CHECK: stablehlo.add %[[RA]], %[[RB]] : tensor<256xf32>


func.func @sink_subtract(%a: tensor<256x1024xf32>, %b: tensor<256x1024xf32>) -> tensor<256xf32> {
  %init = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %0 = stablehlo.subtract %a, %b : tensor<256x1024xf32>
  %1 = stablehlo.reduce(%0 init: %init) applies stablehlo.add across dimensions = [1] : (tensor<256x1024xf32>, tensor<f32>) -> tensor<256xf32>
  return %1 : tensor<256xf32>
}

// CHECK-LABEL: func.func @sink_subtract
// CHECK-NOT: stablehlo.subtract %{{.*}} : tensor<256x1024xf32>
// CHECK: stablehlo.subtract %{{.+}}, %{{.+}} : tensor<256xf32>


func.func @sink_negate(%a: tensor<256x1024xf32>) -> tensor<256xf32> {
  %init = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %0 = stablehlo.negate %a : tensor<256x1024xf32>
  %1 = stablehlo.reduce(%0 init: %init) applies stablehlo.add across dimensions = [1] : (tensor<256x1024xf32>, tensor<f32>) -> tensor<256xf32>
  return %1 : tensor<256xf32>
}

// CHECK-LABEL: func.func @sink_negate
// CHECK-NOT: stablehlo.negate %{{.*}} : tensor<256x1024xf32>
// CHECK: stablehlo.negate %{{.+}} : tensor<256xf32>


// A chain sinks all the way to the leaves, leaving only reduced-shape
// arithmetic behind.
func.func @sink_chain(%a: tensor<8x4xf32>, %b: tensor<8x4xf32>, %c: tensor<8x4xf32>) -> tensor<8xf32> {
  %init = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %0 = stablehlo.add %a, %b : tensor<8x4xf32>
  %1 = stablehlo.subtract %0, %c : tensor<8x4xf32>
  %2 = stablehlo.reduce(%1 init: %init) applies stablehlo.add across dimensions = [1] : (tensor<8x4xf32>, tensor<f32>) -> tensor<8xf32>
  return %2 : tensor<8xf32>
}

// CHECK-LABEL: func.func @sink_chain
// CHECK-SAME: (%[[A:.+]]: tensor<8x4xf32>, %[[B:.+]]: tensor<8x4xf32>, %[[C:.+]]: tensor<8x4xf32>)
// CHECK-NOT: stablehlo.add %{{.+}}, %{{.+}} : tensor<8x4xf32>
// CHECK-NOT: stablehlo.subtract %{{.+}}, %{{.+}} : tensor<8x4xf32>
// CHECK: %[[RA:.+]] = stablehlo.reduce(%[[A]] init: %{{.+}}) applies stablehlo.add across dimensions = [1] : (tensor<8x4xf32>, tensor<f32>) -> tensor<8xf32>
// CHECK: %[[RB:.+]] = stablehlo.reduce(%[[B]] init: %{{.+}}) applies stablehlo.add across dimensions = [1] : (tensor<8x4xf32>, tensor<f32>) -> tensor<8xf32>
// CHECK: %[[SUM:.+]] = stablehlo.add %[[RA]], %[[RB]] : tensor<8xf32>
// CHECK: %[[RC:.+]] = stablehlo.reduce(%[[C]] init: %{{.+}}) applies stablehlo.add across dimensions = [1] : (tensor<8x4xf32>, tensor<f32>) -> tensor<8xf32>
// CHECK: stablehlo.subtract %[[SUM]], %[[RC]] : tensor<8xf32>


// Multi-use arithmetic must not be duplicated: sinking would leave the wide op
// in place and add reductions on top of it.
func.func @no_sink_multi_use(%a: tensor<8x4xf32>, %b: tensor<8x4xf32>) -> (tensor<8xf32>, tensor<8x4xf32>) {
  %init = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %0 = stablehlo.add %a, %b : tensor<8x4xf32>
  %1 = stablehlo.reduce(%0 init: %init) applies stablehlo.add across dimensions = [1] : (tensor<8x4xf32>, tensor<f32>) -> tensor<8xf32>
  return %1, %0 : tensor<8xf32>, tensor<8x4xf32>
}

// CHECK-LABEL: func.func @no_sink_multi_use
// CHECK: %[[SUM:.+]] = stablehlo.add %{{.+}}, %{{.+}} : tensor<8x4xf32>
// CHECK: stablehlo.reduce(%[[SUM]]


// Only sum reductions distribute; max does not.
func.func @no_sink_max(%a: tensor<8x4xf32>, %b: tensor<8x4xf32>) -> tensor<8xf32> {
  %init = stablehlo.constant dense<0xFF800000> : tensor<f32>
  %0 = stablehlo.add %a, %b : tensor<8x4xf32>
  %1 = stablehlo.reduce(%0 init: %init) applies stablehlo.maximum across dimensions = [1] : (tensor<8x4xf32>, tensor<f32>) -> tensor<8xf32>
  return %1 : tensor<8xf32>
}

// CHECK-LABEL: func.func @no_sink_max
// CHECK: %[[SUM:.+]] = stablehlo.add %{{.+}}, %{{.+}} : tensor<8x4xf32>
// CHECK: stablehlo.reduce(%[[SUM]]
