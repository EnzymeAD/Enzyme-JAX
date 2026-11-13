// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt | FileCheck %s
// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(enzyme-hlo-generate-td{patterns=rotate_slice;reuse_common_power_op;slice_slice},transform-interpreter,enzyme-hlo-remove-transform)" | FileCheck %s --check-prefix=TD

// Exact #1203 reproducer: slice of single-use rotate along non-rotated dims
// sinks inside the rotate and fuses with the preceding slice.
// CHECK-LABEL: func.func @slice_of_single_use_rotate(
// CHECK: %[[SL:.*]] = stablehlo.slice %arg0 [8:12, 7:1526, 0:3056] : (tensor<20x1536x3056xf64>) -> tensor<4x1519x3056xf64>
// CHECK: %[[ROT:.*]] = "enzymexla.rotate"(%[[SL]]) <{amount = 1 : i32, dimension = 2 : i32}> : (tensor<4x1519x3056xf64>) -> tensor<4x1519x3056xf64>
// CHECK-NOT: stablehlo.slice
// CHECK: return %[[ROT]] : tensor<4x1519x3056xf64>
// TD-LABEL: func.func @slice_of_single_use_rotate(
// TD: %[[TSL:.*]] = stablehlo.slice %arg0 [8:12, 7:1526, 0:3056] : (tensor<20x1536x3056xf64>) -> tensor<4x1519x3056xf64>
// TD: %[[TROT:.*]] = "enzymexla.rotate"(%[[TSL]]) <{amount = 1 : i32, dimension = 2 : i32}> : (tensor<4x1519x3056xf64>) -> tensor<4x1519x3056xf64>
// TD-NOT: stablehlo.slice
// TD: return %[[TROT]] : tensor<4x1519x3056xf64>
func.func @slice_of_single_use_rotate(%arg0: tensor<20x1536x3056xf64>) -> tensor<4x1519x3056xf64> {
  %0 = stablehlo.slice %arg0 [8:12, 6:1526, 0:3056] : (tensor<20x1536x3056xf64>) -> tensor<4x1520x3056xf64>
  %1 = "enzymexla.rotate"(%0) <{amount = 1 : i32, dimension = 2 : i32}> : (tensor<4x1520x3056xf64>) -> tensor<4x1520x3056xf64>
  %2 = stablehlo.slice %1 [0:4, 1:1520, 0:3056] : (tensor<4x1520x3056xf64>) -> tensor<4x1519x3056xf64>
  return %2 : tensor<4x1519x3056xf64>
}

// Negative test: slicing the rotated dimension itself must NOT commute naively.
// CHECK-LABEL: func.func @do_not_sink_slice_on_rotated_dim(
// CHECK: %[[ROT:.*]] = "enzymexla.rotate"(%arg0)
// CHECK: %[[SL:.*]] = stablehlo.slice %[[ROT]]
// CHECK: return %[[SL]]
func.func @do_not_sink_slice_on_rotated_dim(%arg0: tensor<4x10xf32>) -> tensor<4x8xf32> {
  %0 = "enzymexla.rotate"(%arg0) <{amount = 2 : i32, dimension = 1 : i32}> : (tensor<4x10xf32>) -> tensor<4x10xf32>
  %1 = stablehlo.slice %0 [0:4, 1:9] : (tensor<4x10xf32>) -> tensor<4x8xf32>
  return %1 : tensor<4x8xf32>
}

// Exact #1080 reproducer: x^8 alongside 8 * x^7 reuses x^7 via x^7 * x
// even when x^8 appears before x^7 in the block.
// CHECK-LABEL: func.func @reuse_common_power_reverse_ad(
// CHECK-DAG: %[[C7:.*]] = stablehlo.constant dense<7.000000e+00> : tensor<f64>
// CHECK-DAG: %[[C8:.*]] = stablehlo.constant dense<8.000000e+00> : tensor<f64>
// CHECK: %[[P7:.*]] = stablehlo.power %arg0, %[[C7]] : tensor<f64>
// CHECK-NOT: stablehlo.power
// CHECK-DAG: %[[P8:.*]] = stablehlo.multiply %[[P7]], %arg0 : tensor<f64>
// CHECK-DAG: %[[GRAD:.*]] = stablehlo.multiply %[[C8]], %[[P7]] : tensor<f64>
// CHECK: return %[[GRAD]], %[[P8]] : tensor<f64>, tensor<f64>
// TD-LABEL: func.func @reuse_common_power_reverse_ad(
// TD: %[[TP7:.*]] = stablehlo.power %arg0, %{{.*}} : tensor<f64>
// TD-NOT: stablehlo.power
// TD-DAG: %[[TP8:.*]] = stablehlo.multiply %[[TP7]], %arg0 : tensor<f64>
// TD-DAG: %[[TGRAD:.*]] = stablehlo.multiply %{{.*}}, %[[TP7]] : tensor<f64>
// TD: return %[[TGRAD]], %[[TP8]] : tensor<f64>, tensor<f64>
func.func @reuse_common_power_reverse_ad(%arg0: tensor<f64>) -> (tensor<f64>, tensor<f64>) {
  %cst = stablehlo.constant dense<7.000000e+00> : tensor<f64>
  %cst_0 = stablehlo.constant dense<8.000000e+00> : tensor<f64>
  %0 = stablehlo.power %arg0, %cst_0 : tensor<f64>
  %1 = stablehlo.power %arg0, %cst : tensor<f64>
  %2 = stablehlo.multiply %cst_0, %1 : tensor<f64>
  return %2, %0 : tensor<f64>, tensor<f64>
}

// Chained integer powers: x^3, x^4, x^5 -> single power x^3 + two multiplies.
// CHECK-LABEL: func.func @reuse_chained_int_powers(
// CHECK: %[[P3:.*]] = stablehlo.power %arg0, %{{.*}} : tensor<8xi32>
// CHECK-NOT: stablehlo.power
// CHECK: %[[P4:.*]] = stablehlo.multiply %[[P3]], %arg0 : tensor<8xi32>
// CHECK: %[[P5:.*]] = stablehlo.multiply %[[P4]], %arg0 : tensor<8xi32>
// CHECK: return %[[P3]], %[[P4]], %[[P5]] : tensor<8xi32>, tensor<8xi32>, tensor<8xi32>
func.func @reuse_chained_int_powers(%arg0: tensor<8xi32>) -> (tensor<8xi32>, tensor<8xi32>, tensor<8xi32>) {
  %c3 = stablehlo.constant dense<3> : tensor<8xi32>
  %c4 = stablehlo.constant dense<4> : tensor<8xi32>
  %c5 = stablehlo.constant dense<5> : tensor<8xi32>
  %p3 = stablehlo.power %arg0, %c3 : tensor<8xi32>
  %p4 = stablehlo.power %arg0, %c4 : tensor<8xi32>
  %p5 = stablehlo.power %arg0, %c5 : tensor<8xi32>
  return %p3, %p4, %p5 : tensor<8xi32>, tensor<8xi32>, tensor<8xi32>
}
