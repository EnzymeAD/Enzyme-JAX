// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=pad_computed_to_select<1>(64)" --transform-interpreter --enzyme-hlo-remove-transform --split-input-file | FileCheck %s
// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=pad_computed_to_select<1>(0)" --transform-interpreter --enzyme-hlo-remove-transform --split-input-file | FileCheck %s --check-prefix=SMALL

// A pad of a value computed on its rows from slices of wider arguments is
// the same computation over every row, selected inside the padded rows,
// with the padding value outside.

// CHECK-LABEL: @pad_of_computed
// CHECK-DAG: %[[LO:.+]] = stablehlo.constant dense<1> : tensor<8x4xi32>
// CHECK-DAG: %[[HI:.+]] = stablehlo.constant dense<6> : tensor<8x4xi32>
// CHECK-DAG: %[[IOTA:.+]] = stablehlo.iota dim = 0 : tensor<8x4xi32>
// CHECK-DAG: %[[A:.+]] = stablehlo.slice %arg0 [1:9, 0:4] : (tensor<10x4xf32>) -> tensor<8x4xf32>
// CHECK-DAG: %[[B:.+]] = stablehlo.slice %arg1 [0:8, 0:4] : (tensor<10x4xf32>) -> tensor<8x4xf32>
// CHECK-DAG: %[[M:.+]] = stablehlo.multiply %[[A]], %[[B]] : tensor<8x4xf32>
// CHECK-DAG: %[[GE:.+]] = stablehlo.compare GE, %[[IOTA]], %[[LO]]
// CHECK-DAG: %[[LT:.+]] = stablehlo.compare LT, %[[IOTA]], %[[HI]]
// CHECK-DAG: %[[IN:.+]] = stablehlo.and %[[GE]], %[[LT]]
// CHECK-DAG: %[[FILL:.+]] = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f32>) -> tensor<8x4xf32>
// CHECK: %[[R:.+]] = stablehlo.select %[[IN]], %[[M]], %[[FILL]] : tensor<8x4xi1>, tensor<8x4xf32>
// CHECK: return %[[R]]

// SMALL-LABEL: @pad_of_computed
// SMALL: stablehlo.pad
// SMALL-NOT: stablehlo.select
func.func @pad_of_computed(%a: tensor<10x4xf32>, %b: tensor<10x4xf32>, %v: tensor<f32>) -> tensor<8x4xf32> {
  %a0 = stablehlo.slice %a [2:7, 0:4] : (tensor<10x4xf32>) -> tensor<5x4xf32>
  %b0 = stablehlo.slice %b [1:6, 0:4] : (tensor<10x4xf32>) -> tensor<5x4xf32>
  %m = stablehlo.multiply %a0, %b0 : tensor<5x4xf32>
  %p = stablehlo.pad %m, %v, low = [1, 0], high = [2, 0], interior = [0, 0] : (tensor<5x4xf32>, tensor<f32>) -> tensor<8x4xf32>
  return %p : tensor<8x4xf32>
}

// -----

// A pad of a view is data movement: left alone.

// CHECK-LABEL: @pad_of_view
// CHECK: stablehlo.pad
// CHECK-NOT: stablehlo.select
func.func @pad_of_view(%a: tensor<10x4xf32>, %v: tensor<f32>) -> tensor<8x4xf32> {
  %a0 = stablehlo.slice %a [2:7, 0:4] : (tensor<10x4xf32>) -> tensor<5x4xf32>
  %p = stablehlo.pad %a0, %v, low = [1, 0], high = [2, 0], interior = [0, 0] : (tensor<5x4xf32>, tensor<f32>) -> tensor<8x4xf32>
  return %p : tensor<8x4xf32>
}
