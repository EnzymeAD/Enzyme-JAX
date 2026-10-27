// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=concat_pieces_to_select<1>(64)" --transform-interpreter --enzyme-hlo-remove-transform --split-input-file | FileCheck %s
// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=concat_pieces_to_select<1>(1)" --transform-interpreter --enzyme-hlo-remove-transform --split-input-file | FileCheck %s --check-prefix=SMALL

// Two computed pieces, each over slices of the arguments at its own rows,
// are rebuilt over every row and selected by row: one fused kernel for the
// backend, which pays while the elementwise work of the whole block after
// the rebuilt pieces compute stays within the budget (the second RUN's
// budget of one op is below the two of the rebuild).

// CHECK-LABEL: @two_pieces
// CHECK-DAG: %[[IOTA:.+]] = stablehlo.iota dim = 0 : tensor<8x4xi32>
// CHECK-DAG: %[[K:.+]] = stablehlo.constant dense<3> : tensor<8x4xi32>
// CHECK-DAG: %[[A:.+]] = stablehlo.slice %arg0 [1:9, 0:4] : (tensor<10x4xf32>) -> tensor<8x4xf32>
// CHECK-DAG: %[[B:.+]] = stablehlo.slice %arg1 [1:9, 0:4] : (tensor<10x4xf32>) -> tensor<8x4xf32>
// CHECK-DAG: %[[M:.+]] = stablehlo.multiply %[[A]], %[[B]] : tensor<8x4xf32>
// CHECK-DAG: %[[A2:.+]] = stablehlo.slice %arg0 [2:10, 0:4] : (tensor<10x4xf32>) -> tensor<8x4xf32>
// CHECK-DAG: %[[B2:.+]] = stablehlo.slice %arg1 [1:9, 0:4] : (tensor<10x4xf32>) -> tensor<8x4xf32>
// CHECK-DAG: %[[S:.+]] = stablehlo.add %[[A2]], %[[B2]] : tensor<8x4xf32>
// CHECK-DAG: %[[P:.+]] = stablehlo.compare GE, %[[IOTA]], %[[K]] : (tensor<8x4xi32>, tensor<8x4xi32>) -> tensor<8x4xi1>
// CHECK: %[[R:.+]] = stablehlo.select %[[P]], %[[S]], %[[M]] : tensor<8x4xi1>, tensor<8x4xf32>
// CHECK: return %[[R]]

// SMALL-LABEL: @two_pieces
// SMALL: stablehlo.concatenate
// SMALL-NOT: stablehlo.select
func.func @two_pieces(%a: tensor<10x4xf32>, %b: tensor<10x4xf32>) -> tensor<8x4xf32> {
  %a0 = stablehlo.slice %a [1:4, 0:4] : (tensor<10x4xf32>) -> tensor<3x4xf32>
  %b0 = stablehlo.slice %b [1:4, 0:4] : (tensor<10x4xf32>) -> tensor<3x4xf32>
  %p0 = stablehlo.multiply %a0, %b0 : tensor<3x4xf32>
  %a1 = stablehlo.slice %a [5:10, 0:4] : (tensor<10x4xf32>) -> tensor<5x4xf32>
  %b1 = stablehlo.slice %b [4:9, 0:4] : (tensor<10x4xf32>) -> tensor<5x4xf32>
  %p1 = stablehlo.add %a1, %b1 : tensor<5x4xf32>
  %r = stablehlo.concatenate %p0, %p1, dim = 0 : (tensor<3x4xf32>, tensor<5x4xf32>) -> tensor<8x4xf32>
  return %r : tensor<8x4xf32>
}

// -----

// A piece whose rebuilt rows would read past its source is left alone.

// CHECK-LABEL: @out_of_source
// CHECK: stablehlo.concatenate
// CHECK-NOT: stablehlo.select
func.func @out_of_source(%a: tensor<8x4xf32>, %b: tensor<8x4xf32>) -> tensor<8x4xf32> {
  %a0 = stablehlo.slice %a [0:3, 0:4] : (tensor<8x4xf32>) -> tensor<3x4xf32>
  %p0 = stablehlo.negate %a0 : tensor<3x4xf32>
  %a1 = stablehlo.slice %a [2:7, 0:4] : (tensor<8x4xf32>) -> tensor<5x4xf32>
  %p1 = stablehlo.negate %a1 : tensor<5x4xf32>
  %r = stablehlo.concatenate %p0, %p1, dim = 0 : (tensor<3x4xf32>, tensor<5x4xf32>) -> tensor<8x4xf32>
  return %r : tensor<8x4xf32>
}

// -----

// A concatenate of views is data movement: left alone.

// CHECK-LABEL: @views
// CHECK: stablehlo.concatenate
// CHECK-NOT: stablehlo.select
func.func @views(%a: tensor<8x4xf32>) -> tensor<8x4xf32> {
  %a0 = stablehlo.slice %a [5:8, 0:4] : (tensor<8x4xf32>) -> tensor<3x4xf32>
  %a1 = stablehlo.slice %a [0:5, 0:4] : (tensor<8x4xf32>) -> tensor<5x4xf32>
  %r = stablehlo.concatenate %a0, %a1, dim = 0 : (tensor<3x4xf32>, tensor<5x4xf32>) -> tensor<8x4xf32>
  return %r : tensor<8x4xf32>
}
