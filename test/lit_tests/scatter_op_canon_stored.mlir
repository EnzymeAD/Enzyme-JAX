// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=scatter_op_canon(1)" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

// Each element writes its own 5x5 box of the returned buffer: the result is only
// stored, so the scatter is kept.
func.func @output(%x: tensor<750xf64>, %u: tensor<10x5x5xf64>) -> tensor<750xf64> {
  %e = stablehlo.iota dim = 0 : tensor<10xi64>
  %c15 = stablehlo.constant dense<15> : tensor<10xi64>
  %e15 = stablehlo.multiply %c15, %e : tensor<10xi64>
  %eb = stablehlo.broadcast_in_dim %e15, dims = [0] : (tensor<10xi64>) -> tensor<10x5xi64>
  %i = stablehlo.iota dim = 1 : tensor<10x5xi64>
  %s = stablehlo.add %eb, %i : tensor<10x5xi64>
  %c5 = stablehlo.constant dense<5> : tensor<10x5xi64>
  %s5 = stablehlo.multiply %s, %c5 : tensor<10x5xi64>
  %sb = stablehlo.broadcast_in_dim %s5, dims = [0, 1] : (tensor<10x5xi64>) -> tensor<10x5x5x1xi64>
  %j = stablehlo.iota dim = 2 : tensor<10x5x5x1xi64>
  %idx = stablehlo.add %sb, %j : tensor<10x5x5x1xi64>
  %r = "stablehlo.scatter"(%x, %idx, %u) <{scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 3>}> ({
  ^bb0(%a: tensor<f64>, %b: tensor<f64>):
    stablehlo.return %b : tensor<f64>
  }) : (tensor<750xf64>, tensor<10x5x5x1xi64>, tensor<10x5x5xf64>) -> tensor<750xf64>
  return %r : tensor<750xf64>
}

// CHECK-LABEL: func.func @output(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<750xf64>, %[[V1:[^ ,:)]+]]: tensor<10x5x5xf64>) -> tensor<750xf64> {
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = stablehlo.constant dense<5> : tensor<10x5xi64>
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.constant dense<15> : tensor<10xi64>
// CHECK-NEXT: %[[V4:[^ ,:)]+]] = stablehlo.iota dim = 0 : tensor<10xi64>
// CHECK-NEXT: %[[V5:[^ ,:)]+]] = stablehlo.multiply %[[V3]], %[[V4]] : tensor<10xi64>
// CHECK-NEXT: %[[V6:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[V5]], dims = [0] : (tensor<10xi64>) -> tensor<10x5xi64>
// CHECK-NEXT: %[[V7:[^ ,:)]+]] = stablehlo.iota dim = 1 : tensor<10x5xi64>
// CHECK-NEXT: %[[V8:[^ ,:)]+]] = stablehlo.add %[[V6]], %[[V7]] : tensor<10x5xi64>
// CHECK-NEXT: %[[V9:[^ ,:)]+]] = stablehlo.multiply %[[V8]], %[[V2]] : tensor<10x5xi64>
// CHECK-NEXT: %[[V10:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[V9]], dims = [0, 1] : (tensor<10x5xi64>) -> tensor<10x5x5x1xi64>
// CHECK-NEXT: %[[V11:[^ ,:)]+]] = stablehlo.iota dim = 2 : tensor<10x5x5x1xi64>
// CHECK-NEXT: %[[V12:[^ ,:)]+]] = stablehlo.add %[[V10]], %[[V11]] : tensor<10x5x5x1xi64>
// CHECK-NEXT: %[[V13:[^ ,:)]+]] = "stablehlo.scatter"(%[[V0]], %[[V12]], %[[V1]]) <{scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 3>}> ({
// CHECK-NEXT: ^bb0(%[[V14:[^ ,:)]+]]: tensor<f64>, %[[V15:[^ ,:)]+]]: tensor<f64>):
// CHECK-NEXT: stablehlo.return %[[V15]] : tensor<f64>
// CHECK-NEXT: }) : (tensor<750xf64>, tensor<10x5x5x1xi64>, tensor<10x5x5xf64>) -> tensor<750xf64>
// CHECK-NEXT: return %[[V13]] : tensor<750xf64>
// CHECK-NEXT: }

// An accumulation: the second write reads the buffer only to compute what it
// writes back, so both scatters are kept.
func.func @accumulate(%x: tensor<16xf64>, %u: tensor<4xf64>, %v: tensor<4xf64>) -> tensor<16xf64> {
  %i = stablehlo.iota dim = 0 : tensor<4x1xi64>
  %c4 = stablehlo.constant dense<4> : tensor<4x1xi64>
  %j = stablehlo.add %i, %c4 : tensor<4x1xi64>
  %r0 = "stablehlo.scatter"(%x, %i, %u) <{scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>}> ({
  ^bb0(%a: tensor<f64>, %b: tensor<f64>):
    stablehlo.return %b : tensor<f64>
  }) : (tensor<16xf64>, tensor<4x1xi64>, tensor<4xf64>) -> tensor<16xf64>
  %old = stablehlo.slice %r0 [4:8] : (tensor<16xf64>) -> tensor<4xf64>
  %new = stablehlo.add %old, %v : tensor<4xf64>
  %r1 = "stablehlo.scatter"(%r0, %j, %new) <{scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>}> ({
  ^bb0(%a: tensor<f64>, %b: tensor<f64>):
    stablehlo.return %b : tensor<f64>
  }) : (tensor<16xf64>, tensor<4x1xi64>, tensor<4xf64>) -> tensor<16xf64>
  return %r1 : tensor<16xf64>
}

// CHECK-LABEL: func.func @accumulate(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<16xf64>, %[[V1:[^ ,:)]+]]: tensor<4xf64>, %[[V2:[^ ,:)]+]]: tensor<4xf64>) -> tensor<16xf64> {
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.constant dense<4> : tensor<4x1xi64>
// CHECK-NEXT: %[[V4:[^ ,:)]+]] = stablehlo.iota dim = 0 : tensor<4x1xi64>
// CHECK-NEXT: %[[V5:[^ ,:)]+]] = stablehlo.add %[[V4]], %[[V3]] : tensor<4x1xi64>
// CHECK-NEXT: %[[V6:[^ ,:)]+]] = "stablehlo.scatter"(%[[V0]], %[[V4]], %[[V1]]) <{scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>}> ({
// CHECK-NEXT: ^bb0(%[[V7:[^ ,:)]+]]: tensor<f64>, %[[V8:[^ ,:)]+]]: tensor<f64>):
// CHECK-NEXT: stablehlo.return %[[V8]] : tensor<f64>
// CHECK-NEXT: }) : (tensor<16xf64>, tensor<4x1xi64>, tensor<4xf64>) -> tensor<16xf64>
// CHECK-NEXT: %[[V9:[^ ,:)]+]] = stablehlo.slice %[[V6]] [4:8] : (tensor<16xf64>) -> tensor<4xf64>
// CHECK-NEXT: %[[V10:[^ ,:)]+]] = stablehlo.add %[[V9]], %[[V2]] : tensor<4xf64>
// CHECK-NEXT: %[[V11:[^ ,:)]+]] = "stablehlo.scatter"(%[[V6]], %[[V5]], %[[V10]]) <{scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>}> ({
// CHECK-NEXT: ^bb0(%[[V7]]: tensor<f64>, %[[V8]]: tensor<f64>):
// CHECK-NEXT: stablehlo.return %[[V8]] : tensor<f64>
// CHECK-NEXT: }) : (tensor<16xf64>, tensor<4x1xi64>, tensor<4xf64>) -> tensor<16xf64>
// CHECK-NEXT: return %[[V11]] : tensor<16xf64>
// CHECK-NEXT: }

// The written buffer is read back for another result: an update slice, whose
// slice forwards the write.
func.func @read_back(%x: tensor<16xf64>, %u: tensor<4xf64>) -> (tensor<16xf64>, tensor<4xf64>) {
  %i = stablehlo.iota dim = 0 : tensor<4x1xi64>
  %r = "stablehlo.scatter"(%x, %i, %u) <{scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>}> ({
  ^bb0(%a: tensor<f64>, %b: tensor<f64>):
    stablehlo.return %b : tensor<f64>
  }) : (tensor<16xf64>, tensor<4x1xi64>, tensor<4xf64>) -> tensor<16xf64>
  %s = stablehlo.slice %r [0:4] : (tensor<16xf64>) -> tensor<4xf64>
  %t = stablehlo.multiply %s, %s : tensor<4xf64>
  return %r, %t : tensor<16xf64>, tensor<4xf64>
}

// CHECK-LABEL: func.func @read_back(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<16xf64>, %[[V1:[^ ,:)]+]]: tensor<4xf64>) -> (tensor<16xf64>, tensor<4xf64>) {
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = stablehlo.constant dense<0> : tensor<i32>
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.reshape %[[V1]] : (tensor<4xf64>) -> tensor<4x1xf64>
// CHECK-NEXT: %[[V4:[^ ,:)]+]] = stablehlo.transpose %[[V3]], dims = [0, 1] : (tensor<4x1xf64>) -> tensor<4x1xf64>
// CHECK-NEXT: %[[V5:[^ ,:)]+]] = stablehlo.reshape %[[V4]] : (tensor<4x1xf64>) -> tensor<4xf64>
// CHECK-NEXT: %[[V6:[^ ,:)]+]] = stablehlo.dynamic_update_slice %[[V0]], %[[V5]], %[[V2]] : (tensor<16xf64>, tensor<4xf64>, tensor<i32>) -> tensor<16xf64>
// CHECK-NEXT: %[[V7:[^ ,:)]+]] = stablehlo.slice %[[V6]] [0:4] : (tensor<16xf64>) -> tensor<4xf64>
// CHECK-NEXT: %[[V8:[^ ,:)]+]] = stablehlo.multiply %[[V7]], %[[V7]] : tensor<4xf64>
// CHECK-NEXT: return %[[V6]], %[[V8]] : tensor<16xf64>, tensor<4xf64>
// CHECK-NEXT: }
