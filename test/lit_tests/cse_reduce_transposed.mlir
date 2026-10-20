// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=cse_reduce_transposed" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

// Two reductions of the same product of B and x, each read through
// broadcasts of a different layout (the second's input is a transpose of
// the first's, the reduced dimension staying put): the second is a
// transpose of the first.
func.func @transposed(%B: tensor<3x4xf64>, %x: tensor<8x4x3xf64>) -> (tensor<8x4x4xf64>, tensor<4x8x4xf64>) {
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
  %b0 = stablehlo.broadcast_in_dim %B, dims = [1, 3] : (tensor<3x4xf64>) -> tensor<8x3x4x4xf64>
  %x0 = stablehlo.broadcast_in_dim %x, dims = [0, 2, 1] : (tensor<8x4x3xf64>) -> tensor<8x3x4x4xf64>
  %m0 = stablehlo.multiply %b0, %x0 : tensor<8x3x4x4xf64>
  %r0 = stablehlo.reduce(%m0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<8x3x4x4xf64>, tensor<f64>) -> tensor<8x4x4xf64>
  %b1 = stablehlo.broadcast_in_dim %B, dims = [1, 0] : (tensor<3x4xf64>) -> tensor<4x3x8x4xf64>
  %x1 = stablehlo.broadcast_in_dim %x, dims = [2, 3, 1] : (tensor<8x4x3xf64>) -> tensor<4x3x8x4xf64>
  %m1 = stablehlo.multiply %b1, %x1 : tensor<4x3x8x4xf64>
  %r1 = stablehlo.reduce(%m1 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<4x3x8x4xf64>, tensor<f64>) -> tensor<4x8x4xf64>
  return %r0, %r1 : tensor<8x4x4xf64>, tensor<4x8x4xf64>
}

// CHECK-LABEL: func.func @transposed(
// CHECK-SAME: %[[B:[^ ,:)]+]]: tensor<3x4xf64>, %[[X:[^ ,:)]+]]: tensor<8x4x3xf64>) -> (tensor<8x4x4xf64>, tensor<4x8x4xf64>) {
// CHECK-NEXT: %[[Z:[^ ,:)]+]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT: %[[B0:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[B]], dims = [1, 3] : (tensor<3x4xf64>) -> tensor<8x3x4x4xf64>
// CHECK-NEXT: %[[X0:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[X]], dims = [0, 2, 1] : (tensor<8x4x3xf64>) -> tensor<8x3x4x4xf64>
// CHECK-NEXT: %[[M:[^ ,:)]+]] = stablehlo.multiply %[[B0]], %[[X0]] : tensor<8x3x4x4xf64>
// CHECK-NEXT: %[[R:[^ ,:)]+]] = stablehlo.reduce(%[[M]] init: %[[Z]]) applies stablehlo.add across dimensions = [1] : (tensor<8x3x4x4xf64>, tensor<f64>) -> tensor<8x4x4xf64>
// CHECK-NEXT: %[[T:[^ ,:)]+]] = stablehlo.transpose %[[R]], dims = [2, 0, 1] : (tensor<8x4x4xf64>) -> tensor<4x8x4xf64>
// CHECK-NEXT: return %[[R]], %[[T]] : tensor<8x4x4xf64>, tensor<4x8x4xf64>
// CHECK-NEXT: }

// The same values read with x's dimension of 4 and B's column coinciding
// in one input (an elementwise product along it) and separate in the
// other (an outer product): different computations, left alone.
func.func @different(%B: tensor<3x4xf64>, %x: tensor<3x1x8x4x3xf64>) -> (tensor<3x8x4x4xf64>, tensor<3x8x4x4xf64>) {
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
  %b0 = stablehlo.broadcast_in_dim %B, dims = [1, 3] : (tensor<3x4xf64>) -> tensor<3x3x8x4x4xf64>
  %x0 = stablehlo.broadcast_in_dim %x, dims = [0, 3, 2, 4, 1] : (tensor<3x1x8x4x3xf64>) -> tensor<3x3x8x4x4xf64>
  %m0 = stablehlo.multiply %b0, %x0 : tensor<3x3x8x4x4xf64>
  %r0 = stablehlo.reduce(%m0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<3x3x8x4x4xf64>, tensor<f64>) -> tensor<3x8x4x4xf64>
  %x1 = stablehlo.broadcast_in_dim %x, dims = [0, 4, 2, 3, 1] : (tensor<3x1x8x4x3xf64>) -> tensor<3x3x8x4x4xf64>
  %m1 = stablehlo.multiply %b0, %x1 : tensor<3x3x8x4x4xf64>
  %r1 = stablehlo.reduce(%m1 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<3x3x8x4x4xf64>, tensor<f64>) -> tensor<3x8x4x4xf64>
  return %r0, %r1 : tensor<3x8x4x4xf64>, tensor<3x8x4x4xf64>
}

// CHECK-LABEL: func.func @different(
// CHECK-SAME: %[[B:[^ ,:)]+]]: tensor<3x4xf64>, %[[X:[^ ,:)]+]]: tensor<3x1x8x4x3xf64>) -> (tensor<3x8x4x4xf64>, tensor<3x8x4x4xf64>) {
// CHECK-NEXT: %[[Z:[^ ,:)]+]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT: %[[B0:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[B]], dims = [1, 3] : (tensor<3x4xf64>) -> tensor<3x3x8x4x4xf64>
// CHECK-NEXT: %[[X0:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[X]], dims = [0, 3, 2, 4, 1] : (tensor<3x1x8x4x3xf64>) -> tensor<3x3x8x4x4xf64>
// CHECK-NEXT: %[[M0:[^ ,:)]+]] = stablehlo.multiply %[[B0]], %[[X0]] : tensor<3x3x8x4x4xf64>
// CHECK-NEXT: %[[R0:[^ ,:)]+]] = stablehlo.reduce(%[[M0]] init: %[[Z]]) applies stablehlo.add across dimensions = [1] : (tensor<3x3x8x4x4xf64>, tensor<f64>) -> tensor<3x8x4x4xf64>
// CHECK-NEXT: %[[X1:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[X]], dims = [0, 4, 2, 3, 1] : (tensor<3x1x8x4x3xf64>) -> tensor<3x3x8x4x4xf64>
// CHECK-NEXT: %[[M1:[^ ,:)]+]] = stablehlo.multiply %[[B0]], %[[X1]] : tensor<3x3x8x4x4xf64>
// CHECK-NEXT: %[[R1:[^ ,:)]+]] = stablehlo.reduce(%[[M1]] init: %[[Z]]) applies stablehlo.add across dimensions = [1] : (tensor<3x3x8x4x4xf64>, tensor<f64>) -> tensor<3x8x4x4xf64>
// CHECK-NEXT: return %[[R0]], %[[R1]] : tensor<3x8x4x4xf64>, tensor<3x8x4x4xf64>
// CHECK-NEXT: }

// The same computation with %m0 squared 32 times: a tree walk of it has
// 2^32 nodes, the walk of the DAG 33. The second is a transpose of the first.
func.func @shared(%B: tensor<3x4xf64>, %x: tensor<8x4x3xf64>) -> (tensor<8x4x4xf64>, tensor<4x8x4xf64>) {
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
  %b0 = stablehlo.broadcast_in_dim %B, dims = [1, 3] : (tensor<3x4xf64>) -> tensor<8x3x4x4xf64>
  %x0 = stablehlo.broadcast_in_dim %x, dims = [0, 2, 1] : (tensor<8x4x3xf64>) -> tensor<8x3x4x4xf64>
  %m0 = stablehlo.multiply %b0, %x0 : tensor<8x3x4x4xf64>
  %s0_1 = stablehlo.multiply %m0, %m0 : tensor<8x3x4x4xf64>
  %s0_2 = stablehlo.multiply %s0_1, %s0_1 : tensor<8x3x4x4xf64>
  %s0_3 = stablehlo.multiply %s0_2, %s0_2 : tensor<8x3x4x4xf64>
  %s0_4 = stablehlo.multiply %s0_3, %s0_3 : tensor<8x3x4x4xf64>
  %s0_5 = stablehlo.multiply %s0_4, %s0_4 : tensor<8x3x4x4xf64>
  %s0_6 = stablehlo.multiply %s0_5, %s0_5 : tensor<8x3x4x4xf64>
  %s0_7 = stablehlo.multiply %s0_6, %s0_6 : tensor<8x3x4x4xf64>
  %s0_8 = stablehlo.multiply %s0_7, %s0_7 : tensor<8x3x4x4xf64>
  %s0_9 = stablehlo.multiply %s0_8, %s0_8 : tensor<8x3x4x4xf64>
  %s0_10 = stablehlo.multiply %s0_9, %s0_9 : tensor<8x3x4x4xf64>
  %s0_11 = stablehlo.multiply %s0_10, %s0_10 : tensor<8x3x4x4xf64>
  %s0_12 = stablehlo.multiply %s0_11, %s0_11 : tensor<8x3x4x4xf64>
  %s0_13 = stablehlo.multiply %s0_12, %s0_12 : tensor<8x3x4x4xf64>
  %s0_14 = stablehlo.multiply %s0_13, %s0_13 : tensor<8x3x4x4xf64>
  %s0_15 = stablehlo.multiply %s0_14, %s0_14 : tensor<8x3x4x4xf64>
  %s0_16 = stablehlo.multiply %s0_15, %s0_15 : tensor<8x3x4x4xf64>
  %s0_17 = stablehlo.multiply %s0_16, %s0_16 : tensor<8x3x4x4xf64>
  %s0_18 = stablehlo.multiply %s0_17, %s0_17 : tensor<8x3x4x4xf64>
  %s0_19 = stablehlo.multiply %s0_18, %s0_18 : tensor<8x3x4x4xf64>
  %s0_20 = stablehlo.multiply %s0_19, %s0_19 : tensor<8x3x4x4xf64>
  %s0_21 = stablehlo.multiply %s0_20, %s0_20 : tensor<8x3x4x4xf64>
  %s0_22 = stablehlo.multiply %s0_21, %s0_21 : tensor<8x3x4x4xf64>
  %s0_23 = stablehlo.multiply %s0_22, %s0_22 : tensor<8x3x4x4xf64>
  %s0_24 = stablehlo.multiply %s0_23, %s0_23 : tensor<8x3x4x4xf64>
  %s0_25 = stablehlo.multiply %s0_24, %s0_24 : tensor<8x3x4x4xf64>
  %s0_26 = stablehlo.multiply %s0_25, %s0_25 : tensor<8x3x4x4xf64>
  %s0_27 = stablehlo.multiply %s0_26, %s0_26 : tensor<8x3x4x4xf64>
  %s0_28 = stablehlo.multiply %s0_27, %s0_27 : tensor<8x3x4x4xf64>
  %s0_29 = stablehlo.multiply %s0_28, %s0_28 : tensor<8x3x4x4xf64>
  %s0_30 = stablehlo.multiply %s0_29, %s0_29 : tensor<8x3x4x4xf64>
  %s0_31 = stablehlo.multiply %s0_30, %s0_30 : tensor<8x3x4x4xf64>
  %s0_32 = stablehlo.multiply %s0_31, %s0_31 : tensor<8x3x4x4xf64>
  %r0 = stablehlo.reduce(%s0_32 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<8x3x4x4xf64>, tensor<f64>) -> tensor<8x4x4xf64>
  %b1 = stablehlo.broadcast_in_dim %B, dims = [1, 0] : (tensor<3x4xf64>) -> tensor<4x3x8x4xf64>
  %x1 = stablehlo.broadcast_in_dim %x, dims = [2, 3, 1] : (tensor<8x4x3xf64>) -> tensor<4x3x8x4xf64>
  %m1 = stablehlo.multiply %b1, %x1 : tensor<4x3x8x4xf64>
  %s1_1 = stablehlo.multiply %m1, %m1 : tensor<4x3x8x4xf64>
  %s1_2 = stablehlo.multiply %s1_1, %s1_1 : tensor<4x3x8x4xf64>
  %s1_3 = stablehlo.multiply %s1_2, %s1_2 : tensor<4x3x8x4xf64>
  %s1_4 = stablehlo.multiply %s1_3, %s1_3 : tensor<4x3x8x4xf64>
  %s1_5 = stablehlo.multiply %s1_4, %s1_4 : tensor<4x3x8x4xf64>
  %s1_6 = stablehlo.multiply %s1_5, %s1_5 : tensor<4x3x8x4xf64>
  %s1_7 = stablehlo.multiply %s1_6, %s1_6 : tensor<4x3x8x4xf64>
  %s1_8 = stablehlo.multiply %s1_7, %s1_7 : tensor<4x3x8x4xf64>
  %s1_9 = stablehlo.multiply %s1_8, %s1_8 : tensor<4x3x8x4xf64>
  %s1_10 = stablehlo.multiply %s1_9, %s1_9 : tensor<4x3x8x4xf64>
  %s1_11 = stablehlo.multiply %s1_10, %s1_10 : tensor<4x3x8x4xf64>
  %s1_12 = stablehlo.multiply %s1_11, %s1_11 : tensor<4x3x8x4xf64>
  %s1_13 = stablehlo.multiply %s1_12, %s1_12 : tensor<4x3x8x4xf64>
  %s1_14 = stablehlo.multiply %s1_13, %s1_13 : tensor<4x3x8x4xf64>
  %s1_15 = stablehlo.multiply %s1_14, %s1_14 : tensor<4x3x8x4xf64>
  %s1_16 = stablehlo.multiply %s1_15, %s1_15 : tensor<4x3x8x4xf64>
  %s1_17 = stablehlo.multiply %s1_16, %s1_16 : tensor<4x3x8x4xf64>
  %s1_18 = stablehlo.multiply %s1_17, %s1_17 : tensor<4x3x8x4xf64>
  %s1_19 = stablehlo.multiply %s1_18, %s1_18 : tensor<4x3x8x4xf64>
  %s1_20 = stablehlo.multiply %s1_19, %s1_19 : tensor<4x3x8x4xf64>
  %s1_21 = stablehlo.multiply %s1_20, %s1_20 : tensor<4x3x8x4xf64>
  %s1_22 = stablehlo.multiply %s1_21, %s1_21 : tensor<4x3x8x4xf64>
  %s1_23 = stablehlo.multiply %s1_22, %s1_22 : tensor<4x3x8x4xf64>
  %s1_24 = stablehlo.multiply %s1_23, %s1_23 : tensor<4x3x8x4xf64>
  %s1_25 = stablehlo.multiply %s1_24, %s1_24 : tensor<4x3x8x4xf64>
  %s1_26 = stablehlo.multiply %s1_25, %s1_25 : tensor<4x3x8x4xf64>
  %s1_27 = stablehlo.multiply %s1_26, %s1_26 : tensor<4x3x8x4xf64>
  %s1_28 = stablehlo.multiply %s1_27, %s1_27 : tensor<4x3x8x4xf64>
  %s1_29 = stablehlo.multiply %s1_28, %s1_28 : tensor<4x3x8x4xf64>
  %s1_30 = stablehlo.multiply %s1_29, %s1_29 : tensor<4x3x8x4xf64>
  %s1_31 = stablehlo.multiply %s1_30, %s1_30 : tensor<4x3x8x4xf64>
  %s1_32 = stablehlo.multiply %s1_31, %s1_31 : tensor<4x3x8x4xf64>
  %r1 = stablehlo.reduce(%s1_32 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<4x3x8x4xf64>, tensor<f64>) -> tensor<4x8x4xf64>
  return %r0, %r1 : tensor<8x4x4xf64>, tensor<4x8x4xf64>
}

// CHECK-LABEL: func.func @shared(
// CHECK-SAME: %[[B:[^ ,:)]+]]: tensor<3x4xf64>, %[[X:[^ ,:)]+]]: tensor<8x4x3xf64>) -> (tensor<8x4x4xf64>, tensor<4x8x4xf64>) {
// CHECK-NEXT: %[[Z:[^ ,:)]+]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT: %[[B0:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[B]], dims = [1, 3] : (tensor<3x4xf64>) -> tensor<8x3x4x4xf64>
// CHECK-NEXT: %[[X0:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[X]], dims = [0, 2, 1] : (tensor<8x4x3xf64>) -> tensor<8x3x4x4xf64>
// CHECK-NEXT: %[[M:[^ ,:)]+]] = stablehlo.multiply %[[B0]], %[[X0]] : tensor<8x3x4x4xf64>
// CHECK-NEXT: %[[S1:[^ ,:)]+]] = stablehlo.multiply %[[M]], %[[M]] : tensor<8x3x4x4xf64>
// CHECK-NEXT: %[[S2:[^ ,:)]+]] = stablehlo.multiply %[[S1]], %[[S1]] : tensor<8x3x4x4xf64>
// CHECK-NEXT: %[[S3:[^ ,:)]+]] = stablehlo.multiply %[[S2]], %[[S2]] : tensor<8x3x4x4xf64>
// CHECK-NEXT: %[[S4:[^ ,:)]+]] = stablehlo.multiply %[[S3]], %[[S3]] : tensor<8x3x4x4xf64>
// CHECK-NEXT: %[[S5:[^ ,:)]+]] = stablehlo.multiply %[[S4]], %[[S4]] : tensor<8x3x4x4xf64>
// CHECK-NEXT: %[[S6:[^ ,:)]+]] = stablehlo.multiply %[[S5]], %[[S5]] : tensor<8x3x4x4xf64>
// CHECK-NEXT: %[[S7:[^ ,:)]+]] = stablehlo.multiply %[[S6]], %[[S6]] : tensor<8x3x4x4xf64>
// CHECK-NEXT: %[[S8:[^ ,:)]+]] = stablehlo.multiply %[[S7]], %[[S7]] : tensor<8x3x4x4xf64>
// CHECK-NEXT: %[[S9:[^ ,:)]+]] = stablehlo.multiply %[[S8]], %[[S8]] : tensor<8x3x4x4xf64>
// CHECK-NEXT: %[[S10:[^ ,:)]+]] = stablehlo.multiply %[[S9]], %[[S9]] : tensor<8x3x4x4xf64>
// CHECK-NEXT: %[[S11:[^ ,:)]+]] = stablehlo.multiply %[[S10]], %[[S10]] : tensor<8x3x4x4xf64>
// CHECK-NEXT: %[[S12:[^ ,:)]+]] = stablehlo.multiply %[[S11]], %[[S11]] : tensor<8x3x4x4xf64>
// CHECK-NEXT: %[[S13:[^ ,:)]+]] = stablehlo.multiply %[[S12]], %[[S12]] : tensor<8x3x4x4xf64>
// CHECK-NEXT: %[[S14:[^ ,:)]+]] = stablehlo.multiply %[[S13]], %[[S13]] : tensor<8x3x4x4xf64>
// CHECK-NEXT: %[[S15:[^ ,:)]+]] = stablehlo.multiply %[[S14]], %[[S14]] : tensor<8x3x4x4xf64>
// CHECK-NEXT: %[[S16:[^ ,:)]+]] = stablehlo.multiply %[[S15]], %[[S15]] : tensor<8x3x4x4xf64>
// CHECK-NEXT: %[[S17:[^ ,:)]+]] = stablehlo.multiply %[[S16]], %[[S16]] : tensor<8x3x4x4xf64>
// CHECK-NEXT: %[[S18:[^ ,:)]+]] = stablehlo.multiply %[[S17]], %[[S17]] : tensor<8x3x4x4xf64>
// CHECK-NEXT: %[[S19:[^ ,:)]+]] = stablehlo.multiply %[[S18]], %[[S18]] : tensor<8x3x4x4xf64>
// CHECK-NEXT: %[[S20:[^ ,:)]+]] = stablehlo.multiply %[[S19]], %[[S19]] : tensor<8x3x4x4xf64>
// CHECK-NEXT: %[[S21:[^ ,:)]+]] = stablehlo.multiply %[[S20]], %[[S20]] : tensor<8x3x4x4xf64>
// CHECK-NEXT: %[[S22:[^ ,:)]+]] = stablehlo.multiply %[[S21]], %[[S21]] : tensor<8x3x4x4xf64>
// CHECK-NEXT: %[[S23:[^ ,:)]+]] = stablehlo.multiply %[[S22]], %[[S22]] : tensor<8x3x4x4xf64>
// CHECK-NEXT: %[[S24:[^ ,:)]+]] = stablehlo.multiply %[[S23]], %[[S23]] : tensor<8x3x4x4xf64>
// CHECK-NEXT: %[[S25:[^ ,:)]+]] = stablehlo.multiply %[[S24]], %[[S24]] : tensor<8x3x4x4xf64>
// CHECK-NEXT: %[[S26:[^ ,:)]+]] = stablehlo.multiply %[[S25]], %[[S25]] : tensor<8x3x4x4xf64>
// CHECK-NEXT: %[[S27:[^ ,:)]+]] = stablehlo.multiply %[[S26]], %[[S26]] : tensor<8x3x4x4xf64>
// CHECK-NEXT: %[[S28:[^ ,:)]+]] = stablehlo.multiply %[[S27]], %[[S27]] : tensor<8x3x4x4xf64>
// CHECK-NEXT: %[[S29:[^ ,:)]+]] = stablehlo.multiply %[[S28]], %[[S28]] : tensor<8x3x4x4xf64>
// CHECK-NEXT: %[[S30:[^ ,:)]+]] = stablehlo.multiply %[[S29]], %[[S29]] : tensor<8x3x4x4xf64>
// CHECK-NEXT: %[[S31:[^ ,:)]+]] = stablehlo.multiply %[[S30]], %[[S30]] : tensor<8x3x4x4xf64>
// CHECK-NEXT: %[[S32:[^ ,:)]+]] = stablehlo.multiply %[[S31]], %[[S31]] : tensor<8x3x4x4xf64>
// CHECK-NEXT: %[[R:[^ ,:)]+]] = stablehlo.reduce(%[[S32]] init: %[[Z]]) applies stablehlo.add across dimensions = [1] : (tensor<8x3x4x4xf64>, tensor<f64>) -> tensor<8x4x4xf64>
// CHECK-NEXT: %[[T:[^ ,:)]+]] = stablehlo.transpose %[[R]], dims = [2, 0, 1] : (tensor<8x4x4xf64>) -> tensor<4x8x4xf64>
// CHECK-NEXT: return %[[R]], %[[T]] : tensor<8x4x4xf64>, tensor<4x8x4xf64>
// CHECK-NEXT: }

// The product's first leaf is a splat constant, read by other ops too. The
// second reduce is a transpose of the first.
func.func @constant_first(%x: tensor<8x4x3xf64>) -> (tensor<8x4x4xf64>, tensor<4x8x4xf64>, tensor<8x3x4x4xf64>) {
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
  %two = stablehlo.constant dense<2.000000e+00> : tensor<f64>
  %t0 = stablehlo.broadcast_in_dim %two, dims = [] : (tensor<f64>) -> tensor<8x3x4x4xf64>
  %x0 = stablehlo.broadcast_in_dim %x, dims = [0, 2, 1] : (tensor<8x4x3xf64>) -> tensor<8x3x4x4xf64>
  %m0 = stablehlo.multiply %t0, %x0 : tensor<8x3x4x4xf64>
  %r0 = stablehlo.reduce(%m0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<8x3x4x4xf64>, tensor<f64>) -> tensor<8x4x4xf64>
  %t1 = stablehlo.broadcast_in_dim %two, dims = [] : (tensor<f64>) -> tensor<4x3x8x4xf64>
  %x1 = stablehlo.broadcast_in_dim %x, dims = [2, 3, 1] : (tensor<8x4x3xf64>) -> tensor<4x3x8x4xf64>
  %m1 = stablehlo.multiply %t1, %x1 : tensor<4x3x8x4xf64>
  %r1 = stablehlo.reduce(%m1 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<4x3x8x4xf64>, tensor<f64>) -> tensor<4x8x4xf64>
  %u = stablehlo.add %t0, %t0 : tensor<8x3x4x4xf64>
  return %r0, %r1, %u : tensor<8x4x4xf64>, tensor<4x8x4xf64>, tensor<8x3x4x4xf64>
}

// CHECK-LABEL: func.func @constant_first(
// CHECK-SAME: %[[X:[^ ,:)]+]]: tensor<8x4x3xf64>) -> (tensor<8x4x4xf64>, tensor<4x8x4xf64>, tensor<8x3x4x4xf64>) {
// CHECK-NEXT: %[[Z:[^ ,:)]+]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT: %[[TWO:[^ ,:)]+]] = stablehlo.constant dense<2.000000e+00> : tensor<f64>
// CHECK-NEXT: %[[T0:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[TWO]], dims = [] : (tensor<f64>) -> tensor<8x3x4x4xf64>
// CHECK-NEXT: %[[X0:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[X]], dims = [0, 2, 1] : (tensor<8x4x3xf64>) -> tensor<8x3x4x4xf64>
// CHECK-NEXT: %[[M:[^ ,:)]+]] = stablehlo.multiply %[[T0]], %[[X0]] : tensor<8x3x4x4xf64>
// CHECK-NEXT: %[[R:[^ ,:)]+]] = stablehlo.reduce(%[[M]] init: %[[Z]]) applies stablehlo.add across dimensions = [1] : (tensor<8x3x4x4xf64>, tensor<f64>) -> tensor<8x4x4xf64>
// CHECK-NEXT: %[[T:[^ ,:)]+]] = stablehlo.transpose %[[R]], dims = [2, 0, 1] : (tensor<8x4x4xf64>) -> tensor<4x8x4xf64>
// CHECK-NEXT: %[[U:[^ ,:)]+]] = stablehlo.add %[[T0]], %[[T0]] : tensor<8x3x4x4xf64>
// CHECK-NEXT: return %[[R]], %[[T]], %[[U]] : tensor<8x4x4xf64>, tensor<4x8x4xf64>, tensor<8x3x4x4xf64>
// CHECK-NEXT: }
