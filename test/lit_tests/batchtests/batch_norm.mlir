// RUN: enzymexlamlir-opt %s --enzyme-batch --enzyme-hlo-opt | FileCheck %s
// RUN: enzymexlamlir-opt %s --enzyme-batch --enzyme-hlo-opt --stablehlo-legalize-to-hlo --mhlo-test-unfuse-batch-norm --hlo-legalize-to-stablehlo | stablehlo-translate - --interpret --allow-unregistered-dialect

func.func @bn_infer(%x: tensor<2x2xf32>, %scale: tensor<2xf32>, %offset: tensor<2xf32>, %mean: tensor<2xf32>, %var: tensor<2xf32>) -> tensor<2x2xf32> {
  %0 = "stablehlo.batch_norm_inference"(%x, %scale, %offset, %mean, %var) <{epsilon = 1.000000e-05 : f32, feature_index = 1 : i64}> : (tensor<2x2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>) -> tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
}

func.func @bn_train(%x: tensor<3x2xf32>, %scale: tensor<2xf32>, %offset: tensor<2xf32>) -> (tensor<3x2xf32>, tensor<2xf32>, tensor<2xf32>) {
  %0:3 = "stablehlo.batch_norm_training"(%x, %scale, %offset) <{epsilon = 1.000000e-05 : f32, feature_index = 1 : i64}> : (tensor<3x2xf32>, tensor<2xf32>, tensor<2xf32>) -> (tensor<3x2xf32>, tensor<2xf32>, tensor<2xf32>)
  return %0#0, %0#1, %0#2 : tensor<3x2xf32>, tensor<2xf32>, tensor<2xf32>
}

func.func @bn_grad(%x: tensor<3x2xf32>, %scale: tensor<2xf32>, %mean: tensor<2xf32>, %var: tensor<2xf32>, %dy: tensor<3x2xf32>) -> (tensor<3x2xf32>, tensor<2xf32>, tensor<2xf32>) {
  %0:3 = "stablehlo.batch_norm_grad"(%x, %scale, %mean, %var, %dy) <{epsilon = 1.000000e-05 : f32, feature_index = 1 : i64}> : (tensor<3x2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<3x2xf32>) -> (tensor<3x2xf32>, tensor<2xf32>, tensor<2xf32>)
  return %0#0, %0#1, %0#2 : tensor<3x2xf32>, tensor<2xf32>, tensor<2xf32>
}

func.func @call_bn_grad(%x: tensor<2x3x2xf32>, %scale: tensor<2x2xf32>, %mean: tensor<2x2xf32>, %var: tensor<2x2xf32>, %dy: tensor<2x3x2xf32>) -> (tensor<2x3x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>) {
  %0:3 = enzyme.batch @bn_grad(%x, %scale, %mean, %var, %dy) <{batch_shape = array<i64: 2>}> : (tensor<2x3x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x3x2xf32>) -> (tensor<2x3x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>)
  return %0#0, %0#1, %0#2 : tensor<2x3x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>
}

func.func @main() {
  // 1. BatchNormInference with per-batch varying scale/offset/mean/var
  %xi = stablehlo.constant dense<[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]> : tensor<2x2x2xf32>
  %scale = stablehlo.constant dense<[[1.5, 2.0], [0.5, 3.0]]> : tensor<2x2xf32>
  %offset = stablehlo.constant dense<[[0.1, -0.2], [1.0, 0.5]]> : tensor<2x2xf32>
  %mean = stablehlo.constant dense<[[2.0, 3.0], [6.0, 7.0]]> : tensor<2x2xf32>
  %var = stablehlo.constant dense<[[1.0, 4.0], [0.25, 1.0]]> : tensor<2x2xf32>

  %bi = enzyme.batch @bn_infer(%xi, %scale, %offset, %mean, %var) <{batch_shape = array<i64: 2>}> : (tensor<2x2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2x2xf32>

  %xi0 = stablehlo.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>
  %sc0 = stablehlo.constant dense<[1.5, 2.0]> : tensor<2xf32>
  %of0 = stablehlo.constant dense<[0.1, -0.2]> : tensor<2xf32>
  %mn0 = stablehlo.constant dense<[2.0, 3.0]> : tensor<2xf32>
  %vr0 = stablehlo.constant dense<[1.0, 4.0]> : tensor<2xf32>
  %ref_i0 = func.call @bn_infer(%xi0, %sc0, %of0, %mn0, %vr0) : (tensor<2x2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>) -> tensor<2x2xf32>

  %xi1 = stablehlo.constant dense<[[5.0, 6.0], [7.0, 8.0]]> : tensor<2x2xf32>
  %sc1 = stablehlo.constant dense<[0.5, 3.0]> : tensor<2xf32>
  %of1 = stablehlo.constant dense<[1.0, 0.5]> : tensor<2xf32>
  %mn1 = stablehlo.constant dense<[6.0, 7.0]> : tensor<2xf32>
  %vr1 = stablehlo.constant dense<[0.25, 1.0]> : tensor<2xf32>
  %ref_i1 = func.call @bn_infer(%xi1, %sc1, %of1, %mn1, %vr1) : (tensor<2x2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>) -> tensor<2x2xf32>

  %got_i0_s = stablehlo.slice %bi [0:1, 0:2, 0:2] : (tensor<2x2x2xf32>) -> tensor<1x2x2xf32>
  %got_i0 = stablehlo.reshape %got_i0_s : (tensor<1x2x2xf32>) -> tensor<2x2xf32>
  %got_i1_s = stablehlo.slice %bi [1:2, 0:2, 0:2] : (tensor<2x2x2xf32>) -> tensor<1x2x2xf32>
  %got_i1 = stablehlo.reshape %got_i1_s : (tensor<1x2x2xf32>) -> tensor<2x2xf32>
  "check.expect_almost_eq"(%got_i0, %ref_i0) : (tensor<2x2xf32>, tensor<2x2xf32>) -> ()
  "check.expect_almost_eq"(%got_i1, %ref_i1) : (tensor<2x2xf32>, tensor<2x2xf32>) -> ()

  // 2. BatchNormTraining with per-batch varying scale/offset
  %xt = stablehlo.constant dense<[[[1.0, 2.0], [3.0, 6.0], [5.0, 10.0]], [[-2.0, 4.0], [0.0, 8.0], [8.0, 12.0]]]> : tensor<2x3x2xf32>
  %bt:3 = enzyme.batch @bn_train(%xt, %scale, %offset) <{batch_shape = array<i64: 2>}> : (tensor<2x3x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>) -> (tensor<2x3x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>)

  %xt0 = stablehlo.constant dense<[[1.0, 2.0], [3.0, 6.0], [5.0, 10.0]]> : tensor<3x2xf32>
  %xt1 = stablehlo.constant dense<[[-2.0, 4.0], [0.0, 8.0], [8.0, 12.0]]> : tensor<3x2xf32>
  %ref_t0:3 = func.call @bn_train(%xt0, %sc0, %of0) : (tensor<3x2xf32>, tensor<2xf32>, tensor<2xf32>) -> (tensor<3x2xf32>, tensor<2xf32>, tensor<2xf32>)
  %ref_t1:3 = func.call @bn_train(%xt1, %sc1, %of1) : (tensor<3x2xf32>, tensor<2xf32>, tensor<2xf32>) -> (tensor<3x2xf32>, tensor<2xf32>, tensor<2xf32>)

  %got_t0_s = stablehlo.slice %bt#0 [0:1, 0:3, 0:2] : (tensor<2x3x2xf32>) -> tensor<1x3x2xf32>
  %got_t0 = stablehlo.reshape %got_t0_s : (tensor<1x3x2xf32>) -> tensor<3x2xf32>
  %got_t1_s = stablehlo.slice %bt#0 [1:2, 0:3, 0:2] : (tensor<2x3x2xf32>) -> tensor<1x3x2xf32>
  %got_t1 = stablehlo.reshape %got_t1_s : (tensor<1x3x2xf32>) -> tensor<3x2xf32>
  "check.expect_almost_eq"(%got_t0, %ref_t0#0) : (tensor<3x2xf32>, tensor<3x2xf32>) -> ()
  "check.expect_almost_eq"(%got_t1, %ref_t1#0) : (tensor<3x2xf32>, tensor<3x2xf32>) -> ()

  %got_tm0_s = stablehlo.slice %bt#1 [0:1, 0:2] : (tensor<2x2xf32>) -> tensor<1x2xf32>
  %got_tm0 = stablehlo.reshape %got_tm0_s : (tensor<1x2xf32>) -> tensor<2xf32>
  %got_tv0_s = stablehlo.slice %bt#2 [0:1, 0:2] : (tensor<2x2xf32>) -> tensor<1x2xf32>
  %got_tv0 = stablehlo.reshape %got_tv0_s : (tensor<1x2xf32>) -> tensor<2xf32>
  "check.expect_almost_eq"(%got_tm0, %ref_t0#1) : (tensor<2xf32>, tensor<2xf32>) -> ()
  "check.expect_almost_eq"(%got_tv0, %ref_t0#2) : (tensor<2xf32>, tensor<2xf32>) -> ()

  %got_tm1_s = stablehlo.slice %bt#1 [1:2, 0:2] : (tensor<2x2xf32>) -> tensor<1x2xf32>
  %got_tm1 = stablehlo.reshape %got_tm1_s : (tensor<1x2xf32>) -> tensor<2xf32>
  %got_tv1_s = stablehlo.slice %bt#2 [1:2, 0:2] : (tensor<2x2xf32>) -> tensor<1x2xf32>
  %got_tv1 = stablehlo.reshape %got_tv1_s : (tensor<1x2xf32>) -> tensor<2xf32>
  "check.expect_almost_eq"(%got_tm1, %ref_t1#1) : (tensor<2xf32>, tensor<2xf32>) -> ()
  "check.expect_almost_eq"(%got_tv1, %ref_t1#2) : (tensor<2xf32>, tensor<2xf32>) -> ()
  return
}

// CHECK-LABEL: func.func private @batched_bn_grad(
// CHECK-NEXT: %[[TX:.+]] = stablehlo.transpose %arg0, dims = [1, 0, 2]
// CHECK-NEXT: %[[RX:.+]] = stablehlo.reshape %[[TX]] : (tensor<3x2x2xf32>) -> tensor<3x4xf32>
// CHECK-NEXT: %[[RS:.+]] = stablehlo.reshape %arg1 : (tensor<2x2xf32>) -> tensor<4xf32>
// CHECK-NEXT: %[[RM:.+]] = stablehlo.reshape %arg2 : (tensor<2x2xf32>) -> tensor<4xf32>
// CHECK-NEXT: %[[RV:.+]] = stablehlo.reshape %arg3 : (tensor<2x2xf32>) -> tensor<4xf32>
// CHECK-NEXT: %[[TDY:.+]] = stablehlo.transpose %arg4, dims = [1, 0, 2]
// CHECK-NEXT: %[[RDY:.+]] = stablehlo.reshape %[[TDY]] : (tensor<3x2x2xf32>) -> tensor<3x4xf32>
// CHECK-NEXT: "stablehlo.batch_norm_grad"(%[[RX]], %[[RS]], %[[RM]], %[[RV]], %[[RDY]])

// CHECK-LABEL: func.func private @batched_bn_infer(
// CHECK: "stablehlo.batch_norm_inference"
// CHECK-NOT: stablehlo.while

// CHECK-LABEL: func.func private @batched_bn_train(
// CHECK: "stablehlo.batch_norm_training"
// CHECK-NOT: stablehlo.while
