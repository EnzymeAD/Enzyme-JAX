// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt | FileCheck %s

// Exact reproducer from Enzyme-JAX issue #867:
// sum(sum(x .^ 2; dims=1); dims=3) -> single dot_general contracting both dims + broadcast_in_dim.
// CHECK-LABEL: func.func @issue_867_reduce_reduce_dot(
// CHECK-NOT: stablehlo.reduce
// CHECK: %[[DOT:.*]] = stablehlo.dot_general %arg0, %arg0, batching_dims = [1] x [1], contracting_dims = [2, 0] x [2, 0] : (tensor<5x4x3xf64>, tensor<5x4x3xf64>) -> tensor<4xf64>
// CHECK-NEXT: %[[OUT:.*]] = stablehlo.reshape %[[DOT]] : (tensor<4xf64>) -> tensor<1x4x1xf64>
// CHECK-NEXT: return %[[OUT]] : tensor<1x4x1xf64>
func.func @issue_867_reduce_reduce_dot(%arg0: tensor<5x4x3xf64>) -> tensor<1x4x1xf64> {
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
  %0 = stablehlo.multiply %arg0, %arg0 : tensor<5x4x3xf64>
  %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<5x4x3xf64>, tensor<f64>) -> tensor<5x4xf64>
  %2 = stablehlo.transpose %1, dims = [1, 0] : (tensor<5x4xf64>) -> tensor<4x5xf64>
  %3 = stablehlo.reshape %2 : (tensor<4x5xf64>) -> tensor<1x4x5xf64>
  %4 = stablehlo.reduce(%3 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x4x5xf64>, tensor<f64>) -> tensor<1x4xf64>
  %5 = stablehlo.reshape %4 : (tensor<1x4xf64>) -> tensor<1x4x1xf64>
  return %5 : tensor<1x4x1xf64>
}

// General non-dot Reduce(Transpose + ExpandDims(Reduce(...))) merges into a single Reduce.
// CHECK-LABEL: func.func @reduce_transpose_expand_reduce(
// CHECK: %[[EXP:.*]] = stablehlo.exponential %arg0 : tensor<5x4x3xf64>
// CHECK-NEXT: %[[RED:.*]] = stablehlo.reduce(%[[EXP]] init: %{{.*}}) applies stablehlo.add across dimensions = [2, 0] : (tensor<5x4x3xf64>, tensor<f64>) -> tensor<4xf64>
// CHECK-NOT: stablehlo.reduce
// CHECK-NEXT: %[[OUT:.*]] = stablehlo.reshape %[[RED]] : (tensor<4xf64>) -> tensor<1x4x1xf64>
// CHECK-NEXT: return %[[OUT]] : tensor<1x4x1xf64>
func.func @reduce_transpose_expand_reduce(%arg0: tensor<5x4x3xf64>) -> tensor<1x4x1xf64> {
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
  %0 = stablehlo.exponential %arg0 : tensor<5x4x3xf64>
  %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<5x4x3xf64>, tensor<f64>) -> tensor<5x4xf64>
  %2 = stablehlo.transpose %1, dims = [1, 0] : (tensor<5x4xf64>) -> tensor<4x5xf64>
  %3 = stablehlo.reshape %2 : (tensor<4x5xf64>) -> tensor<1x4x5xf64>
  %4 = stablehlo.reduce(%3 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x4x5xf64>, tensor<f64>) -> tensor<1x4xf64>
  %5 = stablehlo.reshape %4 : (tensor<1x4xf64>) -> tensor<1x4x1xf64>
  return %5 : tensor<1x4x1xf64>
}

// Exact reproducer from Enzyme-JAX issue #1636:
// Slice of While(DUS) at final iteration [15:16, 0:12, 0:4] avoids caching tensor<16x12x4xf32>.
// CHECK-LABEL: func.func @issue_1636_slice_while_dus_last_iter(
// CHECK-NOT: tensor<16x12x4xf32>
// CHECK-NOT: stablehlo.dynamic_update_slice
// CHECK: stablehlo.while
// CHECK-NOT: stablehlo.dynamic_update_slice
// CHECK: return %{{.*}}, %arg5 : tensor<12x4xf32>, tensor<2xui64>
func.func @issue_1636_slice_while_dus_last_iter(%arg0: tensor<12x16x4xf32>, %arg1: tensor<4x4xf32>, %arg2: tensor<4x4xf32>, %arg3: tensor<4xf32>, %arg4: tensor<4xf32>, %arg5: tensor<2xui64>) -> (tensor<12x4xf32>, tensor<2xui64>) {
  %c = stablehlo.constant dense<0> : tensor<i32>
  %c_0 = stablehlo.constant dense<16> : tensor<i64>
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<16x12x4xf32>
  %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<4x12xf32>
  %c_2 = stablehlo.constant dense<1> : tensor<i32>
  %c_3 = stablehlo.constant dense<1> : tensor<i64>
  %c_4 = stablehlo.constant dense<0> : tensor<i64>
  %0 = stablehlo.broadcast_in_dim %arg4, dims = [0] : (tensor<4xf32>) -> tensor<4x12xf32>
  %1 = stablehlo.broadcast_in_dim %arg1, dims = [1, 2] : (tensor<4x4xf32>) -> tensor<16x4x4xf32>
  %2 = stablehlo.dot_general %1, %arg0, batching_dims = [0] x [1], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] : (tensor<16x4x4xf32>, tensor<12x16x4xf32>) -> tensor<16x4x12xf32>
  %3 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<4xf32>) -> tensor<16x4x12xf32>
  %4 = stablehlo.add %2, %3 : tensor<16x4x12xf32>
  %5:3 = stablehlo.while(%iterArg = %c_4, %iterArg_5 = %cst, %iterArg_6 = %cst_1) : tensor<i64>, tensor<16x12x4xf32>, tensor<4x12xf32> attributes {enzyme.disable_mincut}
  cond {
    %8 = stablehlo.compare LT, %iterArg, %c_0 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %8 : tensor<i1>
  } do {
    %8 = stablehlo.add %c_3, %iterArg : tensor<i64>
    %9 = stablehlo.convert %8 : (tensor<i64>) -> tensor<i32>
    %10 = stablehlo.subtract %9, %c_2 : tensor<i32>
    %11 = stablehlo.dot_general %arg2, %iterArg_6, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4x4xf32>, tensor<4x12xf32>) -> tensor<4x12xf32>
    %12 = stablehlo.add %11, %0 : tensor<4x12xf32>
    %13 = stablehlo.dynamic_slice %4, %iterArg, %c_4, %c_4, sizes = [1, 4, 12] : (tensor<16x4x12xf32>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x4x12xf32>
    %14 = stablehlo.reshape %13 : (tensor<1x4x12xf32>) -> tensor<4x12xf32>
    %15 = stablehlo.add %12, %14 : tensor<4x12xf32>
    %16 = stablehlo.tanh %15 : tensor<4x12xf32>
    %17 = stablehlo.broadcast_in_dim %16, dims = [2, 1] : (tensor<4x12xf32>) -> tensor<1x12x4xf32>
    %18 = stablehlo.dynamic_update_slice %iterArg_5, %17, %10, %c, %c : (tensor<16x12x4xf32>, tensor<1x12x4xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<16x12x4xf32>
    stablehlo.return %8, %18, %16 : tensor<i64>, tensor<16x12x4xf32>, tensor<4x12xf32>
  }
  %6 = stablehlo.slice %5#1 [15:16, 0:12, 0:4] : (tensor<16x12x4xf32>) -> tensor<1x12x4xf32>
  %7 = stablehlo.reshape %6 : (tensor<1x12x4xf32>) -> tensor<12x4xf32>
  return %7, %arg5 : tensor<12x4xf32>, tensor<2xui64>
}

// Slice of While(DUS) at an interior iteration [5:6, 0:8] carries tensor<1x8xf32> with select.
// CHECK-LABEL: func.func @slice_while_dus_mid_iter(
// CHECK-NOT: stablehlo.dynamic_update_slice
// CHECK: stablehlo.while({{.*}}) : tensor<i64>, tensor<1x8xf32>
// CHECK-NOT: stablehlo.dynamic_update_slice
// CHECK: stablehlo.compare EQ
// CHECK: stablehlo.select
// CHECK: return %{{.*}} : tensor<8xf32>
func.func @slice_while_dus_mid_iter(%arg0: tensor<16x8xf32>, %init: tensor<8xf32>) -> tensor<8xf32> {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %c16 = stablehlo.constant dense<16> : tensor<i64>
  %zeros = stablehlo.constant dense<0.000000e+00> : tensor<16x8xf32>
  %w:3 = stablehlo.while(%iv = %c0, %buf = %zeros, %acc = %init) : tensor<i64>, tensor<16x8xf32>, tensor<8xf32>
  cond {
    %cmp = stablehlo.compare LT, %iv, %c16 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %cmp : tensor<i1>
  } do {
    %next_iv = stablehlo.add %iv, %c1 : tensor<i64>
    %ds = stablehlo.dynamic_slice %arg0, %iv, %c0, sizes = [1, 8] : (tensor<16x8xf32>, tensor<i64>, tensor<i64>) -> tensor<1x8xf32>
    %row = stablehlo.reshape %ds : (tensor<1x8xf32>) -> tensor<8xf32>
    %next_acc = stablehlo.tanh %row : tensor<8xf32>
    %upd = stablehlo.reshape %next_acc : (tensor<8xf32>) -> tensor<1x8xf32>
    %next_buf = stablehlo.dynamic_update_slice %buf, %upd, %iv, %c0 : (tensor<16x8xf32>, tensor<1x8xf32>, tensor<i64>, tensor<i64>) -> tensor<16x8xf32>
    stablehlo.return %next_iv, %next_buf, %next_acc : tensor<i64>, tensor<16x8xf32>, tensor<8xf32>
  }
  %sl = stablehlo.slice %w#1 [5:6, 0:8] : (tensor<16x8xf32>) -> tensor<1x8xf32>
  %out = stablehlo.reshape %sl : (tensor<1x8xf32>) -> tensor<8xf32>
  return %out : tensor<8xf32>
}
