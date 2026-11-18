// RUN: enzymexlamlir-opt  -allow-unregistered-dialect --pass-pipeline="builtin.module(enzyme-hlo-opt{passses=67584})" %s | FileCheck %s

func.func @test_while_transpose_elimination(%arg0: tensor<2x3xf32>, %arg1: tensor<i64>) -> tensor<3x2xf32> {
  %cond = stablehlo.constant dense<true> : tensor<i1>
  
  %0:2 = "stablehlo.while"(%arg0, %arg1) ({
    ^bb0(%arg2: tensor<2x3xf32>, %arg3: tensor<i64>):
      stablehlo.return %cond : tensor<i1>
  }, {
    ^bb0(%arg2: tensor<2x3xf32>, %arg3: tensor<i64>):
      // Transpose input inside while body
      %transposed = "unregistered.custom_transform"(%arg2) {} : (tensor<2x3xf32>) -> tensor<3x2xf32>
      %transposed_back = stablehlo.transpose %transposed, dims = [1, 0] : (tensor<3x2xf32>) -> tensor<2x3xf32>
      
      // Do something with the iteration count
      %one = stablehlo.constant dense<1> : tensor<i64>
      %next_iter = stablehlo.add %arg3, %one : tensor<i64>
      
      // Yield the transposed value (which becomes a result of the while)
      stablehlo.return %transposed_back, %next_iter : tensor<2x3xf32>, tensor<i64>
  }) : (tensor<2x3xf32>, tensor<i64>) -> (tensor<2x3xf32>, tensor<i64>)
  
  // Transpose the while result back to the original shape (this should be eliminated by the pattern)
  %1 = stablehlo.transpose %0#0, dims = [1, 0] : (tensor<2x3xf32>) -> tensor<3x2xf32>
  
  return %1 : tensor<3x2xf32>
}

// CHECK-LABEL: func.func @test_while_transpose_elimination
// CHECK-SAME: (%[[ARG0:.*]]: tensor<2x3xf32>, %[[ARG1:.*]]: tensor<i64>) -> tensor<3x2xf32>
// CHECK-NEXT: %[[COND:.*]] = stablehlo.constant dense<true> : tensor<i1>
// CHECK-NEXT: %[[TRANSPOSE_BEFORE:.*]] = stablehlo.transpose %[[ARG0]], dims = [1, 0] : (tensor<2x3xf32>) -> tensor<3x2xf32>
// CHECK-NEXT: %[[WHILE_RESULT:.*]] = stablehlo.while(%[[ITER_ARG:.*]] = %[[TRANSPOSE_BEFORE]]) : tensor<3x2xf32>
// CHECK-NEXT:   cond {
// CHECK-NEXT:     stablehlo.return %[[COND]] : tensor<i1>
// CHECK-NEXT:   } do {
// CHECK-NEXT:     %[[BODY_TRANS:.*]] = stablehlo.transpose %[[ITER_ARG]], dims = [1, 0] : (tensor<3x2xf32>) -> tensor<2x3xf32>
// CHECK-NEXT:     %[[CUSTOM:.*]] = "unregistered.custom_transform"(%[[BODY_TRANS]]) : (tensor<2x3xf32>) -> tensor<3x2xf32>
// CHECK-NEXT:     stablehlo.return %[[CUSTOM]] : tensor<3x2xf32>
// CHECK:        }
// CHECK-NEXT: return %[[WHILE_RESULT]] : tensor<3x2xf32>

// Multi-user Reactant.jl#3269 pattern: while result used by both a full
// reduction and a transpose, with matching init/entry/exit transposes -> 0 transposes.
func.func @test_while_transpose_with_reduce_multiuser(%x: tensor<200x100xf64>, %y: tensor<200x100xf64>) -> (tensor<f64>, tensor<200x100xf64>) {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c10 = stablehlo.constant dense<10> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %zero = stablehlo.constant dense<0.0> : tensor<f64>
  %init_t = stablehlo.transpose %y, dims = [1, 0] : (tensor<200x100xf64>) -> tensor<100x200xf64>
  %res:2 = stablehlo.while(%i = %c0, %buf = %init_t) : tensor<i64>, tensor<100x200xf64>
  cond {
    %cmp = stablehlo.compare LT, %i, %c10 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %cmp : tensor<i1>
  } do {
    %next_i = stablehlo.add %i, %c1 : tensor<i64>
    %buf_t = stablehlo.transpose %buf, dims = [1, 0] : (tensor<100x200xf64>) -> tensor<200x100xf64>
    %upd = "unregistered.kernel"(%x, %buf_t) : (tensor<200x100xf64>, tensor<200x100xf64>) -> tensor<200x100xf64>
    %upd_t = stablehlo.transpose %upd, dims = [1, 0] : (tensor<200x100xf64>) -> tensor<100x200xf64>
    stablehlo.return %next_i, %upd_t : tensor<i64>, tensor<100x200xf64>
  }
  %sum = stablehlo.reduce(%res#1 init: %zero) across dimensions = [0, 1] : (tensor<100x200xf64>, tensor<f64>) -> tensor<f64>
  reducer(%a: tensor<f64>, %b: tensor<f64>) {
    %s = stablehlo.add %a, %b : tensor<f64>
    stablehlo.return %s : tensor<f64>
  }
  %out_t = stablehlo.transpose %res#1, dims = [1, 0] : (tensor<100x200xf64>) -> tensor<200x100xf64>
  return %sum, %out_t : tensor<f64>, tensor<200x100xf64>
}

// CHECK-LABEL: func.func @test_while_transpose_with_reduce_multiuser
// CHECK-NOT: stablehlo.transpose
// CHECK: %[[W:.*]]:2 = stablehlo.while(%{{.*}} = %{{.*}}, %[[BUF:.*]] = %arg1) : tensor<i64>, tensor<200x100xf64>
// CHECK-NOT: stablehlo.transpose
// CHECK: %[[UPD:.*]] = "unregistered.kernel"(%arg0, %[[BUF]]) : (tensor<200x100xf64>, tensor<200x100xf64>) -> tensor<200x100xf64>
// CHECK-NEXT: stablehlo.return %{{.*}}, %[[UPD]] : tensor<i64>, tensor<200x100xf64>
// CHECK: %[[SUM:.*]] = stablehlo.reduce(%[[W]]#1 init: %{{.*}}) applies stablehlo.add across dimensions = [0, 1] : (tensor<200x100xf64>, tensor<f64>) -> tensor<f64>
// CHECK-NOT: stablehlo.transpose
// CHECK: return %[[SUM]], %[[W]]#1 : tensor<f64>, tensor<200x100xf64>

// Partial 1D reduce + broadcast_in_dim + dot_general + transpose external users.
func.func @test_while_transpose_partial_reduce_bcast_dot(%y: tensor<5x4xf32>, %v: tensor<5xf32>) -> (tensor<5xf32>, tensor<2x5x4xf32>, tensor<4xf32>, tensor<5x4xf32>) {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c3 = stablehlo.constant dense<3> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %zero = stablehlo.constant dense<0.0> : tensor<f32>
  %init_t = stablehlo.transpose %y, dims = [1, 0] : (tensor<5x4xf32>) -> tensor<4x5xf32>
  %res:2 = stablehlo.while(%i = %c0, %buf = %init_t) : tensor<i64>, tensor<4x5xf32>
  cond {
    %cmp = stablehlo.compare LT, %i, %c3 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %cmp : tensor<i1>
  } do {
    %next_i = stablehlo.add %i, %c1 : tensor<i64>
    %buf_t = stablehlo.transpose %buf, dims = [1, 0] : (tensor<4x5xf32>) -> tensor<5x4xf32>
    %upd = "unregistered.step"(%buf_t) : (tensor<5x4xf32>) -> tensor<5x4xf32>
    %upd_t = stablehlo.transpose %upd, dims = [1, 0] : (tensor<5x4xf32>) -> tensor<4x5xf32>
    stablehlo.return %next_i, %upd_t : tensor<i64>, tensor<4x5xf32>
  }
  %red = stablehlo.reduce(%res#1 init: %zero) across dimensions = [0] : (tensor<4x5xf32>, tensor<f32>) -> tensor<5xf32>
  reducer(%a: tensor<f32>, %b: tensor<f32>) {
    %s = stablehlo.add %a, %b : tensor<f32>
    stablehlo.return %s : tensor<f32>
  }
  %bcast = stablehlo.broadcast_in_dim %res#1, dims = [2, 1] : (tensor<4x5xf32>) -> tensor<2x5x4xf32>
  %dot = stablehlo.dot_general %res#1, %v, contracting_dims = [1] x [0] : (tensor<4x5xf32>, tensor<5xf32>) -> tensor<4xf32>
  %out_t = stablehlo.transpose %res#1, dims = [1, 0] : (tensor<4x5xf32>) -> tensor<5x4xf32>
  return %red, %bcast, %dot, %out_t : tensor<5xf32>, tensor<2x5x4xf32>, tensor<4xf32>, tensor<5x4xf32>
}

// CHECK-LABEL: func.func @test_while_transpose_partial_reduce_bcast_dot
// CHECK-NOT: stablehlo.transpose
// CHECK: %[[W2:.*]]:2 = stablehlo.while(%{{.*}} = %{{.*}}, %[[BUF2:.*]] = %arg0) : tensor<i64>, tensor<5x4xf32>
// CHECK: %[[RED2:.*]] = stablehlo.reduce(%[[W2]]#1 init: %{{.*}}) applies stablehlo.add across dimensions = [1] : (tensor<5x4xf32>, tensor<f32>) -> tensor<5xf32>
// CHECK: %[[BC2:.*]] = stablehlo.broadcast_in_dim %[[W2]]#1, dims = [1, 2] : (tensor<5x4xf32>) -> tensor<2x5x4xf32>
// CHECK: %[[DOT2:.*]] = stablehlo.dot_general %[[W2]]#1, %arg1, contracting_dims = [0] x [0] : (tensor<5x4xf32>, tensor<5xf32>) -> tensor<4xf32>
// CHECK-NOT: stablehlo.transpose
// CHECK: return %[[RED2]], %[[BC2]], %[[DOT2]], %[[W2]]#1

// Seeded from inner transpose when external user has NO transpose (only reduce).
func.func @test_while_inner_transpose_only_reduce_user(%y: tensor<20x10xf32>) -> tensor<20xf32> {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c4 = stablehlo.constant dense<4> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %zero = stablehlo.constant dense<0.0> : tensor<f32>
  %init_t = stablehlo.transpose %y, dims = [1, 0] : (tensor<20x10xf32>) -> tensor<10x20xf32>
  %res:2 = stablehlo.while(%i = %c0, %buf = %init_t) : tensor<i64>, tensor<10x20xf32>
  cond {
    %cmp = stablehlo.compare LT, %i, %c4 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %cmp : tensor<i1>
  } do {
    %next_i = stablehlo.add %i, %c1 : tensor<i64>
    %buf_t = stablehlo.transpose %buf, dims = [1, 0] : (tensor<10x20xf32>) -> tensor<20x10xf32>
    %upd = "unregistered.step"(%buf_t) : (tensor<20x10xf32>) -> tensor<20x10xf32>
    %upd_t = stablehlo.transpose %upd, dims = [1, 0] : (tensor<20x10xf32>) -> tensor<10x20xf32>
    stablehlo.return %next_i, %upd_t : tensor<i64>, tensor<10x20xf32>
  }
  %red = stablehlo.reduce(%res#1 init: %zero) across dimensions = [0] : (tensor<10x20xf32>, tensor<f32>) -> tensor<20xf32>
  reducer(%a: tensor<f32>, %b: tensor<f32>) {
    %s = stablehlo.add %a, %b : tensor<f32>
    stablehlo.return %s : tensor<f32>
  }
  return %red : tensor<20xf32>
}

// CHECK-LABEL: func.func @test_while_inner_transpose_only_reduce_user
// CHECK-NOT: stablehlo.transpose
// CHECK: %[[W3:.*]]:2 = stablehlo.while(%{{.*}} = %{{.*}}, %[[BUF3:.*]] = %arg0) : tensor<i64>, tensor<20x10xf32>
// CHECK-NOT: stablehlo.transpose
// CHECK: %[[RED3:.*]] = stablehlo.reduce(%[[W3]]#1 init: %{{.*}}) applies stablehlo.add across dimensions = [1] : (tensor<20x10xf32>, tensor<f32>) -> tensor<20xf32>
// CHECK: return %[[RED3]] : tensor<20xf32>

// Hoist matching entry+exit inner transposes out of an opaque body even when
// returned directly, without oscillating back in.
func.func @test_while_inner_transpose_hoist_opaque_return(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c5 = stablehlo.constant dense<5> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %res:2 = stablehlo.while(%i = %c0, %buf = %arg0) : tensor<i64>, tensor<2x3xf32>
  cond {
    %cmp = stablehlo.compare LT, %i, %c5 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %cmp : tensor<i1>
  } do {
    %next_i = stablehlo.add %i, %c1 : tensor<i64>
    %t_in = stablehlo.transpose %buf, dims = [1, 0] : (tensor<2x3xf32>) -> tensor<3x2xf32>
    %upd = "unregistered.opaque_step"(%t_in) : (tensor<3x2xf32>) -> tensor<3x2xf32>
    %t_out = stablehlo.transpose %upd, dims = [1, 0] : (tensor<3x2xf32>) -> tensor<2x3xf32>
    stablehlo.return %next_i, %t_out : tensor<i64>, tensor<2x3xf32>
  }
  return %res#1 : tensor<2x3xf32>
}

// CHECK-LABEL: func.func @test_while_inner_transpose_hoist_opaque_return
// CHECK: %[[INIT_T:.*]] = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<2x3xf32>) -> tensor<3x2xf32>
// CHECK-NEXT: %[[W4:.*]]:2 = stablehlo.while(%{{.*}} = %{{.*}}, %[[BUF4:.*]] = %[[INIT_T]]) : tensor<i64>, tensor<3x2xf32>
// CHECK: } do {
// CHECK-NOT: stablehlo.transpose
// CHECK: %[[OPQ:.*]] = "unregistered.opaque_step"(%[[BUF4]]) : (tensor<3x2xf32>) -> tensor<3x2xf32>
// CHECK-NEXT: stablehlo.return %{{.*}}, %[[OPQ]] : tensor<i64>, tensor<3x2xf32>
// CHECK: }
// CHECK-NEXT: %[[OUT_T:.*]] = stablehlo.transpose %[[W4]]#1, dims = [1, 0] : (tensor<3x2xf32>) -> tensor<2x3xf32>
// CHECK-NEXT: return %[[OUT_T]] : tensor<2x3xf32>

// Negative test: do not push a single outer transpose into an opaque while body
// that has no inner transposes to cancel.
func.func @test_while_outer_transpose_not_pushed_into_opaque_body(%arg0: tensor<2x3xf32>) -> tensor<3x2xf32> {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c5 = stablehlo.constant dense<5> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %res:2 = stablehlo.while(%i = %c0, %buf = %arg0) : tensor<i64>, tensor<2x3xf32>
  cond {
    %cmp = stablehlo.compare LT, %i, %c5 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %cmp : tensor<i1>
  } do {
    %next_i = stablehlo.add %i, %c1 : tensor<i64>
    %upd = "unregistered.opaque_step"(%buf) : (tensor<2x3xf32>) -> tensor<2x3xf32>
    stablehlo.return %next_i, %upd : tensor<i64>, tensor<2x3xf32>
  }
  %out_t = stablehlo.transpose %res#1, dims = [1, 0] : (tensor<2x3xf32>) -> tensor<3x2xf32>
  return %out_t : tensor<3x2xf32>
}

// CHECK-LABEL: func.func @test_while_outer_transpose_not_pushed_into_opaque_body
// CHECK-NOT: stablehlo.transpose
// CHECK: %[[W5:.*]]:2 = stablehlo.while(%{{.*}} = %{{.*}}, %[[BUF5:.*]] = %arg0) : tensor<i64>, tensor<2x3xf32>
// CHECK: } do {
// CHECK-NOT: stablehlo.transpose
// CHECK: %[[OPQ5:.*]] = "unregistered.opaque_step"(%[[BUF5]]) : (tensor<2x3xf32>) -> tensor<2x3xf32>
// CHECK-NEXT: stablehlo.return %{{.*}}, %[[OPQ5]] : tensor<i64>, tensor<2x3xf32>
// CHECK: }
// CHECK-NEXT: %[[OUT_T5:.*]] = stablehlo.transpose %[[W5]]#1, dims = [1, 0] : (tensor<2x3xf32>) -> tensor<3x2xf32>
// CHECK-NEXT: return %[[OUT_T5]] : tensor<3x2xf32>

// Negative test: 3D reduction whose two unreduced dimensions flip order under
// invPerm ([2, 0, 1] -> invPerm [1, 2, 0] flips unreduced dims 0 and 2) would
// require creating a new post-reduce transpose, so do not transform via Source A.
func.func @test_while_3d_reduce_unreduced_order_mismatch(%arg0: tensor<2x3x4xf32>) -> (tensor<2x4xf32>, tensor<4x2x3xf32>) {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c5 = stablehlo.constant dense<5> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %zero = stablehlo.constant dense<0.0> : tensor<f32>
  %res:2 = stablehlo.while(%i = %c0, %buf = %arg0) : tensor<i64>, tensor<2x3x4xf32>
  cond {
    %cmp = stablehlo.compare LT, %i, %c5 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %cmp : tensor<i1>
  } do {
    %next_i = stablehlo.add %i, %c1 : tensor<i64>
    %upd = "unregistered.step3d"(%buf) : (tensor<2x3x4xf32>) -> tensor<2x4x3xf32>
    %upd_t = stablehlo.transpose %upd, dims = [0, 2, 1] : (tensor<2x4x3xf32>) -> tensor<2x3x4xf32>
    stablehlo.return %next_i, %upd_t : tensor<i64>, tensor<2x3x4xf32>
  }
  %red = stablehlo.reduce(%res#1 init: %zero) across dimensions = [1] : (tensor<2x3x4xf32>, tensor<f32>) -> tensor<2x4xf32>
  reducer(%a: tensor<f32>, %b: tensor<f32>) {
    %s = stablehlo.add %a, %b : tensor<f32>
    stablehlo.return %s : tensor<f32>
  }
  %out_t = stablehlo.transpose %res#1, dims = [2, 0, 1] : (tensor<2x3x4xf32>) -> tensor<4x2x3xf32>
  return %red, %out_t : tensor<2x4xf32>, tensor<4x2x3xf32>
}

// CHECK-LABEL: func.func @test_while_3d_reduce_unreduced_order_mismatch
// CHECK: %[[W6:.*]]:2 = stablehlo.while(%{{.*}} = %{{.*}}, %[[BUF6:.*]] = %arg0) : tensor<i64>, tensor<2x3x4xf32>
// CHECK: %[[RED6:.*]] = stablehlo.reduce(%[[W6]]#1 init: %{{.*}}) applies stablehlo.add across dimensions = [1] : (tensor<2x3x4xf32>, tensor<f32>) -> tensor<2x4xf32>
// CHECK: %[[OUT6:.*]] = stablehlo.transpose %[[W6]]#1, dims = [2, 0, 1] : (tensor<2x3x4xf32>) -> tensor<4x2x3xf32>
// CHECK: return %[[RED6]], %[[OUT6]]
