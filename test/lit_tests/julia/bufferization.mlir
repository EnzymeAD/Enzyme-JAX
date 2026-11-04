// RUN: enzymexlamlir-opt %s --shlo-to-julia-pipeline="enzyme-hlo-opt=false" | FileCheck %s --check-prefix=IR
// RUN: enzymexlamlir-opt %s --shlo-to-julia-pipeline="enzyme-hlo-opt=false" --emit-julia -o /dev/null | FileCheck %s --check-prefix=JL

module {
  // 1. Donated buffer argument (`tf.aliasing_output = 0`) mutates `%x` directly
  // in-place with ZERO heap allocations!
  // IR-LABEL: julia.func @donated_arg_inplace
  // IR-NOT: julia.alloc
  // IR: julia.kernel! outs(%arg0
  // JL-LABEL: function donated_arg_inplace(
  // JL-NOT: undef
  func.func @donated_arg_inplace(
      %x: tensor<16x16xf32> {tf.aliasing_output = 0 : i32},
      %y: tensor<16x16xf32>) -> tensor<16x16xf32> {
    %s = stablehlo.add %x, %y : tensor<16x16xf32>
    %out = stablehlo.exponential %s : tensor<16x16xf32>
    return %out : tensor<16x16xf32>
  }

  // 2. Concatenate destination `@view` slice forwarding:
  // Both `%left = A * B` (matmul) and `%right = sin(C)` (kernel) have a single
  // use in `stablehlo.concatenate`, so `--julia-bufferize` allocates ONLY the
  // final concatenated output and writes both producers directly into `@view`
  // slices of the destination — ZERO intermediate allocations or `copyto!`!
  // IR-LABEL: julia.func @concat_view_forwarding
  // IR-COUNT-1: julia.alloc
  // IR: julia.view_slice
  // IR: julia.matmul!
  // IR: julia.view_slice
  // IR: julia.kernel!
  // IR-NOT: julia.concat!
  // IR-NOT: julia.copy_to!
  // JL-LABEL: function concat_view_forwarding(
  // JL-COUNT-1: Array{Float32, 2}(undef, 10, 4)
  // JL-NOT: undef
  // JL: LinearAlgebra.mul!(@view(
  // JL: @view(
  func.func @concat_view_forwarding(
      %a: tensor<4x8xf32>,
      %b: tensor<8x6xf32>,
      %c: tensor<4x4xf32>) -> tensor<4x10xf32> {
    %left = stablehlo.dot_general %a, %b, contracting_dims = [1] x [0] : (tensor<4x8xf32>, tensor<8x6xf32>) -> tensor<4x6xf32>
    %right = stablehlo.sine %c : tensor<4x4xf32>
    %cat = stablehlo.concatenate %left, %right, dim = 1 : (tensor<4x6xf32>, tensor<4x4xf32>) -> tensor<4x10xf32>
    return %cat : tensor<4x10xf32>
  }

  // 3. Linear-scan buffer pool coloring + dead-operand reuse:
  // Chain of 3 sequential matmuls (`t1 = A * B`, `t2 = t1 * C`, `out = t2 * D`)
  // where `t1` and `t2` have shape `4x4` and `out` escapes (`4x4`):
  // `t1` and `t2` cannot alias within the same `mul!`, so `t1` needs 1 scratch
  // buffer, `t2` needs a 2nd buffer, and `out = t2 * D` reuses `t1`'s dead
  // scratch buffer if non-escaping or colors non-overlapping temporaries!
  // For a 4-matmul chain (`t1`, `t2`, `t3`, `out` all `4x4`), `t3 = t2 * D`
  // reuses `t1`'s buffer via linear-scan pool coloring, so `t1` and `t3` share
  // ONE scratch allocation (only 2 scratch allocs + 1 escaping return alloc =
  // at most 2 total allocs when `t2` also colors or 3 instead of 4)!
  // Let's verify non-escaping `t1` (used in `t2`) and `t3` (used in scalar sum):
  // `t1` [matmul 1 -> matmul 2] and `t3` [matmul 3 -> reduce] have disjoint
  // lifetimes and identical shape `4x4`, so pool coloring merges them into 1 alloc!
  // IR-LABEL: julia.func @buffer_pool_coloring
  // IR-COUNT-2: julia.alloc
  // IR-NOT: julia.alloc
  // JL-LABEL: function buffer_pool_coloring(
  // JL-COUNT-2: Array{Float32, 2}(undef, 4, 4)
  func.func @buffer_pool_coloring(
      %a: tensor<4x4xf32>,
      %b: tensor<4x4xf32>,
      %c: tensor<4x4xf32>,
      %d: tensor<4x4xf32>) -> tensor<f32> {
    %t1 = stablehlo.dot_general %a, %b, contracting_dims = [1] x [0] : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    %t2 = stablehlo.dot_general %t1, %c, contracting_dims = [1] x [0] : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    %t3 = stablehlo.dot_general %t2, %d, contracting_dims = [1] x [0] : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    %zero = stablehlo.constant dense<0.0> : tensor<f32>
    %sum = stablehlo.reduce(%t3 init: %zero) across dimensions = [0, 1] : (tensor<4x4xf32>, tensor<f32>) -> tensor<f32>
      reducer(%acc: tensor<f32>, %v: tensor<f32>) {
        %s = stablehlo.add %acc, %v : tensor<f32>
        stablehlo.return %s : tensor<f32>
      }
    return %sum : tensor<f32>
  }

  // 4. `stablehlo.while` in-place loop-carried state update:
  // Elementwise state update mutates the loop-carried buffer in-place on every
  // iteration with ZERO allocations inside the `while` body!
  // IR-LABEL: julia.func @while_inplace_elementwise
  // IR: julia.while
  // IR-NOT: julia.alloc
  // JL-LABEL: function while_inplace_elementwise(
  // JL: while
  func.func @while_inplace_elementwise(
      %init: tensor<8xf32> {tf.aliasing_output = 0 : i32},
      %step: tensor<8xf32>,
      %n: tensor<i32>) -> tensor<8xf32> {
    %c0 = stablehlo.constant dense<0> : tensor<i32>
    %c1 = stablehlo.constant dense<1> : tensor<i32>
    %fin_i, %fin_w = "stablehlo.while"(%c0, %init) ({
    ^bb0(%i: tensor<i32>, %w: tensor<8xf32>):
      %cond = stablehlo.compare LT, %i, %n, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %cond : tensor<i1>
    }, {
    ^bb0(%i: tensor<i32>, %w: tensor<8xf32>):
      %ni = stablehlo.add %i, %c1 : tensor<i32>
      %nw = stablehlo.add %w, %step : tensor<8xf32>
      stablehlo.return %ni, %nw : tensor<i32>, tensor<8xf32>
    }) : (tensor<i32>, tensor<8xf32>) -> (tensor<i32>, tensor<8xf32>)
    return %fin_w : tensor<8xf32>
  }

  // 5. `stablehlo.while` ping-pong double-buffer hoisting for BLAS matmul state:
  // `nw = A * w` cannot overwrite `w` in-place inside `LinearAlgebra.mul!`, so
  // `--julia-bufferize` hoists ONE `pong` buffer outside the `while` loop and
  // swaps `(w, pong) = (pong, w)` each iteration — ZERO heap allocations inside
  // the `while` loop!
  // IR-LABEL: julia.func @while_pingpong_matmul
  // IR-COUNT-2: julia.alloc
  // IR: julia.while
  // IR-NOT: julia.alloc
  // JL-LABEL: function while_pingpong_matmul(
  // JL: while
  // JL-NOT: undef
  // JL: LinearAlgebra.mul!(
  func.func @while_pingpong_matmul(
      %a: tensor<4x4xf32>,
      %init: tensor<4x2xf32>,
      %n: tensor<i32>) -> tensor<4x2xf32> {
    %c0 = stablehlo.constant dense<0> : tensor<i32>
    %c1 = stablehlo.constant dense<1> : tensor<i32>
    %fin_i, %fin_w = "stablehlo.while"(%c0, %init) ({
    ^bb0(%i: tensor<i32>, %w: tensor<4x2xf32>):
      %cond = stablehlo.compare LT, %i, %n, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %cond : tensor<i1>
    }, {
    ^bb0(%i: tensor<i32>, %w: tensor<4x2xf32>):
      %ni = stablehlo.add %i, %c1 : tensor<i32>
      %nw = stablehlo.dot_general %a, %w, contracting_dims = [1] x [0] : (tensor<4x4xf32>, tensor<4x2xf32>) -> tensor<4x2xf32>
      stablehlo.return %ni, %nw : tensor<i32>, tensor<4x2xf32>
    }) : (tensor<i32>, tensor<4x2xf32>) -> (tensor<i32>, tensor<4x2xf32>)
    return %fin_w : tensor<4x2xf32>
  }
}
