// RUN: enzymexlamlir-opt %s --shlo-to-julia-pipeline="enzyme-hlo-opt=false" | FileCheck %s --check-prefix=IR
// RUN: enzymexlamlir-opt %s --shlo-to-julia-pipeline="enzyme-hlo-opt=false" --emit-julia -o /dev/null | FileCheck %s --check-prefix=JL

module {
  // Vertical DAG fusion: broadcast + transpose + reverse + slice + elementwise
  // fuses into a single `julia.kernel!` with a single output allocation.
  // IR-LABEL: julia.func @vertical_fusion
  // IR-COUNT-1: julia.alloc
  // IR-COUNT-1: julia.kernel!
  // IR-NOT: julia.kernel!
  // JL-LABEL: function vertical_fusion(
  // JL-COUNT-1: Array{Float32, 2}(undef, 3, 4)
  func.func @vertical_fusion(%a: tensor<4x6xf32>, %b: tensor<6x4xf32>, %bias: tensor<4xf32>) -> tensor<4x3xf32> {
    %bt = stablehlo.transpose %b, dims = [1, 0] : (tensor<6x4xf32>) -> tensor<4x6xf32>
    %brev = stablehlo.reverse %bt, dims = [1] : tensor<4x6xf32>
    %bbias = stablehlo.broadcast_in_dim %bias, dims = [0] : (tensor<4xf32>) -> tensor<4x6xf32>
    %sum = stablehlo.add %a, %brev : tensor<4x6xf32>
    %act = stablehlo.tanh %sum : tensor<4x6xf32>
    %biased = stablehlo.add %act, %bbias : tensor<4x6xf32>
    %sliced = stablehlo.slice %biased [0:4:1, 1:4:1] : (tensor<4x6xf32>) -> tensor<4x3xf32>
    %scale = stablehlo.constant dense<0.5> : tensor<4x3xf32>
    %out = stablehlo.multiply %sliced, %scale : tensor<4x3xf32>
    return %out : tensor<4x3xf32>
  }

  // Horizontal sibling kernel fusion: two independent elementwise outputs with
  // identical shapes fuse into a single multi-output `julia.kernel!` loop.
  // IR-LABEL: julia.func @horizontal_fusion
  // IR-COUNT-1: julia.kernel!
  // IR-NOT: julia.kernel!
  // JL-LABEL: function horizontal_fusion(
  func.func @horizontal_fusion(%x: tensor<16xf64>, %y: tensor<16xf64>) -> (tensor<16xf64>, tensor<16xf64>) {
    %s = stablehlo.add %x, %y : tensor<16xf64>
    %out0 = stablehlo.exponential %s : tensor<16xf64>
    %d = stablehlo.subtract %x, %y : tensor<16xf64>
    %out1 = stablehlo.cosine %d : tensor<16xf64>
    return %out0, %out1 : tensor<16xf64>, tensor<16xf64>
  }

  // 5-arg BLAS GEMM epilogue fusion: `alpha * (A * B) + beta * C` fuses into a
  // single `julia.matmul!` (`LinearAlgebra.BLAS.gemm!`) with zero elementwise
  // epilogue loops and zero heap allocations, reusing donated `%c` in-place!
  // IR-LABEL: julia.func @matmul_epilogue_gemm
  // IR-NOT: julia.alloc
  // IR-NOT: julia.kernel!
  // IR: julia.matmul!
  // JL-LABEL: function matmul_epilogue_gemm(
  // JL-NOT: undef
  // JL: LinearAlgebra.BLAS.gemm!('N', 'N',
  func.func @matmul_epilogue_gemm(
      %a: tensor<8x16xf32>,
      %b: tensor<16x12xf32>,
      %c: tensor<8x12xf32> {tf.aliasing_output = 0 : i32},
      %alpha: tensor<f32>,
      %beta: tensor<f32>) -> tensor<8x12xf32> {
    %mm = stablehlo.dot_general %a, %b, contracting_dims = [1] x [0] : (tensor<8x16xf32>, tensor<16x12xf32>) -> tensor<8x12xf32>
    %b_alpha = stablehlo.broadcast_in_dim %alpha, dims = [] : (tensor<f32>) -> tensor<8x12xf32>
    %b_beta = stablehlo.broadcast_in_dim %beta, dims = [] : (tensor<f32>) -> tensor<8x12xf32>
    %scaled_mm = stablehlo.multiply %b_alpha, %mm : tensor<8x12xf32>
    %scaled_c = stablehlo.multiply %b_beta, %c : tensor<8x12xf32>
    %out = stablehlo.add %scaled_mm, %scaled_c : tensor<8x12xf32>
    return %out : tensor<8x12xf32>
  }
}
