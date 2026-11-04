// RUN: enzymexlamlir-opt %s --shlo-to-julia-pipeline="enzyme-hlo-opt=false" | FileCheck %s --check-prefix=IR
// RUN: enzymexlamlir-opt %s --shlo-to-julia-pipeline="enzyme-hlo-opt=false" --emit-julia -o /dev/null | FileCheck %s --check-prefix=JL

module {
  // 0-D tensors unbox to native Julia scalar SSA values with zero heap allocations.
  // IR-LABEL: julia.func @scalar_unbox(%arg0: f64, %arg1: f64) -> f64
  // IR-NOT: julia.alloc
  // JL-LABEL: function scalar_unbox(
  // JL-SAME: ::Float64,
  // JL-NOT: Array{
  func.func @scalar_unbox(%x: tensor<f64>, %y: tensor<f64>) -> tensor<f64> {
    %sum = stablehlo.add %x, %y : tensor<f64>
    %s = stablehlo.sine %sum : tensor<f64>
    %e = chlo.erf %s : tensor<f64> -> tensor<f64>
    %lg = chlo.lgamma %y : tensor<f64> -> tensor<f64>
    %at = stablehlo.atan2 %e, %lg : tensor<f64>
    return %at : tensor<f64>
  }

  // Scalar helper call + conditional branching.
  func.func private @scale_helper(%v: tensor<f32>) -> tensor<f32> {
    %two = stablehlo.constant dense<2.0> : tensor<f32>
    %r = stablehlo.multiply %v, %two : tensor<f32>
    return %r : tensor<f32>
  }

  // IR-LABEL: julia.func @branch_and_call
  // IR: julia.if
  // JL-LABEL: function branch_and_call(
  // JL: if
  // JL: scale_helper(
  func.func @branch_and_call(%p: tensor<i1>, %a: tensor<f32>, %b: tensor<f32>) -> tensor<f32> {
    %res = "stablehlo.if"(%p) ({
      %ca = func.call @scale_helper(%a) : (tensor<f32>) -> tensor<f32>
      "stablehlo.return"(%ca) : (tensor<f32>) -> ()
    }, {
      %diff = stablehlo.subtract %b, %a : tensor<f32>
      "stablehlo.return"(%diff) : (tensor<f32>) -> ()
    }) : (tensor<i1>) -> tensor<f32>
    return %res : tensor<f32>
  }

  // Array iota, clamp, select, bitcast, and complex construction.
  // IR-LABEL: julia.func @iota_clamp_select
  // IR: julia.kernel!
  // JL-LABEL: function iota_clamp_select(
  // JL: @inbounds
  func.func @iota_clamp_select(%x: tensor<8xf32>, %lo: tensor<f32>, %hi: tensor<f32>) -> (tensor<8xf32>, tensor<8xcomplex<f32>>) {
    %idx = stablehlo.iota dim = 0 : tensor<8xi32>
    %idxf = stablehlo.convert %idx : (tensor<8xi32>) -> tensor<8xf32>
    %blo = stablehlo.broadcast_in_dim %lo, dims = [] : (tensor<f32>) -> tensor<8xf32>
    %bhi = stablehlo.broadcast_in_dim %hi, dims = [] : (tensor<f32>) -> tensor<8xf32>
    %clamped = stablehlo.clamp %blo, %x, %bhi : tensor<8xf32>
    %zero = stablehlo.constant dense<0.0> : tensor<8xf32>
    %mask = stablehlo.compare GT, %x, %zero, FLOAT : (tensor<8xf32>, tensor<8xf32>) -> tensor<8xi1>
    %sel = stablehlo.select %mask, %clamped, %idxf : tensor<8xi1>, tensor<8xf32>
    %cplx = stablehlo.complex %sel, %idxf : tensor<8xcomplex<f32>>
    return %sel, %cplx : tensor<8xf32>, tensor<8xcomplex<f32>>
  }
}
