// Stress tests for the precise-reduce path at N=3. These inputs cannot survive
// non-precise reduce (leading-limb f32 add absorbs the smaller terms), so all
// RUN lines pass precise-reduce=true. The point is to verify the N-limb
// multiFloatAdd inside the reducer body preserves precision end-to-end.
//
// RUN: enzymexlamlir-opt %s --multi-float-conversion="source-type=f64 target-type=f32 expansion-size=3 concat-dimension=first precise-reduce=true" | stablehlo-translate - --interpret --allow-unregistered-dialect
// RUN: enzymexlamlir-opt %s --multi-float-conversion="source-type=f64 target-type=f32 expansion-size=3 concat-dimension=last precise-reduce=true" | stablehlo-translate - --interpret --allow-unregistered-dialect

func.func @do_reduce4(%a: tensor<4xf64>) -> tensor<f64> {
  %init = stablehlo.constant dense<0.0> : tensor<f64>
  %r = "stablehlo.reduce"(%a, %init) ({
    ^bb0(%acc: tensor<f64>, %v: tensor<f64>):
      %s = stablehlo.add %acc, %v : tensor<f64>
      stablehlo.return %s : tensor<f64>
  }) {dimensions = array<i64: 0>} : (tensor<4xf64>, tensor<f64>) -> tensor<f64>
  return %r : tensor<f64>
}

func.func @do_reduce3(%a: tensor<3xf64>) -> tensor<f64> {
  %init = stablehlo.constant dense<0.0> : tensor<f64>
  %r = "stablehlo.reduce"(%a, %init) ({
    ^bb0(%acc: tensor<f64>, %v: tensor<f64>):
      %s = stablehlo.add %acc, %v : tensor<f64>
      stablehlo.return %s : tensor<f64>
  }) {dimensions = array<i64: 0>} : (tensor<3xf64>, tensor<f64>) -> tensor<f64>
  return %r : tensor<f64>
}

func.func @main() attributes {enzyme.no_multifloat} {
  // Case A: mixed magnitudes spanning ~12 orders of magnitude. Non-precise
  // reduce would drop everything below the 1e10 leading limb.
  %a1 = stablehlo.constant dense<[1.0e10, 1.0, 0.1, 0.01]> : tensor<4xf64>
  %e1 = stablehlo.constant dense<10000000001.110001> : tensor<f64>
  %r1 = func.call @do_reduce4(%a1) : (tensor<4xf64>) -> tensor<f64>
  "check.expect_close"(%r1, %e1) {max_ulp_difference = 4 : ui64} : (tensor<f64>, tensor<f64>) -> ()

  // Case B: catastrophic cancellation. f64 left-to-right gives 0.0
  // (1e30 + 1 rounds to 1e30 at f64 precision; the +1 is lost). Multifloat at
  // N=3 has ~72 bits of mantissa, plenty to retain the +1 through the
  // cancellation. Expected = 1.0 exactly. A wrong answer here would be ~0.0,
  // which is ~2^62 ULPs from 1.0 — impossible to mask with any reasonable
  // tolerance.
  %a2 = stablehlo.constant dense<[1.0e30, 1.0, -1.0e30]> : tensor<3xf64>
  %e2 = stablehlo.constant dense<1.0> : tensor<f64>
  %r2 = func.call @do_reduce3(%a2) : (tensor<3xf64>) -> tensor<f64>
  "check.expect_close"(%r2, %e2) {max_ulp_difference = 4 : ui64} : (tensor<f64>, tensor<f64>) -> ()

  return
}
