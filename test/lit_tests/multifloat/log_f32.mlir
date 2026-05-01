// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f32 target-type=bf16 concat-dimension=first convert-signatures=false" %s | stablehlo-translate - --interpret --allow-unregistered-dialect

func.func @log_test_f32(%arg0: tensor<f32>) -> tensor<f32> {
  %0 = stablehlo.log %arg0 : tensor<f32>
  return %0 : tensor<f32>
}

func.func @main() attributes {enzyme.no_multifloat} {
  %c2_f32 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
  %e2_f32 = stablehlo.constant dense<0.691358805> : tensor<f32> // Result from 2xbf16 emulation
  %r1_f32 = func.call @log_test_f32(%c2_f32) : (tensor<f32>) -> tensor<f32>
  "check.expect_close"(%r1_f32, %e2_f32) {max_ulp_difference = 0 : ui64} : (tensor<f32>, tensor<f32>) -> ()
  return
}
