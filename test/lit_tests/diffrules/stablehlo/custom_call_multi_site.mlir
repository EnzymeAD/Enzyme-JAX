// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_active argTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops | FileCheck %s --check-prefix=REVERSE

// Two call sites share one call_target_name and have different adjoints. This is
// the case a registry keyed on call_target_name could not express, and it is the
// real shape of the motivating kernel: `mps.metal_kernel_lib` dispatches _fwd,
// _bwd_q, _bwd_kv and _bwd_b, all under the same target name, configured only by
// backend_config.

func.func @double_rev(%x: tensor<4xf32>, %y: tensor<4xf32>,
                      %dy: tensor<4xf32>) -> tensor<4xf32> {
  %two = stablehlo.constant dense<2.000000e+00> : tensor<4xf32>
  %res = stablehlo.multiply %dy, %two : tensor<4xf32>
  func.return %res : tensor<4xf32>
}

func.func @triple_rev(%x: tensor<4xf32>, %y: tensor<4xf32>,
                      %dy: tensor<4xf32>) -> tensor<4xf32> {
  %three = stablehlo.constant dense<3.000000e+00> : tensor<4xf32>
  %res = stablehlo.multiply %dy, %three : tensor<4xf32>
  func.return %res : tensor<4xf32>
}

func.func @main(%x : tensor<4xf32>) -> tensor<4xf32> {
  %a = stablehlo.custom_call @kernel(%x) {
    enzyme.reverse = @double_rev,
    enzyme.active_operands = array<i64: 0>,
    backend_config = "{\22which\22: \22double\22}"
  } : (tensor<4xf32>) -> tensor<4xf32>
  %b = stablehlo.custom_call @kernel(%a) {
    enzyme.reverse = @triple_rev,
    enzyme.active_operands = array<i64: 0>,
    backend_config = "{\22which\22: \22triple\22}"
  } : (tensor<4xf32>) -> tensor<4xf32>
  func.return %b : tensor<4xf32>
}

// REVERSE: func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
// Reverse order: the second call site's rule runs first, and each site gets its
// own reverse function despite sharing @kernel.
// REVERSE:   call @triple_rev(
// REVERSE:   call @double_rev(
// REVERSE:   return
