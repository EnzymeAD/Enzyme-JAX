// REQUIRES: triton
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=convert outfn=fwd argTys=enzyme_dup retTys=enzyme_dup mode=ForwardMode" --canonicalize | FileCheck %s --check-prefix=FWD
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=convert outfn=rev argTys=enzyme_active retTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --canonicalize | FileCheck %s --check-prefix=REV
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=precise outfn=fwd_precise argTys=enzyme_dup,enzyme_dup retTys=enzyme_dup mode=ForwardMode" --canonicalize | FileCheck %s --check-prefix=PRECISE-FWD
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=precise outfn=rev_precise argTys=enzyme_active,enzyme_active retTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --canonicalize | FileCheck %s --check-prefix=PRECISE-REV

tt.func @convert(%x: tensor<4xf32>) -> tensor<4xf16> {
  %y = tt.fp_to_fp %x, rounding = rtne : tensor<4xf32> -> tensor<4xf16>
  tt.return %y : tensor<4xf16>
}

tt.func @precise(%x: f32, %y: f32) -> f32 {
  %s = tt.precise_sqrt %x : f32
  %d = tt.precise_divf %s, %y : f32
  tt.return %d : f32
}

// FWD-LABEL: tt.func private @fwd(
// FWD: tt.fp_to_fp %arg1, rounding = rtne : tensor<4xf32> -> tensor<4xf16>
// FWD: tt.return
// REV-LABEL: tt.func private @rev(
// REV: tt.fp_to_fp %arg1, rounding = rtne : tensor<4xf16> -> tensor<4xf32>
// REV: tt.return
// PRECISE-FWD-LABEL: tt.func private @fwd_precise(
// PRECISE-FWD: tt.precise_sqrt
// PRECISE-FWD: arith.divf
// PRECISE-FWD: tt.precise_divf
// PRECISE-FWD: tt.return
// PRECISE-REV-LABEL: tt.func private @rev_precise(
// PRECISE-REV: tt.precise_sqrt
// PRECISE-REV: arith.divf
// PRECISE-REV: tt.precise_divf
// PRECISE-REV: tt.return
