// REQUIRES: triton
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=shapes outfn=fwd argTys=enzyme_dup retTys=enzyme_dup mode=ForwardMode" --canonicalize | FileCheck %s --check-prefix=FWD
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=shapes outfn=rev argTys=enzyme_active retTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --canonicalize | FileCheck %s --check-prefix=REV
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=splat outfn=splat_rev argTys=enzyme_active retTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --canonicalize | FileCheck %s --check-prefix=SPLAT
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=splat outfn=splat_fwd argTys=enzyme_dup retTys=enzyme_dup mode=ForwardMode" --canonicalize | FileCheck %s --check-prefix=SPLAT-FWD

tt.func @shapes(%x: tensor<2xf32>) -> tensor<4x2xf32> {
  %a = tt.expand_dims %x {axis = 0 : i32} : tensor<2xf32> -> tensor<1x2xf32>
  %b = tt.broadcast %a : tensor<1x2xf32> -> tensor<4x2xf32>
  %t = tt.trans %b {order = array<i32: 1, 0>} : tensor<4x2xf32> -> tensor<2x4xf32>
  %r = tt.reshape %t : tensor<2x4xf32> -> tensor<4x2xf32>
  tt.return %r : tensor<4x2xf32>
}

tt.func @splat(%x: f32) -> tensor<2x4xf32> {
  %s = tt.splat %x : f32 -> tensor<2x4xf32>
  tt.return %s : tensor<2x4xf32>
}

// FWD-LABEL: tt.func private @fwd(
// FWD: tt.expand_dims %arg1
// FWD: tt.broadcast
// FWD: tt.trans
// FWD: tt.reshape
// FWD: tt.return
// REV-LABEL: tt.func private @rev(
// REV: tt.reshape
// REV: tt.trans
// REV: "tt.reduce"
// REV: arith.addf
// REV: tt.reduce.return
// REV: tt.return
// SPLAT-LABEL: tt.func private @splat_rev(
// SPLAT: "tt.reduce"
// SPLAT: "tt.reduce"
// SPLAT: tt.return {{.*}} : f32
// SPLAT-FWD-LABEL: tt.func private @splat_fwd(
// SPLAT-FWD: tt.splat %arg1
// SPLAT-FWD: tt.return
