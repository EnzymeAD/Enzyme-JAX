// REQUIRES: triton
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=reduce_minimum outfn=fwd_minimum argTys=enzyme_dup retTys=enzyme_dup mode=ForwardMode" --canonicalize | FileCheck %s --check-prefix=MINIMUM
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=reduce_maximum outfn=fwd_maximum argTys=enzyme_dup retTys=enzyme_dup mode=ForwardMode" --canonicalize | FileCheck %s --check-prefix=MAXIMUM
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=reduce_sum outfn=fwd_sum argTys=enzyme_dup retTys=enzyme_dup mode=ForwardMode" --canonicalize | FileCheck %s --check-prefix=SUM-FWD
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=reduce_sum outfn=rev_sum argTys=enzyme_active retTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --canonicalize | FileCheck %s --check-prefix=SUM-REV
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=reduce_max outfn=fwd_max argTys=enzyme_dup retTys=enzyme_dup mode=ForwardMode" --canonicalize | FileCheck %s --check-prefix=MAX-FWD
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=reduce_max outfn=rev_max argTys=enzyme_active retTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --canonicalize | FileCheck %s --check-prefix=MAX-REV
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=reduce_min outfn=fwd_min argTys=enzyme_dup retTys=enzyme_dup mode=ForwardMode" --canonicalize | FileCheck %s --check-prefix=MIN-FWD
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=reduce_min outfn=rev_min argTys=enzyme_active retTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --canonicalize | FileCheck %s --check-prefix=MIN-REV

tt.func @reduce_sum(%x: tensor<4x8xf32>) -> tensor<4xf32> {
  %s = "tt.reduce"(%x) <{axis = 1 : i32}> ({
  ^bb0(%a: f32, %b: f32):
    %v = arith.addf %a, %b : f32
    tt.reduce.return %v : f32
  }) : (tensor<4x8xf32>) -> tensor<4xf32>
  tt.return %s : tensor<4xf32>
}

tt.func @reduce_max(%x: tensor<8xf32>) -> f32 {
  %s = "tt.reduce"(%x) <{axis = 0 : i32}> ({
  ^bb0(%a: f32, %b: f32):
    %v = arith.maxnumf %a, %b : f32
    tt.reduce.return %v : f32
  }) : (tensor<8xf32>) -> f32
  tt.return %s : f32
}

tt.func @reduce_min(%x: tensor<8xf32>) -> f32 {
  %s = "tt.reduce"(%x) <{axis = 0 : i32}> ({
  ^bb0(%a: f32, %b: f32):
    %v = arith.minnumf %a, %b : f32
    tt.reduce.return %v : f32
  }) : (tensor<8xf32>) -> f32
  tt.return %s : f32
}

// SUM-FWD-LABEL: tt.func private @fwd_sum(
// SUM-FWD: "tt.reduce"(%arg1)
// SUM-FWD: arith.addf
// SUM-FWD: tt.return
// SUM-REV-LABEL: tt.func private @rev_sum(
// SUM-REV: %[[E:.*]] = tt.expand_dims %arg1 {axis = 1 : i32}
// SUM-REV: %[[B:.*]] = tt.broadcast %[[E]] : tensor<4x1xf32> -> tensor<4x8xf32>
// SUM-REV: tt.return %[[B]] : tensor<4x8xf32>

// The same normalized equality weights are used in both modes, so tied
// extrema distribute the adjoint and average the tangent consistently.
// MAX-FWD-LABEL: tt.func private @fwd_max(
// MAX-FWD: arith.maxnumf
// MAX-FWD: arith.cmpf oeq
// MAX-FWD: arith.uitofp
// MAX-FWD: "tt.reduce"
// MAX-FWD: arith.divf
// MAX-FWD: arith.mulf
// MAX-FWD: "tt.reduce"
// MAX-FWD: tt.return
// MAX-REV-LABEL: tt.func private @rev_max(
// MAX-REV: arith.maxnumf
// MAX-REV: arith.cmpf oeq
// MAX-REV: arith.uitofp
// MAX-REV: "tt.reduce"
// MAX-REV: arith.divf
// MAX-REV: arith.mulf
// MAX-REV: tt.return
// MIN-FWD-LABEL: tt.func private @fwd_min(
// MIN-FWD: arith.minnumf
// MIN-FWD: arith.cmpf oeq
// MIN-FWD: arith.divf
// MIN-FWD: tt.return
// MIN-REV-LABEL: tt.func private @rev_min(
// MIN-REV: arith.minnumf
// MIN-REV: arith.cmpf oeq
// MIN-REV: arith.divf
// MIN-REV: tt.return

// Preserve the forward support supplied by the base PR for arith.maximumf.
tt.func @reduce_maximum(%x: tensor<8xf32>) -> f32 {
  %s = "tt.reduce"(%x) <{axis = 0 : i32}> ({
  ^bb0(%a: f32, %b: f32):
    %v = arith.maximumf %a, %b : f32
    tt.reduce.return %v : f32
  }) : (tensor<8xf32>) -> f32
  tt.return %s : f32
}

// MAXIMUM-LABEL: tt.func private @fwd_maximum(
// MAXIMUM: %[[R:.*]]:2 = "tt.reduce"(%arg0, %arg1)
// MAXIMUM: arith.maximumf
// MAXIMUM: tt.reduce.return {{.*}} : f32, f32
// MAXIMUM: tt.return %[[R]]#0, %[[R]]#1

// Preserve the forward support supplied by the base PR for arith.minimumf.
tt.func @reduce_minimum(%x: tensor<8xf32>) -> f32 {
  %s = "tt.reduce"(%x) <{axis = 0 : i32}> ({
  ^bb0(%a: f32, %b: f32):
    %v = arith.minimumf %a, %b : f32
    tt.reduce.return %v : f32
  }) : (tensor<8xf32>) -> f32
  tt.return %s : f32
}

// MINIMUM-LABEL: tt.func private @fwd_minimum(
// MINIMUM: %[[R:.*]]:2 = "tt.reduce"(%arg0, %arg1)
// MINIMUM: arith.minimumf
// MINIMUM: tt.reduce.return {{.*}} : f32, f32
// MINIMUM: tt.return %[[R]]#0, %[[R]]#1
