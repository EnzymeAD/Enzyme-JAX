// REQUIRES: triton
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=loop outfn=fwd argTys=enzyme_dup,enzyme_const retTys=enzyme_dup mode=ForwardMode" --canonicalize | FileCheck %s --check-prefix=FWD
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=loop outfn=rev argTys=enzyme_dup,enzyme_const retTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --canonicalize | FileCheck %s --check-prefix=REV

// The pointer result is unused, but its loop argument remains active. Forward
// AD must retain its shadow result slot before the live accumulator. Reverse
// AD must carry the shadow pointer in the primal loop, without adding pointers.
tt.func @loop(%in: !tt.ptr<f32>, %n: i32) -> tensor<4xf32> {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %step = arith.constant dense<4> : tensor<4xi32>
  %zero = arith.constant dense<0.0> : tensor<4xf32>
  %range = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32>
  %base = tt.splat %in : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
  %start = tt.addptr %base, %range : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
  %result:2 = scf.for %i = %c0 to %n step %c1 iter_args(%ptr = %start, %acc = %zero) -> (tensor<4x!tt.ptr<f32>>, tensor<4xf32>) : i32 {
    %v = tt.load %ptr : tensor<4x!tt.ptr<f32>>
    %sum = arith.addf %acc, %v : tensor<4xf32>
    %next = tt.addptr %ptr, %step : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    scf.yield %next, %sum : tensor<4x!tt.ptr<f32>>, tensor<4xf32>
  }
  tt.return %result#1 : tensor<4xf32>
}

// FWD-LABEL: tt.func private @fwd(
// FWD: scf.for
// FWD-SAME: -> (tensor<4x!tt.ptr<f32>>, tensor<4x!tt.ptr<f32>>, tensor<4xf32>, tensor<4xf32>)
// FWD: tt.load
// FWD: tt.load
// FWD: scf.yield
// FWD: tt.return {{.*}} : tensor<4xf32>, tensor<4xf32>
// REV-LABEL: tt.func private @rev(
// REV: scf.for
// REV: scf.for
// REV: tt.atomic_rmw fadd, relaxed, gpu
// REV: tt.return
