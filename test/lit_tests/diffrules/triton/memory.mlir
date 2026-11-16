// REQUIRES: triton
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=masked outfn=fwd argTys=enzyme_dup,enzyme_dup,enzyme_dup,enzyme_const mode=ForwardMode" --canonicalize --remove-unnecessary-enzyme-ops --canonicalize | FileCheck %s --check-prefix=FWD
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=masked outfn=rev argTys=enzyme_dup,enzyme_dup,enzyme_active,enzyme_const mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --canonicalize | FileCheck %s --check-prefix=REV
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=constant_store outfn=fwd_constant argTys=enzyme_dup mode=ForwardMode" --canonicalize | FileCheck %s --check-prefix=CONST-FWD
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=constant_store outfn=rev_constant argTys=enzyme_dup mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --canonicalize | FileCheck %s --check-prefix=CONST-REV
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=masked outfn=fwd_const_ptr argTys=enzyme_const,enzyme_dup,enzyme_dup,enzyme_const mode=ForwardMode" --canonicalize | FileCheck %s --check-prefix=FALLBACK-FWD
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=masked outfn=rev_const_ptr argTys=enzyme_const,enzyme_dup,enzyme_active,enzyme_const mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --canonicalize | FileCheck %s --check-prefix=FALLBACK-REV

// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=masked_store outfn=fwd_store argTys=enzyme_dup,enzyme_dup,enzyme_const mode=ForwardMode" --canonicalize | FileCheck %s --check-prefix=STORE-FWD
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=masked_store outfn=rev_store argTys=enzyme_dup,enzyme_active,enzyme_const mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --canonicalize | FileCheck %s --check-prefix=STORE-REV

// Every lane reads the same address. Reverse loads must accumulate atomically.
// The fallback is differentiable even when the pointer is inactive.
tt.func @masked(%in: !tt.ptr<f32>, %out: !tt.ptr<f32>, %other: tensor<4xf32>, %mask: tensor<4xi1>) {
  %ptrs = tt.splat %in : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
  %x = tt.load %ptrs, %mask, %other : tensor<4x!tt.ptr<f32>>
  %range = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32>
  %outs = tt.splat %out : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
  %dst = tt.addptr %outs, %range : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
  tt.store %dst, %x : tensor<4x!tt.ptr<f32>>
  tt.return
}

tt.func @constant_store(%out: !tt.ptr<f32>) {
  %c = arith.constant 7.0 : f32
  tt.store %out, %c : !tt.ptr<f32>
  tt.return
}

// FWD-LABEL: tt.func private @fwd(
// FWD-SAME: %[[IN:arg[0-9]+]]: !tt.ptr<f32>, %[[DIN:arg[0-9]+]]: !tt.ptr<f32>, %[[OUT:arg[0-9]+]]: !tt.ptr<f32>, %[[DOUT:arg[0-9]+]]: !tt.ptr<f32>, %[[OTHER:arg[0-9]+]]: tensor<4xf32>, %[[DOTHER:arg[0-9]+]]: tensor<4xf32>, %[[MASK:arg[0-9]+]]: tensor<4xi1>
// FWD: %[[PTRS:.*]] = tt.splat %[[DIN]]
// FWD: %[[DX:.*]] = tt.load %[[PTRS]], %[[MASK]], %[[DOTHER]]
// FWD: tt.store {{.*}}, %[[DX]]

// REV-LABEL: tt.func private @rev(
// REV: %[[ZERO:.*]] = arith.constant dense<0.000000e+00> : tensor<4xf32>
// REV: %[[DY:.*]] = tt.load %[[DST:[a-zA-Z0-9_]+]] : tensor<4x!tt.ptr<f32>>
// REV: tt.store %[[DST]], %[[ZERO]]
// REV: tt.atomic_rmw fadd, relaxed, gpu, {{.*}}, %[[DY]], %[[MASK:.*]] :
// REV: %[[OTHER:.*]] = arith.select %[[MASK]], %[[ZERO]], %[[DY]]
// REV: tt.return %[[OTHER]] : tensor<4xf32>

// CONST-FWD-LABEL: tt.func private @fwd_constant(
// CONST-FWD-SAME: %[[P:arg[0-9]+]]: !tt.ptr<f32>, %[[DP:arg[0-9]+]]: !tt.ptr<f32>
// CONST-FWD: %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
// CONST-FWD: tt.store %[[DP]], %[[ZERO]]
// CONST-REV-LABEL: tt.func private @rev_constant(
// CONST-REV-SAME: %[[P:arg[0-9]+]]: !tt.ptr<f32>, %[[DP:arg[0-9]+]]: !tt.ptr<f32>
// CONST-REV: %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
// CONST-REV: tt.store %[[DP]], %[[ZERO]]
// CONST-REV: tt.return

// FALLBACK-FWD-LABEL: tt.func private @fwd_const_ptr(
// FALLBACK-FWD: arith.select
// FALLBACK-FWD: tt.store
// FALLBACK-REV-LABEL: tt.func private @rev_const_ptr(
// FALLBACK-REV-NOT: tt.atomic_rmw
// FALLBACK-REV: arith.select
// FALLBACK-REV: tt.return

tt.func @masked_store(%out: !tt.ptr<f32>, %value: f32, %mask: i1) {
  tt.store %out, %value, %mask : !tt.ptr<f32>
  tt.return
}

// STORE-FWD-LABEL: tt.func private @fwd_store(
// STORE-FWD: tt.store %arg1, %arg3, %arg4 : !tt.ptr<f32>
// STORE-REV-LABEL: tt.func private @rev_store(
// STORE-REV: %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
// STORE-REV: %[[D:.*]] = tt.load %arg1, %arg3, %[[ZERO]] : !tt.ptr<f32>
// STORE-REV: tt.store %arg1, %[[ZERO]], %arg3 : !tt.ptr<f32>
// STORE-REV: tt.return %[[D]] : f32
