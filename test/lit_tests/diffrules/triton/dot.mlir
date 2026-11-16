// REQUIRES: triton
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=dot outfn=fwd argTys=enzyme_dup,enzyme_dup,enzyme_dup retTys=enzyme_dup mode=ForwardMode" --canonicalize | FileCheck %s --check-prefix=FWD
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=dot outfn=rev argTys=enzyme_active,enzyme_active,enzyme_active retTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --canonicalize | FileCheck %s --check-prefix=REV
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=dot outfn=partial argTys=enzyme_active,enzyme_const,enzyme_const retTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --canonicalize | FileCheck %s --check-prefix=PARTIAL

tt.func @dot(%a: tensor<16x32xf16>, %b: tensor<32x64xf16>, %c: tensor<16x64xf32>) -> tensor<16x64xf32> {
  %d = tt.dot %a, %b, %c, inputPrecision = tf32 : tensor<16x32xf16> * tensor<32x64xf16> -> tensor<16x64xf32>
  tt.return %d : tensor<16x64xf32>
}

// FWD-LABEL: tt.func private @fwd(
// FWD: %[[DA:.*]] = tt.dot %arg1, %arg2, %arg5
// FWD: %[[DB:.*]] = tt.dot %arg0, %arg3, %[[DA]]
// FWD: tt.return {{.*}}%[[DB]]
// REV-LABEL: tt.func private @rev(
// REV: tt.trans {{.*}} : tensor<32x64xf16> -> tensor<64x32xf16>
// REV: arith.extf {{.*}} : tensor<64x32xf16> to tensor<64x32xf32>
// REV: tt.dot {{.*}} : tensor<16x64xf32> * tensor<64x32xf32> -> tensor<16x32xf32>
// REV: arith.truncf {{.*}} : tensor<16x32xf32> to tensor<16x32xf16>
// REV: tt.trans {{.*}} : tensor<16x32xf16> -> tensor<32x16xf16>
// REV: tt.dot {{.*}} : tensor<32x16xf32> * tensor<16x64xf32> -> tensor<32x64xf32>
// REV: arith.truncf {{.*}} : tensor<32x64xf32> to tensor<32x64xf16>
// REV: tt.return {{.*}} : tensor<16x32xf16>, tensor<32x64xf16>, tensor<16x64xf32>
// PARTIAL-LABEL: tt.func private @partial(
// PARTIAL: tt.trans
// PARTIAL: tt.dot
// PARTIAL-NOT: tt.dot
// PARTIAL: tt.return {{.*}} : tensor<16x32xf16>
