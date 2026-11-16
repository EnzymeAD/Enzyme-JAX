// REQUIRES: triton
// RUN: not enzymexlamlir-opt %s --enzyme-wrap="infn=product outfn=diff argTys=enzyme_dup retTys=enzyme_dup mode=ForwardMode" 2>&1 | FileCheck %s
// RUN: not enzymexlamlir-opt %s --enzyme-wrap="infn=product outfn=diff argTys=enzyme_active retTys=enzyme_active mode=ReverseModeCombined" 2>&1 | FileCheck %s

// CHECK: autodiff supports single-input addf/maxnumf/minnumf reductions
// A product reduction must never silently get the sum pullback.
tt.func @product(%x: tensor<8xf32>) -> f32 {
  %s = "tt.reduce"(%x) <{axis = 0 : i32}> ({
  ^bb0(%a: f32, %b: f32):
    %v = arith.mulf %a, %b : f32
    tt.reduce.return %v : f32
  }) : (tensor<8xf32>) -> f32
  tt.return %s : f32
}
