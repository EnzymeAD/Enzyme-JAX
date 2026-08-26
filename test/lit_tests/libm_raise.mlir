// RUN: enzymexlamlir-opt %s --libdevice-funcs-raise | FileCheck %s

// CPU code that keeps math errno calls libm by name -- sqrt, not
// llvm.intr.sqrt -- and a call the raising does not recognize is opaque to
// everything downstream: the forward-mode AD saw no way to differentiate
// MFEM's minimal-surface energy and returned zero gradients. The plain libm
// names raise to math ops exactly like their __nv_ device twins.

module {
  llvm.func local_unnamed_addr @sqrt(f64) -> f64
  llvm.func local_unnamed_addr @sqrtf(f32) -> f32
  llvm.func local_unnamed_addr @cos(f64) -> f64
  llvm.func local_unnamed_addr @powf(f32, f32) -> f32

  llvm.func @raise_libm(%x: f64, %y: f32) -> f64 {
    %0 = llvm.call @sqrt(%x) : (f64) -> f64
    %1 = llvm.call @cos(%0) : (f64) -> f64
    %2 = llvm.call @sqrtf(%y) : (f32) -> f32
    %3 = llvm.call @powf(%2, %y) : (f32, f32) -> f32
    %4 = llvm.fpext %3 : f32 to f64
    %5 = llvm.fadd %1, %4 : f64
    llvm.return %5 : f64
  }
}

// CHECK-LABEL: llvm.func @raise_libm
// CHECK: %[[S:.+]] = math.sqrt %arg0 : f64
// CHECK: %[[C:.+]] = math.cos %[[S]] : f64
// CHECK: %[[SF:.+]] = math.sqrt %arg1 : f32
// CHECK: math.powf %[[SF]], %arg1 : f32
// CHECK-NOT: llvm.call @sqrt
