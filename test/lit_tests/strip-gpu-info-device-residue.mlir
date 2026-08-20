// RUN: enzymexlamlir-opt %s --strip-gpu-info --split-input-file | FileCheck %s

// Extracting device code into gpu modules can leave renamed host-side residue
// behind: internal-linkage clones of device functions, held only by equally
// dead device vtable globals. LLVM internal linkage is not MLIR private
// visibility, so symbol-dce keeps them, and the host backend fatals on their
// sm_* subtargets. strip-gpu-info erases the residue.

module {
  llvm.func internal @_ZNK4mfem6metricEvalW.53(%arg0: !llvm.ptr) -> f64 attributes {dso_local, target_cpu = "sm_120"} {
    %0 = llvm.mlir.constant(0.0 : f64) : f64
    llvm.return %0 : f64
  }
  llvm.mlir.global internal unnamed_addr constant @_ZTVN4mfem6metricE.51() {addr_space = 0 : i32, alignment = 8 : i64, dso_local} : !llvm.struct<(array<1 x ptr>)> {
    %0 = llvm.mlir.undef : !llvm.struct<(array<1 x ptr>)>
    %1 = llvm.mlir.addressof @_ZNK4mfem6metricEvalW.53 : !llvm.ptr
    %2 = llvm.mlir.undef : !llvm.array<1 x ptr>
    %3 = llvm.insertvalue %1, %2[0] : !llvm.array<1 x ptr>
    %4 = llvm.insertvalue %3, %0[0] : !llvm.struct<(array<1 x ptr>)>
    llvm.return %4 : !llvm.struct<(array<1 x ptr>)>
  }
  llvm.func @host(%p: !llvm.ptr) -> f64 attributes {target_cpu = "x86-64"} {
    %0 = llvm.mlir.constant(1.0 : f64) : f64
    llvm.return %0 : f64
  }
}

// CHECK-NOT: EvalW
// CHECK-NOT: _ZTVN4mfem6metricE
// CHECK: llvm.func @host

// -----

// A device function something still holds beyond the dead vtable stays.

module {
  llvm.func internal @kept.1(%arg0: !llvm.ptr) -> f64 attributes {target_cpu = "sm_120"} {
    %0 = llvm.mlir.constant(0.0 : f64) : f64
    llvm.return %0 : f64
  }
  llvm.func @host2() -> !llvm.ptr {
    %0 = llvm.mlir.addressof @kept.1 : !llvm.ptr
    llvm.return %0 : !llvm.ptr
  }
}

// CHECK: llvm.func internal @kept.1
// CHECK: llvm.func @host2
