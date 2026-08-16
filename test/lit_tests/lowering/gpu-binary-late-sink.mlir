// RUN: env REACTANT_SINK_DEBUG=1 enzymexlamlir-opt %s --pass-pipeline="builtin.module(enzymexla-gpu-module-to-binary{format=isa sink=2})" -o /dev/null 2>&1 | FileCheck %s
// RUN: env REACTANT_SINK_DEBUG=1 enzymexlamlir-opt %s --pass-pipeline="builtin.module(enzymexla-gpu-module-to-binary{format=isa sink=0})" -o /dev/null 2>&1 | FileCheck %s --check-prefix=NOSINK

// The base offset %e * 522 feeds stores on both sides of the branch, so the
// target's LLVM optimization pipeline keeps a single multiply (folded with the
// f64 element size into * 4176) live across the branch in the entry block.
// The late sink runs after that pipeline and rematerializes the multiply and
// the base getelementptr in each using block, where no further IR-level
// CSE/GVN can re-hoist them; REACTANT_SINK_DEBUG dumps the IR exactly as it
// is handed to instruction selection.

module attributes {gpu.container_module} {
  gpu.module @kmod [#nvvm.target<O = 3, chip = "sm_90", features = "+ptx80,+sm_90", flags = {}>] {
    llvm.func @walled(%out: !llvm.ptr, %e: i64, %cond: i1) attributes {gpu.kernel, nvvm.kernel} {
      %c522 = llvm.mlir.constant(522 : i64) : i64
      %c1 = llvm.mlir.constant(1 : i64) : i64
      %c2 = llvm.mlir.constant(2 : i64) : i64
      %c3 = llvm.mlir.constant(3 : i64) : i64
      %f1 = llvm.mlir.constant(1.000000e+00 : f64) : f64
      %f2 = llvm.mlir.constant(2.000000e+00 : f64) : f64
      %base = llvm.mul %e, %c522 : i64
      llvm.cond_br %cond, ^bb1, ^bb2
    ^bb1:
      %o1 = llvm.add %base, %c1 : i64
      %p1 = llvm.getelementptr %out[%o1] : (!llvm.ptr, i64) -> !llvm.ptr, f64
      llvm.store %f1, %p1 : f64, !llvm.ptr
      %o3 = llvm.add %base, %c3 : i64
      %p3 = llvm.getelementptr %out[%o3] : (!llvm.ptr, i64) -> !llvm.ptr, f64
      llvm.store %f1, %p3 : f64, !llvm.ptr
      llvm.br ^bb3
    ^bb2:
      %o2 = llvm.add %base, %c2 : i64
      %p2 = llvm.getelementptr %out[%o2] : (!llvm.ptr, i64) -> !llvm.ptr, f64
      llvm.store %f2, %p2 : f64, !llvm.ptr
      llvm.br ^bb3
    ^bb3:
      llvm.return
    }
  }
}

// With sinking the entry block ends at the branch and each arm recomputes the
// multiply locally.
// CHECK: define ptx_kernel void @walled
// CHECK-NEXT: br i1 %2, label %[[THEN:[0-9]+]], label %[[TAIL:[0-9]+]]
// CHECK: [[THEN]]:
// CHECK-NEXT: mul i64 %1, 4176
// CHECK: [[TAIL]]:
// CHECK: phi
// CHECK: mul i64 %1, 4176

// Without sinking the multiply stays hoisted in the entry block, live across
// the branch.
// NOSINK: define ptx_kernel void @walled
// NOSINK-NEXT: mul i64 %1, 4176
// NOSINK: br i1
// NOSINK-NOT: mul i64
