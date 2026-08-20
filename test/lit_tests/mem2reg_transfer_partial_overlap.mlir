// RUN: enzymexlamlir-opt %s -polygeist-mem2reg -split-input-file | FileCheck %s

// A memcpy whose destination range covers a cached slot from a different
// starting offset still overwrites it. Here the loads read a[24..32), and the
// copy fills a[16..32) between them: the second load must not be forwarded to
// the first -- the copy clobbered those bytes. matches() reported the copy as
// neither the same slot nor one lying inside the other (None) and, before the
// overlap check, the clobber was dropped and the stale value forwarded. This
// is the shape MFEM's collocated-gradient kernel raised into, where the two
// halves of a shared tile are filled by adjacent copies.

llvm.func @use(f64)

// CHECK-LABEL: llvm.func @partial_overlap_clobber
llvm.func @partial_overlap_clobber(%src: !llvm.ptr) {
  %c1 = llvm.mlir.constant(1 : i32) : i32
  %c16 = llvm.mlir.constant(16 : i64) : i64
  %a = llvm.alloca %c1 x !llvm.array<8 x f64> {alignment = 16 : i64} : (i32) -> !llvm.ptr
  %at24 = llvm.getelementptr %a[24] : (!llvm.ptr) -> !llvm.ptr, i8
  %at16 = llvm.getelementptr %a[16] : (!llvm.ptr) -> !llvm.ptr, i8
  // CHECK: %[[V1:.+]] = llvm.load
  %v1 = llvm.load %at24 : !llvm.ptr -> f64
  llvm.call @use(%v1) : (f64) -> ()
  "llvm.intr.memcpy"(%at16, %src, %c16) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
  // The clobber forces a fresh load rather than reusing %[[V1]].
  // CHECK: %[[V2:.+]] = llvm.load
  // CHECK: llvm.call @use(%[[V2]])
  %v2 = llvm.load %at24 : !llvm.ptr -> f64
  llvm.call @use(%v2) : (f64) -> ()
  llvm.return
}

// -----

// A copy that does not reach the slot leaves the cached value alone: filling
// a[0..16) does not touch a[24..32), so the two loads still fold to one.

llvm.func @use(f64)

// CHECK-LABEL: llvm.func @disjoint_copy_still_forwards
llvm.func @disjoint_copy_still_forwards(%src: !llvm.ptr) {
  %c1 = llvm.mlir.constant(1 : i32) : i32
  %c16 = llvm.mlir.constant(16 : i64) : i64
  %a = llvm.alloca %c1 x !llvm.array<8 x f64> {alignment = 16 : i64} : (i32) -> !llvm.ptr
  %at24 = llvm.getelementptr %a[24] : (!llvm.ptr) -> !llvm.ptr, i8
  %at0 = llvm.getelementptr %a[0] : (!llvm.ptr) -> !llvm.ptr, i8
  // CHECK: %[[V:.+]] = llvm.load
  %v1 = llvm.load %at24 : !llvm.ptr -> f64
  llvm.call @use(%v1) : (f64) -> ()
  "llvm.intr.memcpy"(%at0, %src, %c16) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
  // CHECK-NOT: llvm.load
  // CHECK: llvm.call @use(%[[V]])
  %v2 = llvm.load %at24 : !llvm.ptr -> f64
  llvm.call @use(%v2) : (f64) -> ()
  llvm.return
}

// -----

// A direct (same-period) overlap, distinct from the partial-overlap above:
// the load reads [8,16) and the copy fills [12,20), sharing [12,16). This is
// the k=0 conflict that the stride comparison, on its own, never checked --
// it only ruled out a spill into the next period.

llvm.func @use(f64)

// CHECK-LABEL: llvm.func @direct_overlap_clobber
llvm.func @direct_overlap_clobber(%src: !llvm.ptr) {
  %c1 = llvm.mlir.constant(1 : i32) : i32
  %c8 = llvm.mlir.constant(8 : i64) : i64
  %a = llvm.alloca %c1 x !llvm.array<8 x f64> {alignment = 16 : i64} : (i32) -> !llvm.ptr
  %at8 = llvm.getelementptr %a[8] : (!llvm.ptr) -> !llvm.ptr, i8
  %at12 = llvm.getelementptr %a[12] : (!llvm.ptr) -> !llvm.ptr, i8
  // CHECK: %[[V1:.+]] = llvm.load
  %v1 = llvm.load %at8 : !llvm.ptr -> f64
  llvm.call @use(%v1) : (f64) -> ()
  "llvm.intr.memcpy"(%at12, %src, %c8) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
  // CHECK: %[[V2:.+]] = llvm.load
  // CHECK: llvm.call @use(%[[V2]])
  %v2 = llvm.load %at8 : !llvm.ptr -> f64
  llvm.call @use(%v2) : (f64) -> ()
  llvm.return
}
