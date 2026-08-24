// RUN: enzymexlamlir-opt %s -polygeist-mem2reg -split-input-file | FileCheck %s

// A slot built one field at a time holds the fields folded over the undefined
// value the allocation began with, and a read of the whole takes that.
llvm.func @use(!llvm.struct<(i32, f64)>)
llvm.func @built_by_fields(%x: i32, %y: f64) {
  %c1 = llvm.mlir.constant(1 : i32) : i32
  %mem = llvm.alloca %c1 x !llvm.struct<(i32, f64)> : (i32) -> !llvm.ptr
  %f0 = llvm.getelementptr %mem[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, f64)>
  llvm.store %x, %f0 : i32, !llvm.ptr
  %f1 = llvm.getelementptr %mem[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, f64)>
  llvm.store %y, %f1 : f64, !llvm.ptr
  %v = llvm.load %mem : !llvm.ptr -> !llvm.struct<(i32, f64)>
  llvm.call @use(%v) : (!llvm.struct<(i32, f64)>) -> ()
  llvm.return
}

// CHECK-LABEL: llvm.func @built_by_fields(
// CHECK-SAME: %[[X:[a-z0-9]+]]: i32, %[[Y:[a-z0-9]+]]: f64
// CHECK-NOT: llvm.alloca
// CHECK: %[[U:.+]] = llvm.mlir.undef : !llvm.struct<(i32, f64)>
// CHECK: %[[I0:.+]] = llvm.insertvalue %[[X]], %[[U]][0]
// CHECK: %[[I1:.+]] = llvm.insertvalue %[[Y]], %[[I0]][1]
// CHECK: llvm.call @use(%[[I1]])

// -----

// A field written into an element of an array member lands at its path.
llvm.func @usef(f64, f64)
llvm.func @field_in_array(%v: !llvm.struct<(i32, array<4 x f64>)>, %e: f64) {
  %c1 = llvm.mlir.constant(1 : i32) : i32
  %mem = llvm.alloca %c1 x !llvm.struct<(i32, array<4 x f64>)> : (i32) -> !llvm.ptr
  llvm.store %v, %mem : !llvm.struct<(i32, array<4 x f64>)>, !llvm.ptr
  %g = llvm.getelementptr %mem[0, 1, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, array<4 x f64>)>
  llvm.store %e, %g : f64, !llvm.ptr
  %g1 = llvm.getelementptr %mem[0, 1, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, array<4 x f64>)>
  %a = llvm.load %g1 : !llvm.ptr -> f64
  %g2 = llvm.getelementptr %mem[0, 1, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, array<4 x f64>)>
  %b = llvm.load %g2 : !llvm.ptr -> f64
  llvm.call @usef(%a, %b) : (f64, f64) -> ()
  llvm.return
}

// CHECK-LABEL: llvm.func @field_in_array(
// CHECK-SAME: %[[VAL:[a-z0-9]+]]: !llvm.struct<(i32, array<4 x f64>)>, %[[E:[a-z0-9]+]]: f64
// CHECK-NOT: llvm.alloca
// CHECK: %[[INS:.+]] = llvm.insertvalue %[[E]], %[[VAL]][1, 2]
// CHECK: %[[A:.+]] = llvm.extractvalue %[[INS]][1, 1]
// CHECK: llvm.call @usef(%[[A]], %[[E]])

// -----

// A piece store after a write this cannot see has nothing to fold into: what
// the slot holds after it is unknown, and the read after stays a read.
llvm.func @clobber(!llvm.ptr)
llvm.func @piece_after_unknown(%x: i32) -> !llvm.struct<(i32, i32)> {
  %c1 = llvm.mlir.constant(1 : i32) : i32
  %mem = llvm.alloca %c1 x !llvm.struct<(i32, i32)> : (i32) -> !llvm.ptr
  llvm.call @clobber(%mem) : (!llvm.ptr) -> ()
  %f0 = llvm.getelementptr %mem[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32)>
  llvm.store %x, %f0 : i32, !llvm.ptr
  %v = llvm.load %mem : !llvm.ptr -> !llvm.struct<(i32, i32)>
  llvm.return %v : !llvm.struct<(i32, i32)>
}

// CHECK-LABEL: llvm.func @piece_after_unknown(
// CHECK: llvm.call @clobber(
// CHECK: llvm.store
// CHECK: %[[LD:.+]] = llvm.load
// CHECK: llvm.return %[[LD]]

// -----

// The fields are written in one block and the whole read in another: the
// value crosses as a block argument, undef seed and all.
llvm.func @useB(!llvm.struct<(i32, f64)>)
llvm.func @across_blocks(%x: i32, %y: f64, %c: i1) {
  %c1 = llvm.mlir.constant(1 : i32) : i32
  %p = llvm.alloca %c1 x !llvm.struct<(i32, f64)> : (i32) -> !llvm.ptr
  cf.cond_br %c, ^bb1, ^bb1
^bb1:
  %g0 = llvm.getelementptr %p[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, f64)>
  llvm.store %x, %g0 : i32, !llvm.ptr
  %g1 = llvm.getelementptr %p[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, f64)>
  llvm.store %y, %g1 : f64, !llvm.ptr
  %v = llvm.load %p : !llvm.ptr -> !llvm.struct<(i32, f64)>
  llvm.call @useB(%v) : (!llvm.struct<(i32, f64)>) -> ()
  llvm.return
}

// CHECK-LABEL: llvm.func @across_blocks(
// CHECK-SAME: %[[X:[a-z0-9]+]]: i32, %[[Y:[a-z0-9]+]]: f64
// CHECK-NOT: llvm.alloca
// CHECK: %[[U:.+]] = llvm.mlir.undef : !llvm.struct<(i32, f64)>
// CHECK: %[[I0:.+]] = llvm.insertvalue %[[X]], %[[U]][0]
// CHECK: %[[I1:.+]] = llvm.insertvalue %[[Y]], %[[I0]][1]
// CHECK: llvm.call @useB(%[[I1]])

// -----

// An integer written over a field and the padding after it lands its slice on
// the field; one written over two fields lands a slice on each.
llvm.func @usew(!llvm.struct<(struct<(i32, ptr)>, i32, i32)>)
llvm.func @wide_stores(%pair: i64, %two: i64) {
  %c1 = llvm.mlir.constant(1 : i32) : i32
  %p = llvm.alloca %c1 x !llvm.struct<(struct<(i32, ptr)>, i32, i32)> : (i32) -> !llvm.ptr
  llvm.store %pair, %p : i64, !llvm.ptr
  %g = llvm.getelementptr %p[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(struct<(i32, ptr)>, i32, i32)>
  llvm.store %two, %g : i64, !llvm.ptr
  %v = llvm.load %p : !llvm.ptr -> !llvm.struct<(struct<(i32, ptr)>, i32, i32)>
  llvm.call @usew(%v) : (!llvm.struct<(struct<(i32, ptr)>, i32, i32)>) -> ()
  llvm.return
}

// CHECK-LABEL: llvm.func @wide_stores(
// CHECK-SAME: %[[PAIR:[a-z0-9]+]]: i64, %[[TWO:[a-z0-9]+]]: i64
// CHECK-NOT: llvm.alloca
// CHECK-DAG: %[[PLO:.+]] = llvm.trunc %[[PAIR]] : i64 to i32
// CHECK: llvm.insertvalue %[[PLO]], %{{.+}}[0, 0]
// CHECK-DAG: %[[TLO:.+]] = llvm.trunc %[[TWO]] : i64 to i32
// CHECK: llvm.insertvalue %[[TLO]], %{{.+}}[1]
// CHECK: %[[C32:.+]] = llvm.mlir.constant(32 : i64) : i64
// CHECK: %[[SH:.+]] = llvm.lshr %[[TWO]], %[[C32]]
// CHECK: %[[THI:.+]] = llvm.trunc %[[SH]] : i64 to i32
// CHECK: %[[FIN:.+]] = llvm.insertvalue %[[THI]], %{{.+}}[2]
// CHECK: llvm.call @usew(%[[FIN]])

// -----

// An integer stored over an aggregate member of the same extent -- a dim3
// pair written as one i64 -- lands as the aggregate, spelled through the
// matching vector.
llvm.func @used(!llvm.struct<(i32, ptr, array<2 x i32>)>)
llvm.func @dims_pair(%v: !llvm.struct<(i32, ptr, array<2 x i32>)>, %pair: i64) {
  %c1 = llvm.mlir.constant(1 : i32) : i32
  %p = llvm.alloca %c1 x !llvm.struct<(i32, ptr, array<2 x i32>)> : (i32) -> !llvm.ptr
  llvm.store %v, %p : !llvm.struct<(i32, ptr, array<2 x i32>)>, !llvm.ptr
  %g = llvm.getelementptr %p[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, ptr, array<2 x i32>)>
  llvm.store %pair, %g : i64, !llvm.ptr
  %w = llvm.load %p : !llvm.ptr -> !llvm.struct<(i32, ptr, array<2 x i32>)>
  llvm.call @used(%w) : (!llvm.struct<(i32, ptr, array<2 x i32>)>) -> ()
  llvm.return
}

// CHECK-LABEL: llvm.func @dims_pair(
// CHECK-SAME: %[[V:[a-z0-9]+]]: !llvm.struct<(i32, ptr, array<2 x i32>)>, %[[PAIR:[a-z0-9]+]]: i64
// CHECK-NOT: llvm.alloca
// CHECK: %[[VEC:.+]] = llvm.bitcast %[[PAIR]] : i64 to vector<2xi32>
// CHECK: %[[E0:.+]] = llvm.extractelement %[[VEC]]
// CHECK: llvm.insertvalue %[[E0]], %{{.+}}[0]
// CHECK: %[[E1:.+]] = llvm.extractelement %[[VEC]]
// CHECK: %[[ARR:.+]] = llvm.insertvalue %[[E1]], %{{.+}}[1]
// CHECK: %[[FIN:.+]] = llvm.insertvalue %[[ARR]], %[[V]][2]
// CHECK: llvm.call @used(%[[FIN]])
