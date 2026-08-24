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
