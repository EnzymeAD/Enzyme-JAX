// RUN: enzymexlamlir-opt %s -polygeist-mem2reg -split-input-file | FileCheck %s

// A copy reading exactly what an allocation holds reads the value in it, so it
// is a store of that value into wherever it was copying to.
llvm.func @memcpy_src(%dst: !llvm.ptr) -> i32 {
  %c1 = llvm.mlir.constant(1 : i32) : i32
  %len = llvm.mlir.constant(4 : i64) : i64
  %val = llvm.mlir.constant(42 : i32) : i32
  %mem = llvm.alloca %c1 x i32 : (i32) -> !llvm.ptr
  llvm.store %val, %mem : i32, !llvm.ptr
  "llvm.intr.memcpy"(%dst, %mem, %len) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
  %loaded = llvm.load %mem : !llvm.ptr -> i32
  llvm.return %loaded : i32
}

// CHECK-LABEL: llvm.func @memcpy_src(
// CHECK-SAME: %[[DST:[a-z0-9]+]]: !llvm.ptr
// CHECK: %[[C42:.+]] = llvm.mlir.constant(42 : i32) : i32
// CHECK-NOT: llvm.alloca
// CHECK-NOT: "llvm.intr.memcpy"
// CHECK: llvm.store %[[C42]], %[[DST]]
// CHECK: llvm.return %[[C42]] : i32

// -----

// The same for a memmove, which moves what it reads before anything else sees
// it either way.
llvm.func @memmove_src(%dst: !llvm.ptr) -> i32 {
  %c1 = llvm.mlir.constant(1 : i32) : i32
  %len = llvm.mlir.constant(4 : i64) : i64
  %val = llvm.mlir.constant(42 : i32) : i32
  %mem = llvm.alloca %c1 x i32 : (i32) -> !llvm.ptr
  llvm.store %val, %mem : i32, !llvm.ptr
  "llvm.intr.memmove"(%dst, %mem, %len) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
  %loaded = llvm.load %mem : !llvm.ptr -> i32
  llvm.return %loaded : i32
}

// CHECK-LABEL: llvm.func @memmove_src(
// CHECK-SAME: %[[DST:[a-z0-9]+]]: !llvm.ptr
// CHECK: %[[C42:.+]] = llvm.mlir.constant(42 : i32) : i32
// CHECK-NOT: "llvm.intr.memmove"
// CHECK: llvm.store %[[C42]], %[[DST]]
// CHECK: llvm.return %[[C42]] : i32

// -----

// A copy writing the allocation puts what it read there, which is not what the
// store before it put there: that store does not reach the load after it, and
// what the copy read does.
llvm.func @memcpy_dst(%src: !llvm.ptr) -> i32 {
  %c1 = llvm.mlir.constant(1 : i32) : i32
  %len = llvm.mlir.constant(4 : i64) : i64
  %val = llvm.mlir.constant(42 : i32) : i32
  %mem = llvm.alloca %c1 x i32 : (i32) -> !llvm.ptr
  llvm.store %val, %mem : i32, !llvm.ptr
  "llvm.intr.memcpy"(%mem, %src, %len) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
  %loaded = llvm.load %mem : !llvm.ptr -> i32
  llvm.return %loaded : i32
}

// CHECK-LABEL: llvm.func @memcpy_dst(
// CHECK-SAME: %[[SRC:[a-z0-9]+]]: !llvm.ptr
// CHECK-NOT: "llvm.intr.memcpy"
// CHECK: %[[LD:.+]] = llvm.load %[[SRC]]
// CHECK: llvm.return %[[LD]] : i32

// -----

// Filling the allocation with zero bytes leaves a zero of what it holds, which
// is what a read of it finds.
llvm.func @memset_dst() -> i32 {
  %c1 = llvm.mlir.constant(1 : i32) : i32
  %len = llvm.mlir.constant(4 : i64) : i64
  %byte = llvm.mlir.constant(0 : i8) : i8
  %val = llvm.mlir.constant(42 : i32) : i32
  %mem = llvm.alloca %c1 x i32 : (i32) -> !llvm.ptr
  llvm.store %val, %mem : i32, !llvm.ptr
  "llvm.intr.memset"(%mem, %byte, %len) <{isVolatile = false}> : (!llvm.ptr, i8, i64) -> ()
  %loaded = llvm.load %mem : !llvm.ptr -> i32
  llvm.return %loaded : i32
}

// CHECK-LABEL: llvm.func @memset_dst(
// CHECK-NOT: "llvm.intr.memset"
// CHECK: %[[Z:.+]] = llvm.mlir.zero : i32
// CHECK: llvm.return %[[Z]] : i32

// -----

// Filled with something else, what is there is only known once it is said what
// those bytes make of the value.
llvm.func @memset_nonzero() -> i32 {
  %c1 = llvm.mlir.constant(1 : i32) : i32
  %len = llvm.mlir.constant(4 : i64) : i64
  %byte = llvm.mlir.constant(7 : i8) : i8
  %val = llvm.mlir.constant(42 : i32) : i32
  %mem = llvm.alloca %c1 x i32 : (i32) -> !llvm.ptr
  llvm.store %val, %mem : i32, !llvm.ptr
  "llvm.intr.memset"(%mem, %byte, %len) <{isVolatile = false}> : (!llvm.ptr, i8, i64) -> ()
  %loaded = llvm.load %mem : !llvm.ptr -> i32
  llvm.return %loaded : i32
}

// CHECK-LABEL: llvm.func @memset_nonzero(
// CHECK: "llvm.intr.memset"
// CHECK: %[[LD:.+]] = llvm.load
// CHECK: llvm.return %[[LD]] : i32

// -----

// Reading part of what an allocation holds is not reading the value in it, so
// the copy stays and so does what it copies from.
llvm.func @partial_copy(%dst: !llvm.ptr) {
  %c1 = llvm.mlir.constant(1 : i32) : i32
  %len = llvm.mlir.constant(2 : i64) : i64
  %val = llvm.mlir.constant(42 : i32) : i32
  %mem = llvm.alloca %c1 x i32 : (i32) -> !llvm.ptr
  llvm.store %val, %mem : i32, !llvm.ptr
  "llvm.intr.memcpy"(%dst, %mem, %len) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
  llvm.return
}

// CHECK-LABEL: llvm.func @partial_copy(
// CHECK: llvm.alloca
// CHECK: "llvm.intr.memcpy"

// -----

// A copy into one field says nothing about another, so what was written there
// is still what is read there.
llvm.func @copy_into_other_field(%src: !llvm.ptr, %val: f32) -> f32 {
  %c1 = llvm.mlir.constant(1 : i32) : i32
  %len = llvm.mlir.constant(4 : i64) : i64
  %mem = llvm.alloca %c1 x !llvm.struct<(f32, f32)> : (i32) -> !llvm.ptr
  %f0 = llvm.getelementptr %mem[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(f32, f32)>
  llvm.store %val, %f0 : f32, !llvm.ptr
  %f1 = llvm.getelementptr %mem[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(f32, f32)>
  "llvm.intr.memcpy"(%f1, %src, %len) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
  %loaded = llvm.load %f0 : !llvm.ptr -> f32
  llvm.return %loaded : f32
}

// CHECK-LABEL: llvm.func @copy_into_other_field(
// CHECK-SAME: %{{[a-z0-9]+}}: !llvm.ptr, %[[VAL:[a-z0-9]+]]: f32
// CHECK-NOT: llvm.load
// CHECK: llvm.return %[[VAL]] : f32

// -----

// An allocation only ever written -- through a view, an offset of it, and a
// transfer into it -- is dead and goes away with all of them.
llvm.func @dead_alloca(%src: !llvm.ptr) {
  %c0 = arith.constant 0 : index
  %c1 = llvm.mlir.constant(1 : i32) : i32
  %c2 = llvm.mlir.constant(2 : i64) : i64
  %len = llvm.mlir.constant(4 : i64) : i64
  %byte = llvm.mlir.constant(0 : i8) : i8
  %val = llvm.mlir.constant(42 : i32) : i32
  %mem = llvm.alloca %c1 x !llvm.array<4 x i32> : (i32) -> !llvm.ptr
  llvm.intr.lifetime.start %mem : !llvm.ptr
  %gep = llvm.getelementptr %mem[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<4 x i32>
  llvm.store %val, %gep : i32, !llvm.ptr
  %view = "enzymexla.pointer2memref"(%mem) : (!llvm.ptr) -> memref<?xi32>
  memref.store %val, %view[%c0] : memref<?xi32>
  "llvm.intr.memcpy"(%mem, %src, %len) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
  "llvm.intr.memset"(%mem, %byte, %len) <{isVolatile = false}> : (!llvm.ptr, i8, i64) -> ()
  llvm.intr.lifetime.end %mem : !llvm.ptr
  llvm.return
}

// CHECK-LABEL: llvm.func @dead_alloca(
// CHECK-NOT: llvm.alloca
// CHECK-NOT: llvm.intr.mem
// CHECK: llvm.return

// -----

// Storing the pointer itself hands it to whoever reads that memory, so nothing
// may be forwarded through it.
llvm.func @escaped_through_store(%out: !llvm.ptr) -> i32 {
  %c1 = llvm.mlir.constant(1 : i32) : i32
  %val = llvm.mlir.constant(42 : i32) : i32
  %mem = llvm.alloca %c1 x i32 : (i32) -> !llvm.ptr
  llvm.store %val, %mem : i32, !llvm.ptr
  llvm.store %mem, %out : !llvm.ptr, !llvm.ptr
  %loaded = llvm.load %mem : !llvm.ptr -> i32
  llvm.return %loaded : i32
}

// CHECK-LABEL: llvm.func @escaped_through_store(
// CHECK: %[[LD:.+]] = llvm.load
// CHECK: llvm.return %[[LD]] : i32

// -----

// A store put where a copy was may assume of the destination only what the copy
// said of it, which is what the copy carries rather than what the source it
// read from happened to be.
llvm.func @copy_states_alignment(%dst: !llvm.ptr, %val: i32) {
  %c1 = llvm.mlir.constant(1 : i32) : i32
  %len = llvm.mlir.constant(4 : i64) : i64
  %mem = llvm.alloca %c1 x i32 : (i32) -> !llvm.ptr
  llvm.store %val, %mem : i32, !llvm.ptr
  "llvm.intr.memcpy"(%dst, %mem, %len) <{arg_attrs = [{llvm.align = 4 : i64}, {}, {}], isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
  llvm.return
}

// CHECK-LABEL: llvm.func @copy_states_alignment(
// CHECK-SAME: %[[DST:[a-z0-9]+]]: !llvm.ptr, %[[VAL:[a-z0-9]+]]: i32
// CHECK: llvm.store %[[VAL]], %[[DST]] {alignment = 4 : i64}

// -----

// Saying nothing of it, a byte is all that may be assumed -- whatever the
// allocation it copied out of was aligned to.
llvm.func @copy_states_nothing(%dst: !llvm.ptr, %val: i32) {
  %c1 = llvm.mlir.constant(1 : i32) : i32
  %len = llvm.mlir.constant(4 : i64) : i64
  %mem = llvm.alloca %c1 x i32 {alignment = 8 : i64} : (i32) -> !llvm.ptr
  llvm.store %val, %mem : i32, !llvm.ptr
  "llvm.intr.memcpy"(%dst, %mem, %len) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
  llvm.return
}

// CHECK-LABEL: llvm.func @copy_states_nothing(
// CHECK-SAME: %[[DST:[a-z0-9]+]]: !llvm.ptr, %[[VAL:[a-z0-9]+]]: i32
// CHECK: llvm.store %[[VAL]], %[[DST]] {alignment = 1 : i64}

// -----

// A copy into exactly what an allocation holds leaves what it read there, so a
// read of it afterwards is a read of where the copy came from.
llvm.func @copy_in(%src: !llvm.ptr) -> i32 {
  %c1 = llvm.mlir.constant(1 : i32) : i32
  %len = llvm.mlir.constant(4 : i64) : i64
  %mem = llvm.alloca %c1 x i32 : (i32) -> !llvm.ptr
  "llvm.intr.memcpy"(%mem, %src, %len) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
  %loaded = llvm.load %mem : !llvm.ptr -> i32
  llvm.return %loaded : i32
}

// CHECK-LABEL: llvm.func @copy_in(
// CHECK-SAME: %[[SRC:[a-z0-9]+]]: !llvm.ptr
// CHECK-NOT: llvm.alloca
// CHECK-NOT: "llvm.intr.memcpy"
// CHECK: %[[LD:.+]] = llvm.load %[[SRC]]
// CHECK: llvm.return %[[LD]] : i32

// -----

// Copied from one allocation to another and read back, the value goes straight
// across and neither allocation is left.
llvm.func @copy_between(%val: i32) -> i32 {
  %c1 = llvm.mlir.constant(1 : i32) : i32
  %len = llvm.mlir.constant(4 : i64) : i64
  %a = llvm.alloca %c1 x i32 : (i32) -> !llvm.ptr
  %b = llvm.alloca %c1 x i32 : (i32) -> !llvm.ptr
  llvm.store %val, %a : i32, !llvm.ptr
  "llvm.intr.memcpy"(%b, %a, %len) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
  %loaded = llvm.load %b : !llvm.ptr -> i32
  llvm.return %loaded : i32
}

// CHECK-LABEL: llvm.func @copy_between(
// CHECK-SAME: %[[VAL:[a-z0-9]+]]: i32
// CHECK-NOT: llvm.alloca
// CHECK-NOT: "llvm.intr.memcpy"
// CHECK: llvm.return %[[VAL]] : i32

// -----

// What a transfer leaves in a slot is threaded to the reads of it like anything
// else, which for a read past a join means through a block argument -- so it
// has to be of what the slot holds, not of the bytes that were moved.
func.func @transfer_through_branch(%cond: i1, %src: !llvm.ptr) -> i32 {
  %c1 = llvm.mlir.constant(1 : i32) : i32
  %len = llvm.mlir.constant(4 : i64) : i64
  %byte = llvm.mlir.constant(0 : i8) : i8
  %mem = llvm.alloca %c1 x i32 : (i32) -> !llvm.ptr
  cf.cond_br %cond, ^bb1, ^bb2
^bb1:
  "llvm.intr.memcpy"(%mem, %src, %len) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
  cf.br ^bb3
^bb2:
  "llvm.intr.memset"(%mem, %byte, %len) <{isVolatile = false}> : (!llvm.ptr, i8, i64) -> ()
  cf.br ^bb3
^bb3:
  %loaded = llvm.load %mem : !llvm.ptr -> i32
  return %loaded : i32
}

// CHECK-LABEL: func.func @transfer_through_branch(
// CHECK-SAME: %{{[a-z0-9]+}}: i1, %[[SRC:[a-z0-9]+]]: !llvm.ptr
// CHECK-NOT: llvm.alloca
// CHECK: %[[LD:.+]] = llvm.load %[[SRC]]
// CHECK: cf.br ^[[JOIN:.+]](%[[LD]] : i32)
// CHECK: %[[Z:.+]] = llvm.mlir.zero : i32
// CHECK: cf.br ^[[JOIN]](%[[Z]] : i32)
// CHECK: ^[[JOIN]](%[[ARG:.+]]: i32):
// CHECK: return %[[ARG]] : i32

// -----

// One value of an allocation may be spelled two ways -- here a pointer and the
// struct that holds one -- and what reaches a read of it through a block
// argument has to be spelled as that argument is.
func.func @transfer_of_wrapped_pointer(%cond: i1, %src: !llvm.ptr, %p: !llvm.ptr) -> !llvm.ptr {
  %c1 = llvm.mlir.constant(1 : i32) : i32
  %len = llvm.mlir.constant(8 : i64) : i64
  %undef = llvm.mlir.undef : !llvm.struct<(ptr)>
  %wrapped = llvm.insertvalue %p, %undef[0] : !llvm.struct<(ptr)>
  %mem = llvm.alloca %c1 x !llvm.struct<(ptr)> : (i32) -> !llvm.ptr
  cf.cond_br %cond, ^bb1, ^bb2
^bb1:
  llvm.store %wrapped, %mem : !llvm.struct<(ptr)>, !llvm.ptr
  cf.br ^bb3
^bb2:
  "llvm.intr.memcpy"(%mem, %src, %len) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
  cf.br ^bb3
^bb3:
  %loaded = llvm.load %mem : !llvm.ptr -> !llvm.ptr
  return %loaded : !llvm.ptr
}

// CHECK-LABEL: func.func @transfer_of_wrapped_pointer(
// CHECK-NOT: llvm.alloca
// CHECK: ^{{.+}}(%[[ARG:.+]]: !llvm.ptr):
// CHECK: return %[[ARG]] : !llvm.ptr
