// RUN: enzymexlamlir-opt --polygeist-mem2reg --split-input-file %s | FileCheck %s

// A lane's copy of a by-value capture, stored whole and read back at an index
// that depends on the lane: the members read become an allocation of their own,
// and the members never read (the pointer among them) are dropped.
module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f64 = dense<64> : vector<2xi64>>} {
  func.func @lane(%cap: !llvm.ptr, %out: memref<2x2xf64>) {
    %one = arith.constant 1 : i64
    %v = "enzymexla.pointer2memref"(%cap) : (!llvm.ptr) -> memref<?x!llvm.struct<(i32, ptr, array<2 x array<2 x f64>>)>>
    %agg = affine.load %v[0] : memref<?x!llvm.struct<(i32, ptr, array<2 x array<2 x f64>>)>>
    affine.parallel (%qx) = (0) to (2) {
      %copy = llvm.alloca %one x !llvm.struct<(i32, ptr, array<2 x array<2 x f64>>)> : (i64) -> !llvm.ptr
      %w = "enzymexla.pointer2memref"(%copy) : (!llvm.ptr) -> memref<?x!llvm.struct<(i32, ptr, array<2 x array<2 x f64>>)>>
      affine.store %agg, %w[0] : memref<?x!llvm.struct<(i32, ptr, array<2 x array<2 x f64>>)>>
      %d = "enzymexla.pointer2memref"(%copy) : (!llvm.ptr) -> memref<?xf64>
      %b0 = affine.load %d[%qx * 2 + 2] : memref<?xf64>
      %b1 = affine.load %d[%qx * 2 + 3] : memref<?xf64>
      affine.store %b0, %out[%qx, 0] : memref<2x2xf64>
      affine.store %b1, %out[%qx, 1] : memref<2x2xf64>
    }
    return
  }
}

// CHECK-LABEL:   func.func @lane(
// CHECK-SAME:                    %[[ARG0:[a-z0-9]+]]: !llvm.ptr,
// CHECK-SAME:                    %[[ARG1:[a-z0-9]+]]: memref<2x2xf64>) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 1 : i64
// CHECK:           %[[POINTER2MEMREF_0:.*]] = "enzymexla.pointer2memref"(%[[ARG0]]) : (!llvm.ptr) -> memref<?x!llvm.struct<(i32, ptr, array<2 x array<2 x f64>>)>>
// CHECK:           %[[LOAD_0:.*]] = affine.load %[[POINTER2MEMREF_0]][0] : memref<?x!llvm.struct<(i32, ptr, array<2 x array<2 x f64>>)>>
// CHECK:           affine.parallel (%[[VAL_0:.*]]) = (0) to (2) {
// CHECK:             %[[ALLOCA_0:.*]] = memref.alloca() : memref<4xf64>
// CHECK:             %[[EXTRACTVALUE_0:.*]] = llvm.extractvalue %[[LOAD_0]][2, 0, 0] : !llvm.struct<(i32, ptr, array<2 x array<2 x f64>>)>
// CHECK:             affine.store %[[EXTRACTVALUE_0]], %[[ALLOCA_0]][0] : memref<4xf64>
// CHECK:             %[[EXTRACTVALUE_1:.*]] = llvm.extractvalue %[[LOAD_0]][2, 0, 1] : !llvm.struct<(i32, ptr, array<2 x array<2 x f64>>)>
// CHECK:             affine.store %[[EXTRACTVALUE_1]], %[[ALLOCA_0]][1] : memref<4xf64>
// CHECK:             %[[EXTRACTVALUE_2:.*]] = llvm.extractvalue %[[LOAD_0]][2, 1, 0] : !llvm.struct<(i32, ptr, array<2 x array<2 x f64>>)>
// CHECK:             affine.store %[[EXTRACTVALUE_2]], %[[ALLOCA_0]][2] : memref<4xf64>
// CHECK:             %[[EXTRACTVALUE_3:.*]] = llvm.extractvalue %[[LOAD_0]][2, 1, 1] : !llvm.struct<(i32, ptr, array<2 x array<2 x f64>>)>
// CHECK:             affine.store %[[EXTRACTVALUE_3]], %[[ALLOCA_0]][3] : memref<4xf64>
// CHECK:             %[[LOAD_1:.*]] = affine.load %[[ALLOCA_0]][%[[VAL_0]] * 2] : memref<4xf64>
// CHECK:             %[[LOAD_2:.*]] = affine.load %[[ALLOCA_0]][%[[VAL_0]] * 2 + 1] : memref<4xf64>
// CHECK:             affine.store %[[LOAD_1]], %[[ARG1]][%[[VAL_0]], 0] : memref<2x2xf64>
// CHECK:             affine.store %[[LOAD_2]], %[[ARG1]][%[[VAL_0]], 1] : memref<2x2xf64>
// CHECK:           }
// CHECK:           return
// CHECK:         }

// -----

// The same before loops are affine: an scf.for, and memref accesses at an
// index computed in arith.
module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f64 = dense<64> : vector<2xi64>>} {
  func.func @scf(%cap: !llvm.ptr, %out: memref<2xf64>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %one = arith.constant 1 : i64
    %v = "enzymexla.pointer2memref"(%cap) : (!llvm.ptr) -> memref<?x!llvm.struct<(i32, ptr, array<2 x f64>)>>
    %agg = memref.load %v[%c0] : memref<?x!llvm.struct<(i32, ptr, array<2 x f64>)>>
    %copy = llvm.alloca %one x !llvm.struct<(i32, ptr, array<2 x f64>)> : (i64) -> !llvm.ptr
    %w = "enzymexla.pointer2memref"(%copy) : (!llvm.ptr) -> memref<?x!llvm.struct<(i32, ptr, array<2 x f64>)>>
    memref.store %agg, %w[%c0] : memref<?x!llvm.struct<(i32, ptr, array<2 x f64>)>>
    %d = "enzymexla.pointer2memref"(%copy) : (!llvm.ptr) -> memref<?xf64>
    scf.for %k = %c0 to %c2 step %c1 {
      %i = arith.addi %k, %c2 : index
      %x = memref.load %d[%i] : memref<?xf64>
      memref.store %x, %out[%k] : memref<2xf64>
    }
    return
  }
}

// CHECK-LABEL:   func.func @scf(
// CHECK-SAME:                   %[[ARG0:[a-z0-9]+]]: !llvm.ptr,
// CHECK-SAME:                   %[[ARG1:[a-z0-9]+]]: memref<2xf64>) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : index
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 1 : index
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 2 : index
// CHECK:           %[[CONSTANT_3:.*]] = arith.constant 1 : i64
// CHECK:           %[[POINTER2MEMREF_0:.*]] = "enzymexla.pointer2memref"(%[[ARG0]]) : (!llvm.ptr) -> memref<?x!llvm.struct<(i32, ptr, array<2 x f64>)>>
// CHECK:           %[[LOAD_0:.*]] = memref.load %[[POINTER2MEMREF_0]][%[[CONSTANT_0]]] : memref<?x!llvm.struct<(i32, ptr, array<2 x f64>)>>
// CHECK:           %[[ALLOCA_0:.*]] = memref.alloca() : memref<2xf64>
// CHECK:           %[[EXTRACTVALUE_0:.*]] = llvm.extractvalue %[[LOAD_0]][2, 0] : !llvm.struct<(i32, ptr, array<2 x f64>)>
// CHECK:           affine.store %[[EXTRACTVALUE_0]], %[[ALLOCA_0]][0] : memref<2xf64>
// CHECK:           %[[EXTRACTVALUE_1:.*]] = llvm.extractvalue %[[LOAD_0]][2, 1] : !llvm.struct<(i32, ptr, array<2 x f64>)>
// CHECK:           affine.store %[[EXTRACTVALUE_1]], %[[ALLOCA_0]][1] : memref<2xf64>
// CHECK:           scf.for %[[VAL_0:.*]] = %[[CONSTANT_0]] to %[[CONSTANT_2]] step %[[CONSTANT_1]] {
// CHECK:             %[[ADDI_0:.*]] = arith.addi %[[VAL_0]], %[[CONSTANT_2]] : index
// CHECK:             %[[CONSTANT_4:.*]] = arith.constant 2 : index
// CHECK:             %[[SUBI_0:.*]] = arith.subi %[[ADDI_0]], %[[CONSTANT_4]] : index
// CHECK:             %[[LOAD_1:.*]] = memref.load %[[ALLOCA_0]][%[[SUBI_0]]] : memref<2xf64>
// CHECK:             memref.store %[[LOAD_1]], %[[ARG1]][%[[VAL_0]]] : memref<2xf64>
// CHECK:           }
// CHECK:           return
// CHECK:         }

// -----

// A capture built member by member, one member copied in from memory it does
// not own, and loaded whole only for the members read: each read member is
// answered by what was stored or copied into it.
module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f64 = dense<64> : vector<2xi64>>} {
  func.func @copied(%mat: !llvm.ptr, %a: f64, %p: !llvm.ptr, %out: memref<3xf64>) {
    %one = arith.constant 1 : i64
    %c16 = arith.constant 16 : i64
    %cap = llvm.alloca %one x !llvm.struct<(f64, ptr, struct<(f64, f64)>)> : (i64) -> !llvm.ptr
    %f = "enzymexla.pointer2memref"(%cap) : (!llvm.ptr) -> memref<?xf64>
    affine.store %a, %f[0] : memref<?xf64>
    %q = "enzymexla.pointer2memref"(%cap) : (!llvm.ptr) -> memref<?x!llvm.ptr>
    affine.store %p, %q[1] : memref<?x!llvm.ptr>
    %m = llvm.getelementptr %cap[16] : (!llvm.ptr) -> !llvm.ptr, i8
    "llvm.intr.memcpy"(%m, %mat, %c16) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %v = "enzymexla.pointer2memref"(%cap) : (!llvm.ptr) -> memref<?x!llvm.struct<(f64, ptr, struct<(f64, f64)>)>>
    %agg = affine.load %v[0] : memref<?x!llvm.struct<(f64, ptr, struct<(f64, f64)>)>>
    %x = llvm.extractvalue %agg[0] : !llvm.struct<(f64, ptr, struct<(f64, f64)>)>
    %y = llvm.extractvalue %agg[2, 0] : !llvm.struct<(f64, ptr, struct<(f64, f64)>)>
    %z = llvm.extractvalue %agg[2, 1] : !llvm.struct<(f64, ptr, struct<(f64, f64)>)>
    affine.store %x, %out[0] : memref<3xf64>
    affine.store %y, %out[1] : memref<3xf64>
    affine.store %z, %out[2] : memref<3xf64>
    return
  }
}

// CHECK-LABEL: func.func @copied(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: !llvm.ptr, %[[V1:[^ ,:)]+]]: f64, %[[V2:[^ ,:)]+]]: !llvm.ptr, %[[V3:[^ ,:)]+]]: memref<3xf64>) {
// CHECK-NEXT: %[[V4:[^ ,:)]+]] = arith.constant 1 : i64
// CHECK-NEXT: %[[V5:[^ ,:)]+]] = arith.constant 16 : i64
// CHECK-NEXT: %[[V6:[^ ,:)]+]] = llvm.load %[[V0]] <alignment = 1> : !llvm.ptr -> f64
// CHECK-NEXT: %[[V7:[^ ,:)]+]] = llvm.getelementptr %[[V0]][8] : (!llvm.ptr) -> !llvm.ptr, i8
// CHECK-NEXT: %[[V8:[^ ,:)]+]] = llvm.load %[[V7]] <alignment = 1> : !llvm.ptr -> f64
// CHECK-NEXT: affine.store %[[V1]], %[[V3]][0] : memref<3xf64>
// CHECK-NEXT: affine.store %[[V6]], %[[V3]][1] : memref<3xf64>
// CHECK-NEXT: affine.store %[[V8]], %[[V3]][2] : memref<3xf64>
// CHECK-NEXT: return
// CHECK-NEXT: }

// -----

// A pointer to the copy escapes: nothing is known of what reads it, so it
// stays whole.
module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f64 = dense<64> : vector<2xi64>>} {
  llvm.func @use(!llvm.ptr)
  func.func @escapes(%agg: !llvm.struct<(i32, ptr, array<2 x f64>)>, %out: memref<2xf64>) {
    %one = arith.constant 1 : i64
    %copy = llvm.alloca %one x !llvm.struct<(i32, ptr, array<2 x f64>)> : (i64) -> !llvm.ptr
    %w = "enzymexla.pointer2memref"(%copy) : (!llvm.ptr) -> memref<?x!llvm.struct<(i32, ptr, array<2 x f64>)>>
    affine.store %agg, %w[0] : memref<?x!llvm.struct<(i32, ptr, array<2 x f64>)>>
    llvm.call @use(%copy) : (!llvm.ptr) -> ()
    %d = "enzymexla.pointer2memref"(%copy) : (!llvm.ptr) -> memref<?xf64>
    affine.for %k = 0 to 2 {
      %x = affine.load %d[%k + 2] : memref<?xf64>
      affine.store %x, %out[%k] : memref<2xf64>
    }
    return
  }
}

// CHECK-LABEL:   func.func @escapes(
// CHECK-SAME:                       %[[ARG0:[a-z0-9]+]]: !llvm.struct<(i32, ptr, array<2 x f64>)>,
// CHECK-SAME:                       %[[ARG1:[a-z0-9]+]]: memref<2xf64>) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 1 : i64
// CHECK:           %[[ALLOCA_0:.*]] = llvm.alloca %[[CONSTANT_0]] x !llvm.struct<(i32, ptr, array<2 x f64>)> : (i64) -> !llvm.ptr
// CHECK:           %[[POINTER2MEMREF_0:.*]] = "enzymexla.pointer2memref"(%[[ALLOCA_0]]) : (!llvm.ptr) -> memref<?x!llvm.struct<(i32, ptr, array<2 x f64>)>>
// CHECK:           affine.store %[[ARG0]], %[[POINTER2MEMREF_0]][0] : memref<?x!llvm.struct<(i32, ptr, array<2 x f64>)>>
// CHECK:           llvm.call @use(%[[ALLOCA_0]]) : (!llvm.ptr) -> ()
// CHECK:           %[[POINTER2MEMREF_1:.*]] = "enzymexla.pointer2memref"(%[[ALLOCA_0]]) : (!llvm.ptr) -> memref<?xf64>
// CHECK:           affine.for %[[VAL_0:.*]] = 0 to 2 {
// CHECK:             %[[LOAD_0:.*]] = affine.load %[[POINTER2MEMREF_1]][%[[VAL_0]] + 2] : memref<?xf64>
// CHECK:             affine.store %[[LOAD_0]], %[[ARG1]][%[[VAL_0]]] : memref<2xf64>
// CHECK:           }
// CHECK:           return
// CHECK:         }

// -----

// A temporary stored whole, written a member at a time, and copied out: the
// copy reads its runs, which go where the copy put them, and the temporary
// goes.
module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, f64 = dense<64> : vector<2xi64>>} {
  func.func @copy_out(%init: !llvm.struct<(array<8 x i8>, f64, array<3 x f64>)>, %x: f64, %dst: !llvm.ptr) {
    %one = arith.constant 1 : i64
    %c24 = arith.constant 24 : i64
    %t = llvm.alloca %one x !llvm.struct<(array<8 x i8>, f64, array<3 x f64>)> : (i64) -> !llvm.ptr
    %w = "enzymexla.pointer2memref"(%t) : (!llvm.ptr) -> memref<?x!llvm.struct<(array<8 x i8>, f64, array<3 x f64>)>>
    affine.store %init, %w[0] : memref<?x!llvm.struct<(array<8 x i8>, f64, array<3 x f64>)>>
    %d = "enzymexla.pointer2memref"(%t) : (!llvm.ptr) -> memref<?xf64>
    affine.store %x, %d[2] : memref<?xf64>
    affine.store %x, %d[3] : memref<?xf64>
    affine.store %x, %d[4] : memref<?xf64>
    %r = llvm.getelementptr %t[16] : (!llvm.ptr) -> !llvm.ptr, i8
    "llvm.intr.memcpy"(%dst, %r, %c24) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    return
  }
}

// CHECK-LABEL: func.func @copy_out(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: !llvm.struct<(array<8 x i8>, f64, array<3 x f64>)>, %[[V1:[^ ,:)]+]]: f64, %[[V2:[^ ,:)]+]]: !llvm.ptr) {
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = arith.constant 1 : i64
// CHECK-NEXT: %[[V4:[^ ,:)]+]] = arith.constant 24 : i64
// CHECK-NEXT: %[[V5:[^ ,:)]+]] = llvm.extractvalue %[[V0]][2, 0] : !llvm.struct<(array<8 x i8>, f64, array<3 x f64>)>
// CHECK-NEXT: %[[V6:[^ ,:)]+]] = llvm.extractvalue %[[V0]][2, 1] : !llvm.struct<(array<8 x i8>, f64, array<3 x f64>)>
// CHECK-NEXT: %[[V7:[^ ,:)]+]] = llvm.extractvalue %[[V0]][2, 2] : !llvm.struct<(array<8 x i8>, f64, array<3 x f64>)>
// CHECK-NEXT: %[[V8:[^ ,:)]+]] = "enzymexla.pointer2memref"(%[[V2]]) : (!llvm.ptr) -> memref<?xf64>
// CHECK-NEXT: affine.store %[[V1]], %[[V8]][0] : memref<?xf64>
// CHECK-NEXT: %[[V9:[^ ,:)]+]] = llvm.getelementptr %[[V2]][8] : (!llvm.ptr) -> !llvm.ptr, i8
// CHECK-NEXT: %[[V10:[^ ,:)]+]] = "enzymexla.pointer2memref"(%[[V9]]) : (!llvm.ptr) -> memref<?xf64>
// CHECK-NEXT: affine.store %[[V1]], %[[V10]][0] : memref<?xf64>
// CHECK-NEXT: %[[V11:[^ ,:)]+]] = llvm.getelementptr %[[V2]][16] : (!llvm.ptr) -> !llvm.ptr, i8
// CHECK-NEXT: %[[V12:[^ ,:)]+]] = "enzymexla.pointer2memref"(%[[V11]]) : (!llvm.ptr) -> memref<?xf64>
// CHECK-NEXT: affine.store %[[V1]], %[[V12]][0] : memref<?xf64>
// CHECK-NEXT: return
// CHECK-NEXT: }
