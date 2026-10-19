// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(canonicalize-parallel)" --split-input-file | FileCheck %s

// A select over geps off one base (MFEM's symmetric/nonsymmetric slice
// picks, after clang folds the slice constants into the addressing) sinks
// back into the index, so the address chain stays a single gep.

func.func private @const_arms(%p: !llvm.ptr, %c: i1) -> !llvm.ptr {
  %a = llvm.getelementptr inbounds|nuw %p[1152] : (!llvm.ptr) -> !llvm.ptr, i8
  %b = llvm.getelementptr inbounds|nuw %p[1440] : (!llvm.ptr) -> !llvm.ptr, i8
  %s = arith.select %c, %a, %b : !llvm.ptr
  return %s : !llvm.ptr
}

// CHECK:  func.func private @const_arms(%[[p:.+]]: !llvm.ptr, %[[c:.+]]: i1) -> !llvm.ptr {
// CHECK-NEXT:  %[[cA:.+]] = arith.constant 1152 : i64
// CHECK-NEXT:  %[[cB:.+]] = arith.constant 1440 : i64
// CHECK-NEXT:  %[[idx:.+]] = arith.select %[[c]], %[[cA]], %[[cB]] : i64
// CHECK-NEXT:  %[[g:.+]] = llvm.getelementptr inbounds|nuw %[[p]][%[[idx]]] : (!llvm.ptr, i64) -> !llvm.ptr, i8
// CHECK-NEXT:  return %[[g]] : !llvm.ptr
// CHECK-NEXT:  }

// -----

// A dynamic arm keeps its index type; the constant arm joins it.

func.func private @mixed_arms(%p: !llvm.ptr, %c: i1, %d: i64) -> !llvm.ptr {
  %a = llvm.getelementptr %p[%d] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  %b = llvm.getelementptr %p[36] : (!llvm.ptr) -> !llvm.ptr, f64
  %s = arith.select %c, %a, %b : !llvm.ptr
  return %s : !llvm.ptr
}

// CHECK:  func.func private @mixed_arms(%[[p:.+]]: !llvm.ptr, %[[c:.+]]: i1, %[[d:.+]]: i64) -> !llvm.ptr {
// CHECK-NEXT:  %[[c36:.+]] = arith.constant 36 : i64
// CHECK-NEXT:  %[[idx:.+]] = arith.select %[[c]], %[[d]], %[[c36]] : i64
// CHECK-NEXT:  %[[g:.+]] = llvm.getelementptr %[[p]][%[[idx]]] : (!llvm.ptr, i64) -> !llvm.ptr, f64
// CHECK-NEXT:  return %[[g]] : !llvm.ptr
// CHECK-NEXT:  }

// -----

// Different bases stay a pointer select.

func.func private @different_bases(%p: !llvm.ptr, %q: !llvm.ptr, %c: i1) -> !llvm.ptr {
  %a = llvm.getelementptr %p[8] : (!llvm.ptr) -> !llvm.ptr, i8
  %b = llvm.getelementptr %q[16] : (!llvm.ptr) -> !llvm.ptr, i8
  %s = arith.select %c, %a, %b : !llvm.ptr
  return %s : !llvm.ptr
}

// CHECK:  func.func private @different_bases(%[[p:.+]]: !llvm.ptr, %[[q:.+]]: !llvm.ptr, %[[c:.+]]: i1) -> !llvm.ptr {
// CHECK-NEXT:  %[[a:.+]] = llvm.getelementptr %[[p]][8] : (!llvm.ptr) -> !llvm.ptr, i8
// CHECK-NEXT:  %[[b:.+]] = llvm.getelementptr %[[q]][16] : (!llvm.ptr) -> !llvm.ptr, i8
// CHECK-NEXT:  %[[s:.+]] = arith.select %[[c]], %[[a]], %[[b]] : !llvm.ptr
// CHECK-NEXT:  return %[[s]] : !llvm.ptr
// CHECK-NEXT:  }

// -----

// Different element types stay a pointer select.

func.func private @different_elem(%p: !llvm.ptr, %c: i1, %d: i64) -> !llvm.ptr {
  %a = llvm.getelementptr %p[%d] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  %b = llvm.getelementptr %p[%d] : (!llvm.ptr, i64) -> !llvm.ptr, i32
  %s = arith.select %c, %a, %b : !llvm.ptr
  return %s : !llvm.ptr
}

// CHECK:  func.func private @different_elem(%[[p:.+]]: !llvm.ptr, %[[c:.+]]: i1, %[[d:.+]]: i64) -> !llvm.ptr {
// CHECK-NEXT:  %[[a:.+]] = llvm.getelementptr %[[p]][%[[d]]] : (!llvm.ptr, i64) -> !llvm.ptr, f64
// CHECK-NEXT:  %[[b:.+]] = llvm.getelementptr %[[p]][%[[d]]] : (!llvm.ptr, i64) -> !llvm.ptr, i32
// CHECK-NEXT:  %[[s:.+]] = arith.select %[[c]], %[[a]], %[[b]] : !llvm.ptr
// CHECK-NEXT:  return %[[s]] : !llvm.ptr
// CHECK-NEXT:  }

// -----

// End to end: the sunk select feeds a pointer2memref view, the load rebases
// onto the base buffer through the existing gep-view fold, and the byte
// offset's cast and division sink through the select into element offsets.

func.func private @through_view(%p: !llvm.ptr, %c: i1, %i: index) -> f64 {
  %a = llvm.getelementptr inbounds|nuw %p[288] : (!llvm.ptr) -> !llvm.ptr, i8
  %b = llvm.getelementptr inbounds|nuw %p[576] : (!llvm.ptr) -> !llvm.ptr, i8
  %s = arith.select %c, %a, %b : !llvm.ptr
  %v = "enzymexla.pointer2memref"(%s) : (!llvm.ptr) -> memref<?xf64>
  %x = memref.load %v[%i] : memref<?xf64>
  return %x : f64
}

// CHECK:  func.func private @through_view(%[[p:.+]]: !llvm.ptr, %[[c:.+]]: i1, %[[i:.+]]: index) -> f64 {
// CHECK-NEXT:  %[[cB:.+]] = arith.constant 72 : index
// CHECK-NEXT:  %[[cA:.+]] = arith.constant 36 : index
// CHECK-NEXT:  %[[view:.+]] = "enzymexla.pointer2memref"(%[[p]]) : (!llvm.ptr) -> memref<?xf64>
// CHECK-NEXT:  %[[sel:.+]] = arith.select %[[c]], %[[cA]], %[[cB]] : index
// CHECK-NEXT:  %[[add:.+]] = arith.addi %[[i]], %[[sel]] : index
// CHECK-NEXT:  %[[x:.+]] = memref.load %[[view]][%[[add]]] : memref<?xf64>
// CHECK-NEXT:  return %[[x]] : f64
// CHECK-NEXT:  }
