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

// Different element types scale onto the largest common element type whose
// byte size divides all steps (here i32: f64 is 2 x i32). Narrower dynamic
// indices sign-extend to the pointer index width before arithmetic.

func.func private @different_elem(%p: !llvm.ptr, %c: i1, %d: i64) -> !llvm.ptr {
  %a = llvm.getelementptr %p[%d] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  %b = llvm.getelementptr %p[%d] : (!llvm.ptr, i64) -> !llvm.ptr, i32
  %s = arith.select %c, %a, %b : !llvm.ptr
  return %s : !llvm.ptr
}

// CHECK:  func.func private @different_elem(%[[p:.+]]: !llvm.ptr, %[[c:.+]]: i1, %[[d:.+]]: i64) -> !llvm.ptr {
// CHECK-NEXT:  %[[c2:.+]] = arith.constant 2 : i64
// CHECK-NEXT:  %[[scaled:.+]] = arith.muli %[[d]], %[[c2]] : i64
// CHECK-NEXT:  %[[idx:.+]] = arith.select %[[c]], %[[scaled]], %[[d]] : i64
// CHECK-NEXT:  %[[g:.+]] = llvm.getelementptr %[[p]][%[[idx]]] : (!llvm.ptr, i64) -> !llvm.ptr, i32
// CHECK-NEXT:  return %[[g]] : !llvm.ptr
// CHECK-NEXT:  }

// -----

func.func private @different_elem_i32_idx(%p: !llvm.ptr, %c: i1, %d: i32) -> !llvm.ptr {
  %a = llvm.getelementptr %p[%d] : (!llvm.ptr, i32) -> !llvm.ptr, f64
  %b = llvm.getelementptr %p[%d] : (!llvm.ptr, i32) -> !llvm.ptr, i32
  %s = arith.select %c, %a, %b : !llvm.ptr
  return %s : !llvm.ptr
}

// CHECK-LABEL: func.func private @different_elem_i32_idx(
// CHECK-SAME: %[[p:.+]]: !llvm.ptr, %[[c:.+]]: i1, %[[d:.+]]: i32) -> !llvm.ptr {
// CHECK-NEXT: %[[c2:.+]] = arith.constant 2 : i64
// CHECK-NEXT: %[[ext1:.+]] = arith.extsi %[[d]] : i32 to i64
// CHECK-NEXT: %[[scaled:.+]] = arith.muli %[[ext1]], %[[c2]] : i64
// CHECK-NEXT: %[[ext2:.+]] = arith.extsi %[[d]] : i32 to i64
// CHECK-NEXT: %[[idx:.+]] = arith.select %[[c]], %[[scaled]], %[[ext2]] : i64
// CHECK-NEXT: %[[g:.+]] = llvm.getelementptr %[[p]][%[[idx]]] : (!llvm.ptr, i64) -> !llvm.ptr, i32
// CHECK-NEXT: return %[[g]] : !llvm.ptr
// CHECK-NEXT: }

// -----

// On a 32-bit pointer address space, i32 is already the pointer index width so
// mixed element type arithmetic stays in i32 without sign-extension.

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<3>, dense<[32, 32, 32, 32]> : vector<4xi64>>>} {
  func.func private @different_elem_i32_ptr(%p: !llvm.ptr<3>, %c: i1, %d: i32) -> !llvm.ptr<3> {
    %a = llvm.getelementptr %p[%d] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f64
    %b = llvm.getelementptr %p[%d] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, i32
    %s = arith.select %c, %a, %b : !llvm.ptr<3>
    return %s : !llvm.ptr<3>
  }
}

// CHECK-LABEL: func.func private @different_elem_i32_ptr(
// CHECK-SAME: %[[p:.+]]: !llvm.ptr<3>, %[[c:.+]]: i1, %[[d:.+]]: i32) -> !llvm.ptr<3> {
// CHECK-NEXT: %[[c2:.+]] = arith.constant 2 : i32
// CHECK-NEXT: %[[scaled:.+]] = arith.muli %[[d]], %[[c2]] : i32
// CHECK-NEXT: %[[idx:.+]] = arith.select %[[c]], %[[scaled]], %[[d]] : i32
// CHECK-NEXT: %[[g:.+]] = llvm.getelementptr %[[p]][%[[idx]]] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, i32
// CHECK-NEXT: return %[[g]] : !llvm.ptr<3>
// CHECK-NEXT: }

// -----

// Narrow dynamic indices (< 32 bits) stay narrow when constants fit and widen
// to the pointer index width when a constant would truncate.

func.func private @narrow_idx_fits(%p: !llvm.ptr, %c: i1, %d: i16) -> !llvm.ptr {
  %a = llvm.getelementptr %p[%d] : (!llvm.ptr, i16) -> !llvm.ptr, f64
  %b = llvm.getelementptr %p[36] : (!llvm.ptr) -> !llvm.ptr, f64
  %s = arith.select %c, %a, %b : !llvm.ptr
  return %s : !llvm.ptr
}

// CHECK-LABEL: func.func private @narrow_idx_fits(
// CHECK-SAME: %[[p:.+]]: !llvm.ptr, %[[c:.+]]: i1, %[[d:.+]]: i16) -> !llvm.ptr {
// CHECK-NEXT: %[[c36:.+]] = arith.constant 36 : i16
// CHECK-NEXT: %[[idx:.+]] = arith.select %[[c]], %[[d]], %[[c36]] : i16
// CHECK-NEXT: %[[g:.+]] = llvm.getelementptr %[[p]][%[[idx]]] : (!llvm.ptr, i16) -> !llvm.ptr, f64
// CHECK-NEXT: return %[[g]] : !llvm.ptr
// CHECK-NEXT: }

func.func private @narrow_idx_widens(%p: !llvm.ptr, %c: i1, %d: i16) -> !llvm.ptr {
  %a = llvm.getelementptr %p[%d] : (!llvm.ptr, i16) -> !llvm.ptr, f64
  %b = llvm.getelementptr %p[40000] : (!llvm.ptr) -> !llvm.ptr, f64
  %s = arith.select %c, %a, %b : !llvm.ptr
  return %s : !llvm.ptr
}

// CHECK-LABEL: func.func private @narrow_idx_widens(
// CHECK-SAME: %[[p:.+]]: !llvm.ptr, %[[c:.+]]: i1, %[[d:.+]]: i16) -> !llvm.ptr {
// CHECK-DAG: %[[c40000:.+]] = arith.constant 40000 : i64
// CHECK-DAG: %[[ext:.+]] = arith.extsi %[[d]] : i16 to i64
// CHECK: %[[idx:.+]] = arith.select %[[c]], %[[ext]], %[[c40000]] : i64
// CHECK-NEXT: %[[g:.+]] = llvm.getelementptr %[[p]][%[[idx]]] : (!llvm.ptr, i64) -> !llvm.ptr, f64
// CHECK-NEXT: return %[[g]] : !llvm.ptr
// CHECK-NEXT: }

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
