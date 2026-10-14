// RUN: enzymexlamlir-opt %s --affine-cfg | FileCheck %s

// A scratch laid out for 5 points a side, filled over two counts c and o:
// staying in bounds gives only 26 * (c - 1) + 5 * (o - 1) <= 124, which
// allows c = 2 and o = 20, where two iterations write one element. The
// verify before it (failing into a call that does not return) gives o <= c,
// with which c and o are at most 5, and the nest is parallel.
llvm.func @abort() attributes {noreturn}
func.func @verified(%c32: i32, %o32: i32, %x: memref<?xf64>, %y: memref<?xf64>) {
  %bad = arith.cmpi sgt, %o32, %c32 : i32
  cf.cond_br %bad, ^fail, ^ok
^fail:
  llvm.call @abort() : () -> ()
  llvm.unreachable
^ok:
  %c = arith.index_cast %c32 : i32 to index
  %o = arith.index_cast %o32 : i32 to index
  %a = memref.alloca() : memref<125xf64>
  affine.for %i = 0 to %c {
    affine.for %j = 0 to %o {
      affine.for %k = 0 to %c {
        %v = affine.load %x[%i] : memref<?xf64>
        affine.store %v, %a[%i + %k * 25 + %j * 5] : memref<125xf64>
      }
    }
  }
  %r = affine.load %a[7] : memref<125xf64>
  affine.store %r, %y[0] : memref<?xf64>
  return
}

// CHECK:  func.func @verified(%arg0: i32, %arg1: i32, %arg2: memref<?xf64>, %arg3: memref<?xf64>) {
// CHECK-NEXT:   %0 = arith.cmpi sgt, %arg1, %arg0 : i32
// CHECK-NEXT:   cf.cond_br %0, ^bb1, ^bb2
// CHECK-NEXT: ^bb1:  // pred: ^bb0
// CHECK-NEXT:   llvm.call @abort() : () -> ()
// CHECK-NEXT:   llvm.unreachable
// CHECK-NEXT: ^bb2:  // pred: ^bb0
// CHECK-NEXT:   %1 = arith.index_cast %arg0 : i32 to index
// CHECK-NEXT:   %2 = arith.index_cast %arg1 : i32 to index
// CHECK-NEXT:   %alloca = memref.alloca() : memref<125xf64>
// CHECK-NEXT:   affine.parallel (%arg4, %arg5, %arg6) = (0, 0, 0) to (symbol(%1), symbol(%2), symbol(%1)) {
// CHECK-NEXT:     %4 = affine.load %arg2[%arg4] : memref<?xf64>
// CHECK-NEXT:     affine.store %4, %alloca[%arg4 + %arg6 * 25 + %arg5 * 5] : memref<125xf64>
// CHECK-NEXT:   }
// CHECK-NEXT:   %3 = affine.load %alloca[7] : memref<125xf64>
// CHECK-NEXT:   affine.store %3, %arg3[0] : memref<?xf64>
// CHECK-NEXT:   return
// CHECK-NEXT: }

// The same check, but its failing side joins the nest's path again: it says
// nothing there, and the nest stays serial.
func.func @rejoined(%c32: i32, %o32: i32, %x: memref<?xf64>, %y: memref<?xf64>) {
  %bad = arith.cmpi sgt, %o32, %c32 : i32
  cf.cond_br %bad, ^fail, ^ok
^fail:
  cf.br ^ok
^ok:
  %c = arith.index_cast %c32 : i32 to index
  %o = arith.index_cast %o32 : i32 to index
  %a = memref.alloca() : memref<125xf64>
  affine.for %i = 0 to %c {
    affine.for %j = 0 to %o {
      affine.for %k = 0 to %c {
        %v = affine.load %x[%i] : memref<?xf64>
        affine.store %v, %a[%i + %k * 25 + %j * 5] : memref<125xf64>
      }
    }
  }
  %r = affine.load %a[7] : memref<125xf64>
  affine.store %r, %y[0] : memref<?xf64>
  return
}

// CHECK:  func.func @rejoined(%arg0: i32, %arg1: i32, %arg2: memref<?xf64>, %arg3: memref<?xf64>) {
// CHECK-NEXT:   %0 = arith.cmpi sgt, %arg1, %arg0 : i32
// CHECK-NEXT:   cf.cond_br %0, ^bb1, ^bb2
// CHECK-NEXT: ^bb1:  // pred: ^bb0
// CHECK-NEXT:   cf.br ^bb2
// CHECK-NEXT: ^bb2:  // 2 preds: ^bb0, ^bb1
// CHECK-NEXT:   %1 = arith.index_cast %arg0 : i32 to index
// CHECK-NEXT:   %2 = arith.index_cast %arg1 : i32 to index
// CHECK-NEXT:   %alloca = memref.alloca() : memref<125xf64>
// CHECK-NEXT:   affine.parallel (%arg4) = (0) to (symbol(%1)) {
// CHECK-NEXT:     affine.for %arg5 = 0 to %2 {
// CHECK-NEXT:       affine.parallel (%arg6) = (0) to (symbol(%1)) {
// CHECK-NEXT:         %4 = affine.load %arg2[%arg4] : memref<?xf64>
// CHECK-NEXT:         affine.store %4, %alloca[%arg4 + %arg6 * 25 + %arg5 * 5] : memref<125xf64>
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   %3 = affine.load %alloca[7] : memref<125xf64>
// CHECK-NEXT:   affine.store %3, %arg3[0] : memref<?xf64>
// CHECK-NEXT:   return
// CHECK-NEXT: }

// The check as an scf.if the nest is under: it only runs where o <= c.
func.func @under_if(%c32: i32, %o32: i32, %x: memref<?xf64>, %y: memref<?xf64>) {
  %ok = arith.cmpi sle, %o32, %c32 : i32
  %c = arith.index_cast %c32 : i32 to index
  %o = arith.index_cast %o32 : i32 to index
  scf.if %ok {
    %a = memref.alloca() : memref<125xf64>
    affine.for %i = 0 to %c {
      affine.for %j = 0 to %o {
        affine.for %k = 0 to %c {
          %v = affine.load %x[%i] : memref<?xf64>
          affine.store %v, %a[%i + %k * 25 + %j * 5] : memref<125xf64>
        }
      }
    }
    %r = affine.load %a[7] : memref<125xf64>
    affine.store %r, %y[0] : memref<?xf64>
  }
  return
}

// CHECK:  func.func @under_if(%arg0: i32, %arg1: i32, %arg2: memref<?xf64>, %arg3: memref<?xf64>) {
// CHECK-NEXT:   %0 = arith.cmpi sle, %arg1, %arg0 : i32
// CHECK-NEXT:   %1 = arith.index_cast %arg0 : i32 to index
// CHECK-NEXT:   %2 = arith.index_cast %arg1 : i32 to index
// CHECK-NEXT:   scf.if %0 {
// CHECK-NEXT:     %alloca = memref.alloca() : memref<125xf64>
// CHECK-NEXT:     affine.parallel (%arg4, %arg5, %arg6) = (0, 0, 0) to (symbol(%1), symbol(%2), symbol(%1)) {
// CHECK-NEXT:       %4 = affine.load %arg2[%arg4] : memref<?xf64>
// CHECK-NEXT:       affine.store %4, %alloca[%arg4 + %arg6 * 25 + %arg5 * 5] : memref<125xf64>
// CHECK-NEXT:     }
// CHECK-NEXT:     %3 = affine.load %alloca[7] : memref<125xf64>
// CHECK-NEXT:     affine.store %3, %arg3[0] : memref<?xf64>
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }
