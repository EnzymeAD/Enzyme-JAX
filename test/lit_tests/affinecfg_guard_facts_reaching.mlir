// RUN: enzymexlamlir-opt %s --affine-cfg | FileCheck %s

// The nest of affinecfg_parallel_guard_facts.mlir (parallel once o <= c is
// known), where the verifies give o <= m and m <= c, with m a value the nest
// does not read: both facts reach the nest's values through m, and the proof
// goes through. The verifies on p and q are about nothing the nest reads,
// and the chain of casts of r is read by nothing; they are left out of the
// dependence tests.
llvm.func @abort() attributes {noreturn}
func.func @chained(%c32: i32, %o32: i32, %m32: i32, %p32: i32, %q32: i32, %r32: i32, %x: memref<?xf64>, %y: memref<?xf64>) {
  %bad0 = arith.cmpi sgt, %o32, %m32 : i32
  cf.cond_br %bad0, ^fail, ^ok0
^fail:
  llvm.call @abort() : () -> ()
  llvm.unreachable
^ok0:
  %bad1 = arith.cmpi sgt, %m32, %c32 : i32
  cf.cond_br %bad1, ^fail, ^ok1
^ok1:
  %bad2 = arith.cmpi sle, %p32, %q32 : i32
  cf.cond_br %bad2, ^fail, ^ok2
^ok2:
  %bad3 = arith.cmpi slt, %r32, %p32 : i32
  cf.cond_br %bad3, ^fail, ^ok3
^ok3:
  %m = arith.index_cast %m32 : i32 to index
  %r0 = arith.index_cast %r32 : i32 to index
  %r1 = arith.index_cast %r32 : i32 to index
  %r2 = arith.index_cast %r32 : i32 to index
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
  affine.store %r, %y[%m] : memref<?xf64>
  affine.store %r, %y[%r0] : memref<?xf64>
  affine.store %r, %y[%r1] : memref<?xf64>
  affine.store %r, %y[%r2] : memref<?xf64>
  return
}

// CHECK:  llvm.func @abort() attributes {noreturn}
// CHECK-NEXT: func.func @chained(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: memref<?xf64>, %arg7: memref<?xf64>) {
// CHECK-NEXT:   %0 = arith.cmpi sgt, %arg1, %arg2 : i32
// CHECK-NEXT:   cf.cond_br %0, ^bb1, ^bb2
// CHECK-NEXT: ^bb1:  // 4 preds: ^bb0, ^bb2, ^bb3, ^bb4
// CHECK-NEXT:   llvm.call @abort() : () -> ()
// CHECK-NEXT:   llvm.unreachable
// CHECK-NEXT: ^bb2:  // pred: ^bb0
// CHECK-NEXT:   %1 = arith.cmpi sgt, %arg2, %arg0 : i32
// CHECK-NEXT:   cf.cond_br %1, ^bb1, ^bb3
// CHECK-NEXT: ^bb3:  // pred: ^bb2
// CHECK-NEXT:   %2 = arith.cmpi sle, %arg3, %arg4 : i32
// CHECK-NEXT:   cf.cond_br %2, ^bb1, ^bb4
// CHECK-NEXT: ^bb4:  // pred: ^bb3
// CHECK-NEXT:   %3 = arith.cmpi slt, %arg5, %arg3 : i32
// CHECK-NEXT:   cf.cond_br %3, ^bb1, ^bb5
// CHECK-NEXT: ^bb5:  // pred: ^bb4
// CHECK-NEXT:   %4 = arith.index_cast %arg2 : i32 to index
// CHECK-NEXT:   %5 = arith.index_cast %arg5 : i32 to index
// CHECK-NEXT:   %6 = arith.index_cast %arg5 : i32 to index
// CHECK-NEXT:   %7 = arith.index_cast %arg5 : i32 to index
// CHECK-NEXT:   %8 = arith.index_cast %arg0 : i32 to index
// CHECK-NEXT:   %9 = arith.index_cast %arg1 : i32 to index
// CHECK-NEXT:   %alloca = memref.alloca() : memref<125xf64>
// CHECK-NEXT:   affine.parallel (%arg8, %arg9, %arg10) = (0, 0, 0) to (symbol(%8), symbol(%9), symbol(%8)) {
// CHECK-NEXT:     %11 = affine.load %arg6[%arg8] : memref<?xf64>
// CHECK-NEXT:     affine.store %11, %alloca[%arg8 + %arg10 * 25 + %arg9 * 5] : memref<125xf64>
// CHECK-NEXT:   }
// CHECK-NEXT:   %10 = affine.load %alloca[7] : memref<125xf64>
// CHECK-NEXT:   affine.store %10, %arg7[symbol(%4)] : memref<?xf64>
// CHECK-NEXT:   affine.store %10, %arg7[symbol(%5)] : memref<?xf64>
// CHECK-NEXT:   affine.store %10, %arg7[symbol(%6)] : memref<?xf64>
// CHECK-NEXT:   affine.store %10, %arg7[symbol(%7)] : memref<?xf64>
// CHECK-NEXT:   return
// CHECK-NEXT: }
