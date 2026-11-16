// RUN: enzymexlamlir-opt %s --canonicalize --split-input-file | FileCheck %s

// A pointer round trip is a view of the source buffer. The memory space
// changes on the source's own shape and the shape cast lands last, next to
// the accesses, where it folds into them; the space cast left directly on
// the access is what the raising preprocessing strips.

func.func private @space(%m: memref<30xf64>, %i: index) -> f64 {
  %p = "enzymexla.memref2pointer"(%m) : (memref<30xf64>) -> !llvm.ptr<3>
  %v = "enzymexla.pointer2memref"(%p) : (!llvm.ptr<3>) -> memref<?xf64, 3>
  %x = memref.load %v[%i] : memref<?xf64, 3>
  return %x : f64
}

// CHECK:  func.func private @space(%[[m:.+]]: memref<30xf64>, %[[i:.+]]: index) -> f64 {
// CHECK-NEXT:  %[[sp:.+]] = memref.memory_space_cast %[[m]] : memref<30xf64> to memref<30xf64, 3>
// CHECK-NEXT:  %[[x:.+]] = memref.load %[[sp]][%[[i]]] : memref<30xf64, 3>
// CHECK-NEXT:  return %[[x]] : f64
// CHECK-NEXT:  }

// -----

// The same through affine accesses of a static scratch buffer, the shape
// MFEM's shared-memory tensors take after llvm-to-affine-access.

func.func private @space_affine(%i: index) -> f64 {
  %m = memref.alloca() : memref<64xf64>
  %c = arith.constant 1.0 : f64
  memref.store %c, %m[%i] : memref<64xf64>
  %p = "enzymexla.memref2pointer"(%m) : (memref<64xf64>) -> !llvm.ptr<3>
  %v = "enzymexla.pointer2memref"(%p) : (!llvm.ptr<3>) -> memref<?xf64, 3>
  affine.store %c, %v[%i] : memref<?xf64, 3>
  %x = affine.load %v[%i + 1] : memref<?xf64, 3>
  return %x : f64
}

// CHECK:  func.func private @space_affine(%[[i:.+]]: index) -> f64 {
// CHECK-DAG:  %[[c:.+]] = arith.constant 1.000000e+00 : f64
// CHECK-DAG:  %[[m:.+]] = memref.alloca() : memref<64xf64>
// CHECK:  memref.store %[[c]], %[[m]][%[[i]]] : memref<64xf64>
// CHECK-NEXT:  %[[sp:.+]] = memref.memory_space_cast %[[m]] : memref<64xf64> to memref<64xf64, 3>
// CHECK-NEXT:  affine.store %[[c]], %[[sp]][symbol(%[[i]])] : memref<64xf64, 3>
// CHECK-NEXT:  %[[x:.+]] = affine.load %[[sp]][symbol(%[[i]]) + 1] : memref<64xf64, 3>
// CHECK-NEXT:  return %[[x]] : f64
// CHECK-NEXT:  }

// -----

// Same space: the view is just a cast, which folds into the access.

func.func private @same_space(%m: memref<30xf64>, %i: index) -> f64 {
  %p = "enzymexla.memref2pointer"(%m) : (memref<30xf64>) -> !llvm.ptr
  %v = "enzymexla.pointer2memref"(%p) : (!llvm.ptr) -> memref<?xf64>
  %x = memref.load %v[%i] : memref<?xf64>
  return %x : f64
}

// CHECK:  func.func private @same_space(%[[m:.+]]: memref<30xf64>, %[[i:.+]]: index) -> f64 {
// CHECK-NEXT:  %[[x:.+]] = memref.load %[[m]][%[[i]]] : memref<30xf64>
// CHECK-NEXT:  return %[[x]] : f64
// CHECK-NEXT:  }

// -----

// Shapes that are not cast compatible would need a differently sized buffer:
// the round trip stays. (This built an invalid memref.cast before the
// dimension zero check.)

func.func private @incompatible(%m: memref<30xf64>, %i: index) -> f64 {
  %p = "enzymexla.memref2pointer"(%m) : (memref<30xf64>) -> !llvm.ptr
  %v = "enzymexla.pointer2memref"(%p) : (!llvm.ptr) -> memref<20xf64>
  %x = memref.load %v[%i] : memref<20xf64>
  return %x : f64
}

// CHECK:  func.func private @incompatible(%[[m:.+]]: memref<30xf64>, %[[i:.+]]: index) -> f64 {
// CHECK-NEXT:  %[[p:.+]] = "enzymexla.memref2pointer"(%[[m]]) : (memref<30xf64>) -> !llvm.ptr
// CHECK-NEXT:  %[[v:.+]] = "enzymexla.pointer2memref"(%[[p]]) : (!llvm.ptr) -> memref<20xf64>
// CHECK-NEXT:  %[[x:.+]] = memref.load %[[v]][%[[i]]] : memref<20xf64>
// CHECK-NEXT:  return %[[x]] : f64
// CHECK-NEXT:  }

// -----

// A strided source is not the flat view the pointer handed out: rewriting it
// to an access of the source would apply the source's stride.

func.func private @strided(%m: memref<30xf64, strided<[2]>>, %i: index) -> f64 {
  %p = "enzymexla.memref2pointer"(%m) : (memref<30xf64, strided<[2]>>) -> !llvm.ptr
  %v = "enzymexla.pointer2memref"(%p) : (!llvm.ptr) -> memref<?xf64>
  %x = memref.load %v[%i] : memref<?xf64>
  return %x : f64
}

// CHECK:  func.func private @strided(%[[m:.+]]: memref<30xf64, strided<[2]>>, %[[i:.+]]: index) -> f64 {
// CHECK-NEXT:  %[[p:.+]] = "enzymexla.memref2pointer"(%[[m]]) : (memref<30xf64, strided<[2]>>) -> !llvm.ptr
// CHECK-NEXT:  %[[v:.+]] = "enzymexla.pointer2memref"(%[[p]]) : (!llvm.ptr) -> memref<?xf64>
// CHECK-NEXT:  %[[x:.+]] = memref.load %[[v]][%[[i]]] : memref<?xf64>
// CHECK-NEXT:  return %[[x]] : f64
// CHECK-NEXT:  }

// -----

// A rank change is the flat view delinearization rebuilds into typed
// multi-dimensional accesses: leave it alone.

func.func private @rank_change(%m: memref<6x5xf64>, %i: index) -> f64 {
  %p = "enzymexla.memref2pointer"(%m) : (memref<6x5xf64>) -> !llvm.ptr
  %v = "enzymexla.pointer2memref"(%p) : (!llvm.ptr) -> memref<?xf64>
  %x = memref.load %v[%i] : memref<?xf64>
  return %x : f64
}

// CHECK:  func.func private @rank_change(%[[m:.+]]: memref<6x5xf64>, %[[i:.+]]: index) -> f64 {
// CHECK-NEXT:  %[[p:.+]] = "enzymexla.memref2pointer"(%[[m]]) : (memref<6x5xf64>) -> !llvm.ptr
// CHECK-NEXT:  %[[v:.+]] = "enzymexla.pointer2memref"(%[[p]]) : (!llvm.ptr) -> memref<?xf64>
// CHECK-NEXT:  %[[x:.+]] = memref.load %[[v]][%[[i]]] : memref<?xf64>
// CHECK-NEXT:  return %[[x]] : f64
// CHECK-NEXT:  }
