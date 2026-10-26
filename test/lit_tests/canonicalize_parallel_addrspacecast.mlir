// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(canonicalize-parallel)" --split-input-file | FileCheck %s

// Clang lowers every CUDA __shared__ access through one generic-izing
// addrspacecast with all gep arithmetic on the generic pointer. The cast
// sinks through geps and dies at the pointer2memref view, whose type
// already tolerates the space mismatch.

func.func private @sunk(%p: !llvm.ptr<3>, %d: i64) -> !llvm.ptr {
  %g = llvm.addrspacecast %p : !llvm.ptr<3> to !llvm.ptr
  %a = llvm.getelementptr inbounds|nuw %g[%d] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  return %a : !llvm.ptr
}

// CHECK:  func.func private @sunk(%[[v1:.+]]: !llvm.ptr<3>, %[[v2:.+]]: i64) -> !llvm.ptr {
// CHECK-NEXT:  %[[v3:.+]] = llvm.getelementptr inbounds|nuw %[[v1]][%[[v2]]] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f64
// CHECK-NEXT:  %[[v4:.+]] = llvm.addrspacecast %[[v3]] : !llvm.ptr<3> to !llvm.ptr
// CHECK-NEXT:  return %[[v4]] : !llvm.ptr
// CHECK-NEXT:  }

// -----


func.func private @view(%p: !llvm.ptr<3>) -> memref<?xf64> {
  %g = llvm.addrspacecast %p : !llvm.ptr<3> to !llvm.ptr
  %v = "enzymexla.pointer2memref"(%g) : (!llvm.ptr) -> memref<?xf64>
  return %v : memref<?xf64>
}

// CHECK:  func.func private @view(%[[v1:.+]]: !llvm.ptr<3>) -> memref<?xf64> {
// CHECK-NEXT:  %[[v2:.+]] = "enzymexla.pointer2memref"(%[[v1]]) : (!llvm.ptr<3>) -> memref<?xf64>
// CHECK-NEXT:  return %[[v2]] : memref<?xf64>
// CHECK-NEXT:  }

// -----


// End to end on the MFEM shared-slice shape: the cast sinks out of the gep
// chain, the view rebases onto the shared block, and the constant plane
// offset joins the index.

func.func private @smem(%m: memref<9x6x6xf64>, %out: memref<16xf64, 1>, %d: i64, %i: index) {
  affine.parallel (%t) = (0) to (16) {
    %p3 = "enzymexla.memref2pointer"(%m) : (memref<9x6x6xf64>) -> !llvm.ptr<3>
    %p = llvm.addrspacecast %p3 : !llvm.ptr<3> to !llvm.ptr
    %g1 = llvm.getelementptr inbounds|nuw %p[288] : (!llvm.ptr) -> !llvm.ptr, i8
    %g2 = llvm.getelementptr inbounds|nuw %g1[%d] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<8 x i8>
    %v = "enzymexla.pointer2memref"(%g2) : (!llvm.ptr) -> memref<?xf64>
    %x = memref.load %v[%i] : memref<?xf64>
    affine.store %x, %out[%t] : memref<16xf64, 1>
  }
  return
}

// CHECK:  func.func private @smem(%[[v1:.+]]: memref<9x6x6xf64>, %[[v2:.+]]: memref<16xf64, 1>, %[[v3:.+]]: i64, %[[v4:.+]]: index) {
// CHECK-NEXT:  %[[v5:.+]] = arith.constant 36 : index
// CHECK-NEXT:  affine.parallel (%[[v6:.+]]) = (0) to (16) {
// CHECK-NEXT:    %[[v7:.+]] = "enzymexla.memref2pointer"(%[[v1]]) : (memref<9x6x6xf64>) -> !llvm.ptr<3>
// CHECK-NEXT:    %[[v8:.+]] = "enzymexla.pointer2memref"(%[[v7]]) : (!llvm.ptr<3>) -> memref<?xf64>
// CHECK-NEXT:    %[[v9:.+]] = arith.index_cast %[[v3]] : i64 to index
// CHECK-NEXT:    %[[v10:.+]] = arith.addi %[[v9]], %[[v5]] : index
// CHECK-NEXT:    %[[v11:.+]] = arith.addi %[[v4]], %[[v10]] : index
// CHECK-NEXT:    %[[v12:.+]] = memref.load %[[v8]][%[[v11]]] : memref<?xf64>
// CHECK-NEXT:    affine.store %[[v12]], %[[v2]][%[[v6]]] : memref<16xf64, 1>
// CHECK-NEXT:  }
// CHECK-NEXT:  return
// CHECK-NEXT:  }
