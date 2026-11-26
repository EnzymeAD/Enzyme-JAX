// RUN: enzymexlamlir-opt %s --llvm-to-affine-access | FileCheck %s

// A complex number stored at an index with no affine form (a scatter,
// dst[perm[i]] = z; EnzymeAD/Reactant.jl#3497) has no affine access to fold
// into, so it becomes a memref.store through a pointer2memref of the
// getelementptr. That view of the aggregate is still to be split per leaf,
// as at an affine index: with the store left whole, raising fails on the
// memref<?x!llvm.array<2 x f64>> it writes through.

func.func private @complex_scatter(%out: memref<8x2xf64, 1>, %perm: memref<8xi64, 1>, %re: f64, %im: f64) {
  %u = llvm.mlir.undef : !llvm.array<2 x f64>
  %a = llvm.insertvalue %re, %u[0] : !llvm.array<2 x f64>
  %z = llvm.insertvalue %im, %a[1] : !llvm.array<2 x f64>
  %p = "enzymexla.memref2pointer"(%out) : (memref<8x2xf64, 1>) -> !llvm.ptr<1>
  affine.parallel (%i) = (0) to (8) {
    %j = affine.load %perm[%i] : memref<8xi64, 1>
    %q = llvm.getelementptr inbounds %p[%j] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, !llvm.array<2 x f64>
    llvm.store %z, %q <alignment = 8> : !llvm.array<2 x f64>, !llvm.ptr<1>
  }
  return
}

// CHECK-LABEL: func.func private @complex_scatter
// CHECK-SAME:    (%{{.+}}: memref<8x2xf64, 1>, %{{.+}}: memref<8xi64, 1>, %[[RE:.+]]: f64, %[[IM:.+]]: f64)
// CHECK-NOT:     !llvm.array<2 x f64>
// CHECK:         affine.parallel
// CHECK-DAG:       memref.store %[[RE]], %{{.+}} : memref<{{.+}}xf64, 1{{.*}}>
// CHECK-DAG:       memref.store %[[IM]], %{{.+}} : memref<{{.+}}xf64, 1{{.*}}>
// CHECK-NOT:     !llvm.array<2 x f64>
