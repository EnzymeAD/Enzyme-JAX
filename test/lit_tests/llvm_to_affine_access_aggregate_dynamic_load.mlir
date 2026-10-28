// RUN: enzymexlamlir-opt %s --llvm-to-affine-access --affine-cfg | FileCheck %s

// A complex number loaded at an index with no affine form (a gather,
// z = src[perm[i]]; EnzymeAD/Reactant.jl#3497) is split per leaf, each leaf
// read at 2 * j + leaf. That position is an index computation like the one it
// is made from, not an affine one: as an affine.apply of j, affine-cfg would
// take it for an affine index and compose it into an affine.load, where j --
// loaded inside the affine.parallel -- is neither a valid dimension nor a
// valid symbol. Computed in arith, the leaves stay memref loads, as the same
// gather of a real number does.

func.func private @complex_gather(%out: memref<8xf64, 1>, %in: memref<8x2xf64, 1>, %perm: memref<8xi64, 1>) {
  %p = "enzymexla.memref2pointer"(%in) : (memref<8x2xf64, 1>) -> !llvm.ptr<1>
  affine.parallel (%i) = (0) to (8) {
    %j = affine.load %perm[%i] : memref<8xi64, 1>
    %q = llvm.getelementptr inbounds %p[%j] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, !llvm.array<2 x f64>
    %z = llvm.load %q <alignment = 8> : !llvm.ptr<1> -> !llvm.array<2 x f64>
    %re = llvm.extractvalue %z[0] : !llvm.array<2 x f64>
    %im = llvm.extractvalue %z[1] : !llvm.array<2 x f64>
    %s = arith.addf %re, %im : f64
    affine.store %s, %out[%i] : memref<8xf64, 1>
  }
  return
}

// CHECK-LABEL: func.func private @complex_gather
// CHECK:         affine.parallel (%[[I:.+]]) = (0) to (8) {
// CHECK:           %[[J:.+]] = affine.load %{{.+}}[%[[I]]] : memref<8xi64, 1>
// CHECK-NOT:       affine.load
// CHECK-NOT:       affine.apply
// CHECK:           %[[RE:.+]] = memref.load
// CHECK-NOT:       affine.load
// CHECK-NOT:       affine.apply
// CHECK:           %[[IM:.+]] = memref.load
// CHECK:           %[[S:.+]] = arith.addf %[[RE]], %[[IM]] : f64
// CHECK:           affine.store %[[S]], %{{.+}}[%[[I]]] : memref<8xf64, 1>
