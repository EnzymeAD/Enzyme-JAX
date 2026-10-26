// RUN: enzymexlamlir-opt %s --affine-cfg | FileCheck %s

// When a boolean feeds an affine symbol -- here the guard of an if converted
// to an affine set -- the legalization strips the zero-extension and casts
// the raw i1 to index. The signed index_cast sends true to -1, so the set
// s0 - 1 == 0 evaluated false exactly when the condition held: in MFEM's
// PAHcurlL2Apply2D the H1 flag selected the L2 basis and stride for H1 test
// spaces. The cast of an i1 must be unsigned.

module {
  func.func @f(%a: index, %b: index, %out: memref<?xf64>) {
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant 1.0 : f64
    %cst2 = arith.constant 2.0 : f64
    %c = arith.cmpi eq, %a, %b : index
    %ci = arith.extui %c : i1 to i32
    affine.parallel (%q) = (0) to (8) {
      %cond = arith.cmpi eq, %ci, %c1_i32 : i32
      scf.if %cond {
        affine.store %cst, %out[%q] : memref<?xf64>
      } else {
        affine.store %cst2, %out[%q] : memref<?xf64>
      }
    }
    return
  }
}

// CHECK: %[[C:.+]] = arith.cmpi eq, %arg0, %arg1 : index
// CHECK-NEXT: %[[S:.+]] = arith.index_castui %[[C]] : i1 to index
// CHECK: affine.if #set{{[0-9]*}}()[%[[S]]]
