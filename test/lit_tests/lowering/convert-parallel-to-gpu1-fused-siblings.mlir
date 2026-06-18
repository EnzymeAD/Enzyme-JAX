// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(convert-parallel-to-gpu1)" | FileCheck %s

module {
  func.func @fused_siblings(%g0: index, %g1: index, %g2: index,
                            %b0: index, %b1: index, %b2: index,
                            %arg: memref<8xf64>) {
    %r = "enzymexla.gpu_wrapper"(%g0, %g1, %g2, %b0, %b1, %b2) ({
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %cst = arith.constant 1.0 : f64
      scf.parallel (%i0, %i1, %i2, %i3, %i4, %i5) =
          (%c0, %c0, %c0, %c0, %c0, %c0) to (%g0, %g1, %g2, %b0, %b1, %b2)
          step (%c1, %c1, %c1, %c1, %c1, %c1) {
        scf.parallel (%j) = (%c0) to (%c8) step (%c1) {
          memref.store %cst, %arg[%j] : memref<8xf64>
          scf.reduce
        }
        scf.parallel (%k) = (%c0) to (%c8) step (%c1) {
          memref.store %cst, %arg[%k] : memref<8xf64>
          scf.reduce
        }
        scf.parallel (%l) = (%c0) to (%c8) step (%c1) {
          memref.store %cst, %arg[%l] : memref<8xf64>
          scf.reduce
        }
        scf.reduce
      }
      "enzymexla.polygeist_yield"() : () -> ()
    }) : (index, index, index, index, index, index) -> index
    return
  }
}

// CHECK-LABEL: func.func @fused_siblings(
// CHECK:         "enzymexla.alternatives"() ({
// CHECK:           "enzymexla.gpu_error"() ({
// CHECK:             scf.if %{{.*}} {
// CHECK:               gpu.launch blocks(%{{.*}}, %{{.*}}, %{{.*}}) in (%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}) threads(%{{.*}}, %{{.*}}, %{{.*}}) in (%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}) {
// CHECK:                 scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK:                   memref.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<8xf64>
// CHECK:                 }
// CHECK:                 scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK:                   memref.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<8xf64>
// CHECK:                 }
// CHECK:                 scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK:                   memref.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<8xf64>
// CHECK:                 }
// CHECK:                 gpu.terminator
// CHECK:           "enzymexla.polygeist_yield"() : () -> ()
// CHECK:         "enzymexla.polygeist_yield"() : () -> ()
