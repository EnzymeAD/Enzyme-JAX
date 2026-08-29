// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(convert-parallel-to-gpu1)" | FileCheck %s

module {
  func.func @foreach_thread(%n: index, %out: memref<?xf64>, %v: f64) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c288 = arith.constant 288 : index
    "enzymexla.gpu_wrapper"(%n, %c1, %c1, %c8, %c4, %c1) ({
      scf.parallel (%b, %tx, %ty) = (%c0, %c0, %c0) to (%n, %c8, %c4) step (%c1, %c1, %c1) {
        %in = arith.cmpi slt, %b, %n : index
        scf.if %in {
          scf.parallel (%i) = (%c0) to (%c288) step (%c1) {
            memref.store %v, %out[%i] : memref<?xf64>
            scf.reduce
          }
        }
        scf.reduce
      }
      "enzymexla.polygeist_yield"() : () -> ()
    }) : (index, index, index, index, index, index) -> index
    return
  }
}

// CHECK-LABEL: func.func @foreach_thread
// CHECK-NOT: enzymexla.gpu_wrapper
// CHECK: gpu.launch blocks(%{{.+}}, %{{.+}}, %{{.+}}) in (%{{.+}} = %{{.+}}, %{{.+}} = %{{.+}}, %{{.+}} = %{{.+}}) threads(%{{.+}}, %{{.+}}, %{{.+}}) in (%{{.+}} = %[[TX:.+]], %{{.+}} = %[[TY:.+]], %{{.+}} = %{{.+}})
// CHECK: scf.parallel
