// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(convert-parallel-to-gpu1,canonicalize)" | FileCheck %s --check-prefix=DEFAULT
// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(convert-parallel-to-gpu1{emit-alternatives=true})" | FileCheck %s --check-prefix=ALTS

module {
  func.func @square_kernel(%x: memref<64xf64, 1>, %y: memref<64xf64, 1>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %0 = "enzymexla.gpu_wrapper"(%c1, %c1, %c1, %c64, %c1, %c1) ({
      scf.parallel (%i) = (%c0) to (%c64) step (%c1) {
        %vx = memref.load %x[%i] : memref<64xf64, 1>
        %vy = memref.load %y[%i] : memref<64xf64, 1>
        %mul = arith.mulf %vx, %vy : f64
        memref.store %mul, %x[%i] : memref<64xf64, 1>
        scf.reduce
      }
      "enzymexla.polygeist_yield"() : () -> ()
    }) : (index, index, index, index, index, index) -> index
    return
  }
}

// DEFAULT-LABEL: func.func @square_kernel
// DEFAULT-DAG:   %[[C1:.+]] = arith.constant 1 : index
// DEFAULT-DAG:   %[[C64:.+]] = arith.constant 64 : index
// DEFAULT-NOT:   enzymexla.alternatives
// DEFAULT:       gpu.launch blocks(%{{.+}}, %{{.+}}, %{{.+}}) in (%{{.+}} = %[[C1]], %{{.+}} = %[[C1]], %{{.+}} = %[[C1]]) threads(%{{.+}}, %{{.+}}, %{{.+}}) in (%{{.+}} = %[[C64]], %{{.+}} = %[[C1]], %{{.+}} = %[[C1]])
// DEFAULT-NOT:   enzymexla.alternatives

// ALTS-LABEL: func.func @square_kernel
// ALTS:       enzymexla.alternatives
// ALTS:       alternatives.descs = ["block_size=32,", "block_size=64,", "block_size=-1,"]
