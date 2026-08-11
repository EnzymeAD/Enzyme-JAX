// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(convert-parallel-to-gpu1)" | FileCheck %s

// The raising proves MFEM_FOREACH_THREAD's 'for (i = threadIdx.x; i < N;
// i += blockDim.x)' parallel, which leaves a parallel inside the kernel's own
// parallel. split-parallel refuses to split a kernel that still holds one, so
// the kernel loses its grid and block dimensions entirely and the launch is
// never built: the wrapper reaches the LLVM lowering, which has no pattern for
// it. Proving the inner loop parallel must not cost the kernel its launch.
//
// The thread shape comes from the launch geometry the wrapper records -- 8x4
// here -- not from the loops inside, which are serialized as any loop in a
// kernel is.

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
// The launch keeps the kernel's own thread shape ...
// CHECK: gpu.launch blocks(%{{.+}}, %{{.+}}, %{{.+}}) in (%{{.+}} = %{{.+}}, %{{.+}} = %{{.+}}, %{{.+}} = %{{.+}}) threads(%{{.+}}, %{{.+}}, %{{.+}}) in (%{{.+}} = %[[TX:.+]], %{{.+}} = %[[TY:.+]], %{{.+}} = %{{.+}})
// ... and the loop the raising proved parallel is a loop again.
// CHECK: scf.for
// CHECK-NOT: scf.parallel
