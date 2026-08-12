// RUN: env POLYGEIST_GPU_KERNEL_BLOCK_SIZE=128 enzymexlamlir-opt %s --split-input-file --verify-diagnostics --pass-pipeline="builtin.module(convert-parallel-to-gpu1)" | FileCheck %s

// The requested block size cannot split a parallel op with more than three
// dynamic dimensions, and this op offers no fallback: its shape is not the
// six bounds of the wrapper and no original block shape was recorded. Every
// alternative size fails the same way; the kernel collapses to the
// launch-out-of-resources error the failed splits stand for, instead of
// slicing six bounds out of a four-dimensional op.

module {
  func.func @allfail(%n: index, %m: index, %p: index, %q: index, %out: memref<?xf64>, %v: f64) -> index {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    // expected-error @+1 {{no block size splits this kernel}}
    %w = "enzymexla.gpu_wrapper"(%n, %m, %c1, %c1, %c1, %c1) ({
      scf.parallel (%i, %j, %k, %l) = (%c0, %c0, %c0, %c0) to (%n, %m, %p, %q) step (%c1, %c1, %c1, %c1) {
        memref.store %v, %out[%i] : memref<?xf64>
        scf.reduce
      }
      "enzymexla.polygeist_yield"() : () -> ()
    }) : (index, index, index, index, index, index) -> index
    return %w : index
  }
}

// CHECK-LABEL: func.func @allfail
// CHECK: %[[ERR:.+]] = arith.constant 701 : index
// CHECK-NEXT: return %[[ERR]] : index

// -----

// Six bounds matching the wrapper operands exactly: when the requested size
// fails, the original grid/block split is reproduced verbatim.

module {
  func.func @exact(%a: index, %b: index, %c: index, %d: index, %out: memref<?xf64>, %v: f64) {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %w = "enzymexla.gpu_wrapper"(%a, %b, %c, %d, %a, %b) ({
      scf.parallel (%i0, %i1, %i2, %i3, %i4, %i5) = (%c0, %c0, %c0, %c0, %c0, %c0) to (%a, %b, %c, %d, %a, %b) step (%c1, %c1, %c1, %c1, %c1, %c1) {
        memref.store %v, %out[%i0] : memref<?xf64>
        scf.reduce
      }
      "enzymexla.polygeist_yield"() : () -> ()
    }) : (index, index, index, index, index, index) -> index
    return
  }
}

// CHECK-LABEL: func.func @exact(
// CHECK-SAME: %[[A:[^ :]+]]: index, %[[B:[^ :]+]]: index, %[[C:[^ :]+]]: index, %[[D:[^ :]+]]: index
// CHECK: enzymexla.alternatives
// CHECK: gpu.launch blocks(%{{.+}}, %{{.+}}, %{{.+}}) in (%{{.+}} = %[[A]], %{{.+}} = %[[B]], %{{.+}} = %[[C]]) threads(%{{.+}}, %{{.+}}, %{{.+}}) in (%{{.+}} = %[[D]], %{{.+}} = %[[A]], %{{.+}} = %[[B]])
// CHECK: alternatives.descs = ["block_size=-1,"]

// -----

// No exact shape, but the original block dimensions were recorded: the
// original block shape is reproduced from the attribute.

module {
  func.func @recorded(%n: index, %m: index, %p: index, %q: index, %out: memref<?xf64>, %v: f64) {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %w = "enzymexla.gpu_wrapper"(%n, %m, %c1, %c1, %c1, %c1) ({
      scf.parallel (%i, %j, %k, %l) = (%c0, %c0, %c0, %c0) to (%n, %m, %p, %q) step (%c1, %c1, %c1, %c1) {
        memref.store %v, %out[%i] : memref<?xf64>
        scf.reduce
      } {enzymexla.kernel_thread_indices = dense<[2, 3]> : tensor<2xi64>}
      "enzymexla.polygeist_yield"() : () -> ()
    }) : (index, index, index, index, index, index) -> index
    return
  }
}

// CHECK-LABEL: func.func @recorded(
// CHECK-SAME: %[[N:[^ :]+]]: index, %[[M:[^ :]+]]: index, %[[P:[^ :]+]]: index, %[[Q:[^ :]+]]: index
// CHECK: enzymexla.alternatives
// CHECK: gpu.launch blocks(%{{.+}}, %{{.+}}, %{{.+}}) in (%{{.+}} = %[[N]], %{{.+}} = %[[M]], %{{.+}} = %{{.+}}) threads(%{{.+}}, %{{.+}}, %{{.+}}) in (%{{.+}} = %[[P]], %{{.+}} = %[[Q]], %{{.+}} = %{{.+}})

// -----

// The grid and block parallels of a barrier-free kernel are fused into one op,
// and dimensions of extent one are dropped: five bounds carrying the wrapper's
// six. The exact-shape split still applies, reinstating the dropped grid.z.

module {
  func.func @fused_unit_dim(%gx: index, %gy: index, %bx: index, %by: index, %bz: index, %out: memref<?xf64>, %v: f64) {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %w = "enzymexla.gpu_wrapper"(%gx, %gy, %c1, %bx, %by, %bz) ({
      scf.parallel (%i0, %i1, %i2, %i3, %i4) = (%c0, %c0, %c0, %c0, %c0) to (%gx, %gy, %bx, %by, %bz) step (%c1, %c1, %c1, %c1, %c1) {
        memref.store %v, %out[%i0] : memref<?xf64>
        scf.reduce
      }
      "enzymexla.polygeist_yield"() : () -> ()
    }) : (index, index, index, index, index, index) -> index
    return
  }
}

// CHECK-LABEL: func.func @fused_unit_dim(
// CHECK-SAME: %[[GX:[^ :]+]]: index, %[[GY:[^ :]+]]: index, %[[BX:[^ :]+]]: index, %[[BY:[^ :]+]]: index, %[[BZ:[^ :]+]]: index
// CHECK: enzymexla.alternatives
// CHECK: gpu.launch blocks(%{{.+}}, %{{.+}}, %{{.+}}) in (%{{.+}} = %[[GX]], %{{.+}} = %[[GY]], %{{.+}} = %{{.+}}) threads(%{{.+}}, %{{.+}}, %{{.+}}) in (%{{.+}} = %[[BX]], %{{.+}} = %[[BY]], %{{.+}} = %[[BZ]])
// CHECK: alternatives.descs = ["block_size=-1,"]

// -----

// Folding the `k >= N` guard into the loop narrows a bound to a min against the
// kernel's own trip count. That is still the wrapper's dimension, and the
// narrowed bound is the one the launch is built from.

module {
  func.func @min_narrowed_bound(%gx: index, %gy: index, %bx: index, %by: index, %n: index, %out: memref<?xf64>, %v: f64) {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %t = arith.minsi %n, %gx : index
    %w = "enzymexla.gpu_wrapper"(%gx, %gy, %c1, %bx, %by, %c1) ({
      scf.parallel (%i0, %i1, %i2, %i3) = (%c0, %c0, %c0, %c0) to (%t, %gy, %bx, %by) step (%c1, %c1, %c1, %c1) {
        memref.store %v, %out[%i0] : memref<?xf64>
        scf.reduce
      }
      "enzymexla.polygeist_yield"() : () -> ()
    }) : (index, index, index, index, index, index) -> index
    return
  }
}

// CHECK-LABEL: func.func @min_narrowed_bound(
// CHECK-SAME: %[[GX:[^ :]+]]: index, %[[GY:[^ :]+]]: index, %[[BX:[^ :]+]]: index, %[[BY:[^ :]+]]: index
// CHECK: %[[T:.+]] = arith.minsi
// CHECK: enzymexla.alternatives
// CHECK: gpu.launch blocks(%{{.+}}, %{{.+}}, %{{.+}}) in (%{{.+}} = %[[T]], %{{.+}} = %[[GY]], %{{.+}} = %{{.+}}) threads(%{{.+}}, %{{.+}}, %{{.+}}) in (%{{.+}} = %[[BX]], %{{.+}} = %[[BY]], %{{.+}} = %{{.+}})
// CHECK: alternatives.descs = ["block_size=-1,"]
