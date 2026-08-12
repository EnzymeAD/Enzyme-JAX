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

// The block already uses all three thread dimensions, so there is no room to
// take a grid dimension into it however few threads it has. Taking one anyway
// left a fourth block dimension to alias onto an existing thread id, so two of
// the parallel op's dimensions advanced together and the rest of the iteration
// space was never run.

module {
  func.func @full_block_no_split(%gx: index, %out: memref<?xf64>, %v: f64) {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index
    %w = "enzymexla.gpu_wrapper"(%gx, %c1, %c1, %c3, %c3, %c3) ({
      scf.parallel (%i0, %i1, %i2, %i3) = (%c0, %c0, %c0, %c0) to (%gx, %c3, %c3, %c3) step (%c1, %c1, %c1, %c1) {
        %a = arith.muli %i0, %c3 : index
        %b = arith.addi %a, %i3 : index
        %c = arith.muli %b, %c3 : index
        %d = arith.addi %c, %i2 : index
        %e = arith.muli %d, %c3 : index
        %f = arith.addi %e, %i1 : index
        memref.store %v, %out[%f] : memref<?xf64>
        scf.reduce
      }
      "enzymexla.polygeist_yield"() : () -> ()
    }) : (index, index, index, index, index, index) -> index
    return
  }
}

// CHECK-LABEL: func.func @full_block_no_split(
// CHECK-SAME: %[[GX:[^ :]+]]: index
// CHECK: gpu.launch blocks(%{{.+}}, %{{.+}}, %{{.+}}) in (%{{.+}} = %[[GX]], %{{.+}} = %{{.+}}, %{{.+}} = %{{.+}}) threads(%{{.+}}, %{{.+}}, %{{.+}}) in (%{{.+}} = %{{.+}}, %{{.+}} = %{{.+}}, %{{.+}} = %{{.+}})
// CHECK: gpu.thread_id x
// CHECK: gpu.thread_id y
// CHECK: gpu.thread_id z
// CHECK-NOT: gpu.thread_id
