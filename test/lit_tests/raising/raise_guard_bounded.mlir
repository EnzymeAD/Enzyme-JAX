// RUN: enzymexlamlir-opt %s --raise-affine-to-stablehlo --split-input-file | FileCheck %s

// The extent has no clamp of its own, but the launch sits inside the
// surviving branch of a dispatcher check (MFEM_VERIFY-style): inside
// `if (d < 25)` the axis is bounded by 24 and batches behind a guard.
func.func @guarded(%out: memref<32xf64, 1>, %in: memref<32xf64, 1>, %dbuf: memref<i32, 1>, %unused: index) {
  %c1 = arith.constant 1 : index
  %c25 = arith.constant 25 : i32
  %d = affine.load %dbuf[] : memref<i32, 1>
  %ok = arith.cmpi slt, %d, %c25 : i32
  scf.if %ok {
    %di = arith.index_cast %d : i32 to index
    %0 = "enzymexla.gpu_wrapper"(%c1, %c1, %c1, %di, %c1, %c1) ({
      affine.parallel (%e) = (0) to (1) {
        %scr = memref.alloca() : memref<32xf64>
        affine.parallel (%t) = (0) to (symbol(%di)) {
          %v = affine.load %in[%t] : memref<32xf64, 1>
          affine.store %v, %scr[%t] : memref<32xf64>
          "enzymexla.barrier"(%t, %c1, %c1) : (index, index, index) -> ()
          %w = affine.load %scr[0] : memref<32xf64>
          affine.store %w, %out[%t] : memref<32xf64, 1>
        }
      }
      "enzymexla.polygeist_yield"() : () -> ()
    }) : (index, index, index, index, index, index) -> index
  }
  return
}

// CHECK-LABEL: func.func private @rxla$raised_0(
// CHECK-NOT: stablehlo.while
// CHECK: stablehlo.select

// -----

// The extent symbol lives behind a launch-stub boundary: the stub's callers
// pass a clamped value, and the bound flows through the call site.
llvm.func @stub(%out: !llvm.ptr, %in: !llvm.ptr, %bd: i32) {
  %c1 = arith.constant 1 : index
  %bi = arith.index_cast %bd : i32 to index
  %om = "enzymexla.pointer2memref"(%out) : (!llvm.ptr) -> memref<?xf64>
  %im = "enzymexla.pointer2memref"(%in) : (!llvm.ptr) -> memref<?xf64>
  %0 = "enzymexla.gpu_wrapper"(%c1, %c1, %c1, %bi, %c1, %c1) ({
    affine.parallel (%e) = (0) to (1) {
      %scr = memref.alloca() : memref<64xf64>
      affine.parallel (%t) = (0) to (symbol(%bi)) {
        %v = affine.load %im[%t] : memref<?xf64>
        affine.store %v, %scr[%t] : memref<64xf64>
        "enzymexla.barrier"(%t, %c1, %c1) : (index, index, index) -> ()
        %w = affine.load %scr[0] : memref<64xf64>
        affine.store %w, %om[%t] : memref<?xf64>
      }
    }
    "enzymexla.polygeist_yield"() : () -> ()
  }) : (index, index, index, index, index, index) -> index
  llvm.return
}
llvm.func @caller(%out: !llvm.ptr, %in: !llvm.ptr, %n: i32) {
  %c64 = arith.constant 64 : i32
  %b = arith.minsi %n, %c64 : i32
  llvm.call @stub(%out, %in, %b) : (!llvm.ptr, !llvm.ptr, i32) -> ()
  llvm.return
}

// CHECK-LABEL: llvm.func @stub(
// CHECK-NOT: stablehlo.while
// CHECK: enzymexla.xla_wrapper @rxla$raised
