// RUN: enzymexlamlir-opt %s --raise-affine-to-stablehlo | FileCheck %s

// A pointer constant observed only through ptrtoint is a wrapper-invariant
// scalar: the computation hoists out of the kernel and the store raises.
func.func @null_ptr_operand(%out: memref<?xf64>, %n: i64) {
  %c1 = arith.constant 1 : index
  %null = llvm.mlir.zero : !llvm.ptr
  %0 = "enzymexla.gpu_wrapper"(%c1, %c1, %c1, %c1, %c1, %c1) ({
    affine.parallel (%i) = (0) to (4) {
      %pi = llvm.ptrtoint %null : !llvm.ptr to i64
      %v = arith.sitofp %pi : i64 to f64
      affine.store %v, %out[%i] : memref<?xf64>
    }
    "enzymexla.polygeist_yield"() : () -> ()
  }) : (index, index, index, index, index, index) -> index
  return
}

// CHECK-NOT: failed to raise
// CHECK: stablehlo.dynamic_pad
