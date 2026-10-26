// RUN: enzymexlamlir-opt %s --raise-affine-to-stablehlo --split-input-file | FileCheck %s

// The guard flag of an optional buffer is computed inside the kernel as a
// null check of a captured pointer. Both pointers are defined outside the
// wrapper, so the comparison hoists to the host and the kernel captures the
// resulting flag instead of pointers no tensor can stand for; the arithmetic
// on the flag stays in the kernel.
func.func @nullguard(%out: memref<32xf64, 1>, %opt: !llvm.ptr<1>) {
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  %null = llvm.mlir.zero : !llvm.ptr<1>
  %0 = "enzymexla.gpu_wrapper"(%c1, %c1, %c1, %c32, %c1, %c1) ({
    %has = llvm.icmp "ne" %opt, %null : !llvm.ptr<1>
    affine.parallel (%t) = (0) to (32) {
      %v = arith.uitofp %has : i1 to f64
      affine.store %v, %out[%t] : memref<32xf64, 1>
    }
    "enzymexla.polygeist_yield"() : () -> ()
  }) : (index, index, index, index, index, index) -> index
  return
}

// CHECK-LABEL: func.func @nullguard(
// CHECK-SAME: %[[OUT:.+]]: memref<32xf64, 1>, %[[OPT:.+]]: !llvm.ptr<1>
// CHECK: %[[NULL:.+]] = llvm.mlir.zero : !llvm.ptr<1>
// CHECK-NEXT: %[[CMP:.+]] = llvm.icmp "ne" %[[OPT]], %[[NULL]] : !llvm.ptr<1>
// CHECK: affine.store %[[CMP]], %{{.+}}[] : memref<i1>
// CHECK: enzymexla.xla_wrapper @rxla$raised_0 (%{{.+}}, %[[OUT]]) : (memref<i1, 1>, memref<32xf64, 1>) -> ()
// CHECK: func.func private @rxla$raised_0(%{{.+}}: tensor<i1>, %{{.+}}: tensor<32xf64>) -> (tensor<i1>, tensor<32xf64>)
// CHECK: arith.uitofp %{{.+}} : tensor<i1> to tensor<f64>
