// RUN: enzymexlamlir-opt %s --raise-affine-to-stablehlo | FileCheck %s

// A captured scalar ships to the kernel through a one-element staging
// buffer; the copy is sized in bytes, not elements: a count of one only
// moved the low byte, so any captured value past 255 arrived as zero.
func.func @capture(%out: memref<1024xf64, 1>, %n: index) {
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  %0 = "enzymexla.gpu_wrapper"(%c1, %c1, %c1, %c32, %c1, %c1) ({
    affine.parallel (%t) = (0) to (32) {
      %v = arith.index_cast %n : index to i64
      %tv = arith.index_cast %t : index to i64
      %s = arith.addi %v, %tv : i64
      %f = arith.sitofp %s : i64 to f64
      affine.store %f, %out[%t] : memref<1024xf64, 1>
    }
    "enzymexla.polygeist_yield"() : () -> ()
  }) : (index, index, index, index, index, index) -> index
  return
}

// CHECK-LABEL: func.func @capture(
// CHECK: %[[C8:.+]] = arith.constant 8 : index
// CHECK: enzymexla.memcpy %{{.+}}, %{{.+}}, %[[C8]] : memref<i64, 1>, memref<i64>
