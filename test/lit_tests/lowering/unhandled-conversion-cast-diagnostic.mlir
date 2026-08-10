// RUN: not enzymexlamlir-opt %s --convert-polygeist-to-llvm 2>&1 | FileCheck %s

// A gpu_wrapper no pattern converted reaches the lowering, which converts its
// index operands and leaves the materialized casts behind. Saying only that
// an unhandled cast exists leaves nothing to work from; name the cast, the
// function holding it, and who reads it.

module {
  func.func @unconverted(%n: index, %m: memref<?xf64>, %v: f64) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %w = "enzymexla.gpu_wrapper"(%n, %c1, %c1, %c1, %c1, %c1) ({
      memref.store %v, %m[%c0] : memref<?xf64>
      "enzymexla.polygeist_yield"() : () -> ()
    }) : (index, index, index, index, index, index) -> index
    return
  }
}

// CHECK: error: Unhandled unrealized conversion cast %{{.+}} = "builtin.unrealized_conversion_cast"
// CHECK: note: in unconverted
// CHECK: note: used by
