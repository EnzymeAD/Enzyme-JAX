// RUN: enzymexlamlir-opt %s --jit-region-to-jit-call | FileCheck %s

// The gpu_wrappers of a jit_region run on the stream XLA executes it on.

module {
  func.func @main(%arg0: tensor<100xf32>) -> tensor<100xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c100 = arith.constant 100 : index
    %cst = arith.constant 2.0 : f32
    %0 = "enzymexla.jit_region"(%arg0) <{
      output_operand_aliases = [
        #stablehlo.output_operand_alias<output_tuple_indices = [],
                                        operand_index = 0,
                                        operand_tuple_indices = []>
      ]
    }> ({
    ^bb0(%a: memref<100xf32, 1>):
      %1 = "enzymexla.gpu_wrapper"(%c1, %c1, %c1, %c100, %c1, %c1) ({
        scf.parallel (%i) = (%c0) to (%c100) step (%c1) {
          memref.store %cst, %a[%i] : memref<100xf32, 1>
          scf.reduce
        }
        "enzymexla.polygeist_yield"() : () -> ()
      }) : (index, index, index, index, index, index) -> index
      %2 = "enzymexla.gpu_wrapper"(%c1, %c1, %c1, %c100, %c1, %c1) ({
        scf.parallel (%i) = (%c0) to (%c100) step (%c1) {
          memref.store %cst, %a[%i] : memref<100xf32, 1>
          scf.reduce
        }
        "enzymexla.polygeist_yield"() : () -> ()
      }) : (index, index, index, index, index, index) -> index
      "enzymexla.jit_region_yield"() : () -> ()
    }) : (tensor<100xf32>) -> tensor<100xf32>
    return %0 : tensor<100xf32>
  }
}

// CHECK-LABEL: func.func private @main_jit_region(
// CHECK:         %[[S:.+]] = "enzymexla.get_stream"() : () -> !gpu.async.token
// CHECK-NEXT:    %[[T:.+]] = "enzymexla.stream2token"(%[[S]]) : (!gpu.async.token) -> !async.token
// CHECK:         async.execute [%[[T]]] {
// CHECK-NEXT:      "enzymexla.gpu_wrapper"
// CHECK:           async.yield
// CHECK-NEXT:    }
// CHECK-NEXT:    async.execute [%[[T]]] {
// CHECK-NEXT:      "enzymexla.gpu_wrapper"
// CHECK:           async.yield
// CHECK-NEXT:    }
// CHECK-NEXT:    return
