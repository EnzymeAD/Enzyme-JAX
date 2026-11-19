// RUN: enzymexlamlir-opt %s --jit-region-to-jit-call | FileCheck %s

// The jit_call replacing a jit_region keeps the discardable attributes of the
// region, e.g. `enzymexla.require_jit_ordering`.

module {
  func.func @ordered(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    %0 = "enzymexla.jit_region"(%arg0) <{
      output_operand_aliases = [
        #stablehlo.output_operand_alias<output_tuple_indices = [],
                                        operand_index = 0,
                                        operand_tuple_indices = []>
      ]
    }> ({
    ^bb0(%a: memref<4xf32, 1>):
      "enzymexla.jit_region_yield"() : () -> ()
    }) {enzymexla.require_jit_ordering, foo.bar = 1 : i64}
        : (tensor<4xf32>) -> tensor<4xf32>
    return %0 : tensor<4xf32>
  }

  func.func @unordered(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    %0 = "enzymexla.jit_region"(%arg0) <{
      output_operand_aliases = [
        #stablehlo.output_operand_alias<output_tuple_indices = [],
                                        operand_index = 0,
                                        operand_tuple_indices = []>
      ]
    }> ({
    ^bb0(%a: memref<4xf32, 1>):
      "enzymexla.jit_region_yield"() : () -> ()
    }) : (tensor<4xf32>) -> tensor<4xf32>
    return %0 : tensor<4xf32>
  }
}

// CHECK-LABEL: func.func @ordered(
// CHECK:         enzymexla.jit_call @ordered_jit_region
// CHECK-SAME:      {enzymexla.require_jit_ordering, foo.bar = 1 : i64}

// CHECK-LABEL: func.func @unordered(
// CHECK:         enzymexla.jit_call @unordered_jit_region
// CHECK-NOT:     enzymexla.require_jit_ordering
// CHECK-SAME:    : (tensor<4xf32>) -> tensor<4xf32>
