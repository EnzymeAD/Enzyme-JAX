// RUN: enzymexlamlir-opt %s --jit-call-to-jit-region | FileCheck %s
// RUN: enzymexlamlir-opt %s --jit-call-to-jit-region --jit-region-to-jit-call | FileCheck %s --check-prefix=ROUNDTRIP

// The jit_region replacing a jit_call keeps the discardable attributes of the
// call, e.g. `enzymexla.require_jit_ordering`, and so does the jit_call it is
// lowered back to.

module {
  llvm.func private @wrapper(%arg0: !llvm.ptr<1>) {
    llvm.return
  }

  func.func @ordered(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    %0 = enzymexla.jit_call @wrapper (%arg0) <
      output_operand_aliases = [
        #stablehlo.output_operand_alias<output_tuple_indices = [],
                                        operand_index = 0,
                                        operand_tuple_indices = []>
      ]
    > {enzymexla.require_jit_ordering, foo.bar = 1 : i64}
        : (tensor<4xf32>) -> tensor<4xf32>
    return %0 : tensor<4xf32>
  }

  func.func @unordered(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    %0 = enzymexla.jit_call @wrapper (%arg0) <
      output_operand_aliases = [
        #stablehlo.output_operand_alias<output_tuple_indices = [],
                                        operand_index = 0,
                                        operand_tuple_indices = []>
      ]
    > : (tensor<4xf32>) -> tensor<4xf32>
    return %0 : tensor<4xf32>
  }
}

// CHECK-LABEL: func.func @ordered(
// CHECK:         "enzymexla.jit_region"
// CHECK:           enzymexla.jit_region_yield
// CHECK-NEXT:    }) {enzymexla.require_jit_ordering, foo.bar = 1 : i64}
// CHECK-SAME:    : (tensor<4xf32>) -> tensor<4xf32>

// CHECK-LABEL: func.func @unordered(
// CHECK:         "enzymexla.jit_region"
// CHECK:           enzymexla.jit_region_yield
// CHECK-NEXT:    }) : (tensor<4xf32>) -> tensor<4xf32>

// ROUNDTRIP-LABEL: func.func @ordered(
// ROUNDTRIP:         enzymexla.jit_call @ordered_jit_region
// ROUNDTRIP-SAME:      {enzymexla.require_jit_ordering, foo.bar = 1 : i64}

// ROUNDTRIP-LABEL: func.func @unordered(
// ROUNDTRIP:         enzymexla.jit_call @unordered_jit_region
// ROUNDTRIP-NOT:     enzymexla.require_jit_ordering
// ROUNDTRIP-SAME:    : (tensor<4xf32>) -> tensor<4xf32>
