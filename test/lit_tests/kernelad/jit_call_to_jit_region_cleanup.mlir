// RUN: enzymexlamlir-opt %s --jit-call-to-jit-region | FileCheck %s

module {
  // Inlined into one jit_region, but still called by a jit_call that has a
  // backend config.
  func.func private @kept(%arg0: memref<4xf32>) {
    return
  }

  // Called from the body of @outer, which is copied into a jit_region.
  func.func private @inner(%arg0: memref<4xf32>) {
    return
  }

  func.func private @outer(%arg0: memref<4xf32>) {
    func.call @inner(%arg0) : (memref<4xf32>) -> ()
    return
  }

  func.func @main(%a: tensor<4xf32>) -> tensor<4xf32> {
    %0 = enzymexla.jit_call @kept(%a) <
      output_operand_aliases = [
        #stablehlo.output_operand_alias<output_tuple_indices = [],
                                        operand_index = 0,
                                        operand_tuple_indices = []>
      ]
    > : (tensor<4xf32>) -> tensor<4xf32>
    %1 = enzymexla.jit_call @kept(%0) <
      backend_config = "config",
      output_operand_aliases = [
        #stablehlo.output_operand_alias<output_tuple_indices = [],
                                        operand_index = 0,
                                        operand_tuple_indices = []>
      ]
    > : (tensor<4xf32>) -> tensor<4xf32>
    %2 = enzymexla.jit_call @outer(%1) <
      output_operand_aliases = [
        #stablehlo.output_operand_alias<output_tuple_indices = [],
                                        operand_index = 0,
                                        operand_tuple_indices = []>
      ]
    > : (tensor<4xf32>) -> tensor<4xf32>
    %3 = enzymexla.jit_call @inner(%2) <
      output_operand_aliases = [
        #stablehlo.output_operand_alias<output_tuple_indices = [],
                                        operand_index = 0,
                                        operand_tuple_indices = []>
      ]
    > : (tensor<4xf32>) -> tensor<4xf32>
    return %3 : tensor<4xf32>
  }
}

// CHECK:       func.func private @kept(
// CHECK:       func.func private @inner(
// CHECK-NOT:   @outer(
// CHECK-LABEL: func.func @main(
// CHECK:         "enzymexla.jit_region"
// CHECK:         enzymexla.jit_call @kept
// CHECK:         "enzymexla.jit_region"
// CHECK:           func.call @inner(
// CHECK:         "enzymexla.jit_region"
