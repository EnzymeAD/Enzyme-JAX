// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(lower-impulse-trace-ops{backend=cpu},fuse-jit)" %s | FileCheck %s --check-prefix=CSE
// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(lower-impulse-trace-ops{backend=cpu},cse,fuse-jit)" %s | FileCheck %s --check-prefix=CSE

// Metadata operands are hoisted into the fused call, so this real non-MPI
// chain is fuseable with or without CSE.

module {
  func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    %0 = "enzyme.dump"(%arg0) {label = "state"} : (tensor<4xf32>) -> tensor<4xf32>
    %1 = "enzyme.dump"(%0) {label = "state"} : (tensor<4xf32>) -> tensor<4xf32>
    return %1 : tensor<4xf32>
  }
}

// CSE-LABEL: llvm.func @fused__enzyme_probprog_dump_wrapper_{{[0-9]+}}_enzyme_probprog_dump_wrapper_{{[0-9]+}}
// CSE:         llvm.call @enzyme_probprog_dump
// CSE:         llvm.call @enzyme_probprog_dump
// CSE:         llvm.return

// CSE-LABEL: func.func @main
// CSE-NOT:     enzymexla.jit_call @enzyme_probprog_dump_wrapper_
// CSE:         %[[FUSED:.*]] = enzymexla.jit_call @fused__enzyme_probprog_dump_wrapper_{{[0-9]+}}_enzyme_probprog_dump_wrapper_{{[0-9]+}}
// CSE-SAME:    output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>]
// CSE:         return %[[FUSED]] : tensor<4xf32>
