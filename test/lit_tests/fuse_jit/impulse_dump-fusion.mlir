// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(lower-impulse-trace-ops{backend=cpu},fuse-jit)" %s | FileCheck %s --check-prefix=NO-CSE
// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(lower-impulse-trace-ops{backend=cpu},cse,fuse-jit)" %s | FileCheck %s --check-prefix=CSE

// Lowering creates metadata constants next to each dump. Without CSE, the
// second call's metadata cannot dominate a fused call at the first call. CSE
// shares identical metadata and makes this real non-MPI chain fuseable.

module {
  func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    %0 = "enzyme.dump"(%arg0) {label = "state"} : (tensor<4xf32>) -> tensor<4xf32>
    %1 = "enzyme.dump"(%0) {label = "state"} : (tensor<4xf32>) -> tensor<4xf32>
    return %1 : tensor<4xf32>
  }
}

// NO-CSE-LABEL: func.func @main
// NO-CSE-NOT:     enzymexla.jit_call @fused__
// NO-CSE:         %[[FIRST:.*]] = enzymexla.jit_call @enzyme_probprog_dump_wrapper_[[ID:[0-9]+]] (%arg0,
// NO-CSE:         %[[SECOND:.*]] = enzymexla.jit_call @enzyme_probprog_dump_wrapper_[[ID]] (%[[FIRST]],
// NO-CSE-NOT:     enzymexla.jit_call @fused__
// NO-CSE:         return %[[SECOND]] : tensor<4xf32>

// CSE-LABEL: llvm.func @fused__enzyme_probprog_dump_wrapper_[[ID:[0-9]+]]_enzyme_probprog_dump_wrapper_[[ID]]
// CSE:         llvm.call @enzyme_probprog_dump
// CSE:         llvm.call @enzyme_probprog_dump
// CSE:         llvm.return

// CSE-LABEL: func.func @main
// CSE-NOT:     enzymexla.jit_call @enzyme_probprog_dump_wrapper_
// CSE:         %[[FUSED:.*]] = enzymexla.jit_call @fused__enzyme_probprog_dump_wrapper_[[ID]]_enzyme_probprog_dump_wrapper_[[ID]]
// CSE-SAME:    output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>]
// CSE:         return %[[FUSED]] : tensor<4xf32>
