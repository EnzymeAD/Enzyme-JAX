// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(lower-probprog-trace-ops{backend=cpu})" | FileCheck %s --check-prefix=CPU

module {
  // CPU:  func.func @test_select(%arg0: tensor<i1>) -> tensor<ui64> {
  // CPU-NEXT:    %c = stablehlo.constant dense<0> : tensor<ui64>
  // CPU-NEXT:    %0 = enzymexla.jit_call @enzyme_probprog_init_trace_wrapper_0 (%c) {operand_layouts = [dense<> : tensor<0xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<> : tensor<0xindex>]} : (tensor<ui64>) -> tensor<ui64>
  // CPU-NEXT:    %c_0 = stablehlo.constant dense<0> : tensor<ui64>
  // CPU-NEXT:    %1 = enzymexla.jit_call @enzyme_probprog_init_trace_wrapper_0 (%c_0) {operand_layouts = [dense<> : tensor<0xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<> : tensor<0xindex>]} : (tensor<ui64>) -> tensor<ui64>
  // CPU-NEXT:    %2 = stablehlo.select %arg0, %0, %1 : tensor<i1>, tensor<ui64>
  // CPU-NEXT:    return %2 : tensor<ui64>
  // CPU-NEXT:  }
  func.func @test_select(%condition: tensor<i1>) -> !enzyme.Trace {
    %trace_true = enzyme.initTrace : !enzyme.Trace
    %trace_false = enzyme.initTrace : !enzyme.Trace
    %selected = enzyme.select %condition, %trace_true, %trace_false : (tensor<i1>, !enzyme.Trace, !enzyme.Trace) -> !enzyme.Trace
    return %selected : !enzyme.Trace
  }
}
