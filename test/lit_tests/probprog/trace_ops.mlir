// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(lower-probprog-trace-ops{backend=cpu})" | FileCheck %s --check-prefix=CPU

module {
  func.func @test_get_subtrace(%arg0: !enzyme.Trace) -> !enzyme.Trace {
    %0 = enzyme.getSubtrace %arg0 {symbol = #enzyme.symbol<42>} : (!enzyme.Trace) -> !enzyme.Trace
    return %0 : !enzyme.Trace
  }

  func.func @test_get_flattened_samples(%arg0: !enzyme.Trace) -> tensor<3xf64> {
    %0 = enzyme.getFlattenedSamplesFromTrace %arg0 {selection = [[#enzyme.symbol<100>]]} : (!enzyme.Trace) -> tensor<3xf64>
    return %0 : tensor<3xf64>
  }

  func.func @test_get_flattened_samples_multi(%arg0: !enzyme.Trace) -> tensor<5xf64> {
    %0 = enzyme.getFlattenedSamplesFromTrace %arg0 {selection = [[#enzyme.symbol<200>], [#enzyme.symbol<300>]]} : (!enzyme.Trace) -> tensor<5xf64>
    return %0 : tensor<5xf64>
  }

  func.func @test_get_flattened_samples_nested(%arg0: !enzyme.Trace) -> tensor<4xf64> {
    %0 = enzyme.getFlattenedSamplesFromTrace %arg0 {selection = [[#enzyme.symbol<400>, #enzyme.symbol<401>]]} : (!enzyme.Trace) -> tensor<4xf64>
    return %0 : tensor<4xf64>
  }
}

// CPU: llvm.func @enzyme_probprog_get_flattened_samples_from_trace(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr)
// CPU: llvm.func @enzyme_probprog_get_subtrace(!llvm.ptr, !llvm.ptr, !llvm.ptr)

// CPU: llvm.func @enzyme_probprog_get_subtrace_wrapper_0(%[[ARG0:.+]]: !llvm.ptr, %[[ARG1:.+]]: !llvm.ptr, %[[ARG2:.+]]: !llvm.ptr) {
// CPU-NEXT:   llvm.call @enzyme_probprog_get_subtrace(%[[ARG0]], %[[ARG1]], %[[ARG2]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
// CPU-NEXT:   llvm.return
// CPU-NEXT: }

// CPU: func.func @test_get_subtrace(%[[ARG0:.+]]: tensor<ui64>) -> tensor<ui64> {
// CPU-NEXT:   %[[SYMBOL:.+]] = stablehlo.constant dense<42> : tensor<i64>
// CPU-NEXT:   %[[SUBTRACE_PTR:.+]] = stablehlo.constant dense<0> : tensor<ui64>
// CPU-NEXT:   %[[RESULT:.+]] = enzymexla.jit_call @enzyme_probprog_get_subtrace_wrapper_0 (%[[ARG0]], %[[SYMBOL]], %[[SUBTRACE_PTR]]) {output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 2, operand_tuple_indices = []>]} : (tensor<ui64>, tensor<i64>, tensor<ui64>) -> tensor<ui64>
// CPU-NEXT:   return %[[RESULT]] : tensor<ui64>
// CPU-NEXT: }

// CPU: func.func @test_get_flattened_samples(%[[ARG0:.+]]: tensor<ui64>) -> tensor<3xf64> {
// CPU-NEXT:   %[[NUM_ADDR:.+]] = stablehlo.constant dense<1> : tensor<i64>
// CPU-NEXT:   %[[TOTAL_SYM:.+]] = stablehlo.constant dense<1> : tensor<i64>
// CPU-NEXT:   %[[ADDR_LEN:.+]] = stablehlo.constant dense<1> : tensor<1xi64>
// CPU-NEXT:   %[[FLAT_SYM:.+]] = stablehlo.constant dense<100> : tensor<1xi64>
// CPU-NEXT:   %[[POS_BUF:.+]] = stablehlo.constant dense<0.000000e+00> : tensor<3xf64>
// CPU-NEXT:   %[[RESULT:.+]] = enzymexla.jit_call @enzyme_probprog_get_flattened_samples_from_trace_wrapper_0 (%[[ARG0]], %[[NUM_ADDR]], %[[TOTAL_SYM]], %[[ADDR_LEN]], %[[FLAT_SYM]], %[[POS_BUF]]) {output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 5, operand_tuple_indices = []>]} : (tensor<ui64>, tensor<i64>, tensor<i64>, tensor<1xi64>, tensor<1xi64>, tensor<3xf64>) -> tensor<3xf64>
// CPU-NEXT:   return %[[RESULT]] : tensor<3xf64>
// CPU-NEXT: }

// CPU: func.func @test_get_flattened_samples_multi(%[[ARG0:.+]]: tensor<ui64>) -> tensor<5xf64> {
// CPU-NEXT:   %[[NUM_ADDR:.+]] = stablehlo.constant dense<2> : tensor<i64>
// CPU-NEXT:   %[[TOTAL_SYM:.+]] = stablehlo.constant dense<2> : tensor<i64>
// CPU-NEXT:   %[[ADDR_LEN:.+]] = stablehlo.constant dense<1> : tensor<2xi64>
// CPU-NEXT:   %[[FLAT_SYM:.+]] = stablehlo.constant dense<[200, 300]> : tensor<2xi64>
// CPU-NEXT:   %[[POS_BUF:.+]] = stablehlo.constant dense<0.000000e+00> : tensor<5xf64>
// CPU-NEXT:   %[[RESULT:.+]] = enzymexla.jit_call @enzyme_probprog_get_flattened_samples_from_trace_wrapper_1 (%[[ARG0]], %[[NUM_ADDR]], %[[TOTAL_SYM]], %[[ADDR_LEN]], %[[FLAT_SYM]], %[[POS_BUF]]) {output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 5, operand_tuple_indices = []>]} : (tensor<ui64>, tensor<i64>, tensor<i64>, tensor<2xi64>, tensor<2xi64>, tensor<5xf64>) -> tensor<5xf64>
// CPU-NEXT:   return %[[RESULT]] : tensor<5xf64>
// CPU-NEXT: }

// CPU: func.func @test_get_flattened_samples_nested(%[[ARG0:.+]]: tensor<ui64>) -> tensor<4xf64> {
// CPU-NEXT:   %[[NUM_ADDR:.+]] = stablehlo.constant dense<1> : tensor<i64>
// CPU-NEXT:   %[[TOTAL_SYM:.+]] = stablehlo.constant dense<2> : tensor<i64>
// CPU-NEXT:   %[[ADDR_LEN:.+]] = stablehlo.constant dense<2> : tensor<1xi64>
// CPU-NEXT:   %[[FLAT_SYM:.+]] = stablehlo.constant dense<[400, 401]> : tensor<2xi64>
// CPU-NEXT:   %[[POS_BUF:.+]] = stablehlo.constant dense<0.000000e+00> : tensor<4xf64>
// CPU-NEXT:   %[[RESULT:.+]] = enzymexla.jit_call @enzyme_probprog_get_flattened_samples_from_trace_wrapper_2 (%[[ARG0]], %[[NUM_ADDR]], %[[TOTAL_SYM]], %[[ADDR_LEN]], %[[FLAT_SYM]], %[[POS_BUF]]) {output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 5, operand_tuple_indices = []>]} : (tensor<ui64>, tensor<i64>, tensor<i64>, tensor<1xi64>, tensor<2xi64>, tensor<4xf64>) -> tensor<4xf64>
// CPU-NEXT:   return %[[RESULT]] : tensor<4xf64>
// CPU-NEXT: }
