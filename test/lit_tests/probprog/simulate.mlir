// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(lower-enzyme-probprog{backend=cpu})" | FileCheck %s --check-prefix=CPU

module {
  func.func private @normal(tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
  func.func private @logpdf(tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<f64>

  func.func @simulate(%arg0: tensor<2xui64>, %arg1: tensor<f64>, %arg2: tensor<f64>) -> (!enzyme.Trace, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>) {
    %0:5 = call @test.simulate(%arg0, %arg1, %arg2) : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (!enzyme.Trace, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>)
    return %0#0, %0#1, %0#2, %0#3, %0#4 : !enzyme.Trace, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>
  }

  func.func @test.simulate(%arg0: tensor<2xui64>, %arg1: tensor<f64>, %arg2: tensor<f64>) -> (!enzyme.Trace, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>) {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %0 = enzyme.initTrace : !enzyme.Trace
    %1:2 = call @normal(%arg0, %arg1, %arg2) : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    %2 = call @logpdf(%1#1, %arg1, %arg2) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<f64>
    %3 = stablehlo.add %2, %cst : tensor<f64>
    %4 = enzyme.addSampleToTrace(%1#1 : tensor<f64>) into %0 {symbol = #enzyme.symbol<1>}
    %5:5 = call @two_normals.simulate(%1#0, %1#1, %arg2) : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (!enzyme.Trace, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>)
    %6 = enzyme.addSubtrace %5#0 into %4 {symbol = #enzyme.symbol<2>}
    %7 = stablehlo.add %3, %5#1 : tensor<f64>
    %8 = enzyme.addSampleToTrace(%5#3, %5#4 : tensor<f64>, tensor<f64>) into %6 {symbol = #enzyme.symbol<2>}
    %9 = enzyme.addWeightToTrace(%7 : tensor<f64>) into %8
    %10 = enzyme.addRetvalToTrace(%5#3, %5#4 : tensor<f64>, tensor<f64>) into %9
    return %10, %7, %5#2, %5#3, %5#4 : !enzyme.Trace, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>
  }

  func.func @two_normals.simulate(%arg0: tensor<2xui64>, %arg1: tensor<f64>, %arg2: tensor<f64>) -> (!enzyme.Trace, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>) {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %0 = enzyme.initTrace : !enzyme.Trace
    %1:2 = call @normal(%arg0, %arg1, %arg2) : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    %2 = call @logpdf(%1#1, %arg1, %arg2) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<f64>
    %3 = stablehlo.add %2, %cst : tensor<f64>
    %4 = enzyme.addSampleToTrace(%1#1 : tensor<f64>) into %0 {symbol = #enzyme.symbol<3>}
    %5:2 = call @normal(%1#0, %arg1, %arg2) : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    %6 = call @logpdf(%5#1, %arg1, %arg2) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<f64>
    %7 = stablehlo.add %3, %6 : tensor<f64>
    %8 = enzyme.addSampleToTrace(%5#1 : tensor<f64>) into %4 {symbol = #enzyme.symbol<4>}
    %9 = enzyme.addWeightToTrace(%7 : tensor<f64>) into %8
    %10 = enzyme.addRetvalToTrace(%1#1, %5#1 : tensor<f64>, tensor<f64>) into %9
    return %10, %7, %5#0, %1#1, %5#1 : !enzyme.Trace, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>
  }
}

// CPU:  llvm.func @enzyme_probprog_add_retval_to_trace(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr)
// CPU:  llvm.func @enzyme_probprog_add_retval_to_trace_wrapper_0(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr) {
// CPU-NEXT:    %0 = llvm.mlir.constant(1 : i64) : i64
// CPU-NEXT:    %1 = llvm.mlir.constant(2 : i64) : i64
// CPU-NEXT:    %2 = llvm.alloca %0 x i64 : (i64) -> !llvm.ptr
// CPU-NEXT:    llvm.store %1, %2 : i64, !llvm.ptr
// CPU-NEXT:    %3 = llvm.alloca %1 x !llvm.ptr : (i64) -> !llvm.ptr
// CPU-NEXT:    %4 = llvm.alloca %1 x i64 : (i64) -> !llvm.ptr
// CPU-NEXT:    %5 = llvm.alloca %1 x !llvm.ptr : (i64) -> !llvm.ptr
// CPU-NEXT:    %6 = llvm.alloca %1 x i64 : (i64) -> !llvm.ptr
// CPU-NEXT:    %7 = llvm.mlir.constant(0 : i64) : i64
// CPU-NEXT:    %8 = llvm.getelementptr %3[%7] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CPU-NEXT:    llvm.store %arg1, %8 : !llvm.ptr, !llvm.ptr
// CPU-NEXT:    %9 = llvm.mlir.constant(0 : i64) : i64
// CPU-NEXT:    %10 = llvm.mlir.constant(0 : i64) : i64
// CPU-NEXT:    %11 = llvm.getelementptr %4[%10] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CPU-NEXT:    llvm.store %9, %11 : i64, !llvm.ptr
// CPU-NEXT:    %12 = llvm.mlir.constant(64 : i64) : i64
// CPU-NEXT:    %13 = llvm.mlir.constant(0 : i64) : i64
// CPU-NEXT:    %14 = llvm.getelementptr %6[%13] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CPU-NEXT:    llvm.store %12, %14 : i64, !llvm.ptr
// CPU-NEXT:    %15 = llvm.mlir.constant(0 : i64) : i64
// CPU-NEXT:    %16 = llvm.alloca %15 x i64 : (i64) -> !llvm.ptr
// CPU-NEXT:    %17 = llvm.mlir.constant(0 : i64) : i64
// CPU-NEXT:    %18 = llvm.getelementptr %5[%17] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CPU-NEXT:    llvm.store %16, %18 : !llvm.ptr, !llvm.ptr
// CPU-NEXT:    %19 = llvm.mlir.constant(1 : i64) : i64
// CPU-NEXT:    %20 = llvm.getelementptr %3[%19] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CPU-NEXT:    llvm.store %arg2, %20 : !llvm.ptr, !llvm.ptr
// CPU-NEXT:    %21 = llvm.mlir.constant(0 : i64) : i64
// CPU-NEXT:    %22 = llvm.mlir.constant(1 : i64) : i64
// CPU-NEXT:    %23 = llvm.getelementptr %4[%22] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CPU-NEXT:    llvm.store %21, %23 : i64, !llvm.ptr
// CPU-NEXT:    %24 = llvm.mlir.constant(64 : i64) : i64
// CPU-NEXT:    %25 = llvm.mlir.constant(1 : i64) : i64
// CPU-NEXT:    %26 = llvm.getelementptr %6[%25] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CPU-NEXT:    llvm.store %24, %26 : i64, !llvm.ptr
// CPU-NEXT:    %27 = llvm.mlir.constant(0 : i64) : i64
// CPU-NEXT:    %28 = llvm.alloca %27 x i64 : (i64) -> !llvm.ptr
// CPU-NEXT:    %29 = llvm.mlir.constant(1 : i64) : i64
// CPU-NEXT:    %30 = llvm.getelementptr %5[%29] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CPU-NEXT:    llvm.store %28, %30 : !llvm.ptr, !llvm.ptr
// CPU-NEXT:    llvm.call @enzyme_probprog_add_retval_to_trace(%arg0, %3, %2, %4, %5, %6) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
// CPU-NEXT:    llvm.return
// CPU-NEXT:  }

// CPU:  llvm.func @enzyme_probprog_add_weight_to_trace(!llvm.ptr, !llvm.ptr)
// CPU:  llvm.func @enzyme_probprog_add_weight_to_trace_wrapper_0(%arg0: !llvm.ptr, %arg1: !llvm.ptr) {
// CPU-NEXT:    llvm.call @enzyme_probprog_add_weight_to_trace(%arg0, %arg1) : (!llvm.ptr, !llvm.ptr) -> ()
// CPU-NEXT:    llvm.return
// CPU-NEXT:  }

// CPU:  llvm.func @enzyme_probprog_add_sample_to_trace_wrapper_1(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: !llvm.ptr) {
// CPU-NEXT:    %0 = llvm.mlir.constant(1 : i64) : i64
// CPU-NEXT:    %1 = llvm.mlir.constant(2 : i64) : i64
// CPU-NEXT:    %2 = llvm.alloca %0 x i64 : (i64) -> !llvm.ptr
// CPU-NEXT:    llvm.store %1, %2 : i64, !llvm.ptr
// CPU-NEXT:    %3 = llvm.alloca %1 x !llvm.ptr : (i64) -> !llvm.ptr
// CPU-NEXT:    %4 = llvm.alloca %1 x i64 : (i64) -> !llvm.ptr
// CPU-NEXT:    %5 = llvm.alloca %1 x !llvm.ptr : (i64) -> !llvm.ptr
// CPU-NEXT:    %6 = llvm.alloca %1 x i64 : (i64) -> !llvm.ptr
// CPU-NEXT:    %7 = llvm.mlir.constant(0 : i64) : i64
// CPU-NEXT:    %8 = llvm.getelementptr %3[%7] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CPU-NEXT:    llvm.store %arg2, %8 : !llvm.ptr, !llvm.ptr
// CPU-NEXT:    %9 = llvm.mlir.constant(0 : i64) : i64
// CPU-NEXT:    %10 = llvm.mlir.constant(0 : i64) : i64
// CPU-NEXT:    %11 = llvm.getelementptr %4[%10] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CPU-NEXT:    llvm.store %9, %11 : i64, !llvm.ptr
// CPU-NEXT:    %12 = llvm.mlir.constant(64 : i64) : i64
// CPU-NEXT:    %13 = llvm.mlir.constant(0 : i64) : i64
// CPU-NEXT:    %14 = llvm.getelementptr %6[%13] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CPU-NEXT:    llvm.store %12, %14 : i64, !llvm.ptr
// CPU-NEXT:    %15 = llvm.mlir.constant(0 : i64) : i64
// CPU-NEXT:    %16 = llvm.alloca %15 x i64 : (i64) -> !llvm.ptr
// CPU-NEXT:    %17 = llvm.mlir.constant(0 : i64) : i64
// CPU-NEXT:    %18 = llvm.getelementptr %5[%17] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CPU-NEXT:    llvm.store %16, %18 : !llvm.ptr, !llvm.ptr
// CPU-NEXT:    %19 = llvm.mlir.constant(1 : i64) : i64
// CPU-NEXT:    %20 = llvm.getelementptr %3[%19] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CPU-NEXT:    llvm.store %arg3, %20 : !llvm.ptr, !llvm.ptr
// CPU-NEXT:    %21 = llvm.mlir.constant(0 : i64) : i64
// CPU-NEXT:    %22 = llvm.mlir.constant(1 : i64) : i64
// CPU-NEXT:    %23 = llvm.getelementptr %4[%22] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CPU-NEXT:    llvm.store %21, %23 : i64, !llvm.ptr
// CPU-NEXT:    %24 = llvm.mlir.constant(64 : i64) : i64
// CPU-NEXT:    %25 = llvm.mlir.constant(1 : i64) : i64
// CPU-NEXT:    %26 = llvm.getelementptr %6[%25] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CPU-NEXT:    llvm.store %24, %26 : i64, !llvm.ptr
// CPU-NEXT:    %27 = llvm.mlir.constant(0 : i64) : i64
// CPU-NEXT:    %28 = llvm.alloca %27 x i64 : (i64) -> !llvm.ptr
// CPU-NEXT:    %29 = llvm.mlir.constant(1 : i64) : i64
// CPU-NEXT:    %30 = llvm.getelementptr %5[%29] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CPU-NEXT:    llvm.store %28, %30 : !llvm.ptr, !llvm.ptr
// CPU-NEXT:    llvm.call @enzyme_probprog_add_sample_to_trace(%arg0, %arg1, %3, %2, %4, %5, %6) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
// CPU-NEXT:    llvm.return
// CPU-NEXT:  }

// CPU:  llvm.func @enzyme_probprog_add_subtrace(!llvm.ptr, !llvm.ptr, !llvm.ptr)
// CPU:  llvm.func @enzyme_probprog_add_subtrace_wrapper_0(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr) {
// CPU-NEXT:    llvm.call @enzyme_probprog_add_subtrace(%arg0, %arg1, %arg2) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
// CPU-NEXT:    llvm.return
// CPU-NEXT:  }

// CPU:  llvm.func @enzyme_probprog_add_sample_to_trace(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr)
// CPU:  llvm.func @enzyme_probprog_add_sample_to_trace_wrapper_0(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr) {
// CPU-NEXT:    %0 = llvm.mlir.constant(1 : i64) : i64
// CPU-NEXT:    %1 = llvm.mlir.constant(1 : i64) : i64
// CPU-NEXT:    %2 = llvm.alloca %0 x i64 : (i64) -> !llvm.ptr
// CPU-NEXT:    llvm.store %1, %2 : i64, !llvm.ptr
// CPU-NEXT:    %3 = llvm.alloca %1 x !llvm.ptr : (i64) -> !llvm.ptr
// CPU-NEXT:    %4 = llvm.alloca %1 x i64 : (i64) -> !llvm.ptr
// CPU-NEXT:    %5 = llvm.alloca %1 x !llvm.ptr : (i64) -> !llvm.ptr
// CPU-NEXT:    %6 = llvm.alloca %1 x i64 : (i64) -> !llvm.ptr
// CPU-NEXT:    %7 = llvm.mlir.constant(0 : i64) : i64
// CPU-NEXT:    %8 = llvm.getelementptr %3[%7] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CPU-NEXT:    llvm.store %arg2, %8 : !llvm.ptr, !llvm.ptr
// CPU-NEXT:    %9 = llvm.mlir.constant(0 : i64) : i64
// CPU-NEXT:    %10 = llvm.mlir.constant(0 : i64) : i64
// CPU-NEXT:    %11 = llvm.getelementptr %4[%10] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CPU-NEXT:    llvm.store %9, %11 : i64, !llvm.ptr
// CPU-NEXT:    %12 = llvm.mlir.constant(64 : i64) : i64
// CPU-NEXT:    %13 = llvm.mlir.constant(0 : i64) : i64
// CPU-NEXT:    %14 = llvm.getelementptr %6[%13] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CPU-NEXT:    llvm.store %12, %14 : i64, !llvm.ptr
// CPU-NEXT:    %15 = llvm.mlir.constant(0 : i64) : i64
// CPU-NEXT:    %16 = llvm.alloca %15 x i64 : (i64) -> !llvm.ptr
// CPU-NEXT:    %17 = llvm.mlir.constant(0 : i64) : i64
// CPU-NEXT:    %18 = llvm.getelementptr %5[%17] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CPU-NEXT:    llvm.store %16, %18 : !llvm.ptr, !llvm.ptr
// CPU-NEXT:    llvm.call @enzyme_probprog_add_sample_to_trace(%arg0, %arg1, %3, %2, %4, %5, %6) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
// CPU-NEXT:    llvm.return
// CPU-NEXT:  }

// CPU:  llvm.func @enzyme_probprog_init_trace(!llvm.ptr)
// CPU:  llvm.func @enzyme_probprog_init_trace_wrapper_0(%arg0: !llvm.ptr) {
// CPU-NEXT:    llvm.call @enzyme_probprog_init_trace(%arg0) : (!llvm.ptr) -> ()
// CPU-NEXT:    llvm.return
// CPU-NEXT:  }

// CPU:  func.func @simulate(%arg0: tensor<2xui64>, %arg1: tensor<f64>, %arg2: tensor<f64>) -> (tensor<ui64>, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>) {
// CPU-NEXT:    %0:5 = call @test.simulate(%arg0, %arg1, %arg2) : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<ui64>, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>)
// CPU-NEXT:    return %0#0, %0#1, %0#2, %0#3, %0#4 : tensor<ui64>, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>
// CPU-NEXT:  }

// CPU:  func.func @test.simulate(%arg0: tensor<2xui64>, %arg1: tensor<f64>, %arg2: tensor<f64>) -> (tensor<ui64>, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>) {
// CPU-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CPU-NEXT:    %c = stablehlo.constant dense<0> : tensor<ui64>
// CPU-NEXT:    %0 = enzymexla.jit_call @enzyme_probprog_init_trace_wrapper_0 (%c) {operand_layouts = [dense<> : tensor<0xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<> : tensor<0xindex>]} : (tensor<ui64>) -> tensor<ui64>
// CPU-NEXT:    %1:2 = call @normal(%arg0, %arg1, %arg2) : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
// CPU-NEXT:    %2 = call @logpdf(%1#1, %arg1, %arg2) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<f64>
// CPU-NEXT:    %3 = stablehlo.add %2, %cst : tensor<f64>
// CPU-NEXT:    %c_0 = stablehlo.constant dense<1> : tensor<i64>
// CPU-NEXT:    %4 = enzymexla.jit_call @enzyme_probprog_add_sample_to_trace_wrapper_0 (%0, %c_0, %1#1) {output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>]} : (tensor<ui64>, tensor<i64>, tensor<f64>) -> tensor<ui64>
// CPU-NEXT:    %5:5 = call @two_normals.simulate(%1#0, %1#1, %arg2) : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<ui64>, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>)
// CPU-NEXT:    %c_1 = stablehlo.constant dense<2> : tensor<i64>
// CPU-NEXT:    %6 = enzymexla.jit_call @enzyme_probprog_add_subtrace_wrapper_0 (%4, %c_1, %5#0) {output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>]} : (tensor<ui64>, tensor<i64>, tensor<ui64>) -> tensor<ui64>
// CPU-NEXT:    %7 = stablehlo.add %3, %5#1 : tensor<f64>
// CPU-NEXT:    %c_2 = stablehlo.constant dense<2> : tensor<i64>
// CPU-NEXT:    %8 = enzymexla.jit_call @enzyme_probprog_add_sample_to_trace_wrapper_1 (%6, %c_2, %5#3, %5#4) {output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>]} : (tensor<ui64>, tensor<i64>, tensor<f64>, tensor<f64>) -> tensor<ui64>
// CPU-NEXT:    %9 = enzymexla.jit_call @enzyme_probprog_add_weight_to_trace_wrapper_0 (%8, %7) {output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>]} : (tensor<ui64>, tensor<f64>) -> tensor<ui64>
// CPU-NEXT:    %10 = enzymexla.jit_call @enzyme_probprog_add_retval_to_trace_wrapper_0 (%9, %5#3, %5#4) {output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>]} : (tensor<ui64>, tensor<f64>, tensor<f64>) -> tensor<ui64>
// CPU-NEXT:    return %10, %7, %5#2, %5#3, %5#4 : tensor<ui64>, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>
// CPU-NEXT:  }

// CPU:  func.func @two_normals.simulate(%arg0: tensor<2xui64>, %arg1: tensor<f64>, %arg2: tensor<f64>) -> (tensor<ui64>, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>) {
// CPU-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CPU-NEXT:    %c = stablehlo.constant dense<0> : tensor<ui64>
// CPU-NEXT:    %0 = enzymexla.jit_call @enzyme_probprog_init_trace_wrapper_0 (%c) {operand_layouts = [dense<> : tensor<0xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<> : tensor<0xindex>]} : (tensor<ui64>) -> tensor<ui64>
// CPU-NEXT:    %1:2 = call @normal(%arg0, %arg1, %arg2) : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
// CPU-NEXT:    %2 = call @logpdf(%1#1, %arg1, %arg2) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<f64>
// CPU-NEXT:    %3 = stablehlo.add %2, %cst : tensor<f64>
// CPU-NEXT:    %c_0 = stablehlo.constant dense<3> : tensor<i64>
// CPU-NEXT:    %4 = enzymexla.jit_call @enzyme_probprog_add_sample_to_trace_wrapper_0 (%0, %c_0, %1#1) {output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>]} : (tensor<ui64>, tensor<i64>, tensor<f64>) -> tensor<ui64>
// CPU-NEXT:    %5:2 = call @normal(%1#0, %arg1, %arg2) : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
// CPU-NEXT:    %6 = call @logpdf(%5#1, %arg1, %arg2) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<f64>
// CPU-NEXT:    %7 = stablehlo.add %3, %6 : tensor<f64>
// CPU-NEXT:    %c_1 = stablehlo.constant dense<4> : tensor<i64>
// CPU-NEXT:    %8 = enzymexla.jit_call @enzyme_probprog_add_sample_to_trace_wrapper_0 (%4, %c_1, %5#1) {output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>]} : (tensor<ui64>, tensor<i64>, tensor<f64>) -> tensor<ui64>
// CPU-NEXT:    %9 = enzymexla.jit_call @enzyme_probprog_add_weight_to_trace_wrapper_0 (%8, %7) {output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>]} : (tensor<ui64>, tensor<f64>) -> tensor<ui64>
// CPU-NEXT:    %10 = enzymexla.jit_call @enzyme_probprog_add_retval_to_trace_wrapper_0 (%9, %1#1, %5#1) {output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>]} : (tensor<ui64>, tensor<f64>, tensor<f64>) -> tensor<ui64>
// CPU-NEXT:    return %10, %7, %5#0, %1#1, %5#1 : tensor<ui64>, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>
// CPU-NEXT:  }
