// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(lower-enzyme-probprog{backend=cpu})" | FileCheck %s --check-prefix=CPU

module {
  func.func private @normal(tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>)
  func.func private @logpdf(tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<f64>

  func.func @simulate(%arg0: tensor<2xui64>, %arg1: tensor<f64>, %arg2: tensor<f64>) -> (tensor<f64>, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>) {
    %0:7 = call @test.simulate(%arg0, %arg1, %arg2) : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (!enzyme.Trace, tensor<f64>, tensor<f64>, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>)
    return %0#2, %0#3, %0#4, %0#5, %0#6 : tensor<f64>, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>
  }

  func.func @test.simulate(%arg0: tensor<2xui64>, %arg1: tensor<f64>, %arg2: tensor<f64>) -> (!enzyme.Trace, tensor<f64>, tensor<f64>, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>) {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %0 = enzyme.initTrace : !enzyme.Trace
    %1:4 = call @normal(%arg0, %arg1, %arg2) : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>)
    %2 = call @logpdf(%1#0, %arg1, %arg2) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<f64>
    %3 = stablehlo.add %2, %cst : tensor<f64>
    enzyme.addSampleToTrace(%1#0 : tensor<f64>) into %0 {symbol = #enzyme.symbol<1>}
    %4:7 = call @two_normals.simulate(%1#1, %1#0, %arg2) : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (!enzyme.Trace, tensor<f64>, tensor<f64>, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>)
    enzyme.addSubtrace %4#0 into %0 {symbol = #enzyme.symbol<2>}
    %5 = stablehlo.add %3, %4#1 : tensor<f64>
    enzyme.addSampleToTrace(%4#2, %4#3 : tensor<f64>, tensor<f64>) into %0 {symbol = #enzyme.symbol<2>}
    return %0, %5, %4#2, %4#3, %4#4, %4#5, %4#6 : !enzyme.Trace, tensor<f64>, tensor<f64>, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>
  }

  func.func @two_normals.simulate(%arg0: tensor<2xui64>, %arg1: tensor<f64>, %arg2: tensor<f64>) -> (!enzyme.Trace, tensor<f64>, tensor<f64>, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>) {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %0 = enzyme.initTrace : !enzyme.Trace
    %1:4 = call @normal(%arg0, %arg1, %arg2) : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>)
    %2 = call @logpdf(%1#0, %arg1, %arg2) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<f64>
    %3 = stablehlo.add %2, %cst : tensor<f64>
    enzyme.addSampleToTrace(%1#0 : tensor<f64>) into %0 {symbol = #enzyme.symbol<3>}
    %4:4 = call @normal(%1#1, %arg1, %arg2) : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>)
    %5 = call @logpdf(%4#0, %arg1, %arg2) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<f64>
    %6 = stablehlo.add %3, %5 : tensor<f64>
    enzyme.addSampleToTrace(%4#0 : tensor<f64>) into %0 {symbol = #enzyme.symbol<4>}
    return %0, %6, %1#0, %4#0, %4#1, %4#2, %4#3 : !enzyme.Trace, tensor<f64>, tensor<f64>, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>
  }
}

// CPU:  llvm.func @enzyme_probprog_add_sample_to_trace_wrapper_3(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr) {
// CPU-NEXT:     %0 = llvm.mlir.constant(1 : i64) : i64
// CPU-NEXT:     %1 = llvm.mlir.constant(1 : i64) : i64
// CPU-NEXT:     %2 = llvm.alloca %0 x i64 : (i64) -> !llvm.ptr
// CPU-NEXT:     llvm.store %1, %2 : i64, !llvm.ptr
// CPU-NEXT:     %3 = llvm.alloca %1 x !llvm.ptr : (i64) -> !llvm.ptr
// CPU-NEXT:     %4 = llvm.alloca %1 x i64 : (i64) -> !llvm.ptr
// CPU-NEXT:     %5 = llvm.alloca %1 x !llvm.ptr : (i64) -> !llvm.ptr
// CPU-NEXT:     %6 = llvm.alloca %1 x i64 : (i64) -> !llvm.ptr
// CPU-NEXT:     %7 = llvm.mlir.constant(0 : i64) : i64
// CPU-NEXT:     %8 = llvm.getelementptr %3[%7] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CPU-NEXT:     llvm.store %arg2, %8 : !llvm.ptr, !llvm.ptr
// CPU-NEXT:     %9 = llvm.mlir.constant(0 : i64) : i64
// CPU-NEXT:     %10 = llvm.mlir.constant(0 : i64) : i64
// CPU-NEXT:     %11 = llvm.getelementptr %4[%10] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CPU-NEXT:     llvm.store %9, %11 : i64, !llvm.ptr
// CPU-NEXT:     %12 = llvm.mlir.constant(64 : i64) : i64
// CPU-NEXT:     %13 = llvm.mlir.constant(0 : i64) : i64
// CPU-NEXT:     %14 = llvm.getelementptr %6[%13] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CPU-NEXT:     llvm.store %12, %14 : i64, !llvm.ptr
// CPU-NEXT:     %15 = llvm.mlir.constant(0 : i64) : i64
// CPU-NEXT:     %16 = llvm.alloca %15 x i64 : (i64) -> !llvm.ptr
// CPU-NEXT:     %17 = llvm.mlir.constant(0 : i64) : i64
// CPU-NEXT:     %18 = llvm.getelementptr %5[%17] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CPU-NEXT:     llvm.store %16, %18 : !llvm.ptr, !llvm.ptr
// CPU-NEXT:     llvm.call @enzyme_probprog_add_sample_to_trace(%arg0, %arg1, %3, %2, %4, %5, %6) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
// CPU-NEXT:     llvm.return
// CPU-NEXT:   }

// CPU:  llvm.func @enzyme_probprog_add_sample_to_trace_wrapper_2(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr) {
// CPU-NEXT:     %0 = llvm.mlir.constant(1 : i64) : i64
// CPU-NEXT:     %1 = llvm.mlir.constant(1 : i64) : i64
// CPU-NEXT:     %2 = llvm.alloca %0 x i64 : (i64) -> !llvm.ptr
// CPU-NEXT:     llvm.store %1, %2 : i64, !llvm.ptr
// CPU-NEXT:     %3 = llvm.alloca %1 x !llvm.ptr : (i64) -> !llvm.ptr
// CPU-NEXT:     %4 = llvm.alloca %1 x i64 : (i64) -> !llvm.ptr
// CPU-NEXT:     %5 = llvm.alloca %1 x !llvm.ptr : (i64) -> !llvm.ptr
// CPU-NEXT:     %6 = llvm.alloca %1 x i64 : (i64) -> !llvm.ptr
// CPU-NEXT:     %7 = llvm.mlir.constant(0 : i64) : i64
// CPU-NEXT:     %8 = llvm.getelementptr %3[%7] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CPU-NEXT:     llvm.store %arg2, %8 : !llvm.ptr, !llvm.ptr
// CPU-NEXT:     %9 = llvm.mlir.constant(0 : i64) : i64
// CPU-NEXT:     %10 = llvm.mlir.constant(0 : i64) : i64
// CPU-NEXT:     %11 = llvm.getelementptr %4[%10] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CPU-NEXT:     llvm.store %9, %11 : i64, !llvm.ptr
// CPU-NEXT:     %12 = llvm.mlir.constant(64 : i64) : i64
// CPU-NEXT:     %13 = llvm.mlir.constant(0 : i64) : i64
// CPU-NEXT:     %14 = llvm.getelementptr %6[%13] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CPU-NEXT:     llvm.store %12, %14 : i64, !llvm.ptr
// CPU-NEXT:     %15 = llvm.mlir.constant(0 : i64) : i64
// CPU-NEXT:     %16 = llvm.alloca %15 x i64 : (i64) -> !llvm.ptr
// CPU-NEXT:     %17 = llvm.mlir.constant(0 : i64) : i64
// CPU-NEXT:     %18 = llvm.getelementptr %5[%17] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CPU-NEXT:     llvm.store %16, %18 : !llvm.ptr, !llvm.ptr
// CPU-NEXT:     llvm.call @enzyme_probprog_add_sample_to_trace(%arg0, %arg1, %3, %2, %4, %5, %6) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
// CPU-NEXT:     llvm.return
// CPU-NEXT:   }

// CPU:  llvm.func @enzyme_probprog_init_trace_wrapper_1(%arg0: !llvm.ptr) {
// CPU-NEXT:     llvm.call @enzyme_probprog_init_trace(%arg0) : (!llvm.ptr) -> ()
// CPU-NEXT:     llvm.return
// CPU-NEXT:   }

// CPU:  llvm.func @enzyme_probprog_add_sample_to_trace_wrapper_1(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: !llvm.ptr) {
// CPU-NEXT:     %0 = llvm.mlir.constant(1 : i64) : i64
// CPU-NEXT:     %1 = llvm.mlir.constant(2 : i64) : i64
// CPU-NEXT:     %2 = llvm.alloca %0 x i64 : (i64) -> !llvm.ptr
// CPU-NEXT:     llvm.store %1, %2 : i64, !llvm.ptr
// CPU-NEXT:     %3 = llvm.alloca %1 x !llvm.ptr : (i64) -> !llvm.ptr
// CPU-NEXT:     %4 = llvm.alloca %1 x i64 : (i64) -> !llvm.ptr
// CPU-NEXT:     %5 = llvm.alloca %1 x !llvm.ptr : (i64) -> !llvm.ptr
// CPU-NEXT:     %6 = llvm.alloca %1 x i64 : (i64) -> !llvm.ptr
// CPU-NEXT:     %7 = llvm.mlir.constant(0 : i64) : i64
// CPU-NEXT:     %8 = llvm.getelementptr %3[%7] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CPU-NEXT:     llvm.store %arg2, %8 : !llvm.ptr, !llvm.ptr
// CPU-NEXT:     %9 = llvm.mlir.constant(0 : i64) : i64
// CPU-NEXT:     %10 = llvm.mlir.constant(0 : i64) : i64
// CPU-NEXT:     %11 = llvm.getelementptr %4[%10] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CPU-NEXT:     llvm.store %9, %11 : i64, !llvm.ptr
// CPU-NEXT:     %12 = llvm.mlir.constant(64 : i64) : i64
// CPU-NEXT:     %13 = llvm.mlir.constant(0 : i64) : i64
// CPU-NEXT:     %14 = llvm.getelementptr %6[%13] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CPU-NEXT:     llvm.store %12, %14 : i64, !llvm.ptr
// CPU-NEXT:     %15 = llvm.mlir.constant(0 : i64) : i64
// CPU-NEXT:     %16 = llvm.alloca %15 x i64 : (i64) -> !llvm.ptr
// CPU-NEXT:     %17 = llvm.mlir.constant(0 : i64) : i64
// CPU-NEXT:     %18 = llvm.getelementptr %5[%17] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CPU-NEXT:     llvm.store %16, %18 : !llvm.ptr, !llvm.ptr
// CPU-NEXT:     %19 = llvm.mlir.constant(1 : i64) : i64
// CPU-NEXT:     %20 = llvm.getelementptr %3[%19] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CPU-NEXT:     llvm.store %arg3, %20 : !llvm.ptr, !llvm.ptr
// CPU-NEXT:     %21 = llvm.mlir.constant(0 : i64) : i64
// CPU-NEXT:     %22 = llvm.mlir.constant(1 : i64) : i64
// CPU-NEXT:     %23 = llvm.getelementptr %4[%22] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CPU-NEXT:     llvm.store %21, %23 : i64, !llvm.ptr
// CPU-NEXT:     %24 = llvm.mlir.constant(64 : i64) : i64
// CPU-NEXT:     %25 = llvm.mlir.constant(1 : i64) : i64
// CPU-NEXT:     %26 = llvm.getelementptr %6[%25] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CPU-NEXT:     llvm.store %24, %26 : i64, !llvm.ptr
// CPU-NEXT:     %27 = llvm.mlir.constant(0 : i64) : i64
// CPU-NEXT:     %28 = llvm.alloca %27 x i64 : (i64) -> !llvm.ptr
// CPU-NEXT:     %29 = llvm.mlir.constant(1 : i64) : i64
// CPU-NEXT:     %30 = llvm.getelementptr %5[%29] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CPU-NEXT:     llvm.store %28, %30 : !llvm.ptr, !llvm.ptr
// CPU-NEXT:     llvm.call @enzyme_probprog_add_sample_to_trace(%arg0, %arg1, %3, %2, %4, %5, %6) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
// CPU-NEXT:     llvm.return
// CPU-NEXT:   }

// CPU:  llvm.func @enzyme_probprog_add_subtrace(!llvm.ptr, !llvm.ptr, !llvm.ptr)

// CPU:  llvm.func @enzyme_probprog_add_subtrace_wrapper_0(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr) {
// CPU-NEXT:     llvm.call @enzyme_probprog_add_subtrace(%arg0, %arg1, %arg2) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
// CPU-NEXT:     llvm.return
// CPU-NEXT:   }

// CPU:  llvm.func @enzyme_probprog_add_sample_to_trace(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr)

// CPU:  llvm.func @enzyme_probprog_add_sample_to_trace_wrapper_0(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr) {
// CPU-NEXT:     %0 = llvm.mlir.constant(1 : i64) : i64
// CPU-NEXT:     %1 = llvm.mlir.constant(1 : i64) : i64
// CPU-NEXT:     %2 = llvm.alloca %0 x i64 : (i64) -> !llvm.ptr
// CPU-NEXT:     llvm.store %1, %2 : i64, !llvm.ptr
// CPU-NEXT:     %3 = llvm.alloca %1 x !llvm.ptr : (i64) -> !llvm.ptr
// CPU-NEXT:     %4 = llvm.alloca %1 x i64 : (i64) -> !llvm.ptr
// CPU-NEXT:     %5 = llvm.alloca %1 x !llvm.ptr : (i64) -> !llvm.ptr
// CPU-NEXT:     %6 = llvm.alloca %1 x i64 : (i64) -> !llvm.ptr
// CPU-NEXT:     %7 = llvm.mlir.constant(0 : i64) : i64
// CPU-NEXT:     %8 = llvm.getelementptr %3[%7] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CPU-NEXT:     llvm.store %arg2, %8 : !llvm.ptr, !llvm.ptr
// CPU-NEXT:     %9 = llvm.mlir.constant(0 : i64) : i64
// CPU-NEXT:     %10 = llvm.mlir.constant(0 : i64) : i64
// CPU-NEXT:     %11 = llvm.getelementptr %4[%10] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CPU-NEXT:     llvm.store %9, %11 : i64, !llvm.ptr
// CPU-NEXT:     %12 = llvm.mlir.constant(64 : i64) : i64
// CPU-NEXT:     %13 = llvm.mlir.constant(0 : i64) : i64
// CPU-NEXT:     %14 = llvm.getelementptr %6[%13] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CPU-NEXT:     llvm.store %12, %14 : i64, !llvm.ptr
// CPU-NEXT:     %15 = llvm.mlir.constant(0 : i64) : i64
// CPU-NEXT:     %16 = llvm.alloca %15 x i64 : (i64) -> !llvm.ptr
// CPU-NEXT:     %17 = llvm.mlir.constant(0 : i64) : i64
// CPU-NEXT:     %18 = llvm.getelementptr %5[%17] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CPU-NEXT:     llvm.store %16, %18 : !llvm.ptr, !llvm.ptr
// CPU-NEXT:     llvm.call @enzyme_probprog_add_sample_to_trace(%arg0, %arg1, %3, %2, %4, %5, %6) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
// CPU-NEXT:     llvm.return
// CPU-NEXT:   }

// CPU:  llvm.func @enzyme_probprog_init_trace_wrapper_0(%arg0: !llvm.ptr) {
// CPU-NEXT:     llvm.call @enzyme_probprog_init_trace(%arg0) : (!llvm.ptr) -> ()
// CPU-NEXT:     llvm.return
// CPU-NEXT:   }

// CPU:  func.func @test.simulate(%arg0: tensor<2xui64>, %arg1: tensor<f64>, %arg2: tensor<f64>) -> (tensor<ui64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>) {
// CPU-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CPU-NEXT:     %c = stablehlo.constant dense<0> : tensor<ui64>
// CPU-NEXT:     %0 = enzymexla.jit_call @enzyme_probprog_init_trace_wrapper_0 (%c) {operand_layouts = [dense<0> : tensor<1xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<0> : tensor<1xindex>], xla_side_effect_free} : (tensor<ui64>) -> tensor<ui64>
// CPU-NEXT:     %1:4 = call @normal(%arg0, %arg1, %arg2) : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>)
// CPU-NEXT:     %2 = call @logpdf(%1#0, %arg1, %arg2) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<f64>
// CPU-NEXT:     %3 = stablehlo.add %2, %cst : tensor<f64>
// CPU-NEXT:     %c_0 = stablehlo.constant dense<1> : tensor<i64>
// CPU-NEXT:     enzymexla.jit_call @enzyme_probprog_add_sample_to_trace_wrapper_0 (%0, %c_0, %1#0) : (tensor<ui64>, tensor<i64>, tensor<f64>) -> ()
// CPU-NEXT:     %4:7 = call @two_normals.simulate(%1#1, %1#0, %arg2) : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<ui64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>)
// CPU-NEXT:     %c_1 = stablehlo.constant dense<2> : tensor<i64>
// CPU-NEXT:     enzymexla.jit_call @enzyme_probprog_add_subtrace_wrapper_0 (%0, %c_1, %4#0) : (tensor<ui64>, tensor<i64>, tensor<ui64>) -> ()
// CPU-NEXT:     %5 = stablehlo.add %3, %4#1 : tensor<f64>
// CPU-NEXT:     %c_2 = stablehlo.constant dense<2> : tensor<i64>
// CPU-NEXT:     enzymexla.jit_call @enzyme_probprog_add_sample_to_trace_wrapper_1 (%0, %c_2, %4#2, %4#3) : (tensor<ui64>, tensor<i64>, tensor<f64>, tensor<f64>) -> ()
// CPU-NEXT:     return %0, %5, %4#2, %4#3, %4#4, %4#5, %4#6 : tensor<ui64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>
// CPU-NEXT:   }

// CPU:  func.func @two_normals.simulate(%arg0: tensor<2xui64>, %arg1: tensor<f64>, %arg2: tensor<f64>) -> (tensor<ui64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>) {
// CPU-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CPU-NEXT:     %c = stablehlo.constant dense<0> : tensor<ui64>
// CPU-NEXT:     %0 = enzymexla.jit_call @enzyme_probprog_init_trace_wrapper_1 (%c) {operand_layouts = [dense<0> : tensor<1xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<0> : tensor<1xindex>], xla_side_effect_free} : (tensor<ui64>) -> tensor<ui64>
// CPU-NEXT:     %1:4 = call @normal(%arg0, %arg1, %arg2) : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>)
// CPU-NEXT:     %2 = call @logpdf(%1#0, %arg1, %arg2) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<f64>
// CPU-NEXT:     %3 = stablehlo.add %2, %cst : tensor<f64>
// CPU-NEXT:     %c_0 = stablehlo.constant dense<3> : tensor<i64>
// CPU-NEXT:     enzymexla.jit_call @enzyme_probprog_add_sample_to_trace_wrapper_2 (%0, %c_0, %1#0) : (tensor<ui64>, tensor<i64>, tensor<f64>) -> ()
// CPU-NEXT:     %4:4 = call @normal(%1#1, %arg1, %arg2) : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>)
// CPU-NEXT:     %5 = call @logpdf(%4#0, %arg1, %arg2) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<f64>
// CPU-NEXT:     %6 = stablehlo.add %3, %5 : tensor<f64>
// CPU-NEXT:     %c_1 = stablehlo.constant dense<4> : tensor<i64>
// CPU-NEXT:     enzymexla.jit_call @enzyme_probprog_add_sample_to_trace_wrapper_3 (%0, %c_1, %4#0) : (tensor<ui64>, tensor<i64>, tensor<f64>) -> ()
// CPU-NEXT:     return %0, %6, %1#0, %4#0, %4#1, %4#2, %4#3 : tensor<ui64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>
// CPU-NEXT:   }