// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(lower-probprog-trace-ops{backend=cpu})" | FileCheck %s --check-prefix=CPU

module {
  func.func private @joint(tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>, tensor<2xf64>)
  func.func private @joint_logpdf(tensor<f64>, tensor<2xf64>, tensor<f64>, tensor<f64>) -> tensor<f64>

  func.func @foo(%arg0: tensor<2xui64>, %arg1: tensor<f64>, %arg2: tensor<f64>) -> (tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<2xf64>) {
    %0 = stablehlo.constant dense<42> : tensor<ui64>
    %constraint = builtin.unrealized_conversion_cast %0 : tensor<ui64> to !enzyme.Constraint
    %1:5 = call @test.generate(%constraint, %arg0, %arg1, %arg2) : (!enzyme.Constraint, tensor<2xui64>, tensor<f64>, tensor<f64>) -> (!enzyme.Trace, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<2xf64>)
    return %1#1, %1#2, %1#3, %1#4 : tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<2xf64>
  }

  func.func @test.generate(%arg0: !enzyme.Constraint, %arg1: tensor<2xui64>, %arg2: tensor<f64>, %arg3: tensor<f64>) -> (!enzyme.Trace, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<2xf64>) {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %0 = enzyme.initTrace : !enzyme.Trace
    %1:2 = enzyme.getSampleFromConstraint %arg0 {symbol = #enzyme.symbol<5>} : tensor<f64>, tensor<2xf64>
    %2 = call @joint_logpdf(%1#0, %1#1, %arg2, %arg3) : (tensor<f64>, tensor<2xf64>, tensor<f64>, tensor<f64>) -> tensor<f64>
    %3 = stablehlo.add %2, %cst : tensor<f64>
    %4 = enzyme.addSampleToTrace %1#0, %1#1 into %0 {symbol = #enzyme.symbol<5>} : (!enzyme.Trace, tensor<f64>, tensor<2xf64>) -> !enzyme.Trace
    %5 = enzyme.addWeightToTrace %3 into %4 : (!enzyme.Trace, tensor<f64>) -> !enzyme.Trace
    %6 = enzyme.addRetvalToTrace %1#0, %1#1 into %5 : (!enzyme.Trace, tensor<f64>, tensor<2xf64>) -> !enzyme.Trace
    return %6, %3, %arg1, %1#0, %1#1 : !enzyme.Trace, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<2xf64>
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
// CPU-NEXT:    %21 = llvm.mlir.constant(1 : i64) : i64
// CPU-NEXT:    %22 = llvm.mlir.constant(1 : i64) : i64
// CPU-NEXT:    %23 = llvm.getelementptr %4[%22] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CPU-NEXT:    llvm.store %21, %23 : i64, !llvm.ptr
// CPU-NEXT:    %24 = llvm.mlir.constant(64 : i64) : i64
// CPU-NEXT:    %25 = llvm.mlir.constant(1 : i64) : i64
// CPU-NEXT:    %26 = llvm.getelementptr %6[%25] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CPU-NEXT:    llvm.store %24, %26 : i64, !llvm.ptr
// CPU-NEXT:    %27 = llvm.mlir.constant(1 : i64) : i64
// CPU-NEXT:    %28 = llvm.alloca %27 x i64 : (i64) -> !llvm.ptr
// CPU-NEXT:    %29 = llvm.mlir.constant(2 : i64) : i64
// CPU-NEXT:    %30 = llvm.mlir.constant(0 : i64) : i64
// CPU-NEXT:    %31 = llvm.getelementptr %28[%30] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CPU-NEXT:    llvm.store %29, %31 : i64, !llvm.ptr
// CPU-NEXT:    %32 = llvm.mlir.constant(1 : i64) : i64
// CPU-NEXT:    %33 = llvm.getelementptr %5[%32] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CPU-NEXT:    llvm.store %28, %33 : !llvm.ptr, !llvm.ptr
// CPU-NEXT:    llvm.call @enzyme_probprog_add_retval_to_trace(%arg0, %3, %2, %4, %5, %6) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
// CPU-NEXT:    llvm.return
// CPU-NEXT:  }

// CPU:  llvm.func @enzyme_probprog_add_weight_to_trace(!llvm.ptr, !llvm.ptr)
// CPU:  llvm.func @enzyme_probprog_add_weight_to_trace_wrapper_0(%arg0: !llvm.ptr, %arg1: !llvm.ptr) {
// CPU-NEXT:    llvm.call @enzyme_probprog_add_weight_to_trace(%arg0, %arg1) : (!llvm.ptr, !llvm.ptr) -> ()
// CPU-NEXT:    llvm.return
// CPU-NEXT:  }

// CPU:  llvm.func @enzyme_probprog_add_sample_to_trace(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr)
// CPU:  llvm.func @enzyme_probprog_add_sample_to_trace_wrapper_0(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: !llvm.ptr) {
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
// CPU-NEXT:    %21 = llvm.mlir.constant(1 : i64) : i64
// CPU-NEXT:    %22 = llvm.mlir.constant(1 : i64) : i64
// CPU-NEXT:    %23 = llvm.getelementptr %4[%22] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CPU-NEXT:    llvm.store %21, %23 : i64, !llvm.ptr
// CPU-NEXT:    %24 = llvm.mlir.constant(64 : i64) : i64
// CPU-NEXT:    %25 = llvm.mlir.constant(1 : i64) : i64
// CPU-NEXT:    %26 = llvm.getelementptr %6[%25] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CPU-NEXT:    llvm.store %24, %26 : i64, !llvm.ptr
// CPU-NEXT:    %27 = llvm.mlir.constant(1 : i64) : i64
// CPU-NEXT:    %28 = llvm.alloca %27 x i64 : (i64) -> !llvm.ptr
// CPU-NEXT:    %29 = llvm.mlir.constant(2 : i64) : i64
// CPU-NEXT:    %30 = llvm.mlir.constant(0 : i64) : i64
// CPU-NEXT:    %31 = llvm.getelementptr %28[%30] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CPU-NEXT:    llvm.store %29, %31 : i64, !llvm.ptr
// CPU-NEXT:    %32 = llvm.mlir.constant(1 : i64) : i64
// CPU-NEXT:    %33 = llvm.getelementptr %5[%32] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CPU-NEXT:    llvm.store %28, %33 : !llvm.ptr, !llvm.ptr
// CPU-NEXT:    llvm.call @enzyme_probprog_add_sample_to_trace(%arg0, %arg1, %3, %2, %4, %5, %6) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
// CPU-NEXT:    llvm.return
// CPU-NEXT:  }

// CPU:  llvm.func @enzyme_probprog_get_sample_from_constraint(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr)
// CPU:  llvm.func @enzyme_probprog_get_sample_from_constraint_wrapper_0(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: !llvm.ptr) {
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
// CPU-NEXT:    %21 = llvm.mlir.constant(1 : i64) : i64
// CPU-NEXT:    %22 = llvm.mlir.constant(1 : i64) : i64
// CPU-NEXT:    %23 = llvm.getelementptr %4[%22] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CPU-NEXT:    llvm.store %21, %23 : i64, !llvm.ptr
// CPU-NEXT:    %24 = llvm.mlir.constant(64 : i64) : i64
// CPU-NEXT:    %25 = llvm.mlir.constant(1 : i64) : i64
// CPU-NEXT:    %26 = llvm.getelementptr %6[%25] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CPU-NEXT:    llvm.store %24, %26 : i64, !llvm.ptr
// CPU-NEXT:    %27 = llvm.mlir.constant(1 : i64) : i64
// CPU-NEXT:    %28 = llvm.alloca %27 x i64 : (i64) -> !llvm.ptr
// CPU-NEXT:    %29 = llvm.mlir.constant(2 : i64) : i64
// CPU-NEXT:    %30 = llvm.mlir.constant(0 : i64) : i64
// CPU-NEXT:    %31 = llvm.getelementptr %28[%30] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CPU-NEXT:    llvm.store %29, %31 : i64, !llvm.ptr
// CPU-NEXT:    %32 = llvm.mlir.constant(1 : i64) : i64
// CPU-NEXT:    %33 = llvm.getelementptr %5[%32] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CPU-NEXT:    llvm.store %28, %33 : !llvm.ptr, !llvm.ptr
// CPU-NEXT:    llvm.call @enzyme_probprog_get_sample_from_constraint(%arg0, %arg1, %3, %2, %4, %5, %6) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
// CPU-NEXT:    llvm.return
// CPU-NEXT:  }

// CPU:  llvm.func @enzyme_probprog_init_trace(!llvm.ptr)
// CPU:  llvm.func @enzyme_probprog_init_trace_wrapper_0(%arg0: !llvm.ptr) {
// CPU-NEXT:    llvm.call @enzyme_probprog_init_trace(%arg0) : (!llvm.ptr) -> ()
// CPU-NEXT:    llvm.return
// CPU-NEXT:  }

// CPU:  func.func @foo(%arg0: tensor<2xui64>, %arg1: tensor<f64>, %arg2: tensor<f64>) -> (tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<2xf64>) {
// CPU-NEXT:    %c = stablehlo.constant dense<42> : tensor<ui64>
// CPU-NEXT:    %0:5 = call @test.generate(%c, %arg0, %arg1, %arg2) : (tensor<ui64>, tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<ui64>, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<2xf64>)
// CPU-NEXT:    return %0#1, %0#2, %0#3, %0#4 : tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<2xf64>
// CPU-NEXT:  }

// CPU:  func.func @test.generate(%arg0: tensor<ui64>, %arg1: tensor<2xui64>, %arg2: tensor<f64>, %arg3: tensor<f64>) -> (tensor<ui64>, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<2xf64>) {
// CPU-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CPU-NEXT:    %c = stablehlo.constant dense<0> : tensor<ui64>
// CPU-NEXT:    %0 = enzymexla.jit_call @enzyme_probprog_init_trace_wrapper_0 (%c) {operand_layouts = [dense<> : tensor<0xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<> : tensor<0xindex>]} : (tensor<ui64>) -> tensor<ui64>
// CPU-NEXT:    %c_0 = stablehlo.constant dense<5> : tensor<i64>
// CPU-NEXT:    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CPU-NEXT:    %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<2xf64>
// CPU-NEXT:    %1:2 = enzymexla.jit_call @enzyme_probprog_get_sample_from_constraint_wrapper_0 (%arg0, %c_0, %cst_1, %cst_2) {output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 2, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 3, operand_tuple_indices = []>]} : (tensor<ui64>, tensor<i64>, tensor<f64>, tensor<2xf64>) -> (tensor<f64>, tensor<2xf64>)
// CPU-NEXT:    %2 = call @joint_logpdf(%1#0, %1#1, %arg2, %arg3) : (tensor<f64>, tensor<2xf64>, tensor<f64>, tensor<f64>) -> tensor<f64>
// CPU-NEXT:    %3 = stablehlo.add %2, %cst : tensor<f64>
// CPU-NEXT:    %c_3 = stablehlo.constant dense<5> : tensor<i64>
// CPU-NEXT:    %4 = enzymexla.jit_call @enzyme_probprog_add_sample_to_trace_wrapper_0 (%0, %c_3, %1#0, %1#1) {output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>]} : (tensor<ui64>, tensor<i64>, tensor<f64>, tensor<2xf64>) -> tensor<ui64>
// CPU-NEXT:    %5 = enzymexla.jit_call @enzyme_probprog_add_weight_to_trace_wrapper_0 (%4, %3) {output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>]} : (tensor<ui64>, tensor<f64>) -> tensor<ui64>
// CPU-NEXT:    %6 = enzymexla.jit_call @enzyme_probprog_add_retval_to_trace_wrapper_0 (%5, %1#0, %1#1) {output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>]} : (tensor<ui64>, tensor<f64>, tensor<2xf64>) -> tensor<ui64>
// CPU-NEXT:    return %6, %3, %arg1, %1#0, %1#1 : tensor<ui64>, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<2xf64>
// CPU-NEXT:  }
