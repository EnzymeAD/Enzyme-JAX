// RUN: enzymexlamlir-opt %s --arith-raise --lower-probprog-to-stablehlo --lower-probprog-trace-ops | FileCheck %s --check-prefix=CPU
module {
  func.func private @model.regenerate(%arg0: !enzyme.Trace, %arg1: tensor<2xui64>) -> (!enzyme.Trace, tensor<f64>, tensor<2xui64>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<f64>
    %0 = enzyme.initTrace : !enzyme.Trace
    %1 = enzyme.getSampleFromTrace %arg0 {symbol = #enzyme.symbol<1>} : tensor<2xf64>
    %2 = enzyme.addSampleToTrace(%1 : tensor<2xf64>) into %0 {symbol = #enzyme.symbol<2>}
    return %2, %cst, %arg1 : !enzyme.Trace, tensor<f64>, tensor<2xui64>
  }

  func.func @mh_program(%arg0: tensor<2xui64>) -> (tensor<ui64>, tensor<2xui64>) {
    %zero = arith.constant dense<0.000000e+00> : tensor<f64>
    %one = arith.constant dense<1.000000e+00> : tensor<f64>
    %c0 = stablehlo.constant dense<0> : tensor<i64>
    %c100 = stablehlo.constant dense<100> : tensor<i64>
    %c1 = stablehlo.constant dense<1> : tensor<i64>
    %init_trace = stablehlo.constant dense<0> : tensor<ui64>

    %0:3 = stablehlo.while(%iterArg = %c0, %iterArg_trace = %init_trace, %iterArg_rng = %arg0) : tensor<i64>, tensor<ui64>, tensor<2xui64> attributes {enzymexla.disable_min_cut}
    cond {
      %cond = stablehlo.compare LT, %iterArg, %c100 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %cond : tensor<i1>
    } do {
      %iter_next = stablehlo.add %iterArg, %c1 : tensor<i64>
      %old_trace = builtin.unrealized_conversion_cast %iterArg_trace : tensor<ui64> to !enzyme.Trace
      %new_trace, %new_weight, %rng1 = func.call @model.regenerate(%old_trace, %iterArg_rng) : (!enzyme.Trace, tensor<2xui64>) -> (!enzyme.Trace, tensor<f64>, tensor<2xui64>)
      %old_weight = enzyme.getWeightFromTrace %old_trace : tensor<f64>
      %log_alpha = arith.subf %new_weight, %old_weight : tensor<f64>
      %rng2, %uniform = enzyme.random %rng1, %zero, %one {rng_distribution = #enzyme<rng_distribution UNIFORM>} : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
      %log_uniform = math.log %uniform : tensor<f64>
      %accept = arith.cmpf olt, %log_uniform, %log_alpha : tensor<f64>
      %selected_trace = enzyme.selectTrace %accept, %new_trace, %old_trace : tensor<i1>
      %selected_trace_ui64 = builtin.unrealized_conversion_cast %selected_trace : !enzyme.Trace to tensor<ui64>
      stablehlo.return %iter_next, %selected_trace_ui64, %rng2 : tensor<i64>, tensor<ui64>, tensor<2xui64>
    }
    return %0#1, %0#2 : tensor<ui64>, tensor<2xui64>
  }
}

// CPU:  llvm.func @enzyme_probprog_get_weight_from_trace(!llvm.ptr, !llvm.ptr)
// CPU:  llvm.func @enzyme_probprog_get_weight_from_trace_wrapper_0(%arg0: !llvm.ptr, %arg1: !llvm.ptr) {
// CPU-NEXT:    llvm.call @enzyme_probprog_get_weight_from_trace(%arg0, %arg1) : (!llvm.ptr, !llvm.ptr) -> ()
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
// CPU-NEXT:    %9 = llvm.mlir.constant(1 : i64) : i64
// CPU-NEXT:    %10 = llvm.mlir.constant(0 : i64) : i64
// CPU-NEXT:    %11 = llvm.getelementptr %4[%10] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CPU-NEXT:    llvm.store %9, %11 : i64, !llvm.ptr
// CPU-NEXT:    %12 = llvm.mlir.constant(64 : i64) : i64
// CPU-NEXT:    %13 = llvm.mlir.constant(0 : i64) : i64
// CPU-NEXT:    %14 = llvm.getelementptr %6[%13] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CPU-NEXT:    llvm.store %12, %14 : i64, !llvm.ptr
// CPU-NEXT:    %15 = llvm.mlir.constant(1 : i64) : i64
// CPU-NEXT:    %16 = llvm.alloca %15 x i64 : (i64) -> !llvm.ptr
// CPU-NEXT:    %17 = llvm.mlir.constant(2 : i64) : i64
// CPU-NEXT:    %18 = llvm.mlir.constant(0 : i64) : i64
// CPU-NEXT:    %19 = llvm.getelementptr %16[%18] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CPU-NEXT:    llvm.store %17, %19 : i64, !llvm.ptr
// CPU-NEXT:    %20 = llvm.mlir.constant(0 : i64) : i64
// CPU-NEXT:    %21 = llvm.getelementptr %5[%20] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CPU-NEXT:    llvm.store %16, %21 : !llvm.ptr, !llvm.ptr
// CPU-NEXT:    llvm.call @enzyme_probprog_add_sample_to_trace(%arg0, %arg1, %3, %2, %4, %5, %6) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
// CPU-NEXT:    llvm.return
// CPU-NEXT:  }

// CPU:  llvm.func @enzyme_probprog_get_sample_from_trace(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr)
// CPU:  llvm.func @enzyme_probprog_get_sample_from_trace_wrapper_0(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr) {
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
// CPU-NEXT:    %9 = llvm.mlir.constant(1 : i64) : i64
// CPU-NEXT:    %10 = llvm.mlir.constant(0 : i64) : i64
// CPU-NEXT:    %11 = llvm.getelementptr %4[%10] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CPU-NEXT:    llvm.store %9, %11 : i64, !llvm.ptr
// CPU-NEXT:    %12 = llvm.mlir.constant(64 : i64) : i64
// CPU-NEXT:    %13 = llvm.mlir.constant(0 : i64) : i64
// CPU-NEXT:    %14 = llvm.getelementptr %6[%13] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CPU-NEXT:    llvm.store %12, %14 : i64, !llvm.ptr
// CPU-NEXT:    %15 = llvm.mlir.constant(1 : i64) : i64
// CPU-NEXT:    %16 = llvm.alloca %15 x i64 : (i64) -> !llvm.ptr
// CPU-NEXT:    %17 = llvm.mlir.constant(2 : i64) : i64
// CPU-NEXT:    %18 = llvm.mlir.constant(0 : i64) : i64
// CPU-NEXT:    %19 = llvm.getelementptr %16[%18] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CPU-NEXT:    llvm.store %17, %19 : i64, !llvm.ptr
// CPU-NEXT:    %20 = llvm.mlir.constant(0 : i64) : i64
// CPU-NEXT:    %21 = llvm.getelementptr %5[%20] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CPU-NEXT:    llvm.store %16, %21 : !llvm.ptr, !llvm.ptr
// CPU-NEXT:    llvm.call @enzyme_probprog_get_sample_from_trace(%arg0, %arg1, %3, %2, %4, %5, %6) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
// CPU-NEXT:    llvm.return
// CPU-NEXT:  }

// CPU:  llvm.func @enzyme_probprog_init_trace(!llvm.ptr)
// CPU:  llvm.func @enzyme_probprog_init_trace_wrapper_0(%arg0: !llvm.ptr) {
// CPU-NEXT:    llvm.call @enzyme_probprog_init_trace(%arg0) : (!llvm.ptr) -> ()
// CPU-NEXT:    llvm.return
// CPU-NEXT:  }

// CPU:  func.func private @model.regenerate(%arg0: tensor<ui64>, %arg1: tensor<2xui64>) -> (tensor<ui64>, tensor<f64>, tensor<2xui64>) {
// CPU-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CPU-NEXT:    %c = stablehlo.constant dense<0> : tensor<ui64>
// CPU-NEXT:    %0 = enzymexla.jit_call @enzyme_probprog_init_trace_wrapper_0 (%c) {operand_layouts = [dense<> : tensor<0xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<> : tensor<0xindex>]} : (tensor<ui64>) -> tensor<ui64>
// CPU-NEXT:    %c_0 = stablehlo.constant dense<1> : tensor<i64>
// CPU-NEXT:    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<2xf64>
// CPU-NEXT:    %1 = enzymexla.jit_call @enzyme_probprog_get_sample_from_trace_wrapper_0 (%arg0, %c_0, %cst_1) {output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 2, operand_tuple_indices = []>]} : (tensor<ui64>, tensor<i64>, tensor<2xf64>) -> tensor<2xf64>
// CPU-NEXT:    %c_2 = stablehlo.constant dense<2> : tensor<i64>
// CPU-NEXT:    %2 = enzymexla.jit_call @enzyme_probprog_add_sample_to_trace_wrapper_0 (%0, %c_2, %1) {output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>]} : (tensor<ui64>, tensor<i64>, tensor<2xf64>) -> tensor<ui64>
// CPU-NEXT:    return %2, %cst, %arg1 : tensor<ui64>, tensor<f64>, tensor<2xui64>
// CPU-NEXT:  }

// CPU:  func.func @mh_program(%arg0: tensor<2xui64>) -> (tensor<ui64>, tensor<2xui64>) {
// CPU-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CPU-NEXT:    %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
// CPU-NEXT:    %c = stablehlo.constant dense<0> : tensor<i64>
// CPU-NEXT:    %c_1 = stablehlo.constant dense<100> : tensor<i64>
// CPU-NEXT:    %c_2 = stablehlo.constant dense<1> : tensor<i64>
// CPU-NEXT:    %c_3 = stablehlo.constant dense<0> : tensor<ui64>
// CPU-NEXT:    %0:3 = stablehlo.while(%iterArg = %c, %iterArg_4 = %c_3, %iterArg_5 = %arg0) : tensor<i64>, tensor<ui64>, tensor<2xui64> attributes {enzymexla.disable_min_cut}
// CPU-NEXT:    cond {
// CPU-NEXT:      %1 = stablehlo.compare  LT, %iterArg, %c_1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CPU-NEXT:      stablehlo.return %1 : tensor<i1>
// CPU-NEXT:    } do {
// CPU-NEXT:      %1 = stablehlo.add %iterArg, %c_2 : tensor<i64>
// CPU-NEXT:      %2:3 = func.call @model.regenerate(%iterArg_4, %iterArg_5) : (tensor<ui64>, tensor<2xui64>) -> (tensor<ui64>, tensor<f64>, tensor<2xui64>)
// CPU-NEXT:      %cst_6 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CPU-NEXT:      %3 = enzymexla.jit_call @enzyme_probprog_get_weight_from_trace_wrapper_0 (%iterArg_4, %cst_6) {output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 1, operand_tuple_indices = []>]} : (tensor<ui64>, tensor<f64>) -> tensor<f64>
// CPU-NEXT:      %4 = stablehlo.subtract %2#1, %3 : tensor<f64>
// CPU-NEXT:      %output_state, %output = stablehlo.rng_bit_generator %2#2, algorithm =  DEFAULT : (tensor<2xui64>) -> (tensor<2xui64>, tensor<ui64>)
// CPU-NEXT:      %c_7 = stablehlo.constant dense<12> : tensor<ui64>
// CPU-NEXT:      %5 = stablehlo.shift_right_logical %output, %c_7 : tensor<ui64>
// CPU-NEXT:      %c_8 = stablehlo.constant dense<4607182418800017408> : tensor<ui64>
// CPU-NEXT:      %6 = stablehlo.or %5, %c_8 : tensor<ui64>
// CPU-NEXT:      %7 = stablehlo.bitcast_convert %6 : (tensor<ui64>) -> tensor<f64>
// CPU-NEXT:      %cst_9 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
// CPU-NEXT:      %8 = stablehlo.subtract %7, %cst_9 : tensor<f64>
// CPU-NEXT:      %9 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<f64>
// CPU-NEXT:      %10 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f64>) -> tensor<f64>
// CPU-NEXT:      %11 = stablehlo.subtract %10, %9 : tensor<f64>
// CPU-NEXT:      %12 = stablehlo.multiply %11, %8 : tensor<f64>
// CPU-NEXT:      %13 = stablehlo.add %9, %12 : tensor<f64>
// CPU-NEXT:      %14 = stablehlo.log %13 : tensor<f64>
// CPU-NEXT:      %15 = stablehlo.compare  LT, %14, %4,  FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
// CPU-NEXT:      %16 = stablehlo.select %15, %2#0, %iterArg_4 : tensor<i1>, tensor<ui64>
// CPU-NEXT:      stablehlo.return %1, %16, %output_state : tensor<i64>, tensor<ui64>, tensor<2xui64>
// CPU-NEXT:    }
// CPU-NEXT:    return %0#1, %0#2 : tensor<ui64>, tensor<2xui64>
// CPU-NEXT:  }
