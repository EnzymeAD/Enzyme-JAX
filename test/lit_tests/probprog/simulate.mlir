// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(lower-enzyme-probprog{backend=cpu})" | FileCheck %s --check-prefix=CPU

module {
  func.func private @model() -> tensor<10000xf64>

  func.func @simulate_model() -> !enzyme.Trace {
    %symbol = llvm.mlir.constant(2 : i64) : i64
    %0 = enzyme.initTrace : !enzyme.Trace
    %1 = func.call @model() : () -> tensor<10000xf64>
    enzyme.addSampleToTrace [%symbol : i64] %0 : !enzyme.Trace, %1 : tensor<10000xf64>
    return %0 : !enzyme.Trace
  }
}

// CPU:  llvm.func @enzyme_probprog_add_sample_to_trace(!llvm.ptr, !llvm.ptr, !llvm.ptr)
// CPU-NEXT:  llvm.func @enzyme_probprog_add_sample_to_trace_wrapper_0(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr)
// CPU-NEXT:    llvm.call @enzyme_probprog_add_sample_to_trace(%arg0, %arg1, %arg2) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
// CPU-NEXT:    llvm.return
// CPU-NEXT:  }
// CPU-NEXT:  llvm.func @enzyme_probprog_init_trace() -> !llvm.ptr
// CPU-NEXT:  llvm.func @enzyme_probprog_init_trace_wrapper_0() -> !llvm.ptr
// CPU-NEXT:    %0 = llvm.call @enzyme_probprog_init_trace() : () -> !llvm.ptr
// CPU-NEXT:    llvm.return %0 : !llvm.ptr
// CPU-NEXT:  }
// CPU-NEXT:  func.func private @model() -> tensor<10000xf64>
// CPU-NEXT:  func.func @simulate_model() -> tensor<1xui64> {
// CPU-NEXT:    %0 = llvm.mlir.constant(2 : i64) : i64
// CPU-NEXT:    %1 = enzymexla.jit_call @enzyme_probprog_init_trace_wrapper_0 () {operand_layouts = [], result_layouts = [dense<0> : tensor<1xindex>]} : () -> tensor<1xui64>
// CPU-NEXT:    %2 = call @model() : () -> tensor<10000xf64>
// CPU-NEXT:    enzymexla.jit_call @enzyme_probprog_add_sample_to_trace_wrapper_0 (%1, %0, %2) {operand_layouts = [dense<0> : tensor<1xindex>, dense<0> : tensor<1xindex>, dense<0> : tensor<1xindex>], result_layouts = []} : (tensor<1xui64>, i64, tensor<10000xf64>) -> ()
// CPU-NEXT:    return %1 : tensor<1xui64>
// CPU-NEXT:  }
