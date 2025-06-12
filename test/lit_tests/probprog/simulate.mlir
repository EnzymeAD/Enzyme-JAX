// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(lower-enzyme-probprog{backend=cpu})" | FileCheck %s --check-prefix=CPU

module {
  func.func private @model() -> tensor<10000xf64>

  func.func @simulate_model() -> tensor<10000xf64> {
    %1 = func.call @model() : () -> tensor<10000xf64>
    enzyme.addSampleToTrace %1 {name = "model", symbol = 42 : ui64, trace = 43 : ui64} : (tensor<10000xf64>) -> ()
    return %1 : tensor<10000xf64>
  }
}

// CPU:  llvm.func @enzyme_probprog_add_sample_to_trace(!llvm.ptr, !llvm.ptr, !llvm.ptr)
// CPU-NEXT:  llvm.func @enzyme_probprog_add_sample_to_trace_wrapper_0(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr)
// CPU-NEXT:    llvm.call @enzyme_probprog_add_sample_to_trace(%arg0, %arg1, %arg2) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
// CPU-NEXT:    llvm.return
// CPU-NEXT:  }
// CPU-NEXT:  func.func private @model() -> tensor<10000xf64>
// CPU-NEXT:  func.func @simulate_model() -> tensor<10000xf64> {
// CPU-NEXT:    %0 = call @model() : () -> tensor<10000xf64>
// CPU-NEXT:    %c = stablehlo.constant dense<43> : tensor<i64>
// CPU-NEXT:    %c_0 = stablehlo.constant dense<42> : tensor<i64> 
// CPU-NEXT:    enzymexla.jit_call @enzyme_probprog_add_sample_to_trace_wrapper_0 (%c, %c_0, %0) : (tensor<i64>, tensor<i64>, tensor<10000xf64>) -> ()
// CPU-NEXT:    return %0 : tensor<10000xf64>
// CPU-NEXT:  }
