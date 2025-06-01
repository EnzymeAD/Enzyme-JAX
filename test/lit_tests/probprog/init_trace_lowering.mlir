// RUN: enzymexlamlir-opt %s --lower-enzyme-probprog=backend=cpu | FileCheck %s --check-prefix=CPU

module {
  func.func @main() -> tensor<1xui64> {
    %0 = enzyme.initTrace : !enzyme.Trace
    %1 = builtin.unrealized_conversion_cast %0 : !enzyme.Trace to tensor<1xui64>
    return %1 : tensor<1xui64>
  }
}

// CPU:  llvm.func @enzyme_probprog_init_trace() -> !llvm.ptr
// CPU-NEXT:  llvm.func @enzyme_probprog_init_trace_wrapper_0() -> !llvm.ptr
// CPU-NEXT:    %0 = llvm.call @enzyme_probprog_init_trace() : () -> !llvm.ptr
// CPU-NEXT:    llvm.return %0 : !llvm.ptr
// CPU-NEXT:  }
// CPU:  func.func @main() -> tensor<1xui64> {
// CPU-NEXT:    %0 = enzymexla.jit_call @enzyme_probprog_init_trace_wrapper_0 () {operand_layouts = [], result_layouts = []} : () -> tensor<1xui64> 
// CPU-NEXT:    return %0 : tensor<1xui64>
// CPU-NEXT:  }
