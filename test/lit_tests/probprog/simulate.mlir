// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(lower-enzyme-probprog{backend=cpu})" | FileCheck %s --check-prefix=CPU

module {
  func.func private @model() -> tensor<3x4xf64>

  func.func @simulate_model() -> tensor<3x4xf64> {
    %1 = func.call @model() : () -> tensor<3x4xf64>
    enzyme.addSampleToTrace %1 {name = "model", symbol = 42 : ui64, trace = 43 : ui64} : (tensor<3x4xf64>) -> ()
    return %1 : tensor<3x4xf64>
  }
}

// CPU:  llvm.func @enzyme_probprog_add_sample_to_trace(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr)
// CPU-NEXT:  llvm.func @enzyme_probprog_add_sample_to_trace_wrapper_0(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr) {
// CPU-NEXT:    %0 = llvm.mlir.constant(1 : i64) : i64
// CPU-NEXT:    %1 = llvm.mlir.constant(2 : i64) : i64
// CPU-NEXT:    %2 = llvm.mlir.constant(64 : i64) : i64
// CPU-NEXT:    %3 = llvm.alloca %0 x i64 : (i64) -> !llvm.ptr
// CPU-NEXT:    %4 = llvm.alloca %0 x i64 : (i64) -> !llvm.ptr
// CPU-NEXT:    llvm.store %1, %3 : i64, !llvm.ptr
// CPU-NEXT:    llvm.store %2, %4 : i64, !llvm.ptr
// CPU-NEXT:    %5 = llvm.mlir.constant(2 : i64) : i64
// CPU-NEXT:    %6 = llvm.alloca %5 x i64 : (i64) -> !llvm.ptr
// CPU-NEXT:    %7 = llvm.mlir.constant(0 : i64) : i64
// CPU-NEXT:    %8 = llvm.mlir.constant(3 : i64) : i64
// CPU-NEXT:    %9 = llvm.getelementptr %6[%7] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CPU-NEXT:    llvm.store %8, %9 : i64, !llvm.ptr
// CPU-NEXT:    %10 = llvm.mlir.constant(1 : i64) : i64
// CPU-NEXT:    %11 = llvm.mlir.constant(4 : i64) : i64
// CPU-NEXT:    %12 = llvm.getelementptr %6[%10] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CPU-NEXT:    llvm.store %11, %12 : i64, !llvm.ptr
// CPU-NEXT:    llvm.call @enzyme_probprog_add_sample_to_trace(%arg0, %arg1, %arg2, %3, %6, %4) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
// CPU-NEXT:    llvm.return
// CPU-NEXT:  }
// CPU-NEXT:  func.func private @model() -> tensor<3x4xf64>
// CPU-NEXT:  func.func @simulate_model() -> tensor<3x4xf64> {
// CPU-NEXT:    %0 = call @model() : () -> tensor<3x4xf64>
// CPU-NEXT:    %c = stablehlo.constant dense<43> : tensor<i64>
// CPU-NEXT:    %c_0 = stablehlo.constant dense<42> : tensor<i64>
// CPU-NEXT:    enzymexla.jit_call @enzyme_probprog_add_sample_to_trace_wrapper_0 (%c, %c_0, %0) : (tensor<i64>, tensor<i64>, tensor<3x4xf64>) -> ()
// CPU-NEXT:    return %0 : tensor<3x4xf64>
// CPU-NEXT:  }
