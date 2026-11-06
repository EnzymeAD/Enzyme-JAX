// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(convert-polygeist-to-llvm{preserve-xla-funcs=true},lower-jit{parallel=true})" | FileCheck %s
// RUN: enzymexlamlir-opt %s --mlir-disable-threading --pass-pipeline="builtin.module(convert-polygeist-to-llvm{preserve-xla-funcs=true},lower-jit{parallel=true})" | FileCheck %s

// Several jit_calls compiled concurrently: @scale and @shift are compiled
// once each, the second call to @scale reuses the entry point of the first.
// @shift also has a static constructor, run once its module is jitted.

module {
  llvm.mlir.global internal @offset(0.000000e+00 : f64) : f64
  llvm.func @init_offset() {
    %cst = llvm.mlir.constant(1.000000e+00 : f64) : f64
    %0 = llvm.mlir.addressof @offset : !llvm.ptr
    llvm.store %cst, %0 : f64, !llvm.ptr
    llvm.return
  }
  llvm.mlir.global_ctors ctors = [@init_offset], priorities = [0 : i32], data = [#llvm.zero]
  func.func private @scale(%arg0: !llvm.ptr<1>) {
    %cst = llvm.mlir.constant(5.000000e-01 : f64) : f64
    %0 = llvm.load %arg0 <alignment = 8> : !llvm.ptr<1> -> f64
    %1 = llvm.fmul %cst, %0 : f64
    llvm.store %1, %arg0 <alignment = 8> : f64, !llvm.ptr<1>
    return
  }
  func.func private @shift(%arg0: !llvm.ptr<1>) {
    %0 = llvm.mlir.addressof @offset : !llvm.ptr
    %1 = llvm.load %0 : !llvm.ptr -> f64
    %2 = llvm.load %arg0 <alignment = 8> : !llvm.ptr<1> -> f64
    %3 = llvm.fadd %1, %2 : f64
    llvm.store %3, %arg0 <alignment = 8> : f64, !llvm.ptr<1>
    return
  }
  func.func @main(%arg0: tensor<64xf64>) -> tensor<64xf64> {
    %0 = enzymexla.jit_call @scale (%arg0) <{output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>]}> : (tensor<64xf64>) -> tensor<64xf64>
    %1 = enzymexla.jit_call @shift (%0) <{output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>]}> : (tensor<64xf64>) -> tensor<64xf64>
    %2 = enzymexla.jit_call @scale (%1) <{output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>]}> : (tensor<64xf64>) -> tensor<64xf64>
    return %2 : tensor<64xf64>
  }
}

// CHECK-LABEL: @main
// CHECK: stablehlo.custom_call @enzymexla_compile_cpu({{.*}}) {{.*}}backend_config = {attr = [[SCALE:"[^"]*"]]}
// CHECK: stablehlo.custom_call @enzymexla_compile_cpu({{.*}}) {{.*}}backend_config = {attr = [[SHIFT:"[^"]*"]]}
// CHECK: stablehlo.custom_call @enzymexla_compile_cpu({{.*}}) {{.*}}backend_config = {attr = [[SCALE]]}
// CHECK-NOT: llvm.mlir.global_ctors
