// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(lower-kernel,canonicalize,lower-jit{jit=false})" | FileCheck %s
// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(lower-kernel,canonicalize)" | FileCheck %s --check-prefix=JITCALL

module {
  llvm.func internal ptx_kernelcc @kern(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>) {
    %0 = nvvm.read.ptx.sreg.tid.x : i32
    %1 = llvm.zext %0 : i32 to i64
    %2 = llvm.getelementptr inbounds %arg0[%1] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f64
    %3 = llvm.getelementptr inbounds %arg1[%1] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f64
    %4 = llvm.load %2 {alignment = 8 : i64} : !llvm.ptr<1> -> f64
    %5 = llvm.load %3 {alignment = 8 : i64} : !llvm.ptr<1> -> f64
    llvm.store %5, %2 {alignment = 8 : i64} : f64, !llvm.ptr<1>
    llvm.store %4, %3 {alignment = 8 : i64} : f64, !llvm.ptr<1>
    llvm.return
  }
  func.func @main(%arg0: tensor<3x16xf64>, %arg1: tensor<3x16xf64>) -> (tensor<3x16xf64>, tensor<3x16xf64>) {
    %c0 = stablehlo.constant dense<0> : tensor<i64>
    %c1 = stablehlo.constant dense<1> : tensor<i64>
    %c48 = stablehlo.constant dense<48> : tensor<i64>
    %0:2 = enzymexla.kernel_call @kern blocks in(%c1, %c1, %c1) threads in(%c48, %c1, %c1) shmem = %c0 (%arg0, %arg1) {operand_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<[1, 0]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 1, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [1], operand_index = 0, operand_tuple_indices = []>]} : (tensor<3x16xf64>, tensor<3x16xf64>) -> (tensor<3x16xf64>, tensor<3x16xf64>)
    return %0#0, %0#1 : tensor<3x16xf64>, tensor<3x16xf64>
  }
}

// JITCALL: enzymexla.jit_call @kern$call$1 (%arg0, %arg1) {operand_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<[1, 0]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 1, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [1], operand_index = 0, operand_tuple_indices = []>]} : (tensor<3x16xf64>, tensor<3x16xf64>) -> (tensor<3x16xf64>, tensor<3x16xf64>)

// CHECK: func.func @main(%arg0: tensor<3x16xf64>, %arg1: tensor<3x16xf64>) -> (tensor<3x16xf64>, tensor<3x16xf64>) {
// CHECK-NEXT:    %0:2 = stablehlo.custom_call @enzymexla_compile_gpu(%arg0, %arg1) {api_version = 4 : i32, backend_config = {attr = "\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00"}, has_side_effect = true, operand_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<[1, 0]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 1, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [1], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[1, 0]> : tensor<2xindex>, dense<[0, 1]> : tensor<2xindex>]} : (tensor<3x16xf64>, tensor<3x16xf64>) -> (tensor<3x16xf64>, tensor<3x16xf64>)
// CHECK-NEXT:    return %0#0, %0#1 : tensor<3x16xf64>, tensor<3x16xf64>
// CHECK-NEXT:  }
