// RUN: enzymexlamlir-opt --fuse-jit="dag=true" %s | FileCheck %s

module {
  llvm.func @somefunc1(!llvm.ptr, !llvm.ptr)
  llvm.func @jitwrap1(%arg0: !llvm.ptr, %arg1: !llvm.ptr) {
    llvm.call @somefunc1(%arg0, %arg1) : (!llvm.ptr, !llvm.ptr) -> ()
    llvm.return
  }

  llvm.func @somefunc2(!llvm.ptr, !llvm.ptr, !llvm.ptr)
  llvm.func @jitwrap2(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr) {
    llvm.call @somefunc2(%arg0, %arg1, %arg2) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.return
  }

  func.func @main() -> tensor<4xf64> {
    %0 = stablehlo.constant dense<0.000000e+00> : tensor<4xf64>
    %1:2 = enzymexla.jit_call @jitwrap1 (%0, %0) <{
      output_operand_aliases = [
        #stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>,
        #stablehlo.output_operand_alias<output_tuple_indices = [1], operand_index = 1, operand_tuple_indices = []>
      ]}> : (tensor<4xf64>, tensor<4xf64>) -> (tensor<4xf64>, tensor<4xf64>)
    %2 = enzymexla.jit_call @jitwrap2 (%1#0, %1#1, %0) <{
      output_operand_aliases = [
        #stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 2, operand_tuple_indices = []>
      ]
    }> : (tensor<4xf64>, tensor<4xf64>, tensor<4xf64>) -> tensor<4xf64>
    return %2 : tensor<4xf64>
  }
}

// CHECK: llvm.func @[[FUSED:fused.*]](%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr) {
// CHECK-NEXT:   llvm.call @somefunc1(%arg0, %arg1) : (!llvm.ptr, !llvm.ptr) -> ()
// CHECK-NEXT:   llvm.call @somefunc2(%arg0, %arg1, %arg2) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
// CHECK-NEXT:   llvm.return

// CHECK: func.func @main
// CHECK-NOT: enzymexla.jit_call @jitwrap1
// CHECK-NOT: enzymexla.jit_call @jitwrap2
// CHECK: %[[RES:.*]]:3 = enzymexla.jit_call @[[FUSED]] (%[[ARG0:.*]], %[[ARG1:.*]], %[[ARG2:.*]]) <output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 2, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [1], operand_index = 0, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [2], operand_index = 1, operand_tuple_indices = []>]> : (tensor<4xf64>, tensor<4xf64>, tensor<4xf64>) -> (tensor<4xf64>, tensor<4xf64>, tensor<4xf64>)
// CHECK-NEXT:   return %[[RES]]#0 : tensor<4xf64>
