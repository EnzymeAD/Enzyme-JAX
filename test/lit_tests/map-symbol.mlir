// RUN: enzymexlamlir-opt %s --map-symbol="symbols=mock_function=0x1,MOCK_GLOBAL=2" --lower-jit="backend=cpu" | FileCheck %s


module {
  llvm.mlir.global external constant @MOCK_GLOBAL() {addr_space = 0 : i32} : !llvm.ptr
  llvm.func @mock_function(!llvm.ptr, !llvm.ptr) -> i32
  llvm.func @enzymexla_wrapper_mock_function(%arg0: !llvm.ptr {enzymexla.memory_effects = ["read", "write", "allocate", "free"]}) attributes {enzymexla.memory_effects = ["read", "write", "allocate", "free"]} {
    %0 = llvm.mlir.addressof @MOCK_GLOBAL : !llvm.ptr
    %1 = llvm.call @mock_function(%0, %arg0) : (!llvm.ptr, !llvm.ptr) -> i32
    llvm.return
  }
  func.func @main() -> tensor<i32> {
    %c = stablehlo.constant dense<-1> : tensor<i32>
    %0 = enzymexla.jit_call @enzymexla_wrapper_mock_function (%c) {output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>]} : (tensor<i32>) -> tensor<i32>
    return %0 : tensor<i32>
  }
}

// CHECK: %[[v0:.*]] = stablehlo.custom_call @enzymexla_compile_cpu(%[[CST:.*]])
