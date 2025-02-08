// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(lower-jit{jit=false backend=cpu})" | FileCheck %s

module {
  llvm.func internal unnamed_addr fastcc @throw_boundserror_2676() attributes {dso_local, no_inline, sym_visibility = "private"} {
    llvm.unreachable
  }
  func.func private @foo(%arg0: !llvm.ptr<1>) {
    %c0_i32 = arith.constant 0 : i32
    %0 = llvm.mlir.constant(63 : i32) : i32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c40 = arith.constant 40 : index
    scf.parallel (%arg1) = (%c0) to (%c40) step (%c1) {
      scf.execute_region {
        %1 = llvm.icmp "ugt" %c0_i32, %0 : i32
        llvm.cond_br %1, ^bb2, ^bb1
      ^bb1:  // pred: ^bb0
        %2 = llvm.load %arg0 {alignment = 1 : i64} : !llvm.ptr<1> -> i64
        %3 = llvm.mul %2, %2 : i64
        llvm.store %3, %arg0 {alignment = 1 : i64} : i64, !llvm.ptr<1>
        scf.yield
      ^bb2:  // pred: ^bb0
        llvm.call fastcc @throw_boundserror_2676() : () -> ()
        scf.yield
      }
      scf.reduce
    }
    return
  }
  func.func @main(%arg0: tensor<64xi64>) -> tensor<64xi64> {
    %0 = enzymexla.jit_call @foo (%arg0) {output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>]} : (tensor<64xi64>) -> tensor<64xi64>
    return %0 : tensor<64xi64>
  }
}

// CHECK-LABEL: @main
// CHECK-SAME: (%[[ARG0:.+]]: tensor<64xi64>) -> tensor<64xi64> {
// CHECK-NEXT:    %[[CALL:.+]] = stablehlo.custom_call @enzymexla_compile_cpu(%arg0) 
// CHECK-SAME: {api_version = 3 : i32, backend_config = "\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00", 
// CHECK-SAME: output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>]} : (tensor<64xi64>) -> tensor<64xi64>
// CHECK-NEXT:    return %[[CALL]] : tensor<64xi64>
