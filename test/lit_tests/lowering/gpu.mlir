// RUN: enzymexlamlir-opt %s --lower-kernel | FileCheck %s

module {
  llvm.func internal unnamed_addr fastcc @julia_throw_boundserror_2676() attributes {dso_local, no_inline, sym_visibility = "private"} {
    llvm.unreachable
  }
  llvm.func local_unnamed_addr ptx_kernelcc @kern(%arg0: !llvm.ptr<1>) {
    %0 = llvm.mlir.constant(63 : i32) : i32
    %1 = nvvm.read.ptx.sreg.tid.x : i32
    %2 = llvm.icmp "ugt" %1, %0 : i32
    llvm.cond_br %2, ^bb2, ^bb1
  ^bb1:  // pred: ^bb0
    %4 = llvm.zext %1 : i32 to i64
    %5 = llvm.getelementptr inbounds %arg0[%4] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i64
    %6 = llvm.load %5 {alignment = 1 : i64} : !llvm.ptr<1> -> i64
    %7 = llvm.mul %6, %6 : i64
    llvm.store %7, %5 {alignment = 1 : i64} : i64, !llvm.ptr<1>
    llvm.return
  ^bb2:  // pred: ^bb0
    llvm.call fastcc @julia_throw_boundserror_2676() : () -> ()
    llvm.unreachable
  }
  func.func @main(%arg0: tensor<64xi64>) -> tensor<64xi64> {
    %c1 = arith.constant 1 : index
    %c40 = arith.constant 40 : index
    %0 = enzymexla.kernel_call @kern (%c1, %c1, %c1) (%c1, %c1, %c40) (%arg0) {output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>]} : (tensor<64xi64>) -> tensor<64xi64>
    return %0 : tensor<64xi64>
  }
}

