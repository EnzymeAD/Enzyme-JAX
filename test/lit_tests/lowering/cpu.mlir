// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(lower-kernel{jit=false backend=cpu})" | FileCheck %s

module {
  llvm.func internal unnamed_addr fastcc @throw_boundserror_2676() attributes {dso_local, no_inline, sym_visibility = "private"} {
    llvm.unreachable
  }
  llvm.func internal ptx_kernelcc @kern(%arg0: !llvm.ptr<1>) {
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
    llvm.call fastcc @throw_boundserror_2676() : () -> ()
    llvm.unreachable
  }
  func.func @main(%arg0: tensor<64xi64>) -> tensor<64xi64> {
    %c0 = stablehlo.constant dense<0> : tensor<i64>
    %c1 = stablehlo.constant dense<1> : tensor<i64>
    %c40 = stablehlo.constant dense<40> : tensor<i64>
    %0 = enzymexla.kernel_call @kern blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c40) shmem=%c0 (%arg0) {output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>]} : (tensor<64xi64>) -> tensor<64xi64>
    return %0 : tensor<64xi64>
  }
}

// CHECK: func.func @main(%arg0: tensor<64xi64>) -> tensor<64xi64> {
// CHECK-NEXT: stablehlo.constant
// CHECK-NEXT: stablehlo.constant
// CHECK-NEXT: stablehlo.constant
// CHECK-NEXT:    %0 = stablehlo.custom_call @enzymexla_compile_gpu(%arg0) {api_version = 4 : i32, backend_config = {attr = "\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00"}, output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>]} : (tensor<64xi64>) -> tensor<64xi64>
// CHECK-NEXT:    return %0 : tensor<64xi64>
// CHECK-NEXT:  }
// CHECK-NEXT:}
