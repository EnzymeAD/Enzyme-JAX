// RUN: enzymexlamlir-opt %s --propagate-constant-bounds --split-input-file | FileCheck %s

llvm.func @foo(%arg0: i32) -> i32 {
  llvm.return %arg0 : i32
}

// CHECK-LABEL: ptx_kernelcc
llvm.func ptx_kernelcc @bar() {
    // CHECK: nvvm.read.ptx.sreg.tid.x range <i32, 0, 1> : i32
    %0 = nvvm.read.ptx.sreg.tid.x : i32
    // CHECK-NEXT: nvvm.read.ptx.sreg.ctaid.x range <i32, 0, 2> : i32
    %1 = nvvm.read.ptx.sreg.ctaid.x : i32
    // CHECK-NEXT: %[[CST:.+]] = llvm.mlir.constant(1 : i32) : i32
    %2 = nvvm.read.ptx.sreg.ntid.x : i32
    // CHECK: %{{.+}} = llvm.call @foo(%[[CST]]) : (i32) -> i32
    llvm.call @foo(%2) : (i32) -> i32
    llvm.return
}

func.func @main() {
  %c_1 = stablehlo.constant dense<1> : tensor<i64>
  %c_2 = stablehlo.constant dense<2> : tensor<i64>
  %c_4 = stablehlo.constant dense<4> : tensor<i64>
  %c_6 = stablehlo.constant dense<6> : tensor<i64>
  enzymexla.kernel_call @bar blocks in(%c_2, %c_4, %c_4) threads in(%c_1, %c_4, %c_4) shmem = %c_6 () {} : () -> ()
  return
}

// -----

llvm.func @foo(%arg0: i32) -> i32 {
  llvm.return %arg0 : i32
}

// CHECK-LABEL: ptx_kernelcc
llvm.func ptx_kernelcc @bar() {
    // CHECK: nvvm.read.ptx.sreg.tid.x range <i32, 0, 4> : i32
    %0 = nvvm.read.ptx.sreg.tid.x : i32
    // CHECK-NEXT: nvvm.read.ptx.sreg.ctaid.x range <i32, 0, 4> : i32
    %1 = nvvm.read.ptx.sreg.ctaid.x : i32
    // CHECK: %[[DIM:.+]] = nvvm.read.ptx.sreg.ntid.x range <i32, 0, 4> : i32
    %2 = nvvm.read.ptx.sreg.ntid.x : i32
    // CHECK: %{{.+}} = llvm.call @foo(%[[DIM]]) : (i32) -> i32
    llvm.call @foo(%2) : (i32) -> i32
    llvm.return
}

func.func @main() {
  %c_1 = stablehlo.constant dense<1> : tensor<i64>
  %c_2 = stablehlo.constant dense<2> : tensor<i64>
  %c_4 = stablehlo.constant dense<4> : tensor<i64>
  %c_6 = stablehlo.constant dense<6> : tensor<i64>
  enzymexla.kernel_call @bar blocks in(%c_4, %c_4, %c_4) threads in(%c_4, %c_4, %c_4) shmem = %c_6 () {} : () -> ()
  enzymexla.kernel_call @bar blocks in(%c_2, %c_4, %c_4) threads in(%c_1, %c_4, %c_4) shmem = %c_6 () {} : () -> ()
  return
}

// -----

// CHECK-LABEL: ptx_kernelcc
llvm.func ptx_kernelcc @bar() {
    // CHECK: nvvm.read.ptx.sreg.tid.x range <i32, 0, 6> : i32
    %0 = nvvm.read.ptx.sreg.tid.x : i32
    // CHECK: nvvm.read.ptx.sreg.tid.y range <i32, 0, 4> : i32
    %1 = nvvm.read.ptx.sreg.tid.y : i32
    // CHECK: nvvm.read.ptx.sreg.tid.z range <i32, 0, 2> : i32
    %2 = nvvm.read.ptx.sreg.tid.z : i32

    // CHECK: nvvm.read.ptx.sreg.ctaid.x range <i32, 0, 2> : i32
    %3 = nvvm.read.ptx.sreg.ctaid.x : i32
    // CHECK: nvvm.read.ptx.sreg.ctaid.y range <i32, 0, 4> : i32
    %4 = nvvm.read.ptx.sreg.ctaid.y : i32
    // CHECK: nvvm.read.ptx.sreg.ctaid.z range <i32, 0, 6> : i32
    %5 = nvvm.read.ptx.sreg.ctaid.z : i32
    
    // CHECK: llvm.mlir.constant(6 : i32) : i32
    %6 = nvvm.read.ptx.sreg.ntid.x : i32
    // CHECK: llvm.mlir.constant(4 : i32) : i32
    %7 = nvvm.read.ptx.sreg.ntid.y : i32
    // CHECK: llvm.mlir.constant(2 : i32) : i32
    %8 = nvvm.read.ptx.sreg.ntid.z : i32

    // CHECK: llvm.mlir.constant(2 : i32) : i32
    %9 = nvvm.read.ptx.sreg.nctaid.x : i32
    // CHECK: llvm.mlir.constant(4 : i32) : i32
    %10 = nvvm.read.ptx.sreg.nctaid.y : i32
    // CHECK: llvm.mlir.constant(6 : i32) : i32
    %11 = nvvm.read.ptx.sreg.nctaid.z : i32

    llvm.return
}

func.func @main() {
  %c_1 = stablehlo.constant dense<1> : tensor<i64>
  %c_2 = stablehlo.constant dense<2> : tensor<i64>
  %c_4 = stablehlo.constant dense<4> : tensor<i64>
  %c_6 = stablehlo.constant dense<6> : tensor<i64>
  enzymexla.kernel_call @bar blocks in(%c_2, %c_4, %c_6) threads in(%c_6, %c_4, %c_2) shmem = %c_6 () {} : () -> ()
  return
}

// -----

// CHECK-LABEL: ptx_kernelcc
llvm.func ptx_kernelcc @bar() {
    // CHECK: nvvm.read.ptx.sreg.tid.x range <i32, 0, 6> : i32
    %0 = nvvm.read.ptx.sreg.tid.x : i32
    // CHECK: nvvm.read.ptx.sreg.tid.y range <i32, 0, 4> : i32
    %1 = nvvm.read.ptx.sreg.tid.y : i32
    // CHECK: nvvm.read.ptx.sreg.tid.z range <i32, 0, 6> : i32
    %2 = nvvm.read.ptx.sreg.tid.z : i32

    // CHECK: nvvm.read.ptx.sreg.ctaid.x range <i32, 0, 6> : i32
    %3 = nvvm.read.ptx.sreg.ctaid.x : i32
    // CHECK: nvvm.read.ptx.sreg.ctaid.y range <i32, 0, 4> : i32
    %4 = nvvm.read.ptx.sreg.ctaid.y : i32
    // CHECK: nvvm.read.ptx.sreg.ctaid.z range <i32, 0, 6> : i32
    %5 = nvvm.read.ptx.sreg.ctaid.z : i32
    
    // CHECK: nvvm.read.ptx.sreg.ntid.x range <i32, 0, 6> : i32
    %6 = nvvm.read.ptx.sreg.ntid.x : i32
    // CHECK: nvvm.read.ptx.sreg.ntid.y range <i32, 0, 4> : i32
    %7 = nvvm.read.ptx.sreg.ntid.y : i32
    // CHECK: nvvm.read.ptx.sreg.ntid.z range <i32, 0, 6> : i32
    %8 = nvvm.read.ptx.sreg.ntid.z : i32

    // CHECK: nvvm.read.ptx.sreg.nctaid.x range <i32, 0, 6> : i32
    %9 = nvvm.read.ptx.sreg.nctaid.x : i32
    // CHECK: nvvm.read.ptx.sreg.nctaid.y range <i32, 0, 4> : i32
    %10 = nvvm.read.ptx.sreg.nctaid.y : i32
    // CHECK: nvvm.read.ptx.sreg.nctaid.z range <i32, 0, 6> : i32
    %11 = nvvm.read.ptx.sreg.nctaid.z : i32

    llvm.return
}

func.func @main() {
  %c_1 = stablehlo.constant dense<1> : tensor<i64>
  %c_2 = stablehlo.constant dense<2> : tensor<i64>
  %c_4 = stablehlo.constant dense<4> : tensor<i64>
  %c_6 = stablehlo.constant dense<6> : tensor<i64>
  enzymexla.kernel_call @bar blocks in(%c_2, %c_4, %c_6) threads in(%c_6, %c_4, %c_2) shmem = %c_6 () {} : () -> ()
  enzymexla.kernel_call @bar blocks in(%c_6, %c_4, %c_2) threads in(%c_2, %c_4, %c_6) shmem = %c_6 () {} : () -> ()
  return
}

// -----

// CHECK-LABEL: ptx_kernelcc
llvm.func ptx_kernelcc @bar() {
    // CHECK: nvvm.read.ptx.sreg.tid.x range <i32, 0, 6> : i32
    %0 = nvvm.read.ptx.sreg.tid.x : i32
    // CHECK: nvvm.read.ptx.sreg.tid.y range <i32, 0, 4> : i32
    %1 = nvvm.read.ptx.sreg.tid.y : i32
    // CHECK: nvvm.read.ptx.sreg.tid.z : i32
    // CHECK-NOT: range <i32, 0, 2> : i32
    %2 = nvvm.read.ptx.sreg.tid.z : i32

    // CHECK: nvvm.read.ptx.sreg.ctaid.x
    // CHECK-NOT: range <i32, 0, 2> : i32
    %3 = nvvm.read.ptx.sreg.ctaid.x : i32
    // CHECK: nvvm.read.ptx.sreg.ctaid.y range <i32, 0, 4> : i32
    %4 = nvvm.read.ptx.sreg.ctaid.y : i32
    // CHECK: nvvm.read.ptx.sreg.ctaid.z range <i32, 0, 6> : i32
    %5 = nvvm.read.ptx.sreg.ctaid.z : i32
    
    // CHECK: llvm.mlir.constant(6 : i32) : i32
    %6 = nvvm.read.ptx.sreg.ntid.x : i32
    // CHECK: llvm.mlir.constant(4 : i32) : i32
    %7 = nvvm.read.ptx.sreg.ntid.y : i32
    // CHECK: nvvm.read.ptx.sreg.ntid.z : i32
    %8 = nvvm.read.ptx.sreg.ntid.z : i32

    // CHECK: nvvm.read.ptx.sreg.nctaid.x : i32
    %9 = nvvm.read.ptx.sreg.nctaid.x : i32
    // CHECK: llvm.mlir.constant(4 : i32) : i32
    %10 = nvvm.read.ptx.sreg.nctaid.y : i32
    // CHECK: llvm.mlir.constant(6 : i32) : i32
    %11 = nvvm.read.ptx.sreg.nctaid.z : i32

    llvm.return
}

func.func @main(%c_2 : tensor<i64>) {
  %c_1 = stablehlo.constant dense<1> : tensor<i64>
  %c_4 = stablehlo.constant dense<4> : tensor<i64>
  %c_6 = stablehlo.constant dense<6> : tensor<i64>
  enzymexla.kernel_call @bar blocks in(%c_2, %c_4, %c_6) threads in(%c_6, %c_4, %c_2) shmem = %c_6 () {} : () -> ()
  return
}

// -----

// CHECK-LABEL: ptx_kernelcc
llvm.func ptx_kernelcc @bar() {
  // CHECK: nvvm.read.ptx.sreg.tid.x range <i32, 0, 6> : i32
  %0 = nvvm.read.ptx.sreg.tid.x range <i32, 1, 2000> : i32
  llvm.return
}

func.func @main() {
  %c_1 = stablehlo.constant dense<1> : tensor<i64>
  %c_2 = stablehlo.constant dense<2> : tensor<i64>
  %c_4 = stablehlo.constant dense<4> : tensor<i64>
  %c_6 = stablehlo.constant dense<6> : tensor<i64>
  enzymexla.kernel_call @bar blocks in(%c_2, %c_4, %c_6) threads in(%c_6, %c_4, %c_2) shmem = %c_6 () {} : () -> ()
  return
}

// -----

module {
  llvm.mlir.global private unnamed_addr constant @mlir.llvm.nameless_global_0("ERROR: Out of dynamic GPU memory (trying to allocate %d bytes)\0A\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local, sym_visibility = "private"}
  llvm.mlir.global private unnamed_addr constant @exception16("exception\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local, sym_visibility = "private"}
  llvm.func internal unnamed_addr fastcc @julia_throw_boundserror_12403() attributes {dso_local, no_inline, sym_visibility = "private"} {
    llvm.unreachable
  }
  llvm.func local_unnamed_addr @gpu_malloc(%arg0: i64 {llvm.zeroext}) -> i64 attributes {sym_visibility = "private"} {
    %0 = llvm.call @malloc(%arg0) : (i64) -> !llvm.ptr
    %1 = llvm.ptrtoint %0 : !llvm.ptr to i64
    llvm.return %1 : i64
  }
  llvm.func local_unnamed_addr @jl_bool_type() -> !llvm.ptr attributes {sym_visibility = "private"}
  llvm.func local_unnamed_addr @malloc(i64) -> !llvm.ptr attributes {sym_visibility = "private"}
  llvm.func local_unnamed_addr @vprintf(!llvm.ptr, !llvm.ptr) -> i32 attributes {sym_visibility = "private"}
  llvm.func local_unnamed_addr @jl_int32_type() -> !llvm.ptr attributes {sym_visibility = "private"}
  llvm.func local_unnamed_addr @jl_uint8_type() -> !llvm.ptr attributes {sym_visibility = "private"}
  llvm.func local_unnamed_addr @jl_uint32_type() -> !llvm.ptr attributes {sym_visibility = "private"}
  llvm.func local_unnamed_addr @jl_int8_type() -> !llvm.ptr attributes {sym_visibility = "private"}
  llvm.func local_unnamed_addr @gpu_report_oom(%arg0: i64 {llvm.zeroext}) attributes {sym_visibility = "private"} {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.mlir.addressof @mlir.llvm.nameless_global_0 : !llvm.ptr
    %2 = llvm.alloca %0 x !llvm.struct<"printf_args.0", (i64)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
    llvm.intr.lifetime.start 8, %2 : !llvm.ptr
    %3 = llvm.getelementptr inbounds %2[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"printf_args.0", (i64)>
    llvm.store %arg0, %3 {alignment = 8 : i64} : i64, !llvm.ptr
    %4 = llvm.call @vprintf(%1, %2) : (!llvm.ptr, !llvm.ptr) -> i32
    llvm.intr.lifetime.end 8, %2 : !llvm.ptr
    llvm.return
  }
  llvm.func local_unnamed_addr @jl_float64_type() -> !llvm.ptr attributes {sym_visibility = "private"}
  llvm.func local_unnamed_addr @jl_int64_type() -> !llvm.ptr attributes {sym_visibility = "private"}
  llvm.func local_unnamed_addr @jl_float32_type() -> !llvm.ptr attributes {sym_visibility = "private"}
  llvm.func local_unnamed_addr @jl_uint64_type() -> !llvm.ptr attributes {sym_visibility = "private"}
  llvm.func local_unnamed_addr @jl_uint16_type() -> !llvm.ptr attributes {sym_visibility = "private"}
  llvm.func local_unnamed_addr @jl_int16_type() -> !llvm.ptr attributes {sym_visibility = "private"}
  llvm.func ptx_kernelcc @"##call__Z14square_kernel_13CuTracedArrayI5Int64Li1ELi1E5_64__ES1_#241"(%arg0: !llvm.ptr<1> {llvm.noalias}, %arg1: !llvm.ptr<1> {llvm.noalias}) attributes {sym_visibility = "private"} {
    %0 = llvm.mlir.constant(63 : i32) : i32
    %1 = llvm.mlir.constant(dense<[160, 176, 104, 238, 255, 127, 0, 0]> : tensor<8xui8>) : !llvm.array<8 x i8>
    %2 = llvm.mlir.constant(dense<[112, 176, 104, 238, 255, 127, 0, 0]> : tensor<8xui8>) : !llvm.array<8 x i8>
    %3 = llvm.mlir.constant(1 : i64) : i64
    %4 = llvm.alloca %3 x !llvm.array<1 x ptr<1>> : (i64) -> !llvm.ptr
    llvm.store %2, %4 : !llvm.array<8 x i8>, !llvm.ptr
    %5 = llvm.alloca %3 x !llvm.array<1 x ptr<1>> : (i64) -> !llvm.ptr
    llvm.store %1, %5 : !llvm.array<8 x i8>, !llvm.ptr
    llvm.store %arg0, %4 : !llvm.ptr<1>, !llvm.ptr
    llvm.store %arg1, %5 : !llvm.ptr<1>, !llvm.ptr
    %6 = llvm.load %4 : !llvm.ptr -> !llvm.array<1 x ptr<1>>
    %7 = llvm.load %5 : !llvm.ptr -> !llvm.array<1 x ptr<1>>
    %8 = nvvm.read.ptx.sreg.tid.x : i32
    %9 = llvm.icmp "ugt" %8, %0 : i32
    llvm.cond_br %9, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    llvm.call fastcc @julia_throw_boundserror_12403() : () -> ()
    llvm.unreachable
  ^bb2:  // pred: ^bb0
    %10 = llvm.extractvalue %7[0] : !llvm.array<1 x ptr<1>> 
    %11 = llvm.zext %8 : i32 to i64
    %12 = llvm.getelementptr inbounds %10[%11] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i64
    %13 = llvm.load %12 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "custom_tbaa_addrspace(1)", members = {<#llvm.tbaa_root<id = "custom_tbaa">, 0>}>, access_type = <id = "custom_tbaa_addrspace(1)", members = {<#llvm.tbaa_root<id = "custom_tbaa">, 0>}>, offset = 0>]} : !llvm.ptr<1> -> i64
    %14 = llvm.extractvalue %6[0] : !llvm.array<1 x ptr<1>> 
    %15 = llvm.getelementptr inbounds %14[%11] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i64
    %16 = llvm.load %15 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "custom_tbaa_addrspace(1)", members = {<#llvm.tbaa_root<id = "custom_tbaa">, 0>}>, access_type = <id = "custom_tbaa_addrspace(1)", members = {<#llvm.tbaa_root<id = "custom_tbaa">, 0>}>, offset = 0>]} : !llvm.ptr<1> -> i64
    %17 = llvm.mul %13, %16 : i64
    llvm.store %17, %15 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "custom_tbaa_addrspace(1)", members = {<#llvm.tbaa_root<id = "custom_tbaa">, 0>}>, access_type = <id = "custom_tbaa_addrspace(1)", members = {<#llvm.tbaa_root<id = "custom_tbaa">, 0>}>, offset = 0>]} : i64, !llvm.ptr<1>
    llvm.br ^bb3
  ^bb3:  // pred: ^bb2
    llvm.return
  }
  func.func @main(%arg0: tensor<64xi64>, %arg1: tensor<64xi64>) -> (tensor<64xi64>, tensor<64xi64>) {
    %c = stablehlo.constant dense<0> : tensor<i64>
    %c_0 = stablehlo.constant dense<64> : tensor<i64>
    %c_1 = stablehlo.constant dense<1> : tensor<i64>
    %0 = stablehlo.transpose %arg0, dims = [0] : (tensor<64xi64>) -> tensor<64xi64>
    %1 = stablehlo.transpose %arg1, dims = [0] : (tensor<64xi64>) -> tensor<64xi64>
    %2 = stablehlo.transpose %0, dims = [0] : (tensor<64xi64>) -> tensor<64xi64>
    %3 = stablehlo.transpose %1, dims = [0] : (tensor<64xi64>) -> tensor<64xi64>
    %4:2 = enzymexla.kernel_call @"##call__Z14square_kernel_13CuTracedArrayI5Int64Li1ELi1E5_64__ES1_#241" blocks in(%c_1, %c_1, %c_1) threads in(%c_0, %c_1, %c_1) shmem = %c (%2, %3) {output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [1], operand_index = 1, operand_tuple_indices = []>]} : (tensor<64xi64>, tensor<64xi64>) -> (tensor<64xi64>, tensor<64xi64>)
    %5 = stablehlo.transpose %4#0, dims = [0] : (tensor<64xi64>) -> tensor<64xi64>
    %6 = stablehlo.transpose %4#1, dims = [0] : (tensor<64xi64>) -> tensor<64xi64>
    %7 = stablehlo.transpose %5, dims = [0] : (tensor<64xi64>) -> tensor<64xi64>
    %8 = stablehlo.transpose %6, dims = [0] : (tensor<64xi64>) -> tensor<64xi64>
    return %7, %8 : tensor<64xi64>, tensor<64xi64>
  }
}
