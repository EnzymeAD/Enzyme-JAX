// RUN: enzymexlamlir-opt --lower-triton %s | FileCheck %s --check-prefix=TRITON
// RUN: enzymexlamlir-opt --lower-triton --lower-kernel %s | FileCheck %s --check-prefix=KERNEL
// RUN: enzymexlamlir-opt --lower-triton --lower-kernel --lower-jit="jit=false" %s | FileCheck %s --check-prefix=JIT

module @reactant_JITFunc... attributes {mhlo.num_partitions = 1 : i64, mhlo.num_replicas = 1 : i64} {
  triton_ext.module @add_kernel_tt_module_e72661bb113efd0f {
    builtin.module @add_kernel_module_e72661bb113efd0f attributes {ttg.global_scratch_memory_alignment = 1 : i32, ttg.global_scratch_memory_size = 0 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 0 : i32, ttg.target = "cuda:120", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
      llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>
      llvm.func @add_kernel_call_e72661bb113efd0f(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>, %arg3: !llvm.ptr<1>) attributes {noinline = false, nvvm.kernel = 1 : ui1, nvvm.reqntid = array<i32: 32>, ttg.global_scratch_memory_alignment = 1 : i32, ttg.global_scratch_memory_size = 0 : i32} {
        %0 = llvm.mlir.undef : vector<1xf32>
        %1 = llvm.mlir.constant(0 : i32) : i32
        %2 = llvm.mlir.constant(32 : i32) : i32
        %3 = llvm.mlir.constant(31 : i32) : i32
        %4 = llvm.mlir.constant(0 : index) : i32
        %5 = llvm.mlir.constant(1024 : i32) : i32
        %6 = llvm.mlir.constant(64 : i32) : i32
        %7 = llvm.call_intrinsic "llvm.nvvm.read.ptx.sreg.ctaid.x"() : () -> i32
        %8 = llvm.mul %7, %6 : i32
        %9 = nvvm.read.ptx.sreg.tid.x : i32
        %10 = llvm.and %9, %3 : i32
        %11 = llvm.shl %10, %1 : i32
        %12 = llvm.or %1, %11 : i32
        %13 = llvm.or %12, %1 : i32
        %14 = llvm.and %13, %3 : i32
        %15 = llvm.lshr %14, %1 : i32
        %16 = llvm.xor %1, %15 : i32
        %17 = llvm.xor %1, %16 : i32
        %18 = llvm.xor %17, %1 : i32
        %19 = llvm.xor %17, %2 : i32
        %20 = llvm.add %18, %4 : i32
        %21 = llvm.add %19, %4 : i32
        %22 = llvm.add %8, %20 : i32
        %23 = llvm.add %8, %21 : i32
        %24 = llvm.icmp "slt" %22, %5 : i32
        %25 = llvm.icmp "slt" %23, %5 : i32
        %26 = llvm.getelementptr %arg0[%22] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
        %27 = llvm.getelementptr %arg0[%23] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
        %28 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "mov.u32 $0, 0x0;\0A\09@$2 ld.global.b32 { $0 }, [ $1 + 0 ];", "=r,l,b" %26, %24 : (!llvm.ptr<1>, i1) -> i32
        %29 = llvm.bitcast %28 : i32 to vector<1xf32>
        %30 = llvm.extractelement %29[%4 : i32] : vector<1xf32>
        %31 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "mov.u32 $0, 0x0;\0A\09@$2 ld.global.b32 { $0 }, [ $1 + 0 ];", "=r,l,b" %27, %25 : (!llvm.ptr<1>, i1) -> i32
        %32 = llvm.bitcast %31 : i32 to vector<1xf32>
        %33 = llvm.extractelement %32[%4 : i32] : vector<1xf32>
        %34 = llvm.getelementptr %arg1[%22] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
        %35 = llvm.getelementptr %arg1[%23] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
        %36 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "mov.u32 $0, 0x0;\0A\09@$2 ld.global.b32 { $0 }, [ $1 + 0 ];", "=r,l,b" %34, %24 : (!llvm.ptr<1>, i1) -> i32
        %37 = llvm.bitcast %36 : i32 to vector<1xf32>
        %38 = llvm.extractelement %37[%4 : i32] : vector<1xf32>
        %39 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "mov.u32 $0, 0x0;\0A\09@$2 ld.global.b32 { $0 }, [ $1 + 0 ];", "=r,l,b" %35, %25 : (!llvm.ptr<1>, i1) -> i32
        %40 = llvm.bitcast %39 : i32 to vector<1xf32>
        %41 = llvm.extractelement %40[%4 : i32] : vector<1xf32>
        %42 = llvm.fadd %30, %38 : f32
        %43 = llvm.fadd %33, %41 : f32
        %44 = llvm.getelementptr %arg2[%22] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
        %45 = llvm.getelementptr %arg2[%23] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
        %46 = llvm.insertelement %42, %0[%1 : i32] : vector<1xf32>
        %47 = llvm.bitcast %46 : vector<1xf32> to i32
        %48 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$2 st.global.b32 [ $1 + 0 ], { $0 };", "r,l,b" %47, %44, %24 : (i32, !llvm.ptr<1>, i1) -> !llvm.void
        %49 = llvm.insertelement %43, %0[%1 : i32] : vector<1xf32>
        %50 = llvm.bitcast %49 : vector<1xf32> to i32
        %51 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$2 st.global.b32 [ $1 + 0 ], { $0 };", "r,l,b" %50, %45, %25 : (i32, !llvm.ptr<1>, i1) -> !llvm.void
        llvm.return
      }
    }
  }
  func.func @main(%arg0: tensor<1024xf32>, %arg1: tensor<1024xf32>, %arg2: tensor<1024xf32>) -> (tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>) {
    %c = stablehlo.constant dense<64> : tensor<i64>
    %c_0 = stablehlo.constant dense<1> : tensor<i64>
    %c_1 = stablehlo.constant dense<16> : tensor<i64>
    %0:3 = triton_ext.call @add_kernel_tt_module_e72661bb113efd0f::@add_kernel_module_e72661bb113efd0f::@add_kernel_call_e72661bb113efd0f clusters in (%c_0, %c_0, %c_0) blocks in(%c_1, %c_0, %c_0) threads in(%c, %c_0, %c_0) (%arg0, %arg1, %arg2) {output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [1], operand_index = 1, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [2], operand_index = 2, operand_tuple_indices = []>]} : (tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>) -> (tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>)
    return %0#0, %0#1, %0#2 : tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>
  }
}

// TRITON:     %c_2 = stablehlo.constant dense<0> : tensor<i64>
// TRITON:     %0:3 = enzymexla.kernel_call @add_kernel_tt_module_e72661bb113efd0f::@add_kernel_module_e72661bb113efd0f::@add_kernel_call_e72661bb113efd0f blocks in(%c_1, %c_0, %c_0) threads in(%c, %c_0, %c_0) shmem = %c_2 (%arg0, %arg1, %arg2, %c_3) {output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [1], operand_index = 1, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [2], operand_index = 2, operand_tuple_indices = []>]} : (tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<i64>) -> (tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>)

// KERNEL:     %c_2 = stablehlo.constant dense<0> : tensor<i64>
// KERNEL:     %0:3 = enzymexla.jit_call @add_kernel_tt_module_e72661bb113efd0f::@add_kernel_module_e72661bb113efd0f::@add_kernel_call_e72661bb113efd0f$call$1 (%arg0, %arg1, %arg2, %c_3) {output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [1], operand_index = 1, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [2], operand_index = 2, operand_tuple_indices = []>]} : (tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<i64>) -> (tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>)

// JIT:        %c_2 = stablehlo.constant dense<0> : tensor<i64>
// JIT:        %0:3 = stablehlo.custom_call @enzymexla_compile_gpu(%arg0, %arg1, %arg2, %c_3) {api_version = 4 : i32, backend_config = {attr = "\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00"}, has_side_effect = true, output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [1], operand_index = 1, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [2], operand_index = 2, operand_tuple_indices = []>]} : (tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<i64>) -> (tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>)
