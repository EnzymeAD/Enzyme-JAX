// RUN: enzymexlamlir-opt --mark-func-memory-effects %s | FileCheck %s

module {
    // CHECK: llvm.func @add_kernel(%arg0: !llvm.ptr<1> {enzymexla.memory_effects = ["read", "write"], llvm.nofree}, %arg1: !llvm.ptr<1> {enzymexla.memory_effects = ["read", "write"], llvm.nofree}, %arg2: !llvm.ptr<1> {enzymexla.memory_effects = ["read", "write"], llvm.nofree}, %arg3: !llvm.ptr<1> {enzymexla.memory_effects = [], llvm.nofree}) attributes {enzymexla.memory_effects = ["read", "write", "allocate", "free"], noinline = false, nvvm.kernel = 1 : ui1, nvvm.reqntid = array<i32: 32>, ttg.global_scratch_memory_alignment = 1 : i32, ttg.global_scratch_memory_size = 0 : i32} {
    llvm.func @add_kernel(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>, %arg3: !llvm.ptr<1>) attributes {noinline = false, nvvm.kernel = 1 : ui1, nvvm.reqntid = array<i32: 32>, ttg.global_scratch_memory_alignment = 1 : i32, ttg.global_scratch_memory_size = 0 : i32} {
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

module {
    // CHECK: tt.func @add_kernel_call(%arg0: !tt.ptr<f32> {enzymexla.memory_effects = ["read"], llvm.nofree, llvm.readonly}, %arg1: !tt.ptr<f32> {enzymexla.memory_effects = ["read"], llvm.nofree, llvm.readonly}, %arg2: !tt.ptr<f32> {enzymexla.memory_effects = ["write"], llvm.nofree, llvm.writeonly}) attributes {enzymexla.memory_effects = ["read", "write"], noinline = false}
    tt.func @add_kernel_call(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>) attributes {noinline = false} {
        %cst = arith.constant dense<1024> : tensor<64xi32>
        %c64_i32 = arith.constant 64 : i32
        %0 = tt.get_program_id x : i32
        %1 = arith.muli %0, %c64_i32 : i32
        %2 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
        %3 = tt.splat %1 : i32 -> tensor<64xi32>
        %4 = arith.addi %3, %2 : tensor<64xi32>
        %5 = arith.cmpi slt, %4, %cst : tensor<64xi32>
        %6 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
        %7 = tt.addptr %6, %4 : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
        %8 = tt.load %7, %5 : tensor<64x!tt.ptr<f32>>
        %9 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
        %10 = tt.addptr %9, %4 : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
        %11 = tt.load %10, %5 : tensor<64x!tt.ptr<f32>>
        %12 = arith.addf %8, %11 : tensor<64xf32>
        %13 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
        %14 = tt.addptr %13, %4 : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
        tt.store %14, %12, %5 : tensor<64x!tt.ptr<f32>>
        tt.return
    }
}
