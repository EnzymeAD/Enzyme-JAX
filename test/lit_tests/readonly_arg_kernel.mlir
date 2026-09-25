// RUN: enzymexlamlir-opt %s  --pass-pipeline="builtin.module(canonicalize{cse-between-iterations=false    max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true})" | FileCheck %s

// CHECK: %{{.+}} = enzymexla.kernel_call @"##call__Z10gpu_scale_16CompilerMetadataI11DynamicSize12DynamicCheckv16CartesianIndicesILi2E5TupleI5OneToI5Int64ES6_EE7NDRangeILi2ES0_S0_S8_S8_EE13CuTracedArrayI7Float64Li2ELi1E7_16__3_ESE_SD_#300" blocks in(%c_1, %c_1, %c_1) threads in(%c, %c_1, %c_1) shmem = %c_0 (%{{.+}}, %{{.+}}) {operand_layouts = [dense<[1, 0]> : tensor<2xindex>, dense<[1, 0]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[1, 0]> : tensor<2xindex>], xla_side_effect_free} : (tensor<3x16xf64>, tensor<3x16xf64>) -> tensor<3x16xf64>

#tbaa_root = #llvm.tbaa_root<id = "custom_tbaa">
#tbaa_type_desc = #llvm.tbaa_type_desc<id = "custom_tbaa_addrspace(1)", members = {<#tbaa_root, 0>}>
#tbaa_tag = #llvm.tbaa_tag<base_type = #tbaa_type_desc, access_type = #tbaa_type_desc, offset = 0>
module @reactant_f attributes {llvm.data_layout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64", mhlo.num_partitions = 1 : i64, mhlo.num_replicas = 1 : i64} {
  llvm.module_flags [#llvm.mlir.module_flag<warning, "Dwarf Version", 2 : i32>, #llvm.mlir.module_flag<warning, "Debug Info Version", 3 : i32>, #llvm.mlir.module_flag<override, "nvvm-reflect-ftz", 0 : i32>]
  func.func private @identity_broadcast_scalar(%arg0: tensor<f64> {enzymexla.memory_effects = []}) -> tensor<f64> attributes {enzymexla.memory_effects = []} {
    return %arg0 : tensor<f64>
  }
  func.func @main(%arg0: tensor<3x16xf64> {enzymexla.memory_effects = ["read", "write", "allocate", "free"], tf.aliasing_output = 1 : i32}, %arg1: tensor<3x16xf64> {enzymexla.memory_effects = ["read", "write", "allocate", "free"], tf.aliasing_output = 2 : i32}, %arg2: tensor<i64> {enzymexla.memory_effects = ["read", "write", "allocate", "free"]}) -> (tensor<f64>, tensor<3x16xf64>, tensor<3x16xf64>) attributes {enzymexla.memory_effects = ["read", "write", "allocate", "free"]} {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %c = stablehlo.constant dense<48> : tensor<i64>
    %c_0 = stablehlo.constant dense<0> : tensor<i64>
    %c_1 = stablehlo.constant dense<1> : tensor<i64>
    %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<3x16xf64>) -> tensor<16x3xf64>
    %1 = stablehlo.transpose %arg1, dims = [1, 0] : (tensor<3x16xf64>) -> tensor<16x3xf64>
    %2 = stablehlo.subtract %arg2, %c_1 : tensor<i64>
    %3 = stablehlo.divide %2, %c_1 : tensor<i64>
    %4 = stablehlo.add %3, %c_1 : tensor<i64>
    %5:6 = stablehlo.while(%iterArg = %c_0, %iterArg_2 = %4, %iterArg_3 = %0, %iterArg_4 = %c_1, %iterArg_5 = %1, %iterArg_6 = %c_1) : tensor<i64>, tensor<i64>, tensor<16x3xf64>, tensor<i64>, tensor<16x3xf64>, tensor<i64> attributes {enzyme.disable_mincut}
    cond {
      %10 = stablehlo.compare LT, %iterArg, %iterArg_2 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %10 : tensor<i1>
    } do {
      %10 = stablehlo.add %iterArg, %c_1 : tensor<i64>
      %11 = stablehlo.transpose %iterArg_5, dims = [1, 0] : (tensor<16x3xf64>) -> tensor<3x16xf64>
      %12 = stablehlo.transpose %iterArg_3, dims = [1, 0] : (tensor<16x3xf64>) -> tensor<3x16xf64>
      %13:2 = enzymexla.kernel_call @"##call__Z10gpu_scale_16CompilerMetadataI11DynamicSize12DynamicCheckv16CartesianIndicesILi2E5TupleI5OneToI5Int64ES6_EE7NDRangeILi2ES0_S0_S8_S8_EE13CuTracedArrayI7Float64Li2ELi1E7_16__3_ESE_SD_#300" blocks in(%c_1, %c_1, %c_1) threads in(%c, %c_1, %c_1) shmem = %c_0 (%11, %12) {operand_layouts = [dense<[1, 0]> : tensor<2xindex>, dense<[1, 0]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [1], operand_index = 1, operand_tuple_indices = []>], result_layouts = [dense<[1, 0]> : tensor<2xindex>, dense<[1, 0]> : tensor<2xindex>], xla_side_effect_free} : (tensor<3x16xf64>, tensor<3x16xf64>) -> (tensor<3x16xf64>, tensor<3x16xf64>)
      %14 = stablehlo.transpose %13#0, dims = [1, 0] : (tensor<3x16xf64>) -> tensor<16x3xf64>
      %15 = stablehlo.transpose %13#1, dims = [1, 0] : (tensor<3x16xf64>) -> tensor<16x3xf64>
      stablehlo.return %10, %iterArg_2, %15, %iterArg_4, %14, %iterArg_6 : tensor<i64>, tensor<i64>, tensor<16x3xf64>, tensor<i64>, tensor<16x3xf64>, tensor<i64>
    }
    %6 = enzyme.batch @identity_broadcast_scalar(%5#4) {batch_shape = array<i64: 16, 3>} : (tensor<16x3xf64>) -> tensor<16x3xf64>
    %7 = stablehlo.reduce(%6 init: %cst) applies stablehlo.add across dimensions = [0, 1] : (tensor<16x3xf64>, tensor<f64>) -> tensor<f64>
    %8 = stablehlo.transpose %5#2, dims = [1, 0] : (tensor<16x3xf64>) -> tensor<3x16xf64>
    %9 = stablehlo.transpose %6, dims = [1, 0] : (tensor<16x3xf64>) -> tensor<3x16xf64>
    return %7, %8, %9 : tensor<f64>, tensor<3x16xf64>, tensor<3x16xf64>
  }
  llvm.mlir.global private unnamed_addr constant @mlir.llvm.nameless_global_0("ERROR: Out of dynamic GPU memory (trying to allocate %d bytes)\0A\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local, sym_visibility = "private"}
  llvm.mlir.global private unnamed_addr constant @exception110("exception\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local, sym_visibility = "private"}
  llvm.func local_unnamed_addr @gpu_malloc(%arg0: i64 {llvm.noundef, llvm.zeroext}) -> (!llvm.ptr {llvm.noalias, llvm.noundef}) attributes {enzymexla.memory_effects = ["read", "write", "allocate", "free"], memory_effects = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = readwrite, errnoMem = none, targetMem0 = none, targetMem1 = none>, no_unwind, passthrough = ["mustprogress", "nofree"], sym_visibility = "private", will_return} {
    %0 = llvm.call @malloc(%arg0) {memory_effects = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = readwrite, errnoMem = none, targetMem0 = none, targetMem1 = none>, will_return} : (i64 {llvm.noundef}) -> (!llvm.ptr {llvm.noalias, llvm.noundef})
    llvm.return %0 : !llvm.ptr
  }
  llvm.func local_unnamed_addr @jl_bool_type() -> !llvm.ptr attributes {sym_visibility = "private"}
  llvm.func local_unnamed_addr @malloc(i64 {llvm.noundef}) -> (!llvm.ptr {llvm.noalias, llvm.noundef}) attributes {allocsize = array<i32: 0>, memory_effects = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = readwrite, errnoMem = none, targetMem0 = none, targetMem1 = none>, no_unwind, passthrough = ["mustprogress", "nofree", ["allockind", "9"], ["alloc-family", "malloc"]], sym_visibility = "private", will_return}
  llvm.func local_unnamed_addr @vprintf(!llvm.ptr, !llvm.ptr) -> i32 attributes {sym_visibility = "private"}
  llvm.func local_unnamed_addr @jl_int32_type() -> !llvm.ptr attributes {sym_visibility = "private"}
  llvm.func local_unnamed_addr @jl_uint8_type() -> !llvm.ptr attributes {sym_visibility = "private"}
  llvm.func local_unnamed_addr @jl_uint32_type() -> !llvm.ptr attributes {sym_visibility = "private"}
  llvm.func local_unnamed_addr @jl_int8_type() -> !llvm.ptr attributes {sym_visibility = "private"}
  llvm.func local_unnamed_addr @gpu_report_oom(%arg0: i64 {llvm.zeroext}) attributes {enzymexla.memory_effects = ["read", "write", "allocate", "free"], sym_visibility = "private"} {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.mlir.addressof @mlir.llvm.nameless_global_0 : !llvm.ptr
    %2 = llvm.alloca %0 x !llvm.struct<"printf_args.5.1", (i64)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
    llvm.intr.lifetime.start %2 : !llvm.ptr
    llvm.store %arg0, %2 {alignment = 8 : i64} : i64, !llvm.ptr
    %3 = llvm.call @vprintf(%1, %2) : (!llvm.ptr {llvm.nonnull, llvm.noundef}, !llvm.ptr {llvm.align = 8 : i64, llvm.nonnull, llvm.noundef}) -> i32
    llvm.intr.lifetime.end %2 : !llvm.ptr
    llvm.return
  }
  llvm.func local_unnamed_addr @jl_float64_type() -> !llvm.ptr attributes {sym_visibility = "private"}
  llvm.func local_unnamed_addr @jl_int64_type() -> !llvm.ptr attributes {sym_visibility = "private"}
  llvm.func local_unnamed_addr @jl_float32_type() -> !llvm.ptr attributes {sym_visibility = "private"}
  llvm.func local_unnamed_addr @jl_uint64_type() -> !llvm.ptr attributes {sym_visibility = "private"}
  llvm.func local_unnamed_addr @jl_uint16_type() -> !llvm.ptr attributes {sym_visibility = "private"}
  llvm.func local_unnamed_addr @jl_int16_type() -> !llvm.ptr attributes {sym_visibility = "private"}
  llvm.func ptx_kernelcc @"##call__Z10gpu_scale_16CompilerMetadataI11DynamicSize12DynamicCheckv16CartesianIndicesILi2E5TupleI5OneToI5Int64ES6_EE7NDRangeILi2ES0_S0_S8_S8_EE13CuTracedArrayI7Float64Li2ELi1E7_16__3_ESE_SD_#300"(%arg0: !llvm.ptr<1> {llvm.align = 128 : i64, llvm.dereferenceable = 384 : i64, llvm.dereferenceable_or_null = 384 : i64, llvm.noalias, llvm.nocapture, llvm.nofree, llvm.noundef, llvm.writeonly}, %arg1: !llvm.ptr<1> {llvm.align = 128 : i64, llvm.dereferenceable = 384 : i64, llvm.dereferenceable_or_null = 384 : i64, llvm.noalias, llvm.nocapture, llvm.nofree, llvm.noundef, llvm.readonly}) attributes {enzymexla.memory_effects = ["read", "write", "allocate", "free"], memory_effects = #llvm.memory_effects<other = none, argMem = readwrite, inaccessibleMem = none, errnoMem = none, targetMem0 = none, targetMem1 = none>, no_unwind, passthrough = ["mustprogress", "nofree", "norecurse", "nosync"], sym_visibility = "private", will_return} {
    %0 = llvm.mlir.constant(16 : i64) : i64
    %1 = llvm.mlir.constant(1 : i64) : i64
    %2 = llvm.mlir.constant(3 : i64) : i64
    %3 = llvm.mlir.constant(0 : i64) : i64
    %4 = llvm.mlir.constant(9223372036854775807 : i64) : i64
    %5 = llvm.mlir.constant(4 : i64) : i64
    %6 = llvm.mlir.constant(-1 : i64) : i64
    %7 = llvm.mlir.constant(47 : i64) : i64
    %8 = llvm.mlir.constant(2.000000e+00 : f64) : f64
    %9 = nvvm.read.ptx.sreg.tid.x range <i32, 0, 48> : i32
    %10 = llvm.zext nneg %9 : i32 to i64
    %11 = nvvm.read.ptx.sreg.ctaid.x range <i32, 0, 1> : i32
    %12 = llvm.zext nneg %11 : i32 to i64
    %13 = llvm.udiv %10, %0 : i64
    %14 = llvm.add %12, %13 : i64
    %15 = llvm.sub %12, %14 : i64
    %16 = llvm.mul %15, %0 : i64
    %17 = llvm.add %10, %1 overflow<nsw, nuw> : i64
    %18 = llvm.add %17, %16 : i64
    %19 = llvm.mul %12, %2 : i64
    %20 = llvm.add %19, %13 : i64
    %21 = llvm.add %20, %1 : i64
    %22 = llvm.icmp "sgt" %18, %3 : i64
    %23 = llvm.icmp "sle" %18, %0 : i64
    %24 = llvm.and %22, %23 : i1
    %25 = llvm.icmp "ult" %20, %4 : i64
    %26 = llvm.icmp "sle" %21, %2 : i64
    %27 = llvm.and %25, %26 : i1
    %28 = llvm.and %27, %24 : i1
    llvm.cond_br %28, ^bb4, ^bb3
  ^bb1:  // pred: ^bb4
    llvm.unreachable
  ^bb2:  // pred: ^bb4
    %29 = llvm.getelementptr inbounds %arg1[%35] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f64
    %30 = llvm.load %29 invariant {alignment = 8 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr<1> -> f64
    %31 = llvm.fmul %30, %8 : f64
    %32 = llvm.getelementptr inbounds %arg0[%35] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f64
    llvm.store %31, %32 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : f64, !llvm.ptr<1>
    llvm.br ^bb3
  ^bb3:  // 2 preds: ^bb0, ^bb2
    llvm.br ^bb5
  ^bb4:  // pred: ^bb0
    %33 = llvm.shl %20, %5 : i64
    %34 = llvm.add %33, %6 : i64
    %35 = llvm.add %34, %18 : i64
    %36 = llvm.icmp "ugt" %35, %7 : i64
    llvm.cond_br %36, ^bb1, ^bb2
  ^bb5:  // pred: ^bb3
    llvm.return
  }
}
