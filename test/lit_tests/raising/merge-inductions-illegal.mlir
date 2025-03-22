// RUN: enzymexlamlir-opt --affine-cfg --split-input-file --allow-unregistered-dialect %s | FileCheck %s
//
// Check that we fail to run MergeParallelInductions because it is illegal
// CHECK: affine.parallel ({{.*}}) = (0, 0) to (16, 16) {
module {
  func.func @main(%arg0: tensor<f64>, %arg1: tensor<i64>, %arg2: tensor<1x28x28xf64>, %arg3: tensor<1x28x28xf64>, %arg4: tensor<1x29x28xf64>, %arg5: tensor<1x28x28xf64>, %arg6: tensor<1x28x28xf64>, %arg7: tensor<1x29x28xf64>, %arg8: tensor<28x28x28xf64>, %arg9: tensor<28x29x28xf64>, %arg10: tensor<29x28x28xf64>, %arg11: tensor<28x28x28xf64>, %arg12: tensor<28x28x28xf64>, %arg13: tensor<28x29x28xf64>, %arg14: tensor<1x28x28xf64>, %arg15: tensor<1x29x28xf64>, %arg16: tensor<28x28x28xf64>, %arg17: tensor<28x29x28xf64>, %arg18: tensor<1x28x28xf64>, %arg19: tensor<1x29x28xf64>) -> (tensor<f64>, tensor<i64>, tensor<1x28x28xf64>, tensor<1x28x28xf64>, tensor<1x29x28xf64>, tensor<1x28x28xf64>, tensor<1x28x28xf64>, tensor<1x29x28xf64>, tensor<28x28x28xf64>, tensor<28x29x28xf64>, tensor<29x28x28xf64>, tensor<28x28x28xf64>, tensor<28x28x28xf64>, tensor<28x29x28xf64>, tensor<1x28x28xf64>, tensor<1x29x28xf64>, tensor<28x28x28xf64>, tensor<28x29x28xf64>, tensor<1x28x28xf64>, tensor<1x29x28xf64>) {
    %cst = stablehlo.constant dense<[0.000000e+00, 114719.79651841168, 112980.53433304677, 111187.49953234453, 109341.54550375968, 107443.55082143328, 105494.41882803904, 103495.07720484119, 101446.47753016841, 99349.594826514032, 97205.427096478234, 95014.994847772773, 92779.340607514576, 90499.528426039483, 88176.643370471589, 85811.791008290311, 83406.096881140082, 80960.705969133473, 78476.782145902675, 75955.507624658683, 73398.082395521851, 70805.723654391521, 68179.665223626827, 65521.156964813948, 62831.464183899807, 60111.867028974761, 57363.659880991727, 54588.150737710748]> : tensor<28xf64>
    %0:5 = enzymexla.jit_call @"##call__Z40gpu__split_explicit_barotropic_velocity_16CompilerMetadataI16OffsetStaticSizeI12_1_16__1_16_E12DynamicCheckvv7NDRangeILi2E10StaticSizeI6_1__1_ES4_I8_16__16_E5TupleI5Int64S8_E13KernelOffsetsIS9_EEE7Float6421LatitudeLongitudeGridISE_8Periodic7BoundedSH_28StaticVerticalDiscretizationI11OffsetArrayISE_Li1E18TracedStepRangeLenISE_14TwicePrecisionISE_ESM_S8_EESO_SE_SE_ESE_SE_SO_SO_SE_SE_SO_SO_SJ_ISE_Li1E13CuTracedArrayISE_Li1ELi1E5_28__EESS_SS_SS_SE_SE_vESE_SJ_ISE_Li3ESQ_ISE_Li3ELi1E11_28__28__1_EE5FieldI4Face6CentervvvvSV_SE_vvvESW_ISY_SX_vvvvSJ_ISE_Li3ESQ_ISE_Li3ELi1E11_28__29__1_EESE_vvvESV_SZ_S12_SZ_S12_SE_21ForwardBackwardScheme#237$par0" (%cst, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg14, %arg15) {output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 2, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [1], operand_index = 3, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [2], operand_index = 4, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [3], operand_index = 5, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [4], operand_index = 6, operand_tuple_indices = []>]} : (tensor<28xf64>, tensor<1x28x28xf64>, tensor<1x28x28xf64>, tensor<1x29x28xf64>, tensor<1x28x28xf64>, tensor<1x28x28xf64>, tensor<1x29x28xf64>, tensor<1x28x28xf64>, tensor<1x29x28xf64>) -> (tensor<1x28x28xf64>, tensor<1x29x28xf64>, tensor<1x28x28xf64>, tensor<1x28x28xf64>, tensor<1x29x28xf64>)
    return %arg0, %arg1, %arg2, %0#0, %0#1, %0#2, %0#3, %0#4, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15, %arg16, %arg17, %arg18, %arg19 : tensor<f64>, tensor<i64>, tensor<1x28x28xf64>, tensor<1x28x28xf64>, tensor<1x29x28xf64>, tensor<1x28x28xf64>, tensor<1x28x28xf64>, tensor<1x29x28xf64>, tensor<28x28x28xf64>, tensor<28x29x28xf64>, tensor<29x28x28xf64>, tensor<28x28x28xf64>, tensor<28x28x28xf64>, tensor<28x29x28xf64>, tensor<1x28x28xf64>, tensor<1x29x28xf64>, tensor<28x28x28xf64>, tensor<28x29x28xf64>, tensor<1x28x28xf64>, tensor<1x29x28xf64>
  }
  llvm.mlir.global private unnamed_addr constant @mlir.llvm.nameless_global_0("ERROR: Out of dynamic GPU memory (trying to allocate %d bytes)\0A\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local, sym_visibility = "private"}
  llvm.mlir.global private unnamed_addr constant @exception19("exception\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local, sym_visibility = "private"}
  llvm.func local_unnamed_addr @jl_bool_type() -> !llvm.ptr attributes {sym_visibility = "private"}
  llvm.func local_unnamed_addr @malloc(i64) -> !llvm.ptr attributes {sym_visibility = "private"}
  llvm.func local_unnamed_addr @vprintf(!llvm.ptr, !llvm.ptr) -> i32 attributes {sym_visibility = "private"}
  llvm.func local_unnamed_addr @jl_int32_type() -> !llvm.ptr attributes {sym_visibility = "private"}
  llvm.func local_unnamed_addr @jl_uint8_type() -> !llvm.ptr attributes {sym_visibility = "private"}
  llvm.func local_unnamed_addr @jl_uint32_type() -> !llvm.ptr attributes {sym_visibility = "private"}
  llvm.func local_unnamed_addr @jl_int8_type() -> !llvm.ptr attributes {sym_visibility = "private"}
  llvm.func local_unnamed_addr @jl_float64_type() -> !llvm.ptr attributes {sym_visibility = "private"}
  llvm.func local_unnamed_addr @jl_int64_type() -> !llvm.ptr attributes {sym_visibility = "private"}
  llvm.func local_unnamed_addr @jl_float32_type() -> !llvm.ptr attributes {sym_visibility = "private"}
  llvm.func local_unnamed_addr @jl_uint64_type() -> !llvm.ptr attributes {sym_visibility = "private"}
  llvm.func local_unnamed_addr @jl_uint16_type() -> !llvm.ptr attributes {sym_visibility = "private"}
  llvm.func local_unnamed_addr @jl_int16_type() -> !llvm.ptr attributes {sym_visibility = "private"}
  func.func private @"##call__Z40gpu__split_explicit_barotropic_velocity_16CompilerMetadataI16OffsetStaticSizeI12_1_16__1_16_E12DynamicCheckvv7NDRangeILi2E10StaticSizeI6_1__1_ES4_I8_16__16_E5TupleI5Int64S8_E13KernelOffsetsIS9_EEE7Float6421LatitudeLongitudeGridISE_8Periodic7BoundedSH_28StaticVerticalDiscretizationI11OffsetArrayISE_Li1E18TracedStepRangeLenISE_14TwicePrecisionISE_ESM_S8_EESO_SE_SE_ESE_SE_SO_SO_SE_SE_SO_SO_SJ_ISE_Li1E13CuTracedArrayISE_Li1ELi1E5_28__EESS_SS_SS_SE_SE_vESE_SJ_ISE_Li3ESQ_ISE_Li3ELi1E11_28__28__1_EE5FieldI4Face6CentervvvvSV_SE_vvvESW_ISY_SX_vvvvSJ_ISE_Li3ESQ_ISE_Li3ELi1E11_28__29__1_EESE_vvvESV_SZ_S12_SZ_S12_SE_21ForwardBackwardScheme#237$par0"(%arg0: memref<28xf64, 1>, %arg1: memref<1x28x28xf64, 1>, %arg2: memref<1x28x28xf64, 1>, %arg3: memref<1x29x28xf64, 1>, %arg4: memref<1x28x28xf64, 1>, %arg5: memref<1x28x28xf64, 1>, %arg6: memref<1x29x28xf64, 1>, %arg7: memref<1x28x28xf64, 1>, %arg8: memref<1x29x28xf64, 1>) {
    %c5 = arith.constant 5 : index
    %c21 = arith.constant 21 : index
    %c28 = arith.constant 28 : index
    %c6 = arith.constant 6 : index
    %c-16 = arith.constant -16 : index
    %c16 = arith.constant 16 : index
    %c1 = arith.constant 1 : index
    %c1_i64 = arith.constant 1 : i64
    %c5_i64 = arith.constant 5 : i64
    %c4_i64 = arith.constant 4 : i64
    %cst = arith.constant -9806.6499999999996 : f64
    %cst_0 = arith.constant 1.200000e+01 : f64
    %cst_1 = arith.constant 0.000000e+00 : f64
    %cst_2 = arith.constant 138993.65830569842 : f64
    %cst_3 = arith.constant -0.0098792100961226231 : f64
    %0 = "enzymexla.memref2pointer"(%arg0) : (memref<28xf64, 1>) -> !llvm.ptr<1>
    %1 = "enzymexla.memref2pointer"(%arg1) : (memref<1x28x28xf64, 1>) -> !llvm.ptr<1>
    %2 = "enzymexla.memref2pointer"(%arg2) : (memref<1x28x28xf64, 1>) -> !llvm.ptr<1>
    %3 = "enzymexla.memref2pointer"(%arg3) : (memref<1x29x28xf64, 1>) -> !llvm.ptr<1>
    %4 = "enzymexla.memref2pointer"(%arg4) : (memref<1x28x28xf64, 1>) -> !llvm.ptr<1>
    %5 = "enzymexla.memref2pointer"(%arg5) : (memref<1x28x28xf64, 1>) -> !llvm.ptr<1>
    %6 = "enzymexla.memref2pointer"(%arg6) : (memref<1x29x28xf64, 1>) -> !llvm.ptr<1>
    %7 = "enzymexla.memref2pointer"(%arg7) : (memref<1x28x28xf64, 1>) -> !llvm.ptr<1>
    %8 = "enzymexla.memref2pointer"(%arg8) : (memref<1x29x28xf64, 1>) -> !llvm.ptr<1>
    affine.parallel (%arg9, %arg10) = (0, 0) to (16, 16) {
      %9 = arith.muli %arg9, %c16 overflow<nuw> : index
      %10 = arith.addi %9, %arg10 : index
      %11 = arith.addi %10, %c1 : index
      %12 = arith.muli %arg9, %c-16 : index
      %13 = arith.addi %11, %12 : index
      %14 = arith.index_cast %13 : index to i64
      %15 = arith.addi %arg9, %c1 : index
      %16 = arith.index_castui %15 : index to i64
      %17 = arith.addi %arg9, %c6 : index
      %18 = arith.index_castui %17 : index to i64
      %19 = arith.muli %17, %c28 : index
      %20 = arith.index_castui %19 : index to i64
      %21 = arith.addi %14, %c5_i64 : i64
      %22 = arith.addi %21, %20 : i64
      %23 = llvm.getelementptr inbounds %2[%22] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f64
      %24 = llvm.load %23 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "custom_tbaa_addrspace(1)", members = {<#llvm.tbaa_root<id = "custom_tbaa">, 0>}>, access_type = <id = "custom_tbaa_addrspace(1)", members = {<#llvm.tbaa_root<id = "custom_tbaa">, 0>}>, offset = 0>]} : !llvm.ptr<1> -> f64
      %25 = arith.addi %19, %c6 : index
      %26 = arith.index_castui %25 : index to i64
      %27 = llvm.getelementptr inbounds %1[%26] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f64
      %28 = llvm.load %27 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "custom_tbaa_addrspace(1)", members = {<#llvm.tbaa_root<id = "custom_tbaa">, 0>}>, access_type = <id = "custom_tbaa_addrspace(1)", members = {<#llvm.tbaa_root<id = "custom_tbaa">, 0>}>, offset = 0>]} : !llvm.ptr<1> -> f64
      %29 = arith.addi %19, %c21 : index
      %30 = arith.index_castui %29 : index to i64
      %31 = llvm.getelementptr inbounds %1[%30] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f64
      %32 = llvm.load %31 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "custom_tbaa_addrspace(1)", members = {<#llvm.tbaa_root<id = "custom_tbaa">, 0>}>, access_type = <id = "custom_tbaa_addrspace(1)", members = {<#llvm.tbaa_root<id = "custom_tbaa">, 0>}>, offset = 0>]} : !llvm.ptr<1> -> f64
      %33 = arith.subf %28, %32 {fastmathFlags = #llvm.fastmath<none>} : f64
      %34 = llvm.getelementptr inbounds %1[%22] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f64
      %35 = llvm.load %34 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "custom_tbaa_addrspace(1)", members = {<#llvm.tbaa_root<id = "custom_tbaa">, 0>}>, access_type = <id = "custom_tbaa_addrspace(1)", members = {<#llvm.tbaa_root<id = "custom_tbaa">, 0>}>, offset = 0>]} : !llvm.ptr<1> -> f64
      %36 = arith.addi %14, %c4_i64 : i64
      %37 = arith.addi %36, %20 : i64
      %38 = llvm.getelementptr inbounds %1[%37] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f64
      %39 = llvm.load %38 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "custom_tbaa_addrspace(1)", members = {<#llvm.tbaa_root<id = "custom_tbaa">, 0>}>, access_type = <id = "custom_tbaa_addrspace(1)", members = {<#llvm.tbaa_root<id = "custom_tbaa">, 0>}>, offset = 0>]} : !llvm.ptr<1> -> f64
      %40 = arith.subf %35, %39 {fastmathFlags = #llvm.fastmath<none>} : f64
      %41 = arith.cmpi ne, %14, %c1_i64 : i64
      %42 = arith.select %41, %40, %33 {fastmathFlags = #llvm.fastmath<none>} : f64
      %43 = llvm.getelementptr inbounds %0[%18] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f64
      %44 = llvm.load %43 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "custom_tbaa_addrspace(1)", members = {<#llvm.tbaa_root<id = "custom_tbaa">, 0>}>, access_type = <id = "custom_tbaa_addrspace(1)", members = {<#llvm.tbaa_root<id = "custom_tbaa">, 0>}>, offset = 0>]} : !llvm.ptr<1> -> f64
      %45 = arith.divf %42, %44 {fastmathFlags = #llvm.fastmath<none>} : f64
      %46 = arith.mulf %45, %cst {fastmathFlags = #llvm.fastmath<none>} : f64
      %47 = llvm.getelementptr inbounds %7[%22] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f64
      %48 = llvm.load %47 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "custom_tbaa_addrspace(1)", members = {<#llvm.tbaa_root<id = "custom_tbaa">, 0>}>, access_type = <id = "custom_tbaa_addrspace(1)", members = {<#llvm.tbaa_root<id = "custom_tbaa">, 0>}>, offset = 0>]} : !llvm.ptr<1> -> f64
      %49 = arith.addf %48, %46 {fastmathFlags = #llvm.fastmath<none>} : f64
      %50 = arith.mulf %49, %cst_0 {fastmathFlags = #llvm.fastmath<none>} : f64
      %51 = arith.addf %24, %50 {fastmathFlags = #llvm.fastmath<none>} : f64
      llvm.store %51, %23 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "custom_tbaa_addrspace(1)", members = {<#llvm.tbaa_root<id = "custom_tbaa">, 0>}>, access_type = <id = "custom_tbaa_addrspace(1)", members = {<#llvm.tbaa_root<id = "custom_tbaa">, 0>}>, offset = 0>]} : f64, !llvm.ptr<1>
      %52 = llvm.getelementptr inbounds %3[%22] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f64
      %53 = llvm.load %52 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "custom_tbaa_addrspace(1)", members = {<#llvm.tbaa_root<id = "custom_tbaa">, 0>}>, access_type = <id = "custom_tbaa_addrspace(1)", members = {<#llvm.tbaa_root<id = "custom_tbaa">, 0>}>, offset = 0>]} : !llvm.ptr<1> -> f64
      %54 = llvm.load %34 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "custom_tbaa_addrspace(1)", members = {<#llvm.tbaa_root<id = "custom_tbaa">, 0>}>, access_type = <id = "custom_tbaa_addrspace(1)", members = {<#llvm.tbaa_root<id = "custom_tbaa">, 0>}>, offset = 0>]} : !llvm.ptr<1> -> f64
      %55 = arith.addi %arg9, %c5 : index
      %56 = arith.muli %55, %c28 : index
      %57 = arith.index_castui %56 : index to i64
      %58 = arith.addi %21, %57 : i64
      %59 = llvm.getelementptr inbounds %1[%58] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f64
      %60 = llvm.load %59 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "custom_tbaa_addrspace(1)", members = {<#llvm.tbaa_root<id = "custom_tbaa">, 0>}>, access_type = <id = "custom_tbaa_addrspace(1)", members = {<#llvm.tbaa_root<id = "custom_tbaa">, 0>}>, offset = 0>]} : !llvm.ptr<1> -> f64
      %61 = arith.subf %54, %60 {fastmathFlags = #llvm.fastmath<none>} : f64
      %62 = arith.cmpi ne, %16, %c1_i64 : i64
      %63 = arith.select %62, %61, %cst_1 {fastmathFlags = #llvm.fastmath<none>} : f64
      %64 = arith.divf %63, %cst_2 {fastmathFlags = #llvm.fastmath<none>} : f64
      %65 = arith.mulf %64, %cst {fastmathFlags = #llvm.fastmath<none>} : f64
      %66 = llvm.getelementptr inbounds %8[%22] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f64
      %67 = llvm.load %66 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "custom_tbaa_addrspace(1)", members = {<#llvm.tbaa_root<id = "custom_tbaa">, 0>}>, access_type = <id = "custom_tbaa_addrspace(1)", members = {<#llvm.tbaa_root<id = "custom_tbaa">, 0>}>, offset = 0>]} : !llvm.ptr<1> -> f64
      %68 = arith.addf %67, %65 {fastmathFlags = #llvm.fastmath<none>} : f64
      %69 = arith.mulf %68, %cst_0 {fastmathFlags = #llvm.fastmath<none>} : f64
      %70 = arith.addf %53, %69 {fastmathFlags = #llvm.fastmath<none>} : f64
      llvm.store %70, %52 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "custom_tbaa_addrspace(1)", members = {<#llvm.tbaa_root<id = "custom_tbaa">, 0>}>, access_type = <id = "custom_tbaa_addrspace(1)", members = {<#llvm.tbaa_root<id = "custom_tbaa">, 0>}>, offset = 0>]} : f64, !llvm.ptr<1>
      %71 = llvm.getelementptr inbounds %4[%22] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f64
      %72 = llvm.load %71 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "custom_tbaa_addrspace(1)", members = {<#llvm.tbaa_root<id = "custom_tbaa">, 0>}>, access_type = <id = "custom_tbaa_addrspace(1)", members = {<#llvm.tbaa_root<id = "custom_tbaa">, 0>}>, offset = 0>]} : !llvm.ptr<1> -> f64
      %73 = llvm.load %34 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "custom_tbaa_addrspace(1)", members = {<#llvm.tbaa_root<id = "custom_tbaa">, 0>}>, access_type = <id = "custom_tbaa_addrspace(1)", members = {<#llvm.tbaa_root<id = "custom_tbaa">, 0>}>, offset = 0>]} : !llvm.ptr<1> -> f64
      %74 = arith.mulf %73, %cst_3 {fastmathFlags = #llvm.fastmath<none>} : f64
      %75 = arith.addf %72, %74 {fastmathFlags = #llvm.fastmath<none>} : f64
      llvm.store %75, %71 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "custom_tbaa_addrspace(1)", members = {<#llvm.tbaa_root<id = "custom_tbaa">, 0>}>, access_type = <id = "custom_tbaa_addrspace(1)", members = {<#llvm.tbaa_root<id = "custom_tbaa">, 0>}>, offset = 0>]} : f64, !llvm.ptr<1>
      %76 = llvm.getelementptr inbounds %5[%22] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f64
      %77 = llvm.load %76 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "custom_tbaa_addrspace(1)", members = {<#llvm.tbaa_root<id = "custom_tbaa">, 0>}>, access_type = <id = "custom_tbaa_addrspace(1)", members = {<#llvm.tbaa_root<id = "custom_tbaa">, 0>}>, offset = 0>]} : !llvm.ptr<1> -> f64
      %78 = llvm.load %23 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "custom_tbaa_addrspace(1)", members = {<#llvm.tbaa_root<id = "custom_tbaa">, 0>}>, access_type = <id = "custom_tbaa_addrspace(1)", members = {<#llvm.tbaa_root<id = "custom_tbaa">, 0>}>, offset = 0>]} : !llvm.ptr<1> -> f64
      %79 = arith.mulf %78, %cst_3 {fastmathFlags = #llvm.fastmath<none>} : f64
      %80 = arith.addf %77, %79 {fastmathFlags = #llvm.fastmath<none>} : f64
      llvm.store %80, %76 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "custom_tbaa_addrspace(1)", members = {<#llvm.tbaa_root<id = "custom_tbaa">, 0>}>, access_type = <id = "custom_tbaa_addrspace(1)", members = {<#llvm.tbaa_root<id = "custom_tbaa">, 0>}>, offset = 0>]} : f64, !llvm.ptr<1>
      %81 = llvm.getelementptr inbounds %6[%22] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f64
      %82 = llvm.load %81 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "custom_tbaa_addrspace(1)", members = {<#llvm.tbaa_root<id = "custom_tbaa">, 0>}>, access_type = <id = "custom_tbaa_addrspace(1)", members = {<#llvm.tbaa_root<id = "custom_tbaa">, 0>}>, offset = 0>]} : !llvm.ptr<1> -> f64
      %83 = llvm.load %52 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "custom_tbaa_addrspace(1)", members = {<#llvm.tbaa_root<id = "custom_tbaa">, 0>}>, access_type = <id = "custom_tbaa_addrspace(1)", members = {<#llvm.tbaa_root<id = "custom_tbaa">, 0>}>, offset = 0>]} : !llvm.ptr<1> -> f64
      %84 = arith.mulf %83, %cst_3 {fastmathFlags = #llvm.fastmath<none>} : f64
      %85 = arith.addf %82, %84 {fastmathFlags = #llvm.fastmath<none>} : f64
      llvm.store %85, %81 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "custom_tbaa_addrspace(1)", members = {<#llvm.tbaa_root<id = "custom_tbaa">, 0>}>, access_type = <id = "custom_tbaa_addrspace(1)", members = {<#llvm.tbaa_root<id = "custom_tbaa">, 0>}>, offset = 0>]} : f64, !llvm.ptr<1>
    }
    return
  }
}
