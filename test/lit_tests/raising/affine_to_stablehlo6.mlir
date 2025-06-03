// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(func.func(canonicalize-loops),affine-cfg,canonicalize,raise-affine-to-stablehlo,canonicalize,arith-raise,canonicalize,enzyme-hlo-opt{max_constant_expansion=0},canonicalize)" | FileCheck %s

module {
    func.func private @"##call__Z18gpu_compute__x_Az_16CompilerMetadataI10StaticSizeI6_185__E12DynamicCheckvv7NDRangeILi1ES0_I5_12__ES0_I5_16__EvvEE21LatitudeLongitudeGridI7Float648Periodic7Bounded4Flat28StaticVerticalDiscretizationIvvS9_S9_EvvS9_S9_11OffsetArrayIS9_Li1E12StepRangeLenIS9_14TwicePrecisionIS9_ESI_5Int64EESL_vESF_IS9_Li1E13CuTracedArrayIS9_Li1ELi1E6_186__EESP_SP_SP_SP_SP_SP_SP_#703$par65"(%arg0: memref<186xf64, 1>, %arg1: memref<186xf64, 1>, %arg2: memref<186xf64, 1>, %arg3: memref<186xf64, 1>, %arg4: memref<186xf64, 1>, %arg5: memref<186xf64, 1>, %arg6: memref<186xf64, 1>, %arg7: memref<186xf64, 1>) {
      %c-94_i64 = arith.constant -94 : i64
      %c1 = arith.constant 1 : index
      %c-92_i64 = arith.constant -92 : i64
      %c-93_i64 = arith.constant -93 : i64
      %c4_i64 = arith.constant 4 : i64
      %cst = arith.constant 0.000000e+00 : f64
      %cst_0 = arith.constant 5.000000e-01 : f64
      %cst_1 = arith.constant 3.1415926535897931 : f64
      %cst_2 = arith.constant 1.800000e+02 : f64
      %cst_3 = arith.constant 6.371000e+06 : f64
      %cst_4 = arith.constant 0.017453292519943295 : f64
      %cst_5 = arith.constant 708422877652.48376 : f64
      affine.parallel (%arg8, %arg9) = (0, 0) to (12, 16) {
        %0 = arith.addi %arg9, %c1 : index
        %1 = arith.index_castui %0 : index to i64
        %2 = arith.index_castui %arg8 : index to i64
        %3 = arith.shli %2, %c4_i64 : i64
        %4 = arith.addi %3, %1 : i64
        affine.if affine_set<(d0, d1) : (d0 * -16 - d1 + 184 >= 0)>(%arg8, %arg9) {
          %5 = arith.addi %4, %c-93_i64 : i64
          %6 = arith.sitofp %5 : i64 to f64
          %7 = arith.mulf %6, %cst {fastmathFlags = #llvm.fastmath<none>} : f64
          %8 = math.absf %6 : f64
          %9 = arith.cmpf uge, %cst_0, %8 {fastmathFlags = #llvm.fastmath<none>} : f64
          %10 = arith.select %9, %cst_0, %6 {fastmathFlags = #llvm.fastmath<none>} : f64
          %11 = arith.select %9, %6, %cst_0 {fastmathFlags = #llvm.fastmath<none>} : f64
          %12 = arith.addf %10, %11 {fastmathFlags = #llvm.fastmath<none>} : f64
          %13 = arith.subf %10, %12 {fastmathFlags = #llvm.fastmath<none>} : f64
          %14 = arith.addf %11, %13 {fastmathFlags = #llvm.fastmath<none>} : f64
          %15 = arith.addf %7, %14 {fastmathFlags = #llvm.fastmath<none>} : f64
          %16 = arith.addf %12, %15 {fastmathFlags = #llvm.fastmath<none>} : f64
          %17 = arith.mulf %16, %cst_1 {fastmathFlags = #llvm.fastmath<none>} : f64
          %18 = arith.divf %17, %cst_2 {fastmathFlags = #llvm.fastmath<none>} : f64
          %19 = math.cos %18 : f64
          %20 = arith.mulf %19, %cst_3 {fastmathFlags = #llvm.fastmath<none>} : f64
          %21 = arith.mulf %20, %cst_4 {fastmathFlags = #llvm.fastmath<none>} : f64
          affine.store %21, %arg0[%arg8 * 16 + %arg9 + 1] {alignment = 8 : i64, ordering = 0 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "custom_tbaa_addrspace(1)", members = {<#llvm.tbaa_root<id = "custom_tbaa">, 0>}>, access_type = <id = "custom_tbaa_addrspace(1)", members = {<#llvm.tbaa_root<id = "custom_tbaa">, 0>}>, offset = 0>]} : memref<186xf64, 1>
          %22 = arith.cmpf uge, %cst, %8 {fastmathFlags = #llvm.fastmath<none>} : f64
          %23 = arith.select %22, %cst, %6 {fastmathFlags = #llvm.fastmath<none>} : f64
          %24 = arith.select %22, %6, %cst {fastmathFlags = #llvm.fastmath<none>} : f64
          %25 = arith.addf %23, %24 {fastmathFlags = #llvm.fastmath<none>} : f64
          %26 = arith.subf %23, %25 {fastmathFlags = #llvm.fastmath<none>} : f64
          %27 = arith.addf %24, %26 {fastmathFlags = #llvm.fastmath<none>} : f64
          %28 = arith.addf %7, %27 {fastmathFlags = #llvm.fastmath<none>} : f64
          %29 = arith.addf %25, %28 {fastmathFlags = #llvm.fastmath<none>} : f64
          %30 = arith.mulf %29, %cst_1 {fastmathFlags = #llvm.fastmath<none>} : f64
          %31 = arith.divf %30, %cst_2 {fastmathFlags = #llvm.fastmath<none>} : f64
          %32 = math.cos %31 : f64
          %33 = arith.mulf %32, %cst_3 {fastmathFlags = #llvm.fastmath<none>} : f64
          %34 = arith.mulf %33, %cst_4 {fastmathFlags = #llvm.fastmath<none>} : f64
          affine.store %34, %arg1[%arg8 * 16 + %arg9 + 1] {alignment = 8 : i64, ordering = 0 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "custom_tbaa_addrspace(1)", members = {<#llvm.tbaa_root<id = "custom_tbaa">, 0>}>, access_type = <id = "custom_tbaa_addrspace(1)", members = {<#llvm.tbaa_root<id = "custom_tbaa">, 0>}>, offset = 0>]} : memref<186xf64, 1>
          affine.store %34, %arg2[%arg8 * 16 + %arg9 + 1] {alignment = 8 : i64, ordering = 0 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "custom_tbaa_addrspace(1)", members = {<#llvm.tbaa_root<id = "custom_tbaa">, 0>}>, access_type = <id = "custom_tbaa_addrspace(1)", members = {<#llvm.tbaa_root<id = "custom_tbaa">, 0>}>, offset = 0>]} : memref<186xf64, 1>
          affine.store %21, %arg3[%arg8 * 16 + %arg9 + 1] {alignment = 8 : i64, ordering = 0 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "custom_tbaa_addrspace(1)", members = {<#llvm.tbaa_root<id = "custom_tbaa">, 0>}>, access_type = <id = "custom_tbaa_addrspace(1)", members = {<#llvm.tbaa_root<id = "custom_tbaa">, 0>}>, offset = 0>]} : memref<186xf64, 1>
          %35 = arith.addi %4, %c-92_i64 : i64
          %36 = arith.sitofp %35 : i64 to f64
          %37 = arith.mulf %36, %cst {fastmathFlags = #llvm.fastmath<none>} : f64
          %38 = math.absf %36 : f64
          %39 = arith.cmpf uge, %cst, %38 {fastmathFlags = #llvm.fastmath<none>} : f64
          %40 = arith.select %39, %cst, %36 {fastmathFlags = #llvm.fastmath<none>} : f64
          %41 = arith.select %39, %36, %cst {fastmathFlags = #llvm.fastmath<none>} : f64
          %42 = arith.addf %40, %41 {fastmathFlags = #llvm.fastmath<none>} : f64
          %43 = arith.subf %40, %42 {fastmathFlags = #llvm.fastmath<none>} : f64
          %44 = arith.addf %41, %43 {fastmathFlags = #llvm.fastmath<none>} : f64
          %45 = arith.addf %37, %44 {fastmathFlags = #llvm.fastmath<none>} : f64
          %46 = arith.addf %42, %45 {fastmathFlags = #llvm.fastmath<none>} : f64
          %47 = arith.mulf %46, %cst_1 {fastmathFlags = #llvm.fastmath<none>} : f64
          %48 = arith.divf %47, %cst_2 {fastmathFlags = #llvm.fastmath<none>} : f64
          %49 = math.sin %48 : f64
          %50 = math.sin %31 : f64
          %51 = arith.subf %49, %50 {fastmathFlags = #llvm.fastmath<none>} : f64
          %52 = arith.mulf %51, %cst_5 {fastmathFlags = #llvm.fastmath<none>} : f64
          affine.store %52, %arg4[%arg8 * 16 + %arg9 + 1] {alignment = 8 : i64, ordering = 0 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "custom_tbaa_addrspace(1)", members = {<#llvm.tbaa_root<id = "custom_tbaa">, 0>}>, access_type = <id = "custom_tbaa_addrspace(1)", members = {<#llvm.tbaa_root<id = "custom_tbaa">, 0>}>, offset = 0>]} : memref<186xf64, 1>
          %53 = math.sin %18 : f64
          %54 = arith.addi %4, %c-94_i64 : i64
          %55 = arith.sitofp %54 : i64 to f64
          %56 = arith.mulf %55, %cst {fastmathFlags = #llvm.fastmath<none>} : f64
          %57 = math.absf %55 : f64
          %58 = arith.cmpf uge, %cst_0, %57 {fastmathFlags = #llvm.fastmath<none>} : f64
          %59 = arith.select %58, %cst_0, %55 {fastmathFlags = #llvm.fastmath<none>} : f64
          %60 = arith.select %58, %55, %cst_0 {fastmathFlags = #llvm.fastmath<none>} : f64
          %61 = arith.addf %59, %60 {fastmathFlags = #llvm.fastmath<none>} : f64
          %62 = arith.subf %59, %61 {fastmathFlags = #llvm.fastmath<none>} : f64
          %63 = arith.addf %60, %62 {fastmathFlags = #llvm.fastmath<none>} : f64
          %64 = arith.addf %56, %63 {fastmathFlags = #llvm.fastmath<none>} : f64
          %65 = arith.addf %61, %64 {fastmathFlags = #llvm.fastmath<none>} : f64
          %66 = arith.mulf %65, %cst_1 {fastmathFlags = #llvm.fastmath<none>} : f64
          %67 = arith.divf %66, %cst_2 {fastmathFlags = #llvm.fastmath<none>} : f64
          %68 = math.sin %67 : f64
          %69 = arith.subf %53, %68 {fastmathFlags = #llvm.fastmath<none>} : f64
          %70 = arith.mulf %69, %cst_5 {fastmathFlags = #llvm.fastmath<none>} : f64
          affine.store %70, %arg5[%arg8 * 16 + %arg9 + 1] {alignment = 8 : i64, ordering = 0 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "custom_tbaa_addrspace(1)", members = {<#llvm.tbaa_root<id = "custom_tbaa">, 0>}>, access_type = <id = "custom_tbaa_addrspace(1)", members = {<#llvm.tbaa_root<id = "custom_tbaa">, 0>}>, offset = 0>]} : memref<186xf64, 1>
          affine.store %70, %arg6[%arg8 * 16 + %arg9 + 1] {alignment = 8 : i64, ordering = 0 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "custom_tbaa_addrspace(1)", members = {<#llvm.tbaa_root<id = "custom_tbaa">, 0>}>, access_type = <id = "custom_tbaa_addrspace(1)", members = {<#llvm.tbaa_root<id = "custom_tbaa">, 0>}>, offset = 0>]} : memref<186xf64, 1>
          affine.store %52, %arg7[%arg8 * 16 + %arg9 + 1] {alignment = 8 : i64, ordering = 0 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "custom_tbaa_addrspace(1)", members = {<#llvm.tbaa_root<id = "custom_tbaa">, 0>}>, access_type = <id = "custom_tbaa_addrspace(1)", members = {<#llvm.tbaa_root<id = "custom_tbaa">, 0>}>, offset = 0>]} : memref<186xf64, 1>
        }
      }
      return
    }
}

// CHECK:  func.func private @"##call__Z18gpu_compute__x_Az_16CompilerMetadataI10StaticSizeI6_185__E12DynamicCheckvv7NDRangeILi1ES0_I5_12__ES0_I5_16__EvvEE21LatitudeLongitudeGridI7Float648Periodic7Bounded4Flat28StaticVerticalDiscretizationIvvS9_S9_EvvS9_S9_11OffsetArrayIS9_Li1E12StepRangeLenIS9_14TwicePrecisionIS9_ESI_5Int64EESL_vESF_IS9_Li1E13CuTracedArrayIS9_Li1ELi1E6_186__EESP_SP_SP_SP_SP_SP_SP_#703$par65_raised"(%arg0: tensor<186xf64>, %arg1: tensor<186xf64>, %arg2: tensor<186xf64>, %arg3: tensor<186xf64>, %arg4: tensor<186xf64>, %arg5: tensor<186xf64>, %arg6: tensor<186xf64>, %arg7: tensor<186xf64>) -> (tensor<186xf64>, tensor<186xf64>, tensor<186xf64>, tensor<186xf64>, tensor<186xf64>, tensor<186xf64>, tensor<186xf64>, tensor<186xf64>) {
// CHECK-NEXT:    %cst = stablehlo.constant dense<111194.92664455874> : tensor<185xf64>
// CHECK-NEXT:    %cst_0 = stablehlo.constant dense<0.017453292519943295> : tensor<185xf64>
// CHECK-NEXT:    %c = stablehlo.constant dense<-93> : tensor<185xi64>
// CHECK-NEXT:    %cst_1 = stablehlo.constant dense<708422877652.48376> : tensor<185xf64>
// CHECK-NEXT:    %c_2 = stablehlo.constant dense<-91> : tensor<185xi64>
// CHECK-NEXT:    %cst_3 = stablehlo.constant dense<5.000000e-01> : tensor<185xf64>
// CHECK-NEXT:    %cst_4 = stablehlo.constant dense<0.000000e+00> : tensor<185xf64>
// CHECK-NEXT:    %c_5 = stablehlo.constant dense<-92> : tensor<185xi64>
// CHECK-NEXT:    %0 = stablehlo.iota dim = 0 : tensor<185xi64>
// CHECK-NEXT:    %1 = stablehlo.add %0, %c_5 : tensor<185xi64>
// CHECK-NEXT:    %2 = stablehlo.convert %1 : (tensor<185xi64>) -> tensor<185xf64>
// CHECK-NEXT:    %3 = stablehlo.multiply %2, %cst_4 : tensor<185xf64>
// CHECK-NEXT:    %4 = stablehlo.abs %2 : tensor<185xf64>
// CHECK-NEXT:    %5 = stablehlo.compare  GE, %cst_3, %4,  FLOAT : (tensor<185xf64>, tensor<185xf64>) -> tensor<185xi1>
// CHECK-NEXT:    %6 = stablehlo.select %5, %cst_3, %2 : tensor<185xi1>, tensor<185xf64>
// CHECK-NEXT:    %7 = stablehlo.select %5, %2, %cst_3 : tensor<185xi1>, tensor<185xf64>
// CHECK-NEXT:    %8 = stablehlo.add %6, %7 : tensor<185xf64>
// CHECK-NEXT:    %9 = stablehlo.subtract %6, %8 : tensor<185xf64>
// CHECK-NEXT:    %10 = stablehlo.add %7, %9 : tensor<185xf64>
// CHECK-NEXT:    %11 = stablehlo.add %3, %10 : tensor<185xf64>
// CHECK-NEXT:    %12 = stablehlo.add %8, %11 : tensor<185xf64>
// CHECK-NEXT:    %13 = stablehlo.multiply %12, %cst_0 : tensor<185xf64>
// CHECK-NEXT:    %14 = stablehlo.cosine %13 : tensor<185xf64>
// CHECK-NEXT:    %15 = stablehlo.multiply %14, %cst : tensor<185xf64>
// CHECK-NEXT:    %16 = stablehlo.slice %arg0 [0:1] : (tensor<186xf64>) -> tensor<1xf64>
// CHECK-NEXT:    %17 = stablehlo.concatenate %16, %15, dim = 0 : (tensor<1xf64>, tensor<185xf64>) -> tensor<186xf64>
// CHECK-NEXT:    %18 = stablehlo.compare  GE, %cst_4, %4,  FLOAT : (tensor<185xf64>, tensor<185xf64>) -> tensor<185xi1>
// CHECK-NEXT:    %19 = stablehlo.select %18, %cst_4, %2 : tensor<185xi1>, tensor<185xf64>
// CHECK-NEXT:    %20 = stablehlo.select %18, %2, %cst_4 : tensor<185xi1>, tensor<185xf64>
// CHECK-NEXT:    %21 = stablehlo.add %19, %20 : tensor<185xf64>
// CHECK-NEXT:    %22 = stablehlo.subtract %19, %21 : tensor<185xf64>
// CHECK-NEXT:    %23 = stablehlo.add %20, %22 : tensor<185xf64>
// CHECK-NEXT:    %24 = stablehlo.add %3, %23 : tensor<185xf64>
// CHECK-NEXT:    %25 = stablehlo.add %21, %24 : tensor<185xf64>
// CHECK-NEXT:    %26 = stablehlo.multiply %25, %cst_0 : tensor<185xf64>
// CHECK-NEXT:    %27 = stablehlo.cosine %26 : tensor<185xf64>
// CHECK-NEXT:    %28 = stablehlo.multiply %27, %cst : tensor<185xf64>
// CHECK-NEXT:    %29 = stablehlo.slice %arg1 [0:1] : (tensor<186xf64>) -> tensor<1xf64>
// CHECK-NEXT:    %30 = stablehlo.concatenate %29, %28, dim = 0 : (tensor<1xf64>, tensor<185xf64>) -> tensor<186xf64>
// CHECK-NEXT:    %31 = stablehlo.slice %arg2 [0:1] : (tensor<186xf64>) -> tensor<1xf64>
// CHECK-NEXT:    %32 = stablehlo.concatenate %31, %28, dim = 0 : (tensor<1xf64>, tensor<185xf64>) -> tensor<186xf64>
// CHECK-NEXT:    %33 = stablehlo.slice %arg3 [0:1] : (tensor<186xf64>) -> tensor<1xf64>
// CHECK-NEXT:    %34 = stablehlo.concatenate %33, %15, dim = 0 : (tensor<1xf64>, tensor<185xf64>) -> tensor<186xf64>
// CHECK-NEXT:    %35 = stablehlo.add %0, %c_2 : tensor<185xi64>
// CHECK-NEXT:    %36 = stablehlo.convert %35 : (tensor<185xi64>) -> tensor<185xf64>
// CHECK-NEXT:    %37 = stablehlo.multiply %36, %cst_4 : tensor<185xf64>
// CHECK-NEXT:    %38 = stablehlo.abs %36 : tensor<185xf64>
// CHECK-NEXT:    %39 = stablehlo.compare  GE, %cst_4, %38,  FLOAT : (tensor<185xf64>, tensor<185xf64>) -> tensor<185xi1>
// CHECK-NEXT:    %40 = stablehlo.select %39, %cst_4, %36 : tensor<185xi1>, tensor<185xf64>
// CHECK-NEXT:    %41 = stablehlo.select %39, %36, %cst_4 : tensor<185xi1>, tensor<185xf64>
// CHECK-NEXT:    %42 = stablehlo.add %40, %41 : tensor<185xf64>
// CHECK-NEXT:    %43 = stablehlo.subtract %40, %42 : tensor<185xf64>
// CHECK-NEXT:    %44 = stablehlo.add %41, %43 : tensor<185xf64>
// CHECK-NEXT:    %45 = stablehlo.add %37, %44 : tensor<185xf64>
// CHECK-NEXT:    %46 = stablehlo.add %42, %45 : tensor<185xf64>
// CHECK-NEXT:    %47 = stablehlo.multiply %46, %cst_0 : tensor<185xf64>
// CHECK-NEXT:    %48 = stablehlo.sine %47 : tensor<185xf64>
// CHECK-NEXT:    %49 = stablehlo.sine %26 : tensor<185xf64>
// CHECK-NEXT:    %50 = stablehlo.subtract %48, %49 : tensor<185xf64>
// CHECK-NEXT:    %51 = stablehlo.multiply %50, %cst_1 : tensor<185xf64>
// CHECK-NEXT:    %52 = stablehlo.slice %arg4 [0:1] : (tensor<186xf64>) -> tensor<1xf64>
// CHECK-NEXT:    %53 = stablehlo.concatenate %52, %51, dim = 0 : (tensor<1xf64>, tensor<185xf64>) -> tensor<186xf64>
// CHECK-NEXT:    %54 = stablehlo.sine %13 : tensor<185xf64>
// CHECK-NEXT:    %55 = stablehlo.add %0, %c : tensor<185xi64>
// CHECK-NEXT:    %56 = stablehlo.convert %55 : (tensor<185xi64>) -> tensor<185xf64>
// CHECK-NEXT:    %57 = stablehlo.multiply %56, %cst_4 : tensor<185xf64>
// CHECK-NEXT:    %58 = stablehlo.abs %56 : tensor<185xf64>
// CHECK-NEXT:    %59 = stablehlo.compare  GE, %cst_3, %58,  FLOAT : (tensor<185xf64>, tensor<185xf64>) -> tensor<185xi1>
// CHECK-NEXT:    %60 = stablehlo.select %59, %cst_3, %56 : tensor<185xi1>, tensor<185xf64>
// CHECK-NEXT:    %61 = stablehlo.select %59, %56, %cst_3 : tensor<185xi1>, tensor<185xf64>
// CHECK-NEXT:    %62 = stablehlo.add %60, %61 : tensor<185xf64>
// CHECK-NEXT:    %63 = stablehlo.subtract %60, %62 : tensor<185xf64>
// CHECK-NEXT:    %64 = stablehlo.add %61, %63 : tensor<185xf64>
// CHECK-NEXT:    %65 = stablehlo.add %57, %64 : tensor<185xf64>
// CHECK-NEXT:    %66 = stablehlo.add %62, %65 : tensor<185xf64>
// CHECK-NEXT:    %67 = stablehlo.multiply %66, %cst_0 : tensor<185xf64>
// CHECK-NEXT:    %68 = stablehlo.sine %67 : tensor<185xf64>
// CHECK-NEXT:    %69 = stablehlo.subtract %54, %68 : tensor<185xf64>
// CHECK-NEXT:    %70 = stablehlo.multiply %69, %cst_1 : tensor<185xf64>
// CHECK-NEXT:    %71 = stablehlo.slice %arg5 [0:1] : (tensor<186xf64>) -> tensor<1xf64>
// CHECK-NEXT:    %72 = stablehlo.concatenate %71, %70, dim = 0 : (tensor<1xf64>, tensor<185xf64>) -> tensor<186xf64>
// CHECK-NEXT:    %73 = stablehlo.slice %arg6 [0:1] : (tensor<186xf64>) -> tensor<1xf64>
// CHECK-NEXT:    %74 = stablehlo.concatenate %73, %70, dim = 0 : (tensor<1xf64>, tensor<185xf64>) -> tensor<186xf64>
// CHECK-NEXT:    %75 = stablehlo.slice %arg7 [0:1] : (tensor<186xf64>) -> tensor<1xf64>
// CHECK-NEXT:    %76 = stablehlo.concatenate %75, %51, dim = 0 : (tensor<1xf64>, tensor<185xf64>) -> tensor<186xf64>
// CHECK-NEXT:    return %17, %30, %32, %34, %53, %72, %74, %76 : tensor<186xf64>, tensor<186xf64>, tensor<186xf64>, tensor<186xf64>, tensor<186xf64>, tensor<186xf64>, tensor<186xf64>, tensor<186xf64>
// CHECK-NEXT:  }
