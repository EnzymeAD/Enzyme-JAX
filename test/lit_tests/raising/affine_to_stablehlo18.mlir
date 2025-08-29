// RUN: enzymexlamlir-opt %s --raise-affine-to-stablehlo=prefer_while_raising=false --canonicalize --arith-raise --enzyme-hlo-opt=max_constant_expansion=0 | FileCheck %s

#map = affine_map<(d0) -> (d0 - 32)>
module {
  llvm.module_flags [#llvm.mlir.module_flag<warning, "Dwarf Version", 2 : i32>, #llvm.mlir.module_flag<warning, "Debug Info Version", 3 : i32>]
  func.func @"bad_compute_w_from_continuity!"(%arg0: tensor<1x200x104xf64>, %arg1: tensor<40x200x104xf64>) -> (tensor<1x200x104xf64>, tensor<40x200x104xf64>) attributes {enzymexla.memory_effects = ["read", "write", "allocate", "free"]} {
    %0 = enzymexla.jit_call @"##call__Z35gpu__bad_compute_w_from_continuity_16CompilerMetadataI16OffsetStaticSizeI15__2_99___2_195_E12DynamicCheckvv7NDRangeILi2E10StaticSizeI7_7__13_ES4_I8_16__16_E5TupleI5Int64S8_ES0_I8__3___3_EEE11OffsetArrayI7Float64Li3E13CuTracedArrayISE_Li3ELi1E14_104__200__40_EE20ImmersedBoundaryGridISE_8Periodic7BoundedSK_15RectilinearGridISE_SJ_SK_SK_28StaticVerticalDiscretizationISD_ISE_Li1E12StepRangeLenISE_14TwicePrecisionISE_ESP_S8_EESR_SE_SE_ESE_SE_SR_SR_vE16GridFittedBottomI5FieldI6CenterSW_vvvvSD_ISE_Li3ESF_ISE_Li3ELi1E13_104__200__1_EESE_vvvE23CenterImmersedConditionEvvvE#246$par0" (%arg1, %arg0) {output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>], xla_side_effect_free} : (tensor<40x200x104xf64>, tensor<1x200x104xf64>) -> tensor<40x200x104xf64>
    return %arg0, %0 : tensor<1x200x104xf64>, tensor<40x200x104xf64>
  }
  func.func private @"##call__Z35gpu__bad_compute_w_from_continuity_16CompilerMetadataI16OffsetStaticSizeI15__2_99___2_195_E12DynamicCheckvv7NDRangeILi2E10StaticSizeI7_7__13_ES4_I8_16__16_E5TupleI5Int64S8_ES0_I8__3___3_EEE11OffsetArrayI7Float64Li3E13CuTracedArrayISE_Li3ELi1E14_104__200__40_EE20ImmersedBoundaryGridISE_8Periodic7BoundedSK_15RectilinearGridISE_SJ_SK_SK_28StaticVerticalDiscretizationISD_ISE_Li1E12StepRangeLenISE_14TwicePrecisionISE_ESP_S8_EESR_SE_SE_ESE_SE_SR_SR_vE16GridFittedBottomI5FieldI6CenterSW_vvvvSD_ISE_Li3ESF_ISE_Li3ELi1E13_104__200__1_EESE_vvvE23CenterImmersedConditionEvvvE#246$par0"(%arg0: memref<40x200x104xf64, 1>, %arg1: memref<1x200x104xf64, 1>) {
    %cst = arith.constant 0.000000e+00 : f64
    %cst_0 = arith.constant 68.252109753297191 : f64
    %cst_1 = arith.constant 1.0763811494540183E-12 : f64
    %cst_2 = arith.constant 34.126054876649278 : f64
    %cst_3 = arith.constant -1.3117712044802247E-13 : f64
    %cst_4 = arith.constant 1.000000e-02 : f64
    affine.parallel (%arg2, %arg3) = (0, 0) to (198, 102) {
      affine.store %cst, %arg0[4, %arg2 + 1, %arg3 + 1] : memref<40x200x104xf64, 1>
      %0 = affine.load %arg1[0, %arg2 + 1, %arg3 + 1] : memref<1x200x104xf64, 1>
      %1 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst) -> (f64) {
        %2 = affine.apply #map(%arg4)
        %3 = arith.index_cast %2 : index to i64
        %4 = arith.sitofp %3 : i64 to f64
        %5 = arith.mulf %4, %cst_0 {fastmathFlags = #llvm.fastmath<none>} : f64
        %6 = arith.mulf %4, %cst_1 {fastmathFlags = #llvm.fastmath<none>} : f64
        %7 = math.absf %5 : f64
        %8 = arith.cmpf olt, %cst_2, %7 {fastmathFlags = #llvm.fastmath<none>} : f64
        %9 = arith.select %8, %5, %cst_2 : f64
        %10 = arith.select %8, %cst_2, %5 : f64
        %11 = arith.addf %9, %10 {fastmathFlags = #llvm.fastmath<none>} : f64
        %12 = arith.subf %9, %11 {fastmathFlags = #llvm.fastmath<none>} : f64
        %13 = arith.addf %10, %12 {fastmathFlags = #llvm.fastmath<none>} : f64
        %14 = arith.addf %6, %cst_3 {fastmathFlags = #llvm.fastmath<none>} : f64
        %15 = arith.addf %14, %13 {fastmathFlags = #llvm.fastmath<none>} : f64
        %16 = arith.addf %11, %15 {fastmathFlags = #llvm.fastmath<none>} : f64
        %17 = arith.cmpf ole, %16, %0 {fastmathFlags = #llvm.fastmath<none>} : f64
        %18 = arith.select %17, %cst, %cst_4 {fastmathFlags = #llvm.fastmath<none>} : f64
        %19 = arith.subf %arg5, %18 {fastmathFlags = #llvm.fastmath<none>} : f64
        affine.store %19, %arg0[%arg4 + 5, %arg2 + 1, %arg3 + 1] : memref<40x200x104xf64, 1>
        affine.yield %19 : f64
      }
    }
    return
  }
}

// CHECK:  func.func private @"##call__Z35gpu__bad_compute_w_from_continuity_16CompilerMetadataI16OffsetStaticSizeI15__2_99___2_195_E12DynamicCheckvv7NDRangeILi2E10StaticSizeI7_7__13_ES4_I8_16__16_E5TupleI5Int64S8_ES0_I8__3___3_EEE11OffsetArrayI7Float64Li3E13CuTracedArrayISE_Li3ELi1E14_104__200__40_EE20ImmersedBoundaryGridISE_8Periodic7BoundedSK_15RectilinearGridISE_SJ_SK_SK_28StaticVerticalDiscretizationISD_ISE_Li1E12StepRangeLenISE_14TwicePrecisionISE_ESP_S8_EESR_SE_SE_ESE_SE_SR_SR_vE16GridFittedBottomI5FieldI6CenterSW_vvvvSD_ISE_Li3ESF_ISE_Li3ELi1E13_104__200__1_EESE_vvvE23CenterImmersedConditionEvvvE#246$par0_raised"(%arg0: tensor<40x200x104xf64>, %arg1: tensor<1x200x104xf64>) -> (tensor<40x200x104xf64>, tensor<1x200x104xf64>) {
// CHECK-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:    %cst_0 = stablehlo.constant dense<68.252109753297191> : tensor<f64>
// CHECK-NEXT:    %cst_1 = stablehlo.constant dense<1.0763811494540183E-12> : tensor<f64>
// CHECK-NEXT:    %cst_2 = stablehlo.constant dense<34.126054876649278> : tensor<f64>
// CHECK-NEXT:    %cst_3 = stablehlo.constant dense<-1.3117712044802247E-13> : tensor<f64>
// CHECK-NEXT:    %cst_4 = stablehlo.constant dense<1.000000e-02> : tensor<f64>
// CHECK-NEXT:    %0 = stablehlo.iota dim = 0 : tensor<198xi64>
// CHECK-NEXT:    %c = stablehlo.constant dense<0> : tensor<198xi64>
// CHECK-NEXT:    %1 = stablehlo.add %0, %c : tensor<198xi64>
// CHECK-NEXT:    %c_5 = stablehlo.constant dense<1> : tensor<198xi64>
// CHECK-NEXT:    %2 = stablehlo.multiply %1, %c_5 : tensor<198xi64>
// CHECK-NEXT:    %3 = stablehlo.iota dim = 0 : tensor<102xi64>
// CHECK-NEXT:    %c_6 = stablehlo.constant dense<0> : tensor<102xi64>
// CHECK-NEXT:    %4 = stablehlo.add %3, %c_6 : tensor<102xi64>
// CHECK-NEXT:    %c_7 = stablehlo.constant dense<1> : tensor<102xi64>
// CHECK-NEXT:    %5 = stablehlo.multiply %4, %c_7 : tensor<102xi64>
// CHECK-NEXT:    %c_8 = stablehlo.constant dense<4> : tensor<i64>
// CHECK-NEXT:    %c_9 = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:    %c_10 = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:    %6 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<1x198x102xf64>
// CHECK-NEXT:    %7 = stablehlo.dynamic_update_slice %arg0, %6, %c_8, %c_9, %c_10 : (tensor<40x200x104xf64>, tensor<1x198x102xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<40x200x104xf64>
// CHECK-NEXT:    %8 = stablehlo.slice %arg1 [0:1, 1:199, 1:103] : (tensor<1x200x104xf64>) -> tensor<1x198x102xf64>
// CHECK-NEXT:    %9 = stablehlo.reshape %8 : (tensor<1x198x102xf64>) -> tensor<198x102xf64>
// CHECK-NEXT:    %10 = stablehlo.iota dim = 0 : tensor<4xi64>
// CHECK-NEXT:    %c_11 = stablehlo.constant dense<0> : tensor<4xi64>
// CHECK-NEXT:    %11 = stablehlo.add %10, %c_11 : tensor<4xi64>
// CHECK-NEXT:    %c_12 = stablehlo.constant dense<1> : tensor<4xi64>
// CHECK-NEXT:    %12 = stablehlo.multiply %11, %c_12 : tensor<4xi64>
// CHECK-NEXT:    %c_13 = stablehlo.constant dense<-32> : tensor<i64>
// CHECK-NEXT:    %13 = stablehlo.broadcast_in_dim %c_13, dims = [] : (tensor<i64>) -> tensor<4xi64>
// CHECK-NEXT:    %14 = stablehlo.add %12, %13 : tensor<4xi64>
// CHECK-NEXT:    %15 = arith.sitofp %14 : tensor<4xi64> to tensor<4xf64>
// CHECK-NEXT:    %16 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f64>) -> tensor<4xf64>
// CHECK-NEXT:    %17 = arith.mulf %15, %16 {fastmathFlags = #llvm.fastmath<none>} : tensor<4xf64>
// CHECK-NEXT:    %18 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f64>) -> tensor<4xf64>
// CHECK-NEXT:    %19 = arith.mulf %15, %18 {fastmathFlags = #llvm.fastmath<none>} : tensor<4xf64>
// CHECK-NEXT:    %20 = math.absf %17 : tensor<4xf64>
// CHECK-NEXT:    %21 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f64>) -> tensor<4xf64>
// CHECK-NEXT:    %22 = arith.cmpf olt, %21, %20 {fastmathFlags = #llvm.fastmath<none>} : tensor<4xf64>
// CHECK-NEXT:    %23 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f64>) -> tensor<4xf64>
// CHECK-NEXT:    %24 = arith.select %22, %17, %23 : tensor<4xi1>, tensor<4xf64>
// CHECK-NEXT:    %25 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f64>) -> tensor<4xf64>
// CHECK-NEXT:    %26 = arith.select %22, %25, %17 : tensor<4xi1>, tensor<4xf64>
// CHECK-NEXT:    %27 = arith.addf %24, %26 {fastmathFlags = #llvm.fastmath<none>} : tensor<4xf64>
// CHECK-NEXT:    %28 = arith.subf %24, %27 {fastmathFlags = #llvm.fastmath<none>} : tensor<4xf64>
// CHECK-NEXT:    %29 = arith.addf %26, %28 {fastmathFlags = #llvm.fastmath<none>} : tensor<4xf64>
// CHECK-NEXT:    %30 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f64>) -> tensor<4xf64>
// CHECK-NEXT:    %31 = arith.addf %19, %30 {fastmathFlags = #llvm.fastmath<none>} : tensor<4xf64>
// CHECK-NEXT:    %32 = arith.addf %31, %29 {fastmathFlags = #llvm.fastmath<none>} : tensor<4xf64>
// CHECK-NEXT:    %33 = arith.addf %27, %32 {fastmathFlags = #llvm.fastmath<none>} : tensor<4xf64>
// CHECK-NEXT:    %34 = stablehlo.broadcast_in_dim %33, dims = [0] : (tensor<4xf64>) -> tensor<4x198x102xf64>
// CHECK-NEXT:    %35 = stablehlo.broadcast_in_dim %9, dims = [1, 2] : (tensor<198x102xf64>) -> tensor<4x198x102xf64>
// CHECK-NEXT:    %36 = arith.cmpf ole, %34, %35 {fastmathFlags = #llvm.fastmath<none>} : tensor<4x198x102xf64>
// CHECK-NEXT:    %37 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<4x198x102xf64>
// CHECK-NEXT:    %38 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f64>) -> tensor<4x198x102xf64>
// CHECK-NEXT:    %39 = arith.select %36, %37, %38 {fastmathFlags = #llvm.fastmath<none>} : tensor<4x198x102xi1>, tensor<4x198x102xf64>
// CHECK-NEXT:    %40 = stablehlo.broadcast_in_dim %12, dims = [0] : (tensor<4xi64>) -> tensor<4x198x102xi64>
// CHECK-NEXT:    %41 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<4x198x102xf64>
// CHECK-NEXT:    %cst_14 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:    %42 = "stablehlo.reduce_window"(%39, %cst_14) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<[[3, 0], [0, 0], [0, 0]]> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 4, 1, 1>, window_strides = array<i64: 1, 1, 1>}> ({
// CHECK-NEXT:    ^bb0(%arg2: tensor<f64>, %arg3: tensor<f64>):
// CHECK-NEXT:      %48 = stablehlo.add %arg2, %arg3 : tensor<f64>
// CHECK-NEXT:      stablehlo.return %48 : tensor<f64>
// CHECK-NEXT:    }) : (tensor<4x198x102xf64>, tensor<f64>) -> tensor<4x198x102xf64>
// CHECK-NEXT:    %43 = stablehlo.subtract %41, %42 : tensor<4x198x102xf64>
// CHECK-NEXT:    %c_15 = stablehlo.constant dense<5> : tensor<i64>
// CHECK-NEXT:    %c_16 = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:    %c_17 = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:    %44 = stablehlo.broadcast_in_dim %43, dims = [0, 1, 2] : (tensor<4x198x102xf64>) -> tensor<4x198x102xf64>
// CHECK-NEXT:    %45 = stablehlo.dynamic_update_slice %7, %44, %c_15, %c_16, %c_17 : (tensor<40x200x104xf64>, tensor<4x198x102xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<40x200x104xf64>
// CHECK-NEXT:    %c0 = arith.constant 0 : index
// CHECK-NEXT:    %46 = stablehlo.slice %43 [3:4, 0:198, 0:102] : (tensor<4x198x102xf64>) -> tensor<1x198x102xf64>
// CHECK-NEXT:    %47 = stablehlo.reshape %46 : (tensor<1x198x102xf64>) -> tensor<198x102xf64>
// CHECK-NEXT:    return %45, %arg1 : tensor<40x200x104xf64>, tensor<1x200x104xf64>
// CHECK-NEXT:  }
