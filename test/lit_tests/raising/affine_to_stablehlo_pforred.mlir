// RUN: enzymexlamlir-opt %s --raise-affine-to-stablehlo --canonicalize --enzyme-hlo-opt=max_constant_expansion=0 | FileCheck %s

#tbaa_root = #llvm.tbaa_root<id = "custom_tbaa">
#tbaa_type_desc = #llvm.tbaa_type_desc<id = "custom_tbaa_addrspace(1)", members = {<#tbaa_root, 0>}>
#tbaa_tag = #llvm.tbaa_tag<base_type = #tbaa_type_desc, access_type = #tbaa_type_desc, offset = 0>
module @"reactant_loop!" attributes {mhlo.num_partitions = 1 : i64, mhlo.num_replicas = 1 : i64} {
  func.func private @"##call__Z29gpu__compute_barotropic_mode_16CompilerMetadataI10StaticSizeI8_45__20_E12DynamicCheckvv7NDRangeILi2ES0_I6_3__2_ES0_I8_16__16_EvvEE5FieldI4Face6Centervvvv11OffsetArrayI7Float64Li3E13CuTracedArrayISC_Li3ELi1E11_59__34__1_EESC_vvvES8_ISA_S9_vvvvSB_ISC_Li3ESD_ISC_Li3ELi1E11_59__35__1_EESC_vvvE21LatitudeLongitudeGridISC_8Periodic7BoundedSM_28StaticVerticalDiscretizationISB_ISC_Li1ESD_ISC_Li1ELi1E5_25__EESB_ISC_Li1ESD_ISC_Li1ELi1E5_24__EESP_SR_ESC_SC_SB_ISC_Li1E12StepRangeLenISC_14TwicePrecisionISC_ESV_5Int64EESY_SC_SC_SY_SY_SB_ISC_Li1ESD_ISC_Li1ELi1E5_34__EES10_S10_S10_SC_SC_vSW_EvSB_ISC_Li3ESD_ISC_Li3ELi1E12_59__34__24_EESB_ISC_Li3ESD_ISC_Li3ELi1E12_59__35__24_EESF_#555$par153"(%arg0: memref<1x34x59xf64, 1>, %arg1: memref<1x35x59xf64, 1>, %arg2: memref<24xf64, 1>, %arg3: memref<24x34x59xf64, 1>, %arg4: memref<24x35x59xf64, 1>) {
    affine.parallel (%arg5, %arg6) = (0, 0) to (20, 45) {
      %0 = affine.load %arg2[7] {alignment = 8 : i64, ordering = 0 : i64, tbaa = [#tbaa_tag]} : memref<24xf64, 1>
      %1 = affine.load %arg3[7, %arg5 + 7, %arg6 + 7] : memref<24x34x59xf64, 1>
      %2 = arith.mulf %0, %1 {fastmathFlags = #llvm.fastmath<none>} : f64
      affine.store %2, %arg0[0, %arg5 + 7, %arg6 + 7] : memref<1x34x59xf64, 1>
      %3 = affine.load %arg2[7] {alignment = 8 : i64, ordering = 0 : i64, tbaa = [#tbaa_tag]} : memref<24xf64, 1>
      %4 = affine.load %arg4[7, %arg5 + 7, %arg6 + 7] : memref<24x35x59xf64, 1>
      %5 = arith.mulf %3, %4 {fastmathFlags = #llvm.fastmath<none>} : f64
      affine.store %5, %arg1[0, %arg5 + 7, %arg6 + 7] : memref<1x35x59xf64, 1>
      %6 = affine.load %arg0[0, %arg5 + 7, %arg6 + 7] : memref<1x34x59xf64, 1>
      %7 = affine.load %arg1[0, %arg5 + 7, %arg6 + 7] : memref<1x35x59xf64, 1>
      %8:2 = affine.parallel (%arg7) = (0) to (9) reduce ("addf", "addf") -> (f64, f64) {
        %11 = affine.load %arg2[%arg7 + 8] {alignment = 8 : i64, ordering = 0 : i64, tbaa = [#tbaa_tag]} : memref<24xf64, 1>
        %12 = affine.load %arg3[%arg7 + 8, %arg5 + 7, %arg6 + 7] : memref<24x34x59xf64, 1>
        %13 = arith.mulf %11, %12 {fastmathFlags = #llvm.fastmath<none>} : f64
        %14 = affine.load %arg2[%arg7 + 8] {alignment = 8 : i64, ordering = 0 : i64, tbaa = [#tbaa_tag]} : memref<24xf64, 1>
        %15 = affine.load %arg4[%arg7 + 8, %arg5 + 7, %arg6 + 7] : memref<24x35x59xf64, 1>
        %16 = arith.mulf %14, %15 {fastmathFlags = #llvm.fastmath<none>} : f64
        affine.yield %13, %16 : f64, f64
      }
      %9 = arith.addf %6, %8#0 {fastmathFlags = #llvm.fastmath<none>} : f64
      %10 = arith.addf %7, %8#1 {fastmathFlags = #llvm.fastmath<none>} : f64
      affine.store %10, %arg1[0, %arg5 + 7, %arg6 + 7] : memref<1x35x59xf64, 1>
      affine.store %9, %arg0[0, %arg5 + 7, %arg6 + 7] : memref<1x34x59xf64, 1>
    }
    return
  }
}

// CHECK:  func.func private @s2_raised(%arg0: tensor<f64>, %arg1: tensor<28x12xf64>, %arg2: tensor<9x27x59xf64>) -> (tensor<f64>, tensor<28x12xf64>, tensor<9x27x59xf64>) {
// CHECK-NEXT:    %c = stablehlo.constant dense<3> : tensor<i64>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<7> : tensor<i64>
// CHECK-NEXT:    %c_1 = stablehlo.constant dense<8> : tensor<i64>
// CHECK-NEXT:    %c_2 = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:    %c_3 = stablehlo.constant dense<9> : tensor<i64>
// CHECK-NEXT:    %c_4 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %0 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f64>) -> tensor<20x45xf64>
// CHECK-NEXT:    %1:5 = stablehlo.while(%iterArg = %c_4, %iterArg_5 = %0, %iterArg_6 = %arg0, %iterArg_7 = %arg1, %iterArg_8 = %arg2) : tensor<i64>, tensor<20x45xf64>, tensor<f64>, tensor<28x12xf64>, tensor<9x27x59xf64>
// CHECK-NEXT:     cond {
// CHECK-NEXT:      %2 = stablehlo.compare  LT, %iterArg, %c_3 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:      stablehlo.return %2 : tensor<i1>
// CHECK-NEXT:    } do {
// CHECK-NEXT:      %2 = stablehlo.add %iterArg, %c : tensor<i64>
// CHECK-NEXT:      %3 = stablehlo.dynamic_slice %iterArg_7, %c_1, %2, sizes = [20, 1] : (tensor<28x12xf64>, tensor<i64>, tensor<i64>) -> tensor<20x1xf64>
// CHECK-NEXT:      %4 = stablehlo.reshape %3 : (tensor<20x1xf64>) -> tensor<20xf64>
// CHECK-NEXT:      %5 = stablehlo.broadcast_in_dim %4, dims = [0] : (tensor<20xf64>) -> tensor<20x45xf64>
// CHECK-NEXT:      %6 = arith.subf %5, %iterArg_5 : tensor<20x45xf64>
// CHECK-NEXT:      %7 = stablehlo.reshape %6 : (tensor<20x45xf64>) -> tensor<1x20x45xf64>
// CHECK-NEXT:      %8 = stablehlo.dynamic_update_slice %iterArg_8, %7, %iterArg, %c_0, %c_0 : (tensor<9x27x59xf64>, tensor<1x20x45xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<9x27x59xf64>
// CHECK-NEXT:      %9 = stablehlo.add %iterArg, %c_2 : tensor<i64>
// CHECK-NEXT:      stablehlo.return %9, %6, %iterArg_6, %iterArg_7, %8 : tensor<i64>, tensor<20x45xf64>, tensor<f64>, tensor<28x12xf64>, tensor<9x27x59xf64>
// CHECK-NEXT:    }
// CHECK-NEXT:    return %1#2, %1#3, %1#4 : tensor<f64>, tensor<28x12xf64>, tensor<9x27x59xf64>
// CHECK-NEXT:  }

// CHECK:  func.func private @s_raised(%arg0: tensor<f64>, %arg1: tensor<28x12xf64>, %arg2: tensor<9x27x59xf64>) -> (tensor<f64>, tensor<28x12xf64>, tensor<9x27x59xf64>) {
// CHECK-NEXT:    %c = stablehlo.constant dense<7> : tensor<i64>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:    %0 = stablehlo.slice %arg1 [8:28, 3:12] : (tensor<28x12xf64>) -> tensor<20x9xf64>
// CHECK-NEXT:    %1 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f64>) -> tensor<20x9xf64>
// CHECK-NEXT{LITERAL}:    %2 = "stablehlo.reduce_window"(%0, %cst) <{base_dilations = array<i64: 1, 1>, padding = dense<[[0, 0], [8, 0]]> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 9>, window_strides = array<i64: 1, 1>}> ({
// CHECK-NEXT:    ^bb0(%arg3: tensor<f64>, %arg4: tensor<f64>):
// CHECK-NEXT:      %6 = stablehlo.add %arg3, %arg4 : tensor<f64>
// CHECK-NEXT:      stablehlo.return %6 : tensor<f64>
// CHECK-NEXT:    }) : (tensor<20x9xf64>, tensor<f64>) -> tensor<20x9xf64>
// CHECK-NEXT:    %3 = stablehlo.subtract %1, %2 : tensor<20x9xf64>
// CHECK-NEXT:    %4 = stablehlo.broadcast_in_dim %3, dims = [1, 0] : (tensor<20x9xf64>) -> tensor<9x20x45xf64>
// CHECK-NEXT:    %5 = stablehlo.dynamic_update_slice %arg2, %4, %c_0, %c, %c : (tensor<9x27x59xf64>, tensor<9x20x45xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<9x27x59xf64>
// CHECK-NEXT:    return %arg0, %arg1, %5 : tensor<f64>, tensor<28x12xf64>, tensor<9x27x59xf64>
// CHECK-NEXT:  }

// CHECK:  func.func private @f2_raised(%arg0: tensor<f64>, %arg1: tensor<28x12xf64>, %arg2: tensor<9x27x59xf64>) -> (tensor<f64>, tensor<28x12xf64>, tensor<9x27x59xf64>) {
// CHECK-NEXT:    %c = stablehlo.constant dense<7> : tensor<i64>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:    %0 = stablehlo.slice %arg1 [8:28, 3:12] : (tensor<28x12xf64>) -> tensor<20x9xf64>
// CHECK-NEXT:    %1 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f64>) -> tensor<20x9xf64>
// CHECK-NEXT{LITERAL}:    %2 = "stablehlo.reduce_window"(%0, %cst) <{base_dilations = array<i64: 1, 1>, padding = dense<[[0, 0], [8, 0]]> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 9>, window_strides = array<i64: 1, 1>}> ({
// CHECK-NEXT:    ^bb0(%arg3: tensor<f64>, %arg4: tensor<f64>):
// CHECK-NEXT:      %6 = stablehlo.add %arg3, %arg4 : tensor<f64>
// CHECK-NEXT:      stablehlo.return %6 : tensor<f64>
// CHECK-NEXT:    }) : (tensor<20x9xf64>, tensor<f64>) -> tensor<20x9xf64>
// CHECK-NEXT:    %3 = stablehlo.add %2, %1 : tensor<20x9xf64>
// CHECK-NEXT:    %4 = stablehlo.broadcast_in_dim %3, dims = [1, 0] : (tensor<20x9xf64>) -> tensor<9x20x45xf64>
// CHECK-NEXT:    %5 = stablehlo.dynamic_update_slice %arg2, %4, %c_0, %c, %c : (tensor<9x27x59xf64>, tensor<9x20x45xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<9x27x59xf64>
// CHECK-NEXT:    return %arg0, %arg1, %5 : tensor<f64>, tensor<28x12xf64>, tensor<9x27x59xf64>
// CHECK-NEXT:  }

// CHECK:  func.func private @f_raised(%arg0: tensor<f64>, %arg1: tensor<28x12xf64>, %arg2: tensor<9x27x59xf64>) -> (tensor<f64>, tensor<28x12xf64>, tensor<9x27x59xf64>) {
// CHECK-NEXT:    %c = stablehlo.constant dense<7> : tensor<i64>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:    %0 = stablehlo.slice %arg1 [8:28, 3:12] : (tensor<28x12xf64>) -> tensor<20x9xf64>
// CHECK-NEXT:    %1 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f64>) -> tensor<20x9xf64>
// CHECK-NEXT{LITERAL}:    %2 = "stablehlo.reduce_window"(%0, %cst) <{base_dilations = array<i64: 1, 1>, padding = dense<[[0, 0], [8, 0]]> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 9>, window_strides = array<i64: 1, 1>}> ({
// CHECK-NEXT:    ^bb0(%arg3: tensor<f64>, %arg4: tensor<f64>):
// CHECK-NEXT:      %6 = stablehlo.add %arg3, %arg4 : tensor<f64>
// CHECK-NEXT:      stablehlo.return %6 : tensor<f64>
// CHECK-NEXT:    }) : (tensor<20x9xf64>, tensor<f64>) -> tensor<20x9xf64>
// CHECK-NEXT:    %3 = stablehlo.add %2, %1 : tensor<20x9xf64>
// CHECK-NEXT:    %4 = stablehlo.broadcast_in_dim %3, dims = [1, 0] : (tensor<20x9xf64>) -> tensor<9x20x45xf64>
// CHECK-NEXT:    %5 = stablehlo.dynamic_update_slice %arg2, %4, %c_0, %c, %c : (tensor<9x27x59xf64>, tensor<9x20x45xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<9x27x59xf64>
// CHECK-NEXT:    return %arg0, %arg1, %5 : tensor<f64>, tensor<28x12xf64>, tensor<9x27x59xf64>
// CHECK-NEXT:  }
