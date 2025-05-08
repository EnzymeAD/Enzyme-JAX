// RUN: enzymexlamlir-opt --affine-cfg --split-input-file --allow-unregistered-dialect %s | FileCheck %s

module {
  func.func private @"##call__Z32gpu_compute_CATKE_diffusivities_16CompilerMetadataI10StaticSizeI12_62__62__15_E12DynamicCheckvv7NDRangeILi3ES0_I10_4__4__15_ES0_I11_16__16__1_EvvEE22CATKEDiffusivityFieldsI11OffsetArrayI7Float32Li3E13CuTracedArrayISA_Li3ELi1E12_78__78__31_EESD_5FieldI6CenterSF_vvvvS9_ISA_Li3ESB_ISA_Li3ELi1E11_78__78__1_EESA_vvvE13TracedRNumberISA_E10NamedTupleI8__u___v_5TupleISD_SD_EESL_I12__T___S___e_SM_ISD_SD_SD_EESL_I12__T___S___e_SM_I9ZeroFieldI5Int64Li3EEST_SD_EEE21LatitudeLongitudeGridI15CuTracedRNumberISA_Li1EE8Periodic7BoundedS11_28StaticVerticalDiscretizationIS9_ISA_Li1ESB_ISA_Li1ELi1E5_32__EES9_ISA_Li1ESB_ISA_Li1ELi1E5_31__EES14_S16_ESA_SA_S9_ISA_Li1E12StepRangeLenISA_7Float64S19_SS_EES1B_SA_SA_S1B_S1B_S9_ISA_Li1ESB_ISA_Li1ELi1E5_78__EES1D_S1D_S1D_SA_SA_vSS_E24CATKEVerticalDiffusivityI36VerticallyImplicitTimeDiscretization17CATKEMixingLengthISA_ESA_v13CATKEEquationISA_EESL_I12__u___v___w_SP_ESQ_13BuoyancyForceI16SeawaterBuoyancyISA_25BoussinesqEquationOfStateI24TEOS10SeawaterPolynomialISA_ESA_EvvE18NegativeZDirectionE#459$par103"(%arg0: memref<31x78x78xf32, 1> {llvm.align = 128 : i32, llvm.dereferenceable = 754416 : i64, llvm.noalias}, %arg1: memref<31x78x78xf32, 1> {llvm.align = 128 : i32, llvm.dereferenceable = 754416 : i64, llvm.noalias}, %arg2: memref<31x78x78xf32, 1> {llvm.align = 128 : i32, llvm.dereferenceable = 754416 : i64, llvm.noalias}, %arg3: memref<1x78x78xf32, 1> {llvm.align = 128 : i32, llvm.dereferenceable = 24336 : i64, llvm.noalias}, %arg4: memref<f32, 1> {llvm.align = 128 : i32, llvm.dereferenceable = 4 : i64, llvm.noalias}, %arg5: memref<32xf32, 1> {llvm.align = 128 : i32, llvm.dereferenceable = 128 : i64, llvm.noalias}, %arg6: memref<32xf32, 1> {llvm.align = 128 : i32, llvm.dereferenceable = 128 : i64, llvm.noalias}, %arg7: memref<31xf32, 1> {llvm.align = 128 : i32, llvm.dereferenceable = 124 : i64, llvm.noalias}, %arg8: memref<31x78x78xf32, 1> {llvm.align = 128 : i32, llvm.dereferenceable = 754416 : i64, llvm.noalias}, %arg9: memref<31x78x78xf32, 1> {llvm.align = 128 : i32, llvm.dereferenceable = 754416 : i64, llvm.noalias}, %arg10: memref<31x78x78xf32, 1> {llvm.align = 128 : i32, llvm.dereferenceable = 754416 : i64, llvm.noalias}, %arg11: memref<31x78x78xf32, 1> {llvm.align = 128 : i32, llvm.dereferenceable = 754416 : i64, llvm.noalias}, %arg12: memref<31x78x78xf32, 1> {llvm.align = 128 : i32, llvm.dereferenceable = 754416 : i64, llvm.noalias}) attributes {enzymexla.memory_effects = ["read", "write", "allocate", "free"]} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c62_i64 = arith.constant 62 : i64
    %cst = arith.constant 0.000000e+00 : f32
    affine.parallel (%arg13, %arg14, %arg15) = (0, 0, 0) to (15, 62, 62) {
      %0 = arith.addi %arg14, %c1 : index
      %1 = arith.index_castui %0 : index to i64
      %2 = arith.cmpi sgt, %1, %c62_i64 : i64
      %3 = arith.cmpi eq, %arg13, %c0 : index
      %4 = arith.ori %3, %2 : i1
      %5 = arith.ori %2, %4 : i1
      scf.if %5 {
        affine.store %cst, %arg0[%arg13 + 8, %arg14 + 8, %arg15 + 8] : memref<31x78x78xf32, 1>
        affine.store %cst, %arg1[%arg13 + 8, %arg14 + 8, %arg15 + 8] : memref<31x78x78xf32, 1>
      }
      affine.store %cst, %arg2[%arg13 + 8, %arg14 + 8, %arg15 + 8] : memref<31x78x78xf32, 1>
    }
    return
  }
  func.func private @second(%arg0: memref<31x78x78xf32, 1> {llvm.align = 128 : i32, llvm.dereferenceable = 754416 : i64, llvm.noalias}, %arg1: memref<31x78x78xf32, 1> {llvm.align = 128 : i32, llvm.dereferenceable = 754416 : i64, llvm.noalias}, %arg2: memref<31x78x78xf32, 1> {llvm.align = 128 : i32, llvm.dereferenceable = 754416 : i64, llvm.noalias}, %arg3: memref<1x78x78xf32, 1> {llvm.align = 128 : i32, llvm.dereferenceable = 24336 : i64, llvm.noalias}, %arg4: memref<f32, 1> {llvm.align = 128 : i32, llvm.dereferenceable = 4 : i64, llvm.noalias}, %arg5: memref<32xf32, 1> {llvm.align = 128 : i32, llvm.dereferenceable = 128 : i64, llvm.noalias}, %arg6: memref<32xf32, 1> {llvm.align = 128 : i32, llvm.dereferenceable = 128 : i64, llvm.noalias}, %arg7: memref<31xf32, 1> {llvm.align = 128 : i32, llvm.dereferenceable = 124 : i64, llvm.noalias}, %arg8: memref<31x78x78xf32, 1> {llvm.align = 128 : i32, llvm.dereferenceable = 754416 : i64, llvm.noalias}, %arg9: memref<31x78x78xf32, 1> {llvm.align = 128 : i32, llvm.dereferenceable = 754416 : i64, llvm.noalias}, %arg10: memref<31x78x78xf32, 1> {llvm.align = 128 : i32, llvm.dereferenceable = 754416 : i64, llvm.noalias}, %arg11: memref<31x78x78xf32, 1> {llvm.align = 128 : i32, llvm.dereferenceable = 754416 : i64, llvm.noalias}, %arg12: memref<31x78x78xf32, 1> {llvm.align = 128 : i32, llvm.dereferenceable = 754416 : i64, llvm.noalias}) attributes {enzymexla.memory_effects = ["read", "write", "allocate", "free"]} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c62_i64 = arith.constant 62 : i64
    %cst = arith.constant 0.000000e+00 : f32
    affine.parallel (%arg13, %arg14, %arg15) = (0, 0, 0) to (15, 62, 62) {
      %0 = arith.addi %arg14, %c1 : index
      %1 = arith.index_castui %0 : index to i64
      %2 = arith.cmpi sgt, %1, %c62_i64 : i64
      %3 = arith.cmpi eq, %arg13, %c0 : index
      %4 = arith.ori %3, %2 : i1
      %5 = arith.ori %2, %4 : i1
      scf.if %5 {
        affine.store %cst, %arg0[%arg13 + 8, %arg14 + 8, %arg15 + 8] : memref<31x78x78xf32, 1>
        affine.store %cst, %arg1[%arg13 + 8, %arg14 + 8, %arg15 + 8] : memref<31x78x78xf32, 1>
      } else {
	affine.store %cst, %arg2[%arg13 + 8, %arg14 + 8, %arg15 + 8] : memref<31x78x78xf32, 1>
      }
    }
    return
  }
}

// CHECK: #set = affine_set<(d0, d1) : (-d1 + 61 >= 0, d0 - 1 >= 0, -d1 + 61 >= 0)>

// CHECK:   func.func private @"##call__Z32gpu_compute_CATKE_diffusivities_16CompilerMetadataI10StaticSizeI12_62__62__15_E12DynamicCheckvv7NDRangeILi3ES0_I10_4__4__15_ES0_I11_16__16__1_EvvEE22CATKEDiffusivityFieldsI11OffsetArrayI7Float32Li3E13CuTracedArrayISA_Li3ELi1E12_78__78__31_EESD_5FieldI6CenterSF_vvvvS9_ISA_Li3ESB_ISA_Li3ELi1E11_78__78__1_EESA_vvvE13TracedRNumberISA_E10NamedTupleI8__u___v_5TupleISD_SD_EESL_I12__T___S___e_SM_ISD_SD_SD_EESL_I12__T___S___e_SM_I9ZeroFieldI5Int64Li3EEST_SD_EEE21LatitudeLongitudeGridI15CuTracedRNumberISA_Li1EE8Periodic7BoundedS11_28StaticVerticalDiscretizationIS9_ISA_Li1ESB_ISA_Li1ELi1E5_32__EES9_ISA_Li1ESB_ISA_Li1ELi1E5_31__EES14_S16_ESA_SA_S9_ISA_Li1E12StepRangeLenISA_7Float64S19_SS_EES1B_SA_SA_S1B_S1B_S9_ISA_Li1ESB_ISA_Li1ELi1E5_78__EES1D_S1D_S1D_SA_SA_vSS_E24CATKEVerticalDiffusivityI36VerticallyImplicitTimeDiscretization17CATKEMixingLengthISA_ESA_v13CATKEEquationISA_EESL_I12__u___v___w_SP_ESQ_13BuoyancyForceI16SeawaterBuoyancyISA_25BoussinesqEquationOfStateI24TEOS10SeawaterPolynomialISA_ESA_EvvE18NegativeZDirectionE#459$par103"(%arg0: memref<31x78x78xf32, 1> {llvm.align = 128 : i32, llvm.dereferenceable = 754416 : i64, llvm.noalias}, %arg1: memref<31x78x78xf32, 1> {llvm.align = 128 : i32, llvm.dereferenceable = 754416 : i64, llvm.noalias}, %arg2: memref<31x78x78xf32, 1> {llvm.align = 128 : i32, llvm.dereferenceable = 754416 : i64, llvm.noalias}, %arg3: memref<1x78x78xf32, 1> {llvm.align = 128 : i32, llvm.dereferenceable = 24336 : i64, llvm.noalias}, %arg4: memref<f32, 1> {llvm.align = 128 : i32, llvm.dereferenceable = 4 : i64, llvm.noalias}, %arg5: memref<32xf32, 1> {llvm.align = 128 : i32, llvm.dereferenceable = 128 : i64, llvm.noalias}, %arg6: memref<32xf32, 1> {llvm.align = 128 : i32, llvm.dereferenceable = 128 : i64, llvm.noalias}, %arg7: memref<31xf32, 1> {llvm.align = 128 : i32, llvm.dereferenceable = 124 : i64, llvm.noalias}, %arg8: memref<31x78x78xf32, 1> {llvm.align = 128 : i32, llvm.dereferenceable = 754416 : i64, llvm.noalias}, %arg9: memref<31x78x78xf32, 1> {llvm.align = 128 : i32, llvm.dereferenceable = 754416 : i64, llvm.noalias}, %arg10: memref<31x78x78xf32, 1> {llvm.align = 128 : i32, llvm.dereferenceable = 754416 : i64, llvm.noalias}, %arg11: memref<31x78x78xf32, 1> {llvm.align = 128 : i32, llvm.dereferenceable = 754416 : i64, llvm.noalias}, %arg12: memref<31x78x78xf32, 1> {llvm.align = 128 : i32, llvm.dereferenceable = 754416 : i64, llvm.noalias}) attributes {enzymexla.memory_effects = ["read", "write", "allocate", "free"]} {
// CHECK-NEXT:     %cst = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:     affine.parallel (%arg13, %arg14, %arg15) = (0, 0, 0) to (15, 62, 62) {
// CHECK-NEXT:       affine.if #set(%arg13, %arg14) {
// CHECK-NEXT:       } else {
// CHECK-NEXT:         affine.store %cst, %arg0[%arg13 + 8, %arg14 + 8, %arg15 + 8] : memref<31x78x78xf32, 1>
// CHECK-NEXT:         affine.store %cst, %arg1[%arg13 + 8, %arg14 + 8, %arg15 + 8] : memref<31x78x78xf32, 1>
// CHECK-NEXT:       }
// CHECK-NEXT:       affine.store %cst, %arg2[%arg13 + 8, %arg14 + 8, %arg15 + 8] : memref<31x78x78xf32, 1>
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }

// CHECK:   func.func private @second(%arg0: memref<31x78x78xf32, 1> {llvm.align = 128 : i32, llvm.dereferenceable = 754416 : i64, llvm.noalias}, %arg1: memref<31x78x78xf32, 1> {llvm.align = 128 : i32, llvm.dereferenceable = 754416 : i64, llvm.noalias}, %arg2: memref<31x78x78xf32, 1> {llvm.align = 128 : i32, llvm.dereferenceable = 754416 : i64, llvm.noalias}, %arg3: memref<1x78x78xf32, 1> {llvm.align = 128 : i32, llvm.dereferenceable = 24336 : i64, llvm.noalias}, %arg4: memref<f32, 1> {llvm.align = 128 : i32, llvm.dereferenceable = 4 : i64, llvm.noalias}, %arg5: memref<32xf32, 1> {llvm.align = 128 : i32, llvm.dereferenceable = 128 : i64, llvm.noalias}, %arg6: memref<32xf32, 1> {llvm.align = 128 : i32, llvm.dereferenceable = 128 : i64, llvm.noalias}, %arg7: memref<31xf32, 1> {llvm.align = 128 : i32, llvm.dereferenceable = 124 : i64, llvm.noalias}, %arg8: memref<31x78x78xf32, 1> {llvm.align = 128 : i32, llvm.dereferenceable = 754416 : i64, llvm.noalias}, %arg9: memref<31x78x78xf32, 1> {llvm.align = 128 : i32, llvm.dereferenceable = 754416 : i64, llvm.noalias}, %arg10: memref<31x78x78xf32, 1> {llvm.align = 128 : i32, llvm.dereferenceable = 754416 : i64, llvm.noalias}, %arg11: memref<31x78x78xf32, 1> {llvm.align = 128 : i32, llvm.dereferenceable = 754416 : i64, llvm.noalias}, %arg12: memref<31x78x78xf32, 1> {llvm.align = 128 : i32, llvm.dereferenceable = 754416 : i64, llvm.noalias}) attributes {enzymexla.memory_effects = ["read", "write", "allocate", "free"]} {
// CHECK-NEXT:     %cst = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:     affine.parallel (%arg13, %arg14, %arg15) = (0, 0, 0) to (15, 62, 62) {
// CHECK-NEXT:       affine.if #set(%arg13, %arg14) {
// CHECK-NEXT:         affine.store %cst, %arg2[%arg13 + 8, %arg14 + 8, %arg15 + 8] : memref<31x78x78xf32, 1>
// CHECK-NEXT:       } else {
// CHECK-NEXT:         affine.store %cst, %arg0[%arg13 + 8, %arg14 + 8, %arg15 + 8] : memref<31x78x78xf32, 1>
// CHECK-NEXT:         affine.store %cst, %arg1[%arg13 + 8, %arg14 + 8, %arg15 + 8] : memref<31x78x78xf32, 1>
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }
