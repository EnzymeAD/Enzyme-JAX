// RUN: enzymexlamlir-opt --sort-memory %s | FileCheck %s

#set = affine_set<(d0, d1) : (d1 >= 0, -d1 + 61 >= 0, d0 >= 0, -d0 + 61 >= 0)>
module {
  func.func private @"##call__Z37gpu_substep_turbulent_kinetic_energy_16CompilerMetadataI10StaticSizeI12_62__62__15_E12DynamicCheckvv7NDRangeILi3ES0_I10_4__4__15_ES0_I11_16__16__1_EvvEE11OffsetArrayI7Float32Li3E13CuTracedArrayIS9_Li3ELi1E12_78__78__31_EESC_21LatitudeLongitudeGridI15CuTracedRNumberIS9_Li1EE8Periodic7BoundedSH_28StaticVerticalDiscretizationIS8_IS9_Li1ESA_IS9_Li1ELi1E5_32__EES8_IS9_Li1ESA_IS9_Li1ELi1E5_31__EESK_SM_ES9_S9_S8_IS9_Li1E12StepRangeLenIS9_7Float64SP_5Int64EESS_S9_S9_SS_SS_S8_IS9_Li1ESA_IS9_Li1ELi1E5_78__EESU_SU_SU_S9_S9_vSQ_E24CATKEVerticalDiffusivityI36VerticallyImplicitTimeDiscretization17CATKEMixingLengthIS9_ES9_v13CATKEEquationIS9_EE10NamedTupleI12__u___v___w_5TupleISC_SC_SC_EES13_I8__u___v_S14_ISC_SC_EES13_I12__T___S___e_S15_E13BuoyancyForceI16SeawaterBuoyancyIS9_25BoussinesqEquationOfStateI24TEOS10SeawaterPolynomialIS9_ES9_EvvE18NegativeZDirectionE22CATKEDiffusivityFieldsISC_SC_5FieldI6CenterS1L_vvvvS8_IS9_Li3ESA_IS9_Li3ELi1E11_78__78__1_EES9_vvvE13TracedRNumberIS9_ES18_S19_S13_I12__T___S___e_S14_I9ZeroFieldISQ_Li3EES1S_SC_EEES9_S9_SC_SC_#286$par15"() {
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant 0x7F800000 : f32
    affine.parallel (%arg0, %arg1, %arg2) = (0, 0, 0) to (15, 64, 64) {
      %0 = llvm.alloca %c1_i32 x f32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
      affine.if #set(%arg1, %arg2) {
        llvm.intr.lifetime.start 1, %0 : !llvm.ptr
        %1 = "enzymexla.pointer2memref"(%0) : (!llvm.ptr) -> memref<?xf32>
        affine.store %cst, %1[0] {alignment = 4 : i64, ordering = 5 : i64} : memref<?xf32>
        llvm.intr.lifetime.end 1, %0 : !llvm.ptr
        "llvm.intr.trap"() : () -> ()
      }
    }
    return
  }
}

// CHECK:  func.func private @"##call__Z37gpu_substep_turbulent_kinetic_energy_16CompilerMetadataI10StaticSizeI12_62__62__15_E12DynamicCheckvv7NDRangeILi3ES0_I10_4__4__15_ES0_I11_16__16__1_EvvEE11OffsetArrayI7Float32Li3E13CuTracedArrayIS9_Li3ELi1E12_78__78__31_EESC_21LatitudeLongitudeGridI15CuTracedRNumberIS9_Li1EE8Periodic7BoundedSH_28StaticVerticalDiscretizationIS8_IS9_Li1ESA_IS9_Li1ELi1E5_32__EES8_IS9_Li1ESA_IS9_Li1ELi1E5_31__EESK_SM_ES9_S9_S8_IS9_Li1E12StepRangeLenIS9_7Float64SP_5Int64EESS_S9_S9_SS_SS_S8_IS9_Li1ESA_IS9_Li1ELi1E5_78__EESU_SU_SU_S9_S9_vSQ_E24CATKEVerticalDiffusivityI36VerticallyImplicitTimeDiscretization17CATKEMixingLengthIS9_ES9_v13CATKEEquationIS9_EE10NamedTupleI12__u___v___w_5TupleISC_SC_SC_EES13_I8__u___v_S14_ISC_SC_EES13_I12__T___S___e_S15_E13BuoyancyForceI16SeawaterBuoyancyIS9_25BoussinesqEquationOfStateI24TEOS10SeawaterPolynomialIS9_ES9_EvvE18NegativeZDirectionE22CATKEDiffusivityFieldsISC_SC_5FieldI6CenterS1L_vvvvS8_IS9_Li3ESA_IS9_Li3ELi1E11_78__78__1_EES9_vvvE13TracedRNumberIS9_ES18_S19_S13_I12__T___S___e_S14_I9ZeroFieldISQ_Li3EES1S_SC_EEES9_S9_SC_SC_#286$par15"() {
// CHECK-NEXT:    %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:    %cst = arith.constant 0x7F800000 : f32
// CHECK-NEXT:    affine.parallel (%arg0, %arg1, %arg2) = (0, 0, 0) to (15, 64, 64) {
// CHECK-NEXT:      %0 = llvm.alloca %c1_i32 x f32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
// CHECK-NEXT:      affine.if #set(%arg1, %arg2) {
// CHECK-NEXT:        llvm.intr.lifetime.start 1, %0 : !llvm.ptr
// CHECK-NEXT:        %1 = "enzymexla.pointer2memref"(%0) : (!llvm.ptr) -> memref<?xf32>
// CHECK-NEXT:        affine.store %cst, %1[0] {alignment = 4 : i64, ordering = 5 : i64} : memref<?xf32>
// CHECK-NEXT:        llvm.intr.lifetime.end 1, %0 : !llvm.ptr
// CHECK-NEXT:        "llvm.intr.trap"() : () -> ()
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }
