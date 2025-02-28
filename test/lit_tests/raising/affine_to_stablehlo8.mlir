// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(raise-affine-to-stablehlo,arith-raise,enzyme-hlo-opt{max_constant_expansion=1})" | FileCheck %s

module {
  func.func private @"##call__Z27gpu_store_field_tendencies_16CompilerMetadataI10StaticSizeI13_180__85__20_E12DynamicCheckvv7NDRangeILi3ES0_I11_12__6__20_ES0_I11_16__16__1_EvvEE5FieldI4Face6Centervvvv11OffsetArrayI7Float64Li3E13CuTracedArrayISC_Li3ELi1E13_194__187__1_EESC_vvvESG_#1671$par239"(%arg0: memref<1x187x194xf64, 1>, %arg1: memref<1x187x194xf64, 1>) {
  affine.parallel (%arg2, %arg3, %arg4) = (0, 0, 0) to (20, 85, 180) {
    %0 = affine.load %arg1[0, %arg3 + 51, %arg4 + 7] : memref<1x187x194xf64, 1>
    affine.store %0, %arg0[0, %arg3 + 51, %arg4 + 7] : memref<1x187x194xf64, 1>
  }
  return
  }

  func.func @main(%arg101 : tensor<1x187x194xf64>, %52 : tensor<1x187x194xf64>) -> tensor<1x187x194xf64> {
    %170 = enzymexla.jit_call @"##call__Z27gpu_store_field_tendencies_16CompilerMetadataI10StaticSizeI13_180__85__20_E12DynamicCheckvv7NDRangeILi3ES0_I11_12__6__20_ES0_I11_16__16__1_EvvEE5FieldI4Face6Centervvvv11OffsetArrayI7Float64Li3E13CuTracedArrayISC_Li3ELi1E13_194__187__1_EESC_vvvESG_#1671$par239" (%arg101, %52) {output_operand_aliases = [
#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>]} : (tensor<1x187x194xf64>, tensor<1x187x194xf64>) -> tensor<1x187x194xf64>
    return %170 : tensor<1x187x194xf64>
  }
}

// CHECK:  func.func @main(%arg0: tensor<1x187x194xf64>, %arg1: tensor<1x187x194xf64>) -> tensor<1x187x194xf64> {
// CHECK-NEXT:    %0:2 = call @"##call__Z27gpu_store_field_tendencies_16CompilerMetadataI10StaticSizeI13_180__85__20_E12DynamicCheckvv7NDRangeILi3ES0_I11_12__6__20_ES0_I11_16__16__1_EvvEE5FieldI4Face6Centervvvv11OffsetArrayI7Float64Li3E13CuTracedArrayISC_Li3ELi1E13_194__187__1_EESC_vvvESG_#1671$par239_raised"(%arg0, %arg1) : (tensor<1x187x194xf64>, tensor<1x187x194xf64>) -> (tensor<1x187x194xf64>, tensor<1x187x194xf64>)
// CHECK-NEXT:    return %0#0 : tensor<1x187x194xf64>
// CHECK-NEXT:  }
