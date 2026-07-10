// RUN: enzymexlamlir-opt %s --delinearize-indexing | FileCheck %s

#tbaa_root = #llvm.tbaa_root<id = "custom_tbaa">
#tbaa_type_desc = #llvm.tbaa_type_desc<id = "custom_tbaa_addrspace(1)", members = {<#tbaa_root, 0>}>
#tbaa_tag = #llvm.tbaa_tag<base_type = #tbaa_type_desc, access_type = #tbaa_type_desc, offset = 0>
module @reactant_ui8_bra... attributes {llvm.data_layout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64", mhlo.num_partitions = 1 : i64, mhlo.num_replicas = 1 : i64} {
  llvm.module_flags [#llvm.mlir.module_flag<warning, "Dwarf Version", 2 : i32>, #llvm.mlir.module_flag<warning, "Debug Info Version", 3 : i32>, #llvm.mlir.module_flag<override, "nvvm-reflect-ftz", 0 : i32>]
  func.func @main(%arg0: tensor<2xui8> {enzymexla.memory_effects = ["read", "write", "allocate", "free"], tf.aliasing_output = 0 : i32}, %arg1: tensor<2xf64> {enzymexla.memory_effects = ["read", "write", "allocate", "free"], tf.aliasing_output = 1 : i32}) -> (tensor<2xui8>, tensor<2xf64>) attributes {enzymexla.memory_effects = []} {
    %0 = enzymexla.jit_call @"##call__Z22gpu_ui8_branch_kernel_16CompilerMetadataI11DynamicSize12DynamicCheckv16CartesianIndicesILi1E5TupleI5OneToI5Int64EEE7NDRangeILi1ES0_S0_S8_S8_EE13CuTracedArrayI5UInt8Li1ELi1E4_2__ESC_I7Float64Li1ELi1E4_2__E#280$par0" (%arg0, %arg1) {output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>], xla_side_effect_free} : (tensor<2xui8>, tensor<2xf64>) -> tensor<2xui8>
    return %0, %arg1 : tensor<2xui8>, tensor<2xf64>
  }
  func.func private @"##call__Z22gpu_ui8_branch_kernel_16CompilerMetadataI11DynamicSize12DynamicCheckv16CartesianIndicesILi1E5TupleI5OneToI5Int64EEE7NDRangeILi1ES0_S0_S8_S8_EE13CuTracedArrayI5UInt8Li1ELi1E4_2__ESC_I7Float64Li1ELi1E4_2__E#280$par0"(%arg0: memref<2xui8, 1>, %arg1: memref<2xf64, 1>) attributes {enzymexla.memory_effects = ["read", "write", "allocate", "free"]} {
    %cst = arith.constant 5.000000e-01 : f64
    %c1_i8 = arith.constant 1 : i8
    %c2_i8 = arith.constant 2 : i8
    %0 = "enzymexla.memref2pointer"(%arg0) : (memref<2xui8, 1>) -> !llvm.ptr<1>
    %1 = "enzymexla.pointer2memref"(%0) : (!llvm.ptr<1>) -> memref<?xi8, 1>
    affine.parallel (%arg2) = (0) to (2) {
      %2 = affine.load %arg1[%arg2] {alignment = 8 : i64, invariant, ordering = 0 : i64, tbaa = [#tbaa_tag]} : memref<2xf64, 1>
      %3 = arith.cmpf ult, %2, %cst {fastmathFlags = #llvm.fastmath<none>} : f64
      %4 = arith.select %3, %c1_i8, %c2_i8 {fastmathFlags = #llvm.fastmath<none>} : i8
      affine.store %4, %1[%arg2] {alignment = 1 : i64, ordering = 0 : i64, tbaa = [#tbaa_tag]} : memref<?xi8, 1>
    }
    return
  }
}

// CHECK:  func.func private @"##call__Z22gpu_ui8_branch_kernel_16CompilerMetadataI11DynamicSize12DynamicCheckv16CartesianIndicesILi1E5TupleI5OneToI5Int64EEE7NDRangeILi1ES0_S0_S8_S8_EE13CuTracedArrayI5UInt8Li1ELi1E4_2__ESC_I7Float64Li1ELi1E4_2__E#280$par0"(%arg0: memref<2xui8, 1>, %arg1: memref<2xf64, 1>) attributes {enzymexla.memory_effects = ["read", "write", "allocate", "free"]} {
// CHECK-NEXT:    %cst = arith.constant 5.000000e-01 : f64
// CHECK-NEXT:    %c1_i8 = arith.constant 1 : i8
// CHECK-NEXT:    %c2_i8 = arith.constant 2 : i8
// CHECK-NEXT:    %0 = "enzymexla.memref2pointer"(%arg0) : (memref<2xui8, 1>) -> !llvm.ptr<1>
// CHECK-NEXT:    %1 = "enzymexla.pointer2memref"(%0) : (!llvm.ptr<1>) -> memref<?xi8, 1>
// CHECK-NEXT:    affine.parallel (%arg2) = (0) to (2) {
// CHECK-NEXT:      %2 = affine.load %arg1[%arg2] {alignment = 8 : i64, invariant, ordering = 0 : i64, tbaa = [#tbaa_tag]} : memref<2xf64, 1>
// CHECK-NEXT:      %3 = arith.cmpf ult, %2, %cst {fastmathFlags = #llvm.fastmath<none>} : f64
// CHECK-NEXT:      %4 = arith.select %3, %c1_i8, %c2_i8 {fastmathFlags = #llvm.fastmath<none>} : i8
// CHECK-NEXT:      affine.store %4, %1[%arg2] {alignment = 1 : i64, ordering = 0 : i64, tbaa = [#tbaa_tag]} : memref<?xi8, 1>
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }
