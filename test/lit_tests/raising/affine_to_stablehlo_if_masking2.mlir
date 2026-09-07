// RUN: enzymexlamlir-opt %s --raise-affine-to-stablehlo --canonicalize | FileCheck %s

#map = affine_map<(d0) -> (d0 * 4)>
#map1 = affine_map<(d0) -> (d0 * 4 + 1)>
#map2 = affine_map<(d0) -> (d0 * 4 + 2)>
#map3 = affine_map<(d0) -> (d0 * 4 + 3)>
#map4 = affine_map<()[s0] -> (s0 * 4 - 4)>
#map5 = affine_map<()[s0] -> (s0 * 4 - 3)>
#map6 = affine_map<()[s0] -> (s0 * 4 - 2)>
#map7 = affine_map<()[s0] -> (s0 * 4 - 1)>
#tbaa_root = #llvm.tbaa_root<id = "custom_tbaa">
#tbaa_type_desc = #llvm.tbaa_type_desc<id = "custom_tbaa_addrspace(1)", members = {<#tbaa_root, 0>}>
#tbaa_tag = #llvm.tbaa_tag<base_type = #tbaa_type_desc, access_type = #tbaa_type_desc, offset = 0>
module @reactant_dot_gat... attributes {llvm.data_layout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64", mhlo.num_partitions = 1 : i64, mhlo.num_replicas = 1 : i64} {
  llvm.module_flags [#llvm.mlir.module_flag<warning, "Dwarf Version", 2 : i32>, #llvm.mlir.module_flag<warning, "Debug Info Version", 3 : i32>, #llvm.mlir.module_flag<override, "nvvm-reflect-ftz", 0 : i32>]
  func.func @main(%arg0: tensor<2x4x1xf64> {enzymexla.memory_effects = [], tf.aliasing_output = 0 : i32}, %arg1: tensor<2x3x4x1xf64> {enzymexla.memory_effects = ["read", "write", "allocate", "free"], tf.aliasing_output = 1 : i32}, %arg2: tensor<2x4xf64> {enzymexla.memory_effects = ["read", "write", "allocate", "free"], tf.aliasing_output = 2 : i32}, %arg3: tensor<2xi32> {enzymexla.memory_effects = ["read", "write", "allocate", "free"], tf.aliasing_output = 3 : i32}, %arg4: tensor<2xi32> {enzymexla.memory_effects = ["read", "write", "allocate", "free"], tf.aliasing_output = 4 : i32}, %arg5: tensor<2xi32> {enzymexla.memory_effects = ["read", "write", "allocate", "free"], tf.aliasing_output = 5 : i32}) -> (tensor<2x4x1xf64>, tensor<2x3x4x1xf64>, tensor<2x4xf64>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) attributes {enzymexla.memory_effects = ["read", "write", "allocate", "free"]} {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<2x4x1xf64>
    %0 = enzymexla.jit_call @"##call__Z22gpu_dot_gather_kernel_16CompilerMetadataI11DynamicSize12DynamicCheckv16CartesianIndicesILi3E5TupleI5OneToI5Int64ES6_S6_EE7NDRangeILi3ES0_S0_S8_S8_EE13CuTracedArrayI7Float64Li3ELi1E9_1__4__2_ESC_ISD_Li4ELi1E12_1__4__3__2_ESC_ISD_Li2ELi1E6_4__2_ESC_I5Int32Li1ELi1E4_2__ESI_SI_7Workset#283$par0" (%cst, %arg1, %arg2, %arg3, %arg4, %arg5) {output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>], xla_side_effect_free} : (tensor<2x4x1xf64>, tensor<2x3x4x1xf64>, tensor<2x4xf64>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<2x4x1xf64>
    return %0, %arg1, %arg2, %arg3, %arg4, %arg5 : tensor<2x4x1xf64>, tensor<2x3x4x1xf64>, tensor<2x4xf64>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>
  }
  func.func private @"##call__Z22gpu_dot_gather_kernel_16CompilerMetadataI11DynamicSize12DynamicCheckv16CartesianIndicesILi3E5TupleI5OneToI5Int64ES6_S6_EE7NDRangeILi3ES0_S0_S8_S8_EE13CuTracedArrayI7Float64Li3ELi1E9_1__4__2_ESC_ISD_Li4ELi1E12_1__4__3__2_ESC_ISD_Li2ELi1E6_4__2_ESC_I5Int32Li1ELi1E4_2__ESI_SI_7Workset#283$par0"(%arg0: memref<2x4x1xf64, 1>, %arg1: memref<2x3x4x1xf64, 1>, %arg2: memref<2x4xf64, 1>, %arg3: memref<2xi32, 1>, %arg4: memref<2xi32, 1>, %arg5: memref<2xi32, 1>) attributes {enzymexla.memory_effects = ["read", "write", "allocate", "free"]} {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c3 = arith.constant 3 : index
    %c2 = arith.constant 2 : index
    %c-1 = arith.constant -1 : index
    %c1 = arith.constant 1 : index
    %c-1_i64 = arith.constant -1 : i64
    %c4_i64 = arith.constant 4 : i64
    %c12_i64 = arith.constant 12 : i64
    affine.parallel (%arg6, %arg7) = (0, 0) to (2, 3) {
      %0 = affine.load %arg3[%arg6] {alignment = 4 : i64, invariant, ordering = 0 : i64, tbaa = [#tbaa_tag]} : memref<2xi32, 1>
      %1 = arith.addi %arg7, %c1 : index
      %2 = arith.index_cast %1 : index to i64
      %3 = arith.extsi %0 : i32 to i64
      %4 = arith.addi %3, %c-1_i64 : i64
      %5 = arith.index_cast %0 : i32 to index
      %6 = arith.addi %5, %c-1 : index
      %7 = arith.index_cast %0 : i32 to index
      %8 = arith.addi %7, %c-1 : index
      %9 = memref.load %arg4[%6] : memref<2xi32, 1>
      %10 = arith.extsi %9 : i32 to i64
      %11 = memref.load %arg5[%8] : memref<2xi32, 1>
      %12 = arith.extsi %11 : i32 to i64
      %13 = arith.cmpi sle, %2, %12 : i64
      scf.if %13 {
        %14 = arith.muli %4, %c4_i64 : i64
        %15 = arith.index_cast %14 : i64 to index
        %16 = arith.remui %15, %c4 : index
        %17 = arith.divui %15, %c4 : index
        %18 = memref.load %arg2[%17, %16] : memref<2x4xf64, 1>
        %19 = arith.addi %10, %c-1_i64 : i64
        %20 = arith.muli %19, %c12_i64 : i64
        %21 = arith.index_cast %20 : i64 to index
        %22 = affine.apply #map(%arg7)
        %23 = arith.addi %22, %21 : index
        %24 = arith.remui %23, %c4 : index
        %25 = arith.divui %23, %c4 : index
        %26 = arith.remui %25, %c3 : index
        %27 = arith.divui %25, %c3 : index
        %28 = memref.load %arg1[%27, %26, %24, %c0] : memref<2x3x4x1xf64, 1>
        %29 = arith.mulf %18, %28 {fastmathFlags = #llvm.fastmath<none>} : f64
        %30 = arith.index_cast %14 : i64 to index
        %31 = arith.addi %30, %c1 : index
        %32 = arith.remui %31, %c4 : index
        %33 = arith.divui %31, %c4 : index
        %34 = memref.load %arg2[%33, %32] : memref<2x4xf64, 1>
        %35 = arith.index_cast %20 : i64 to index
        %36 = affine.apply #map1(%arg7)
        %37 = arith.addi %36, %35 : index
        %38 = arith.remui %37, %c4 : index
        %39 = arith.divui %37, %c4 : index
        %40 = arith.remui %39, %c3 : index
        %41 = arith.divui %39, %c3 : index
        %42 = memref.load %arg1[%41, %40, %38, %c0] : memref<2x3x4x1xf64, 1>
        %43 = arith.mulf %34, %42 {fastmathFlags = #llvm.fastmath<none>} : f64
        %44 = arith.addf %29, %43 {fastmathFlags = #llvm.fastmath<none>} : f64
        %45 = arith.index_cast %14 : i64 to index
        %46 = arith.addi %45, %c2 : index
        %47 = arith.remui %46, %c4 : index
        %48 = arith.divui %46, %c4 : index
        %49 = memref.load %arg2[%48, %47] : memref<2x4xf64, 1>
        %50 = arith.index_cast %20 : i64 to index
        %51 = affine.apply #map2(%arg7)
        %52 = arith.addi %51, %50 : index
        %53 = arith.remui %52, %c4 : index
        %54 = arith.divui %52, %c4 : index
        %55 = arith.remui %54, %c3 : index
        %56 = arith.divui %54, %c3 : index
        %57 = memref.load %arg1[%56, %55, %53, %c0] : memref<2x3x4x1xf64, 1>
        %58 = arith.mulf %49, %57 {fastmathFlags = #llvm.fastmath<none>} : f64
        %59 = arith.addf %44, %58 {fastmathFlags = #llvm.fastmath<none>} : f64
        %60 = arith.index_cast %14 : i64 to index
        %61 = arith.addi %60, %c3 : index
        %62 = arith.remui %61, %c4 : index
        %63 = arith.divui %61, %c4 : index
        %64 = memref.load %arg2[%63, %62] : memref<2x4xf64, 1>
        %65 = arith.index_cast %20 : i64 to index
        %66 = affine.apply #map3(%arg7)
        %67 = arith.addi %66, %65 : index
        %68 = arith.remui %67, %c4 : index
        %69 = arith.divui %67, %c4 : index
        %70 = arith.remui %69, %c3 : index
        %71 = arith.divui %69, %c3 : index
        %72 = memref.load %arg1[%71, %70, %68, %c0] : memref<2x3x4x1xf64, 1>
        %73 = arith.mulf %64, %72 {fastmathFlags = #llvm.fastmath<none>} : f64
        %74 = arith.addf %59, %73 {fastmathFlags = #llvm.fastmath<none>} : f64
        %75 = affine.load %arg0[%arg6, %arg7 + 1, 0] : memref<2x4x1xf64, 1>
        %76 = arith.addf %75, %74 {fastmathFlags = #llvm.fastmath<none>} : f64
        affine.store %76, %arg0[%arg6, %arg7 + 1, 0] : memref<2x4x1xf64, 1>
      }
    }
    return
  }
}

// CHECK:  module @reactant_dot_gat... attributes {llvm.data_layout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64", mhlo.num_partitions = 1 : i64, mhlo.num_replicas = 1 : i64} {
// CHECK-NEXT:  llvm.module_flags [#llvm.mlir.module_flag<warning, "Dwarf Version", 2 : i32>, #llvm.mlir.module_flag<warning, "Debug Info Version", 3 : i32>, #llvm.mlir.module_flag<override, "nvvm-reflect-ftz", 0 : i32>]
// CHECK-NEXT:  func.func @main(%[[a1:.+]]: tensor<2x4x1xf64> {enzymexla.memory_effects = [], tf.aliasing_output = 0 : i32}, %[[a2:.+]]: tensor<2x3x4x1xf64> {enzymexla.memory_effects = ["read", "write", "allocate", "free"], tf.aliasing_output = 1 : i32}, %[[a3:.+]]: tensor<2x4xf64> {enzymexla.memory_effects = ["read", "write", "allocate", "free"], tf.aliasing_output = 2 : i32}, %[[a4:.+]]: tensor<2xi32> {enzymexla.memory_effects = ["read", "write", "allocate", "free"], tf.aliasing_output = 3 : i32}, %[[a5:.+]]: tensor<2xi32> {enzymexla.memory_effects = ["read", "write", "allocate", "free"], tf.aliasing_output = 4 : i32}, %[[a6:.+]]: tensor<2xi32> {enzymexla.memory_effects = ["read", "write", "allocate", "free"], tf.aliasing_output = 5 : i32}) -> (tensor<2x4x1xf64>, tensor<2x3x4x1xf64>, tensor<2x4xf64>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) attributes {enzymexla.memory_effects = ["read", "write", "allocate", "free"]} {
// CHECK-NEXT:    %[[a7:.+]] = stablehlo.constant dense<0.000000e+00> : tensor<2x4x1xf64>
// CHECK-NEXT:    %[[a8:.+]]:6 = call @"##call__Z22gpu_dot_gather_kernel_16CompilerMetadataI11DynamicSize12DynamicCheckv16CartesianIndicesILi3E5TupleI5OneToI5Int64ES6_S6_EE7NDRangeILi3ES0_S0_S8_S8_EE13CuTracedArrayI7Float64Li3ELi1E9_1__4__2_ESC_ISD_Li4ELi1E12_1__4__3__2_ESC_ISD_Li2ELi1E6_4__2_ESC_I5Int32Li1ELi1E4_2__ESI_SI_7Workset#283$par0_raised"(%[[a7]], %[[a2]], %[[a3]], %[[a4]], %[[a5]], %[[a6]]) : (tensor<2x4x1xf64>, tensor<2x3x4x1xf64>, tensor<2x4xf64>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> (tensor<2x4x1xf64>, tensor<2x3x4x1xf64>, tensor<2x4xf64>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>)
// CHECK-NEXT:    return %[[a8]]#0, %[[a2]], %[[a3]], %[[a4]], %[[a5]], %[[a6]] : tensor<2x4x1xf64>, tensor<2x3x4x1xf64>, tensor<2x4xf64>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>
// CHECK-NEXT:  }
// CHECK-NEXT:  func.func private @"##call__Z22gpu_dot_gather_kernel_16CompilerMetadataI11DynamicSize12DynamicCheckv16CartesianIndicesILi3E5TupleI5OneToI5Int64ES6_S6_EE7NDRangeILi3ES0_S0_S8_S8_EE13CuTracedArrayI7Float64Li3ELi1E9_1__4__2_ESC_ISD_Li4ELi1E12_1__4__3__2_ESC_ISD_Li2ELi1E6_4__2_ESC_I5Int32Li1ELi1E4_2__ESI_SI_7Workset#283$par0_raised"(%[[a1]]: tensor<2x4x1xf64>, %[[a2]]: tensor<2x3x4x1xf64>, %[[a3]]: tensor<2x4xf64>, %[[a4]]: tensor<2xi32>, %[[a5]]: tensor<2xi32>, %[[a6]]: tensor<2xi32>) -> (tensor<2x4x1xf64>, tensor<2x3x4x1xf64>, tensor<2x4xf64>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) {
// CHECK-NEXT:    %[[a9:.+]] = stablehlo.constant dense<1> : tensor<3xi64>
// CHECK-NEXT:    %[[a10:.+]] = stablehlo.constant dense<0> : tensor<3xi64>
// CHECK-NEXT:    %[[a11:.+]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %[[a12:.+]] = stablehlo.constant dense<4> : tensor<i64>
// CHECK-NEXT:    %[[a13:.+]] = stablehlo.constant dense<3> : tensor<i64>
// CHECK-NEXT:    %[[a14:.+]] = stablehlo.constant dense<2> : tensor<i64>
// CHECK-NEXT:    %[[a15:.+]] = stablehlo.constant dense<-1> : tensor<i64>
// CHECK-NEXT:    %[[a16:.+]] = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:    %[[a17:.+]] = stablehlo.constant dense<12> : tensor<i64>
// CHECK-NEXT:    %[[a8]] = stablehlo.iota dim = 0 : tensor<3xi64>
// CHECK-NEXT:    %[[a18:.+]] = stablehlo.add %[[a8]], %[[a10]] : tensor<3xi64>
// CHECK-NEXT:    %[[a19:.+]] = stablehlo.multiply %[[a18]], %[[a9]] : tensor<3xi64>
// CHECK-NEXT:    %[[a20:.+]] = stablehlo.broadcast_in_dim %[[a16]], dims = [] : (tensor<i64>) -> tensor<3xi64>
// CHECK-NEXT:    %[[a21:.+]] = arith.addi %[[a19]], %[[a20]] : tensor<3xi64>
// CHECK-NEXT:    %[[a22:.+]] = arith.extsi %[[a4]] : tensor<2xi32> to tensor<2xi64>
// CHECK-NEXT:    %[[a23:.+]] = stablehlo.broadcast_in_dim %[[a15]], dims = [] : (tensor<i64>) -> tensor<2xi64>
// CHECK-NEXT:    %[[a24:.+]] = arith.addi %[[a22]], %[[a23]] : tensor<2xi64>
// CHECK-NEXT:    %[[a25:.+]] = stablehlo.convert %[[a4]] : (tensor<2xi32>) -> tensor<2xi64>
// CHECK-NEXT:    %[[a26:.+]] = stablehlo.broadcast_in_dim %[[a15]], dims = [] : (tensor<i64>) -> tensor<2xi64>
// CHECK-NEXT:    %[[a27:.+]] = arith.addi %[[a25]], %[[a26]] : tensor<2xi64>
// CHECK-NEXT:    %[[a28:.+]] = stablehlo.convert %[[a4]] : (tensor<2xi32>) -> tensor<2xi64>
// CHECK-NEXT:    %[[a29:.+]] = stablehlo.broadcast_in_dim %[[a15]], dims = [] : (tensor<i64>) -> tensor<2xi64>
// CHECK-NEXT:    %[[a30:.+]] = arith.addi %[[a28]], %[[a29]] : tensor<2xi64>
// CHECK-NEXT:    %[[a31:.+]] = stablehlo.reshape %[[a27]] : (tensor<2xi64>) -> tensor<2x1xi64>
// CHECK-NEXT:    %[[a32:.+]] = "stablehlo.gather"(%[[a5]], %[[a31]]) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<2xi32>, tensor<2x1xi64>) -> tensor<2xi32>
// CHECK-NEXT:    %[[a33:.+]] = arith.extsi %[[a32]] : tensor<2xi32> to tensor<2xi64>
// CHECK-NEXT:    %[[a34:.+]] = stablehlo.reshape %[[a30]] : (tensor<2xi64>) -> tensor<2x1xi64>
// CHECK-NEXT:    %[[a35:.+]] = "stablehlo.gather"(%[[a6]], %[[a34]]) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<2xi32>, tensor<2x1xi64>) -> tensor<2xi32>
// CHECK-NEXT:    %[[a36:.+]] = arith.extsi %[[a35]] : tensor<2xi32> to tensor<2xi64>
// CHECK-NEXT:    %[[a37:.+]] = stablehlo.broadcast_in_dim %[[a21]], dims = [0] : (tensor<3xi64>) -> tensor<3x2xi64>
// CHECK-NEXT:    %[[a38:.+]] = stablehlo.broadcast_in_dim %[[a36]], dims = [1] : (tensor<2xi64>) -> tensor<3x2xi64>
// CHECK-NEXT:    %[[a39:.+]] = arith.cmpi sle, %[[a37]], %[[a38]] : tensor<3x2xi64>
// CHECK-NEXT:    %[[a40:.+]] = stablehlo.broadcast_in_dim %[[a12]], dims = [] : (tensor<i64>) -> tensor<2xi64>
// CHECK-NEXT:    %[[a41:.+]] = arith.muli %[[a24]], %[[a40]] : tensor<2xi64>
// CHECK-NEXT:    %[[a42:.+]] = stablehlo.broadcast_in_dim %[[a12]], dims = [] : (tensor<i64>) -> tensor<2xi64>
// CHECK-NEXT:    %[[a43:.+]] = arith.remui %[[a41]], %[[a42]] : tensor<2xi64>
// CHECK-NEXT:    %[[a44:.+]] = stablehlo.broadcast_in_dim %[[a12]], dims = [] : (tensor<i64>) -> tensor<2xi64>
// CHECK-NEXT:    %[[a45:.+]] = arith.divui %[[a41]], %[[a44]] : tensor<2xi64>
// CHECK-NEXT:    %[[a46:.+]] = stablehlo.reshape %[[a45]] : (tensor<2xi64>) -> tensor<2x1xi64>
// CHECK-NEXT:    %[[a47:.+]] = stablehlo.broadcast_in_dim %[[a43]], dims = [0] : (tensor<2xi64>) -> tensor<2x1xi64>
// CHECK-NEXT:    %[[a48:.+]] = stablehlo.concatenate %[[a46]], %[[a47]], dim = 1 : (tensor<2x1xi64>, tensor<2x1xi64>) -> tensor<2x2xi64>
// CHECK-NEXT:    %[[a49:.+]] = "stablehlo.gather"(%[[a3]], %[[a48]]) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1>}> : (tensor<2x4xf64>, tensor<2x2xi64>) -> tensor<2xf64>
// CHECK-NEXT:    %[[a50:.+]] = stablehlo.broadcast_in_dim %[[a15]], dims = [] : (tensor<i64>) -> tensor<2xi64>
// CHECK-NEXT:    %[[a51:.+]] = arith.addi %[[a33]], %[[a50]] : tensor<2xi64>
// CHECK-NEXT:    %[[a52:.+]] = stablehlo.broadcast_in_dim %[[a17]], dims = [] : (tensor<i64>) -> tensor<2xi64>
// CHECK-NEXT:    %[[a53:.+]] = arith.muli %[[a51]], %[[a52]] : tensor<2xi64>
// CHECK-NEXT:    %[[a54:.+]] = stablehlo.broadcast_in_dim %[[a12]], dims = [] : (tensor<i64>) -> tensor<3xi64>
// CHECK-NEXT:    %[[a55:.+]] = stablehlo.multiply %[[a19]], %[[a54]] : tensor<3xi64>
// CHECK-NEXT:    %[[a56:.+]] = stablehlo.broadcast_in_dim %[[a55]], dims = [0] : (tensor<3xi64>) -> tensor<3x2xi64>
// CHECK-NEXT:    %[[a57:.+]] = stablehlo.broadcast_in_dim %[[a53]], dims = [1] : (tensor<2xi64>) -> tensor<3x2xi64>
// CHECK-NEXT:    %[[a58:.+]] = arith.addi %[[a56]], %[[a57]] : tensor<3x2xi64>
// CHECK-NEXT:    %[[a59:.+]] = stablehlo.broadcast_in_dim %[[a12]], dims = [] : (tensor<i64>) -> tensor<3x2xi64>
// CHECK-NEXT:    %[[a60:.+]] = arith.remui %[[a58]], %[[a59]] : tensor<3x2xi64>
// CHECK-NEXT:    %[[a61:.+]] = stablehlo.broadcast_in_dim %[[a12]], dims = [] : (tensor<i64>) -> tensor<3x2xi64>
// CHECK-NEXT:    %[[a62:.+]] = arith.divui %[[a58]], %[[a61]] : tensor<3x2xi64>
// CHECK-NEXT:    %[[a63:.+]] = stablehlo.broadcast_in_dim %[[a13]], dims = [] : (tensor<i64>) -> tensor<3x2xi64>
// CHECK-NEXT:    %[[a64:.+]] = arith.remui %[[a62]], %[[a63]] : tensor<3x2xi64>
// CHECK-NEXT:    %[[a65:.+]] = stablehlo.broadcast_in_dim %[[a13]], dims = [] : (tensor<i64>) -> tensor<3x2xi64>
// CHECK-NEXT:    %[[a66:.+]] = arith.divui %[[a62]], %[[a65]] : tensor<3x2xi64>
// CHECK-NEXT:    %[[a67:.+]] = stablehlo.reshape %[[a66]] : (tensor<3x2xi64>) -> tensor<3x2x1xi64>
// CHECK-NEXT:    %[[a68:.+]] = stablehlo.broadcast_in_dim %[[a64]], dims = [0, 1] : (tensor<3x2xi64>) -> tensor<3x2x1xi64>
// CHECK-NEXT:    %[[a69:.+]] = stablehlo.concatenate %[[a67]], %[[a68]], dim = 2 : (tensor<3x2x1xi64>, tensor<3x2x1xi64>) -> tensor<3x2x2xi64>
// CHECK-NEXT:    %[[a70:.+]] = stablehlo.broadcast_in_dim %[[a60]], dims = [0, 1] : (tensor<3x2xi64>) -> tensor<3x2x1xi64>
// CHECK-NEXT:    %[[a71:.+]] = stablehlo.concatenate %[[a69]], %[[a70]], dim = 2 : (tensor<3x2x2xi64>, tensor<3x2x1xi64>) -> tensor<3x2x3xi64>
// CHECK-NEXT:    %[[a72:.+]] = stablehlo.broadcast_in_dim %[[a11]], dims = [] : (tensor<i64>) -> tensor<3x2x1xi64>
// CHECK-NEXT:    %[[a73:.+]] = stablehlo.concatenate %[[a71]], %[[a72]], dim = 2 : (tensor<3x2x3xi64>, tensor<3x2x1xi64>) -> tensor<3x2x4xi64>
// CHECK-NEXT:    %[[a74:.+]] = "stablehlo.gather"(%[[a2]], %[[a73]]) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1, 2, 3], start_index_map = [0, 1, 2, 3], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 1, 1>}> : (tensor<2x3x4x1xf64>, tensor<3x2x4xi64>) -> tensor<3x2xf64>
// CHECK-NEXT:    %[[a75:.+]] = stablehlo.broadcast_in_dim %[[a49]], dims = [0] : (tensor<2xf64>) -> tensor<2x3xf64>
// CHECK-NEXT:    %[[a76:.+]] = stablehlo.broadcast_in_dim %[[a74]], dims = [1, 0] : (tensor<3x2xf64>) -> tensor<2x3xf64>
// CHECK-NEXT:    %[[a77:.+]] = arith.mulf %[[a75]], %[[a76]] {fastmathFlags = #llvm.fastmath<none>} : tensor<2x3xf64>
// CHECK-NEXT:    %[[a78:.+]] = stablehlo.broadcast_in_dim %[[a16]], dims = [] : (tensor<i64>) -> tensor<2xi64>
// CHECK-NEXT:    %[[a79:.+]] = arith.addi %[[a41]], %[[a78]] : tensor<2xi64>
// CHECK-NEXT:    %[[a80:.+]] = stablehlo.broadcast_in_dim %[[a12]], dims = [] : (tensor<i64>) -> tensor<2xi64>
// CHECK-NEXT:    %[[a81:.+]] = arith.remui %[[a79]], %[[a80]] : tensor<2xi64>
// CHECK-NEXT:    %[[a82:.+]] = stablehlo.broadcast_in_dim %[[a12]], dims = [] : (tensor<i64>) -> tensor<2xi64>
// CHECK-NEXT:    %[[a83:.+]] = arith.divui %[[a79]], %[[a82]] : tensor<2xi64>
// CHECK-NEXT:    %[[a84:.+]] = stablehlo.reshape %[[a83]] : (tensor<2xi64>) -> tensor<2x1xi64>
// CHECK-NEXT:    %[[a85:.+]] = stablehlo.broadcast_in_dim %[[a81]], dims = [0] : (tensor<2xi64>) -> tensor<2x1xi64>
// CHECK-NEXT:    %[[a86:.+]] = stablehlo.concatenate %[[a84]], %[[a85]], dim = 1 : (tensor<2x1xi64>, tensor<2x1xi64>) -> tensor<2x2xi64>
// CHECK-NEXT:    %[[a87:.+]] = "stablehlo.gather"(%[[a3]], %[[a86]]) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1>}> : (tensor<2x4xf64>, tensor<2x2xi64>) -> tensor<2xf64>
// CHECK-NEXT:    %[[a88:.+]] = stablehlo.broadcast_in_dim %[[a12]], dims = [] : (tensor<i64>) -> tensor<3xi64>
// CHECK-NEXT:    %[[a89:.+]] = stablehlo.multiply %[[a19]], %[[a88]] : tensor<3xi64>
// CHECK-NEXT:    %[[a90:.+]] = stablehlo.broadcast_in_dim %[[a16]], dims = [] : (tensor<i64>) -> tensor<3xi64>
// CHECK-NEXT:    %[[a91:.+]] = stablehlo.add %[[a89]], %[[a90]] : tensor<3xi64>
// CHECK-NEXT:    %[[a92:.+]] = stablehlo.broadcast_in_dim %[[a91]], dims = [0] : (tensor<3xi64>) -> tensor<3x2xi64>
// CHECK-NEXT:    %[[a93:.+]] = stablehlo.broadcast_in_dim %[[a53]], dims = [1] : (tensor<2xi64>) -> tensor<3x2xi64>
// CHECK-NEXT:    %[[a94:.+]] = arith.addi %[[a92]], %[[a93]] : tensor<3x2xi64>
// CHECK-NEXT:    %[[a95:.+]] = stablehlo.broadcast_in_dim %[[a12]], dims = [] : (tensor<i64>) -> tensor<3x2xi64>
// CHECK-NEXT:    %[[a96:.+]] = arith.remui %[[a94]], %[[a95]] : tensor<3x2xi64>
// CHECK-NEXT:    %[[a97:.+]] = stablehlo.broadcast_in_dim %[[a12]], dims = [] : (tensor<i64>) -> tensor<3x2xi64>
// CHECK-NEXT:    %[[a98:.+]] = arith.divui %[[a94]], %[[a97]] : tensor<3x2xi64>
// CHECK-NEXT:    %[[a99:.+]] = stablehlo.broadcast_in_dim %[[a13]], dims = [] : (tensor<i64>) -> tensor<3x2xi64>
// CHECK-NEXT:    %[[a100:.+]] = arith.remui %[[a98]], %[[a99]] : tensor<3x2xi64>
// CHECK-NEXT:    %[[a101:.+]] = stablehlo.broadcast_in_dim %[[a13]], dims = [] : (tensor<i64>) -> tensor<3x2xi64>
// CHECK-NEXT:    %[[a102:.+]] = arith.divui %[[a98]], %[[a101]] : tensor<3x2xi64>
// CHECK-NEXT:    %[[a103:.+]] = stablehlo.reshape %[[a102]] : (tensor<3x2xi64>) -> tensor<3x2x1xi64>
// CHECK-NEXT:    %[[a104:.+]] = stablehlo.broadcast_in_dim %[[a100]], dims = [0, 1] : (tensor<3x2xi64>) -> tensor<3x2x1xi64>
// CHECK-NEXT:    %[[a105:.+]] = stablehlo.concatenate %[[a103]], %[[a104]], dim = 2 : (tensor<3x2x1xi64>, tensor<3x2x1xi64>) -> tensor<3x2x2xi64>
// CHECK-NEXT:    %[[a106:.+]] = stablehlo.broadcast_in_dim %[[a96]], dims = [0, 1] : (tensor<3x2xi64>) -> tensor<3x2x1xi64>
// CHECK-NEXT:    %[[a107:.+]] = stablehlo.concatenate %[[a105]], %[[a106]], dim = 2 : (tensor<3x2x2xi64>, tensor<3x2x1xi64>) -> tensor<3x2x3xi64>
// CHECK-NEXT:    %[[a108:.+]] = stablehlo.broadcast_in_dim %[[a11]], dims = [] : (tensor<i64>) -> tensor<3x2x1xi64>
// CHECK-NEXT:    %[[a109:.+]] = stablehlo.concatenate %[[a107]], %[[a108]], dim = 2 : (tensor<3x2x3xi64>, tensor<3x2x1xi64>) -> tensor<3x2x4xi64>
// CHECK-NEXT:    %[[a110:.+]] = "stablehlo.gather"(%[[a2]], %[[a109]]) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1, 2, 3], start_index_map = [0, 1, 2, 3], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 1, 1>}> : (tensor<2x3x4x1xf64>, tensor<3x2x4xi64>) -> tensor<3x2xf64>
// CHECK-NEXT:    %[[a111:.+]] = stablehlo.broadcast_in_dim %[[a87]], dims = [0] : (tensor<2xf64>) -> tensor<2x3xf64>
// CHECK-NEXT:    %[[a112:.+]] = stablehlo.broadcast_in_dim %[[a110]], dims = [1, 0] : (tensor<3x2xf64>) -> tensor<2x3xf64>
// CHECK-NEXT:    %[[a113:.+]] = arith.mulf %[[a111]], %[[a112]] {fastmathFlags = #llvm.fastmath<none>} : tensor<2x3xf64>
// CHECK-NEXT:    %[[a114:.+]] = arith.addf %[[a77]], %[[a113]] {fastmathFlags = #llvm.fastmath<none>} : tensor<2x3xf64>
// CHECK-NEXT:    %[[a115:.+]] = stablehlo.broadcast_in_dim %[[a14]], dims = [] : (tensor<i64>) -> tensor<2xi64>
// CHECK-NEXT:    %[[a116:.+]] = arith.addi %[[a41]], %[[a115]] : tensor<2xi64>
// CHECK-NEXT:    %[[a117:.+]] = stablehlo.broadcast_in_dim %[[a12]], dims = [] : (tensor<i64>) -> tensor<2xi64>
// CHECK-NEXT:    %[[a118:.+]] = arith.remui %[[a116]], %[[a117]] : tensor<2xi64>
// CHECK-NEXT:    %[[a119:.+]] = stablehlo.broadcast_in_dim %[[a12]], dims = [] : (tensor<i64>) -> tensor<2xi64>
// CHECK-NEXT:    %[[a120:.+]] = arith.divui %[[a116]], %[[a119]] : tensor<2xi64>
// CHECK-NEXT:    %[[a121:.+]] = stablehlo.reshape %[[a120]] : (tensor<2xi64>) -> tensor<2x1xi64>
// CHECK-NEXT:    %[[a122:.+]] = stablehlo.broadcast_in_dim %[[a118]], dims = [0] : (tensor<2xi64>) -> tensor<2x1xi64>
// CHECK-NEXT:    %[[a123:.+]] = stablehlo.concatenate %[[a121]], %[[a122]], dim = 1 : (tensor<2x1xi64>, tensor<2x1xi64>) -> tensor<2x2xi64>
// CHECK-NEXT:    %[[a124:.+]] = "stablehlo.gather"(%[[a3]], %[[a123]]) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1>}> : (tensor<2x4xf64>, tensor<2x2xi64>) -> tensor<2xf64>
// CHECK-NEXT:    %[[a125:.+]] = stablehlo.broadcast_in_dim %[[a12]], dims = [] : (tensor<i64>) -> tensor<3xi64>
// CHECK-NEXT:    %[[a126:.+]] = stablehlo.multiply %[[a19]], %[[a125]] : tensor<3xi64>
// CHECK-NEXT:    %[[a127:.+]] = stablehlo.broadcast_in_dim %[[a14]], dims = [] : (tensor<i64>) -> tensor<3xi64>
// CHECK-NEXT:    %[[a128:.+]] = stablehlo.add %[[a126]], %[[a127]] : tensor<3xi64>
// CHECK-NEXT:    %[[a129:.+]] = stablehlo.broadcast_in_dim %[[a128]], dims = [0] : (tensor<3xi64>) -> tensor<3x2xi64>
// CHECK-NEXT:    %[[a130:.+]] = stablehlo.broadcast_in_dim %[[a53]], dims = [1] : (tensor<2xi64>) -> tensor<3x2xi64>
// CHECK-NEXT:    %[[a131:.+]] = arith.addi %[[a129]], %[[a130]] : tensor<3x2xi64>
// CHECK-NEXT:    %[[a132:.+]] = stablehlo.broadcast_in_dim %[[a12]], dims = [] : (tensor<i64>) -> tensor<3x2xi64>
// CHECK-NEXT:    %[[a133:.+]] = arith.remui %[[a131]], %[[a132]] : tensor<3x2xi64>
// CHECK-NEXT:    %[[a134:.+]] = stablehlo.broadcast_in_dim %[[a12]], dims = [] : (tensor<i64>) -> tensor<3x2xi64>
// CHECK-NEXT:    %[[a135:.+]] = arith.divui %[[a131]], %[[a134]] : tensor<3x2xi64>
// CHECK-NEXT:    %[[a136:.+]] = stablehlo.broadcast_in_dim %[[a13]], dims = [] : (tensor<i64>) -> tensor<3x2xi64>
// CHECK-NEXT:    %[[a137:.+]] = arith.remui %[[a135]], %[[a136]] : tensor<3x2xi64>
// CHECK-NEXT:    %[[a138:.+]] = stablehlo.broadcast_in_dim %[[a13]], dims = [] : (tensor<i64>) -> tensor<3x2xi64>
// CHECK-NEXT:    %[[a139:.+]] = arith.divui %[[a135]], %[[a138]] : tensor<3x2xi64>
// CHECK-NEXT:    %[[a140:.+]] = stablehlo.reshape %[[a139]] : (tensor<3x2xi64>) -> tensor<3x2x1xi64>
// CHECK-NEXT:    %[[a141:.+]] = stablehlo.broadcast_in_dim %[[a137]], dims = [0, 1] : (tensor<3x2xi64>) -> tensor<3x2x1xi64>
// CHECK-NEXT:    %[[a142:.+]] = stablehlo.concatenate %[[a140]], %[[a141]], dim = 2 : (tensor<3x2x1xi64>, tensor<3x2x1xi64>) -> tensor<3x2x2xi64>
// CHECK-NEXT:    %[[a143:.+]] = stablehlo.broadcast_in_dim %[[a133]], dims = [0, 1] : (tensor<3x2xi64>) -> tensor<3x2x1xi64>
// CHECK-NEXT:    %[[a144:.+]] = stablehlo.concatenate %[[a142]], %[[a143]], dim = 2 : (tensor<3x2x2xi64>, tensor<3x2x1xi64>) -> tensor<3x2x3xi64>
// CHECK-NEXT:    %[[a145:.+]] = stablehlo.broadcast_in_dim %[[a11]], dims = [] : (tensor<i64>) -> tensor<3x2x1xi64>
// CHECK-NEXT:    %[[a146:.+]] = stablehlo.concatenate %[[a144]], %[[a145]], dim = 2 : (tensor<3x2x3xi64>, tensor<3x2x1xi64>) -> tensor<3x2x4xi64>
// CHECK-NEXT:    %[[a147:.+]] = "stablehlo.gather"(%[[a2]], %[[a146]]) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1, 2, 3], start_index_map = [0, 1, 2, 3], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 1, 1>}> : (tensor<2x3x4x1xf64>, tensor<3x2x4xi64>) -> tensor<3x2xf64>
// CHECK-NEXT:    %[[a148:.+]] = stablehlo.broadcast_in_dim %[[a124]], dims = [0] : (tensor<2xf64>) -> tensor<2x3xf64>
// CHECK-NEXT:    %[[a149:.+]] = stablehlo.broadcast_in_dim %[[a147]], dims = [1, 0] : (tensor<3x2xf64>) -> tensor<2x3xf64>
// CHECK-NEXT:    %[[a150:.+]] = arith.mulf %[[a148]], %[[a149]] {fastmathFlags = #llvm.fastmath<none>} : tensor<2x3xf64>
// CHECK-NEXT:    %[[a151:.+]] = arith.addf %[[a114]], %[[a150]] {fastmathFlags = #llvm.fastmath<none>} : tensor<2x3xf64>
// CHECK-NEXT:    %[[a152:.+]] = stablehlo.broadcast_in_dim %[[a13]], dims = [] : (tensor<i64>) -> tensor<2xi64>
// CHECK-NEXT:    %[[a153:.+]] = arith.addi %[[a41]], %[[a152]] : tensor<2xi64>
// CHECK-NEXT:    %[[a154:.+]] = stablehlo.broadcast_in_dim %[[a12]], dims = [] : (tensor<i64>) -> tensor<2xi64>
// CHECK-NEXT:    %[[a155:.+]] = arith.remui %[[a153]], %[[a154]] : tensor<2xi64>
// CHECK-NEXT:    %[[a156:.+]] = stablehlo.broadcast_in_dim %[[a12]], dims = [] : (tensor<i64>) -> tensor<2xi64>
// CHECK-NEXT:    %[[a157:.+]] = arith.divui %[[a153]], %[[a156]] : tensor<2xi64>
// CHECK-NEXT:    %[[a158:.+]] = stablehlo.reshape %[[a157]] : (tensor<2xi64>) -> tensor<2x1xi64>
// CHECK-NEXT:    %[[a159:.+]] = stablehlo.broadcast_in_dim %[[a155]], dims = [0] : (tensor<2xi64>) -> tensor<2x1xi64>
// CHECK-NEXT:    %[[a160:.+]] = stablehlo.concatenate %[[a158]], %[[a159]], dim = 1 : (tensor<2x1xi64>, tensor<2x1xi64>) -> tensor<2x2xi64>
// CHECK-NEXT:    %[[a161:.+]] = "stablehlo.gather"(%[[a3]], %[[a160]]) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1>}> : (tensor<2x4xf64>, tensor<2x2xi64>) -> tensor<2xf64>
// CHECK-NEXT:    %[[a162:.+]] = stablehlo.broadcast_in_dim %[[a12]], dims = [] : (tensor<i64>) -> tensor<3xi64>
// CHECK-NEXT:    %[[a163:.+]] = stablehlo.multiply %[[a19]], %[[a162]] : tensor<3xi64>
// CHECK-NEXT:    %[[a164:.+]] = stablehlo.broadcast_in_dim %[[a13]], dims = [] : (tensor<i64>) -> tensor<3xi64>
// CHECK-NEXT:    %[[a165:.+]] = stablehlo.add %[[a163]], %[[a164]] : tensor<3xi64>
// CHECK-NEXT:    %[[a166:.+]] = stablehlo.broadcast_in_dim %[[a165]], dims = [0] : (tensor<3xi64>) -> tensor<3x2xi64>
// CHECK-NEXT:    %[[a167:.+]] = stablehlo.broadcast_in_dim %[[a53]], dims = [1] : (tensor<2xi64>) -> tensor<3x2xi64>
// CHECK-NEXT:    %[[a168:.+]] = arith.addi %[[a166]], %[[a167]] : tensor<3x2xi64>
// CHECK-NEXT:    %[[a169:.+]] = stablehlo.broadcast_in_dim %[[a12]], dims = [] : (tensor<i64>) -> tensor<3x2xi64>
// CHECK-NEXT:    %[[a170:.+]] = arith.remui %[[a168]], %[[a169]] : tensor<3x2xi64>
// CHECK-NEXT:    %[[a171:.+]] = stablehlo.broadcast_in_dim %[[a12]], dims = [] : (tensor<i64>) -> tensor<3x2xi64>
// CHECK-NEXT:    %[[a172:.+]] = arith.divui %[[a168]], %[[a171]] : tensor<3x2xi64>
// CHECK-NEXT:    %[[a173:.+]] = stablehlo.broadcast_in_dim %[[a13]], dims = [] : (tensor<i64>) -> tensor<3x2xi64>
// CHECK-NEXT:    %[[a174:.+]] = arith.remui %[[a172]], %[[a173]] : tensor<3x2xi64>
// CHECK-NEXT:    %[[a175:.+]] = stablehlo.broadcast_in_dim %[[a13]], dims = [] : (tensor<i64>) -> tensor<3x2xi64>
// CHECK-NEXT:    %[[a176:.+]] = arith.divui %[[a172]], %[[a175]] : tensor<3x2xi64>
// CHECK-NEXT:    %[[a177:.+]] = stablehlo.reshape %[[a176]] : (tensor<3x2xi64>) -> tensor<3x2x1xi64>
// CHECK-NEXT:    %[[a178:.+]] = stablehlo.broadcast_in_dim %[[a174]], dims = [0, 1] : (tensor<3x2xi64>) -> tensor<3x2x1xi64>
// CHECK-NEXT:    %[[a179:.+]] = stablehlo.concatenate %[[a177]], %[[a178]], dim = 2 : (tensor<3x2x1xi64>, tensor<3x2x1xi64>) -> tensor<3x2x2xi64>
// CHECK-NEXT:    %[[a180:.+]] = stablehlo.broadcast_in_dim %[[a170]], dims = [0, 1] : (tensor<3x2xi64>) -> tensor<3x2x1xi64>
// CHECK-NEXT:    %[[a181:.+]] = stablehlo.concatenate %[[a179]], %[[a180]], dim = 2 : (tensor<3x2x2xi64>, tensor<3x2x1xi64>) -> tensor<3x2x3xi64>
// CHECK-NEXT:    %[[a182:.+]] = stablehlo.broadcast_in_dim %[[a11]], dims = [] : (tensor<i64>) -> tensor<3x2x1xi64>
// CHECK-NEXT:    %[[a183:.+]] = stablehlo.concatenate %[[a181]], %[[a182]], dim = 2 : (tensor<3x2x3xi64>, tensor<3x2x1xi64>) -> tensor<3x2x4xi64>
// CHECK-NEXT:    %[[a184:.+]] = "stablehlo.gather"(%[[a2]], %[[a183]]) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1, 2, 3], start_index_map = [0, 1, 2, 3], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 1, 1>}> : (tensor<2x3x4x1xf64>, tensor<3x2x4xi64>) -> tensor<3x2xf64>
// CHECK-NEXT:    %[[a185:.+]] = stablehlo.broadcast_in_dim %[[a161]], dims = [0] : (tensor<2xf64>) -> tensor<2x3xf64>
// CHECK-NEXT:    %[[a186:.+]] = stablehlo.broadcast_in_dim %[[a184]], dims = [1, 0] : (tensor<3x2xf64>) -> tensor<2x3xf64>
// CHECK-NEXT:    %[[a187:.+]] = arith.mulf %[[a185]], %[[a186]] {fastmathFlags = #llvm.fastmath<none>} : tensor<2x3xf64>
// CHECK-NEXT:    %[[a188:.+]] = arith.addf %[[a151]], %[[a187]] {fastmathFlags = #llvm.fastmath<none>} : tensor<2x3xf64>
// CHECK-NEXT:    %[[a189:.+]] = stablehlo.slice %[[a1]] [0:2, 1:4, 0:1] : (tensor<2x4x1xf64>) -> tensor<2x3x1xf64>
// CHECK-NEXT:    %[[a190:.+]] = stablehlo.reshape %[[a189]] : (tensor<2x3x1xf64>) -> tensor<2x3xf64>
// CHECK-NEXT:    %[[a191:.+]] = arith.addf %[[a190]], %[[a188]] {fastmathFlags = #llvm.fastmath<none>} : tensor<2x3xf64>
// CHECK-NEXT:    %[[a192:.+]] = stablehlo.broadcast_in_dim %[[a191]], dims = [0, 1] : (tensor<2x3xf64>) -> tensor<2x3x1xf64>
// CHECK-NEXT:    %[[a193:.+]] = stablehlo.slice %[[a1]] [0:2, 1:4, 0:1] : (tensor<2x4x1xf64>) -> tensor<2x3x1xf64>
// CHECK-NEXT:    %[[a194:.+]] = stablehlo.reshape %[[a192]] : (tensor<2x3x1xf64>) -> tensor<2x3xf64>
// CHECK-NEXT:    %[[a195:.+]] = stablehlo.reshape %[[a193]] : (tensor<2x3x1xf64>) -> tensor<2x3xf64>
// CHECK-NEXT:    %[[a196:.+]] = stablehlo.broadcast_in_dim %[[a194]], dims = [1, 0] : (tensor<2x3xf64>) -> tensor<3x2xf64>
// CHECK-NEXT:    %[[a197:.+]] = stablehlo.broadcast_in_dim %[[a195]], dims = [1, 0] : (tensor<2x3xf64>) -> tensor<3x2xf64>
// CHECK-NEXT:    %[[a198:.+]] = stablehlo.select %[[a39]], %[[a196]], %[[a197]] : tensor<3x2xi1>, tensor<3x2xf64>
// CHECK-NEXT:    %[[a199:.+]] = stablehlo.broadcast_in_dim %[[a198]], dims = [1, 0] : (tensor<3x2xf64>) -> tensor<2x3x1xf64>
// CHECK-NEXT:    %[[a200:.+]] = stablehlo.dynamic_update_slice %[[a1]], %[[a199]], %[[a11]], %[[a16]], %[[a11]] : (tensor<2x4x1xf64>, tensor<2x3x1xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<2x4x1xf64>
// CHECK-NEXT:    return %[[a200]], %[[a2]], %[[a3]], %[[a4]], %[[a5]], %[[a6]] : tensor<2x4x1xf64>, tensor<2x3x4x1xf64>, tensor<2x4xf64>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>
// CHECK-NEXT:  }
