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

// CHECK:  func.func private @"##call__Z22gpu_dot_gather_kernel_16CompilerMetadataI11DynamicSize12DynamicCheckv16CartesianIndicesILi3E5TupleI5OneToI5Int64ES6_S6_EE7NDRangeILi3ES0_S0_S8_S8_EE13CuTracedArrayI7Float64Li3ELi1E9_1__4__2_ESC_ISD_Li4ELi1E12_1__4__3__2_ESC_ISD_Li2ELi1E6_4__2_ESC_I5Int32Li1ELi1E4_2__ESI_SI_7Workset#283$par0_raised"(%arg0: tensor<2x4x1xf64>, %arg1: tensor<2x3x4x1xf64>, %arg2: tensor<2x4xf64>, %arg3: tensor<2xi32>, %arg4: tensor<2xi32>, %arg5: tensor<2xi32>) -> (tensor<2x4x1xf64>, tensor<2x3x4x1xf64>, tensor<2x4xf64>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) {
// CHECK-NEXT:    %c = stablehlo.constant dense<1> : tensor<3xi64>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<0> : tensor<3xi64>
// CHECK-NEXT:    %c_1 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %c_2 = stablehlo.constant dense<4> : tensor<i64>
// CHECK-NEXT:    %c_3 = stablehlo.constant dense<3> : tensor<i64>
// CHECK-NEXT:    %c_4 = stablehlo.constant dense<2> : tensor<i64>
// CHECK-NEXT:    %c_5 = stablehlo.constant dense<-1> : tensor<i64>
// CHECK-NEXT:    %c_6 = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:    %c_7 = stablehlo.constant dense<12> : tensor<i64>
// CHECK-NEXT:    %[[v0:.+]] = stablehlo.iota dim = 0 : tensor<3xi64>
// CHECK-NEXT:    %[[v1:.+]] = stablehlo.add %[[v0]], %c_0 : tensor<3xi64>
// CHECK-NEXT:    %[[v2:.+]] = stablehlo.multiply %[[v1]], %c : tensor<3xi64>
// CHECK-NEXT:    %[[v3:.+]] = stablehlo.reshape %arg3 : (tensor<2xi32>) -> tensor<2xi32>
// CHECK-NEXT:    %[[v4:.+]] = stablehlo.broadcast_in_dim %c_6, dims = [] : (tensor<i64>) -> tensor<3xi64>
// CHECK-NEXT:    %[[v5:.+]] = arith.addi %[[v2]], %[[v4]] : tensor<3xi64>
// CHECK-NEXT:    %[[v6:.+]] = arith.extsi %[[v3]] : tensor<2xi32> to tensor<2xi64>
// CHECK-NEXT:    %[[v7:.+]] = stablehlo.broadcast_in_dim %c_5, dims = [] : (tensor<i64>) -> tensor<2xi64>
// CHECK-NEXT:    %[[v8:.+]] = arith.addi %[[v6]], %[[v7]] : tensor<2xi64>
// CHECK-NEXT:    %[[v9:.+]] = stablehlo.convert %[[v3]] : (tensor<2xi32>) -> tensor<2xi64>
// CHECK-NEXT:    %[[v10:.+]] = stablehlo.broadcast_in_dim %c_5, dims = [] : (tensor<i64>) -> tensor<2xi64>
// CHECK-NEXT:    %[[v11:.+]] = arith.addi %[[v9]], %[[v10]] : tensor<2xi64>
// CHECK-NEXT:    %[[v12:.+]] = stablehlo.convert %[[v3]] : (tensor<2xi32>) -> tensor<2xi64>
// CHECK-NEXT:    %[[v13:.+]] = stablehlo.broadcast_in_dim %c_5, dims = [] : (tensor<i64>) -> tensor<2xi64>
// CHECK-NEXT:    %[[v14:.+]] = arith.addi %[[v12]], %[[v13]] : tensor<2xi64>
// CHECK-NEXT:    %[[v15:.+]] = stablehlo.reshape %[[v11]] : (tensor<2xi64>) -> tensor<2x1xi64>
// CHECK-NEXT:    %[[v16:.+]] = "stablehlo.gather"(%arg4, %[[v15]]) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<2xi32>, tensor<2x1xi64>) -> tensor<2xi32>
// CHECK-NEXT:    %[[v17:.+]] = arith.extsi %[[v16]] : tensor<2xi32> to tensor<2xi64>
// CHECK-NEXT:    %[[v18:.+]] = stablehlo.reshape %[[v14]] : (tensor<2xi64>) -> tensor<2x1xi64>
// CHECK-NEXT:    %[[v19:.+]] = "stablehlo.gather"(%arg5, %[[v18]]) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<2xi32>, tensor<2x1xi64>) -> tensor<2xi32>
// CHECK-NEXT:    %[[v20:.+]] = arith.extsi %[[v19]] : tensor<2xi32> to tensor<2xi64>
// CHECK-NEXT:    %[[v21:.+]] = stablehlo.broadcast_in_dim %[[v5]], dims = [0] : (tensor<3xi64>) -> tensor<3x2xi64>
// CHECK-NEXT:    %[[v22:.+]] = stablehlo.broadcast_in_dim %[[v20]], dims = [1] : (tensor<2xi64>) -> tensor<3x2xi64>
// CHECK-NEXT:    %[[v23:.+]] = arith.cmpi sle, %[[v21]], %[[v22]] : tensor<3x2xi64>
// CHECK-NEXT:    %[[v24:.+]] = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i64>) -> tensor<2xi64>
// CHECK-NEXT:    %[[v25:.+]] = arith.muli %[[v8]], %[[v24]] : tensor<2xi64>
// CHECK-NEXT:    %[[v26:.+]] = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i64>) -> tensor<2xi64>
// CHECK-NEXT:    %[[v27:.+]] = arith.remui %[[v25]], %[[v26]] : tensor<2xi64>
// CHECK-NEXT:    %[[v28:.+]] = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i64>) -> tensor<2xi64>
// CHECK-NEXT:    %[[v29:.+]] = arith.divui %[[v25]], %[[v28]] : tensor<2xi64>
// CHECK-NEXT:    %[[v30:.+]] = stablehlo.reshape %[[v29]] : (tensor<2xi64>) -> tensor<2x1xi64>
// CHECK-NEXT:    %[[v31:.+]] = stablehlo.broadcast_in_dim %[[v27]], dims = [0] : (tensor<2xi64>) -> tensor<2x1xi64>
// CHECK-NEXT:    %[[v32:.+]] = stablehlo.broadcast_in_dim %[[v30]], dims = [0, 1] : (tensor<2x1xi64>) -> tensor<2x1xi64>
// CHECK-NEXT:    %[[v33:.+]] = stablehlo.concatenate %[[v32]], %[[v31]], dim = 1 : (tensor<2x1xi64>, tensor<2x1xi64>) -> tensor<2x2xi64>
// CHECK-NEXT:    %[[v34:.+]] = "stablehlo.gather"(%arg2, %[[v33]]) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1>}> : (tensor<2x4xf64>, tensor<2x2xi64>) -> tensor<2xf64>
// CHECK-NEXT:    %[[v35:.+]] = stablehlo.broadcast_in_dim %c_5, dims = [] : (tensor<i64>) -> tensor<2xi64>
// CHECK-NEXT:    %[[v36:.+]] = arith.addi %[[v17]], %[[v35]] : tensor<2xi64>
// CHECK-NEXT:    %[[v37:.+]] = stablehlo.broadcast_in_dim %c_7, dims = [] : (tensor<i64>) -> tensor<2xi64>
// CHECK-NEXT:    %[[v38:.+]] = arith.muli %[[v36]], %[[v37]] : tensor<2xi64>
// CHECK-NEXT:    %[[v39:.+]] = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i64>) -> tensor<3xi64>
// CHECK-NEXT:    %[[v40:.+]] = stablehlo.multiply %[[v2]], %[[v39]] : tensor<3xi64>
// CHECK-NEXT:    %[[v41:.+]] = stablehlo.broadcast_in_dim %[[v40]], dims = [0] : (tensor<3xi64>) -> tensor<3x2xi64>
// CHECK-NEXT:    %[[v42:.+]] = stablehlo.broadcast_in_dim %[[v38]], dims = [1] : (tensor<2xi64>) -> tensor<3x2xi64>
// CHECK-NEXT:    %[[v43:.+]] = arith.addi %[[v41]], %[[v42]] : tensor<3x2xi64>
// CHECK-NEXT:    %[[v44:.+]] = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i64>) -> tensor<3x2xi64>
// CHECK-NEXT:    %[[v45:.+]] = arith.remui %[[v43]], %[[v44]] : tensor<3x2xi64>
// CHECK-NEXT:    %[[v46:.+]] = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i64>) -> tensor<3x2xi64>
// CHECK-NEXT:    %[[v47:.+]] = arith.divui %[[v43]], %[[v46]] : tensor<3x2xi64>
// CHECK-NEXT:    %[[v48:.+]] = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<i64>) -> tensor<3x2xi64>
// CHECK-NEXT:    %[[v49:.+]] = arith.remui %[[v47]], %[[v48]] : tensor<3x2xi64>
// CHECK-NEXT:    %[[v50:.+]] = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<i64>) -> tensor<3x2xi64>
// CHECK-NEXT:    %[[v51:.+]] = arith.divui %[[v47]], %[[v50]] : tensor<3x2xi64>
// CHECK-NEXT:    %[[v52:.+]] = stablehlo.reshape %[[v51]] : (tensor<3x2xi64>) -> tensor<3x2x1xi64>
// CHECK-NEXT:    %[[v53:.+]] = stablehlo.broadcast_in_dim %[[v49]], dims = [0, 1] : (tensor<3x2xi64>) -> tensor<3x2x1xi64>
// CHECK-NEXT:    %[[v54:.+]] = stablehlo.broadcast_in_dim %[[v52]], dims = [0, 1, 2] : (tensor<3x2x1xi64>) -> tensor<3x2x1xi64>
// CHECK-NEXT:    %[[v55:.+]] = stablehlo.concatenate %[[v54]], %[[v53]], dim = 2 : (tensor<3x2x1xi64>, tensor<3x2x1xi64>) -> tensor<3x2x2xi64>
// CHECK-NEXT:    %[[v56:.+]] = stablehlo.broadcast_in_dim %[[v45]], dims = [0, 1] : (tensor<3x2xi64>) -> tensor<3x2x1xi64>
// CHECK-NEXT:    %[[v57:.+]] = stablehlo.broadcast_in_dim %[[v55]], dims = [0, 1, 2] : (tensor<3x2x2xi64>) -> tensor<3x2x2xi64>
// CHECK-NEXT:    %[[v58:.+]] = stablehlo.concatenate %[[v57]], %[[v56]], dim = 2 : (tensor<3x2x2xi64>, tensor<3x2x1xi64>) -> tensor<3x2x3xi64>
// CHECK-NEXT:    %[[v59:.+]] = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i64>) -> tensor<3x2x1xi64>
// CHECK-NEXT:    %[[v60:.+]] = stablehlo.broadcast_in_dim %[[v58]], dims = [0, 1, 2] : (tensor<3x2x3xi64>) -> tensor<3x2x3xi64>
// CHECK-NEXT:    %[[v61:.+]] = stablehlo.concatenate %[[v60]], %[[v59]], dim = 2 : (tensor<3x2x3xi64>, tensor<3x2x1xi64>) -> tensor<3x2x4xi64>
// CHECK-NEXT:    %[[v62:.+]] = "stablehlo.gather"(%arg1, %[[v61]]) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1, 2, 3], start_index_map = [0, 1, 2, 3], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 1, 1>}> : (tensor<2x3x4x1xf64>, tensor<3x2x4xi64>) -> tensor<3x2xf64>
// CHECK-NEXT:    %[[v63:.+]] = stablehlo.broadcast_in_dim %[[v34]], dims = [0] : (tensor<2xf64>) -> tensor<2x3xf64>
// CHECK-NEXT:    %[[v64:.+]] = stablehlo.broadcast_in_dim %[[v62]], dims = [1, 0] : (tensor<3x2xf64>) -> tensor<2x3xf64>
// CHECK-NEXT:    %[[v65:.+]] = arith.mulf %[[v63]], %[[v64]] {fastmathFlags = #llvm.fastmath<none>} : tensor<2x3xf64>
// CHECK-NEXT:    %[[v66:.+]] = stablehlo.broadcast_in_dim %c_6, dims = [] : (tensor<i64>) -> tensor<2xi64>
// CHECK-NEXT:    %[[v67:.+]] = arith.addi %[[v25]], %[[v66]] : tensor<2xi64>
// CHECK-NEXT:    %[[v68:.+]] = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i64>) -> tensor<2xi64>
// CHECK-NEXT:    %[[v69:.+]] = arith.remui %[[v67]], %[[v68]] : tensor<2xi64>
// CHECK-NEXT:    %[[v70:.+]] = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i64>) -> tensor<2xi64>
// CHECK-NEXT:    %[[v71:.+]] = arith.divui %[[v67]], %[[v70]] : tensor<2xi64>
// CHECK-NEXT:    %[[v72:.+]] = stablehlo.reshape %[[v71]] : (tensor<2xi64>) -> tensor<2x1xi64>
// CHECK-NEXT:    %[[v73:.+]] = stablehlo.broadcast_in_dim %[[v69]], dims = [0] : (tensor<2xi64>) -> tensor<2x1xi64>
// CHECK-NEXT:    %[[v74:.+]] = stablehlo.broadcast_in_dim %[[v72]], dims = [0, 1] : (tensor<2x1xi64>) -> tensor<2x1xi64>
// CHECK-NEXT:    %[[v75:.+]] = stablehlo.concatenate %[[v74]], %[[v73]], dim = 1 : (tensor<2x1xi64>, tensor<2x1xi64>) -> tensor<2x2xi64>
// CHECK-NEXT:    %[[v76:.+]] = "stablehlo.gather"(%arg2, %[[v75]]) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1>}> : (tensor<2x4xf64>, tensor<2x2xi64>) -> tensor<2xf64>
// CHECK-NEXT:    %[[v77:.+]] = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i64>) -> tensor<3xi64>
// CHECK-NEXT:    %[[v78:.+]] = stablehlo.multiply %[[v2]], %[[v77]] : tensor<3xi64>
// CHECK-NEXT:    %[[v79:.+]] = stablehlo.broadcast_in_dim %c_6, dims = [] : (tensor<i64>) -> tensor<3xi64>
// CHECK-NEXT:    %[[v80:.+]] = stablehlo.add %[[v78]], %[[v79]] : tensor<3xi64>
// CHECK-NEXT:    %[[v81:.+]] = stablehlo.broadcast_in_dim %[[v80]], dims = [0] : (tensor<3xi64>) -> tensor<3x2xi64>
// CHECK-NEXT:    %[[v82:.+]] = stablehlo.broadcast_in_dim %[[v38]], dims = [1] : (tensor<2xi64>) -> tensor<3x2xi64>
// CHECK-NEXT:    %[[v83:.+]] = arith.addi %[[v81]], %[[v82]] : tensor<3x2xi64>
// CHECK-NEXT:    %[[v84:.+]] = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i64>) -> tensor<3x2xi64>
// CHECK-NEXT:    %[[v85:.+]] = arith.remui %[[v83]], %[[v84]] : tensor<3x2xi64>
// CHECK-NEXT:    %[[v86:.+]] = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i64>) -> tensor<3x2xi64>
// CHECK-NEXT:    %[[v87:.+]] = arith.divui %[[v83]], %[[v86]] : tensor<3x2xi64>
// CHECK-NEXT:    %[[v88:.+]] = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<i64>) -> tensor<3x2xi64>
// CHECK-NEXT:    %[[v89:.+]] = arith.remui %[[v87]], %[[v88]] : tensor<3x2xi64>
// CHECK-NEXT:    %[[v90:.+]] = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<i64>) -> tensor<3x2xi64>
// CHECK-NEXT:    %[[v91:.+]] = arith.divui %[[v87]], %[[v90]] : tensor<3x2xi64>
// CHECK-NEXT:    %[[v92:.+]] = stablehlo.reshape %[[v91]] : (tensor<3x2xi64>) -> tensor<3x2x1xi64>
// CHECK-NEXT:    %[[v93:.+]] = stablehlo.broadcast_in_dim %[[v89]], dims = [0, 1] : (tensor<3x2xi64>) -> tensor<3x2x1xi64>
// CHECK-NEXT:    %[[v94:.+]] = stablehlo.broadcast_in_dim %[[v92]], dims = [0, 1, 2] : (tensor<3x2x1xi64>) -> tensor<3x2x1xi64>
// CHECK-NEXT:    %[[v95:.+]] = stablehlo.concatenate %[[v94]], %[[v93]], dim = 2 : (tensor<3x2x1xi64>, tensor<3x2x1xi64>) -> tensor<3x2x2xi64>
// CHECK-NEXT:    %[[v96:.+]] = stablehlo.broadcast_in_dim %[[v85]], dims = [0, 1] : (tensor<3x2xi64>) -> tensor<3x2x1xi64>
// CHECK-NEXT:    %[[v97:.+]] = stablehlo.broadcast_in_dim %[[v95]], dims = [0, 1, 2] : (tensor<3x2x2xi64>) -> tensor<3x2x2xi64>
// CHECK-NEXT:    %[[v98:.+]] = stablehlo.concatenate %[[v97]], %[[v96]], dim = 2 : (tensor<3x2x2xi64>, tensor<3x2x1xi64>) -> tensor<3x2x3xi64>
// CHECK-NEXT:    %[[v99:.+]] = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i64>) -> tensor<3x2x1xi64>
// CHECK-NEXT:    %[[v100:.+]] = stablehlo.broadcast_in_dim %[[v98]], dims = [0, 1, 2] : (tensor<3x2x3xi64>) -> tensor<3x2x3xi64>
// CHECK-NEXT:    %[[v101:.+]] = stablehlo.concatenate %[[v100]], %[[v99]], dim = 2 : (tensor<3x2x3xi64>, tensor<3x2x1xi64>) -> tensor<3x2x4xi64>
// CHECK-NEXT:    %[[v102:.+]] = "stablehlo.gather"(%arg1, %[[v101]]) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1, 2, 3], start_index_map = [0, 1, 2, 3], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 1, 1>}> : (tensor<2x3x4x1xf64>, tensor<3x2x4xi64>) -> tensor<3x2xf64>
// CHECK-NEXT:    %[[v103:.+]] = stablehlo.broadcast_in_dim %[[v76]], dims = [0] : (tensor<2xf64>) -> tensor<2x3xf64>
// CHECK-NEXT:    %[[v104:.+]] = stablehlo.broadcast_in_dim %[[v102]], dims = [1, 0] : (tensor<3x2xf64>) -> tensor<2x3xf64>
// CHECK-NEXT:    %[[v105:.+]] = arith.mulf %[[v103]], %[[v104]] {fastmathFlags = #llvm.fastmath<none>} : tensor<2x3xf64>
// CHECK-NEXT:    %[[v106:.+]] = arith.addf %[[v65]], %[[v105]] {fastmathFlags = #llvm.fastmath<none>} : tensor<2x3xf64>
// CHECK-NEXT:    %[[v107:.+]] = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<i64>) -> tensor<2xi64>
// CHECK-NEXT:    %[[v108:.+]] = arith.addi %[[v25]], %[[v107]] : tensor<2xi64>
// CHECK-NEXT:    %[[v109:.+]] = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i64>) -> tensor<2xi64>
// CHECK-NEXT:    %[[v110:.+]] = arith.remui %[[v108]], %[[v109]] : tensor<2xi64>
// CHECK-NEXT:    %[[v111:.+]] = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i64>) -> tensor<2xi64>
// CHECK-NEXT:    %[[v112:.+]] = arith.divui %[[v108]], %[[v111]] : tensor<2xi64>
// CHECK-NEXT:    %[[v113:.+]] = stablehlo.reshape %[[v112]] : (tensor<2xi64>) -> tensor<2x1xi64>
// CHECK-NEXT:    %[[v114:.+]] = stablehlo.broadcast_in_dim %[[v110]], dims = [0] : (tensor<2xi64>) -> tensor<2x1xi64>
// CHECK-NEXT:    %[[v115:.+]] = stablehlo.broadcast_in_dim %[[v113]], dims = [0, 1] : (tensor<2x1xi64>) -> tensor<2x1xi64>
// CHECK-NEXT:    %[[v116:.+]] = stablehlo.concatenate %[[v115]], %[[v114]], dim = 1 : (tensor<2x1xi64>, tensor<2x1xi64>) -> tensor<2x2xi64>
// CHECK-NEXT:    %[[v117:.+]] = "stablehlo.gather"(%arg2, %[[v116]]) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1>}> : (tensor<2x4xf64>, tensor<2x2xi64>) -> tensor<2xf64>
// CHECK-NEXT:    %[[v118:.+]] = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i64>) -> tensor<3xi64>
// CHECK-NEXT:    %[[v119:.+]] = stablehlo.multiply %[[v2]], %[[v118]] : tensor<3xi64>
// CHECK-NEXT:    %[[v120:.+]] = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<i64>) -> tensor<3xi64>
// CHECK-NEXT:    %[[v121:.+]] = stablehlo.add %[[v119]], %[[v120]] : tensor<3xi64>
// CHECK-NEXT:    %[[v122:.+]] = stablehlo.broadcast_in_dim %[[v121]], dims = [0] : (tensor<3xi64>) -> tensor<3x2xi64>
// CHECK-NEXT:    %[[v123:.+]] = stablehlo.broadcast_in_dim %[[v38]], dims = [1] : (tensor<2xi64>) -> tensor<3x2xi64>
// CHECK-NEXT:    %[[v124:.+]] = arith.addi %[[v122]], %[[v123]] : tensor<3x2xi64>
// CHECK-NEXT:    %[[v125:.+]] = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i64>) -> tensor<3x2xi64>
// CHECK-NEXT:    %[[v126:.+]] = arith.remui %[[v124]], %[[v125]] : tensor<3x2xi64>
// CHECK-NEXT:    %[[v127:.+]] = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i64>) -> tensor<3x2xi64>
// CHECK-NEXT:    %[[v128:.+]] = arith.divui %[[v124]], %[[v127]] : tensor<3x2xi64>
// CHECK-NEXT:    %[[v129:.+]] = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<i64>) -> tensor<3x2xi64>
// CHECK-NEXT:    %[[v130:.+]] = arith.remui %[[v128]], %[[v129]] : tensor<3x2xi64>
// CHECK-NEXT:    %[[v131:.+]] = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<i64>) -> tensor<3x2xi64>
// CHECK-NEXT:    %[[v132:.+]] = arith.divui %[[v128]], %[[v131]] : tensor<3x2xi64>
// CHECK-NEXT:    %[[v133:.+]] = stablehlo.reshape %[[v132]] : (tensor<3x2xi64>) -> tensor<3x2x1xi64>
// CHECK-NEXT:    %[[v134:.+]] = stablehlo.broadcast_in_dim %[[v130]], dims = [0, 1] : (tensor<3x2xi64>) -> tensor<3x2x1xi64>
// CHECK-NEXT:    %[[v135:.+]] = stablehlo.broadcast_in_dim %[[v133]], dims = [0, 1, 2] : (tensor<3x2x1xi64>) -> tensor<3x2x1xi64>
// CHECK-NEXT:    %[[v136:.+]] = stablehlo.concatenate %[[v135]], %[[v134]], dim = 2 : (tensor<3x2x1xi64>, tensor<3x2x1xi64>) -> tensor<3x2x2xi64>
// CHECK-NEXT:    %[[v137:.+]] = stablehlo.broadcast_in_dim %[[v126]], dims = [0, 1] : (tensor<3x2xi64>) -> tensor<3x2x1xi64>
// CHECK-NEXT:    %[[v138:.+]] = stablehlo.broadcast_in_dim %[[v136]], dims = [0, 1, 2] : (tensor<3x2x2xi64>) -> tensor<3x2x2xi64>
// CHECK-NEXT:    %[[v139:.+]] = stablehlo.concatenate %[[v138]], %[[v137]], dim = 2 : (tensor<3x2x2xi64>, tensor<3x2x1xi64>) -> tensor<3x2x3xi64>
// CHECK-NEXT:    %[[v140:.+]] = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i64>) -> tensor<3x2x1xi64>
// CHECK-NEXT:    %[[v141:.+]] = stablehlo.broadcast_in_dim %[[v139]], dims = [0, 1, 2] : (tensor<3x2x3xi64>) -> tensor<3x2x3xi64>
// CHECK-NEXT:    %[[v142:.+]] = stablehlo.concatenate %[[v141]], %[[v140]], dim = 2 : (tensor<3x2x3xi64>, tensor<3x2x1xi64>) -> tensor<3x2x4xi64>
// CHECK-NEXT:    %[[v143:.+]] = "stablehlo.gather"(%arg1, %[[v142]]) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1, 2, 3], start_index_map = [0, 1, 2, 3], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 1, 1>}> : (tensor<2x3x4x1xf64>, tensor<3x2x4xi64>) -> tensor<3x2xf64>
// CHECK-NEXT:    %[[v144:.+]] = stablehlo.broadcast_in_dim %[[v117]], dims = [0] : (tensor<2xf64>) -> tensor<2x3xf64>
// CHECK-NEXT:    %[[v145:.+]] = stablehlo.broadcast_in_dim %[[v143]], dims = [1, 0] : (tensor<3x2xf64>) -> tensor<2x3xf64>
// CHECK-NEXT:    %[[v146:.+]] = arith.mulf %[[v144]], %[[v145]] {fastmathFlags = #llvm.fastmath<none>} : tensor<2x3xf64>
// CHECK-NEXT:    %[[v147:.+]] = arith.addf %[[v106]], %[[v146]] {fastmathFlags = #llvm.fastmath<none>} : tensor<2x3xf64>
// CHECK-NEXT:    %[[v148:.+]] = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<i64>) -> tensor<2xi64>
// CHECK-NEXT:    %[[v149:.+]] = arith.addi %[[v25]], %[[v148]] : tensor<2xi64>
// CHECK-NEXT:    %[[v150:.+]] = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i64>) -> tensor<2xi64>
// CHECK-NEXT:    %[[v151:.+]] = arith.remui %[[v149]], %[[v150]] : tensor<2xi64>
// CHECK-NEXT:    %[[v152:.+]] = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i64>) -> tensor<2xi64>
// CHECK-NEXT:    %[[v153:.+]] = arith.divui %[[v149]], %[[v152]] : tensor<2xi64>
// CHECK-NEXT:    %[[v154:.+]] = stablehlo.reshape %[[v153]] : (tensor<2xi64>) -> tensor<2x1xi64>
// CHECK-NEXT:    %[[v155:.+]] = stablehlo.broadcast_in_dim %[[v151]], dims = [0] : (tensor<2xi64>) -> tensor<2x1xi64>
// CHECK-NEXT:    %[[v156:.+]] = stablehlo.broadcast_in_dim %[[v154]], dims = [0, 1] : (tensor<2x1xi64>) -> tensor<2x1xi64>
// CHECK-NEXT:    %[[v157:.+]] = stablehlo.concatenate %[[v156]], %[[v155]], dim = 1 : (tensor<2x1xi64>, tensor<2x1xi64>) -> tensor<2x2xi64>
// CHECK-NEXT:    %[[v158:.+]] = "stablehlo.gather"(%arg2, %[[v157]]) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1>}> : (tensor<2x4xf64>, tensor<2x2xi64>) -> tensor<2xf64>
// CHECK-NEXT:    %[[v159:.+]] = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i64>) -> tensor<3xi64>
// CHECK-NEXT:    %[[v160:.+]] = stablehlo.multiply %[[v2]], %[[v159]] : tensor<3xi64>
// CHECK-NEXT:    %[[v161:.+]] = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<i64>) -> tensor<3xi64>
// CHECK-NEXT:    %[[v162:.+]] = stablehlo.add %[[v160]], %[[v161]] : tensor<3xi64>
// CHECK-NEXT:    %[[v163:.+]] = stablehlo.broadcast_in_dim %[[v162]], dims = [0] : (tensor<3xi64>) -> tensor<3x2xi64>
// CHECK-NEXT:    %[[v164:.+]] = stablehlo.broadcast_in_dim %[[v38]], dims = [1] : (tensor<2xi64>) -> tensor<3x2xi64>
// CHECK-NEXT:    %[[v165:.+]] = arith.addi %[[v163]], %[[v164]] : tensor<3x2xi64>
// CHECK-NEXT:    %[[v166:.+]] = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i64>) -> tensor<3x2xi64>
// CHECK-NEXT:    %[[v167:.+]] = arith.remui %[[v165]], %[[v166]] : tensor<3x2xi64>
// CHECK-NEXT:    %[[v168:.+]] = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i64>) -> tensor<3x2xi64>
// CHECK-NEXT:    %[[v169:.+]] = arith.divui %[[v165]], %[[v168]] : tensor<3x2xi64>
// CHECK-NEXT:    %[[v170:.+]] = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<i64>) -> tensor<3x2xi64>
// CHECK-NEXT:    %[[v171:.+]] = arith.remui %[[v169]], %[[v170]] : tensor<3x2xi64>
// CHECK-NEXT:    %[[v172:.+]] = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<i64>) -> tensor<3x2xi64>
// CHECK-NEXT:    %[[v173:.+]] = arith.divui %[[v169]], %[[v172]] : tensor<3x2xi64>
// CHECK-NEXT:    %[[v174:.+]] = stablehlo.reshape %[[v173]] : (tensor<3x2xi64>) -> tensor<3x2x1xi64>
// CHECK-NEXT:    %[[v175:.+]] = stablehlo.broadcast_in_dim %[[v171]], dims = [0, 1] : (tensor<3x2xi64>) -> tensor<3x2x1xi64>
// CHECK-NEXT:    %[[v176:.+]] = stablehlo.broadcast_in_dim %[[v174]], dims = [0, 1, 2] : (tensor<3x2x1xi64>) -> tensor<3x2x1xi64>
// CHECK-NEXT:    %[[v177:.+]] = stablehlo.concatenate %[[v176]], %[[v175]], dim = 2 : (tensor<3x2x1xi64>, tensor<3x2x1xi64>) -> tensor<3x2x2xi64>
// CHECK-NEXT:    %[[v178:.+]] = stablehlo.broadcast_in_dim %[[v167]], dims = [0, 1] : (tensor<3x2xi64>) -> tensor<3x2x1xi64>
// CHECK-NEXT:    %[[v179:.+]] = stablehlo.broadcast_in_dim %[[v177]], dims = [0, 1, 2] : (tensor<3x2x2xi64>) -> tensor<3x2x2xi64>
// CHECK-NEXT:    %[[v180:.+]] = stablehlo.concatenate %[[v179]], %[[v178]], dim = 2 : (tensor<3x2x2xi64>, tensor<3x2x1xi64>) -> tensor<3x2x3xi64>
// CHECK-NEXT:    %[[v181:.+]] = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i64>) -> tensor<3x2x1xi64>
// CHECK-NEXT:    %[[v182:.+]] = stablehlo.broadcast_in_dim %[[v180]], dims = [0, 1, 2] : (tensor<3x2x3xi64>) -> tensor<3x2x3xi64>
// CHECK-NEXT:    %[[v183:.+]] = stablehlo.concatenate %[[v182]], %[[v181]], dim = 2 : (tensor<3x2x3xi64>, tensor<3x2x1xi64>) -> tensor<3x2x4xi64>
// CHECK-NEXT:    %[[v184:.+]] = "stablehlo.gather"(%arg1, %[[v183]]) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1, 2, 3], start_index_map = [0, 1, 2, 3], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 1, 1>}> : (tensor<2x3x4x1xf64>, tensor<3x2x4xi64>) -> tensor<3x2xf64>
// CHECK-NEXT:    %[[v185:.+]] = stablehlo.broadcast_in_dim %[[v158]], dims = [0] : (tensor<2xf64>) -> tensor<2x3xf64>
// CHECK-NEXT:    %[[v186:.+]] = stablehlo.broadcast_in_dim %[[v184]], dims = [1, 0] : (tensor<3x2xf64>) -> tensor<2x3xf64>
// CHECK-NEXT:    %[[v187:.+]] = arith.mulf %[[v185]], %[[v186]] {fastmathFlags = #llvm.fastmath<none>} : tensor<2x3xf64>
// CHECK-NEXT:    %[[v188:.+]] = arith.addf %[[v147]], %[[v187]] {fastmathFlags = #llvm.fastmath<none>} : tensor<2x3xf64>
// CHECK-NEXT:    %[[v189:.+]] = stablehlo.slice %arg0 [0:2, 1:4, 0:1] : (tensor<2x4x1xf64>) -> tensor<2x3x1xf64>
// CHECK-NEXT:    %[[v190:.+]] = stablehlo.reshape %[[v189]] : (tensor<2x3x1xf64>) -> tensor<2x3xf64>
// CHECK-NEXT:    %[[v191:.+]] = arith.addf %[[v190]], %[[v188]] {fastmathFlags = #llvm.fastmath<none>} : tensor<2x3xf64>
// CHECK-NEXT:    %[[v192:.+]] = stablehlo.broadcast_in_dim %[[v191]], dims = [0, 1] : (tensor<2x3xf64>) -> tensor<2x3x1xf64>
// CHECK-NEXT:    %[[v193:.+]] = stablehlo.dynamic_slice %arg0, %c_1, %c_6, %c_1, sizes = [2, 3, 1] : (tensor<2x4x1xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<2x3x1xf64>
// CHECK-NEXT:    %[[v194:.+]] = stablehlo.reshape %[[v192]] : (tensor<2x3x1xf64>) -> tensor<2x3xf64>
// CHECK-NEXT:    %[[v195:.+]] = stablehlo.reshape %[[v193]] : (tensor<2x3x1xf64>) -> tensor<2x3xf64>
// CHECK-NEXT:    %[[v196:.+]] = stablehlo.broadcast_in_dim %[[v194]], dims = [1, 0] : (tensor<2x3xf64>) -> tensor<3x2xf64>
// CHECK-NEXT:    %[[v197:.+]] = stablehlo.broadcast_in_dim %[[v195]], dims = [1, 0] : (tensor<2x3xf64>) -> tensor<3x2xf64>
// CHECK-NEXT:    %[[v198:.+]] = stablehlo.select %[[v23]], %[[v196]], %[[v197]] : tensor<3x2xi1>, tensor<3x2xf64>
// CHECK-NEXT:    %[[v199:.+]] = stablehlo.broadcast_in_dim %[[v198]], dims = [1, 0] : (tensor<3x2xf64>) -> tensor<2x3x1xf64>
// CHECK-NEXT:    %[[v200:.+]] = stablehlo.dynamic_update_slice %arg0, %[[v199]], %c_1, %c_6, %c_1 : (tensor<2x4x1xf64>, tensor<2x3x1xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<2x4x1xf64>
// CHECK-NEXT:    return %[[v200]], %arg1, %arg2, %arg3, %arg4, %arg5 : tensor<2x4x1xf64>, tensor<2x3x4x1xf64>, tensor<2x4xf64>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>
// CHECK-NEXT:  }

