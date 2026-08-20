// RUN: enzymexlamlir-opt %s --raise-affine-to-stablehlo=prefer_while_raising=false --arith-raise --inline --enzyme-hlo-opt | FileCheck %s
#set = affine_set<(d0) : (d0 - 1 == 0)>
#tbaa_root = #llvm.tbaa_root<id = "custom_tbaa">
#tbaa_type_desc = #llvm.tbaa_type_desc<id = "custom_tbaa_addrspace(1)", members = {<#tbaa_root, 0>}>
#tbaa_tag = #llvm.tbaa_tag<base_type = #tbaa_type_desc, access_type = #tbaa_type_desc, offset = 0>
module @"reactant_step!" attributes {llvm.data_layout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64", mhlo.num_partitions = 1 : i64, mhlo.num_replicas = 1 : i64} {
  llvm.module_flags [#llvm.mlir.module_flag<warning, "Dwarf Version", 2 : i32>, #llvm.mlir.module_flag<warning, "Debug Info Version", 3 : i32>, #llvm.mlir.module_flag<override, "nvvm-reflect-ftz", 0 : i32>]
  func.func @main(%arg0: tensor<1x2x1xf64> {enzymexla.memory_effects = ["read", "write", "allocate", "free"], tf.aliasing_output = 0 : i32}, %arg1: tensor<1x1xf64> {enzymexla.memory_effects = ["read", "write", "allocate", "free"], tf.aliasing_output = 1 : i32}, %arg2: tensor<2xf64> {enzymexla.memory_effects = ["read", "write", "allocate", "free"], tf.aliasing_output = 2 : i32}, %arg3: tensor<1xi32> {enzymexla.memory_effects = ["read", "write", "allocate", "free"], tf.aliasing_output = 3 : i32}) -> (tensor<1x2x1xf64>, tensor<1x1xf64>, tensor<2xf64>, tensor<1xi32>) attributes {enzymexla.memory_effects = []} {
    %0 = enzymexla.jit_call @"##call__Z11gpu_kernel_16CompilerMetadataI11DynamicSize12DynamicCheckv16CartesianIndicesILi2E5TupleI5OneToI5Int64ES6_EE7NDRangeILi2ES0_S0_S8_S8_EE13CuTracedArrayI7Float64Li3ELi1E9_1__2__1_ESC_ISD_Li2ELi1E6_1__1_ESC_ISD_Li1ELi1E4_2__ESC_I5Int32Li1ELi1E4_1__ES5_S5_S5_#280$par0" (%arg0, %arg1, %arg2, %arg3) {output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 1, operand_tuple_indices = []>], xla_side_effect_free} : (tensor<1x2x1xf64>, tensor<1x1xf64>, tensor<2xf64>, tensor<1xi32>) -> tensor<1x1xf64>
    return %arg0, %0, %arg2, %arg3 : tensor<1x2x1xf64>, tensor<1x1xf64>, tensor<2xf64>, tensor<1xi32>
  }
  func.func private @"##call__Z11gpu_kernel_16CompilerMetadataI11DynamicSize12DynamicCheckv16CartesianIndicesILi2E5TupleI5OneToI5Int64ES6_EE7NDRangeILi2ES0_S0_S8_S8_EE13CuTracedArrayI7Float64Li3ELi1E9_1__2__1_ESC_ISD_Li2ELi1E6_1__1_ESC_ISD_Li1ELi1E4_2__ESC_I5Int32Li1ELi1E4_1__ES5_S5_S5_#280$par0"(%arg0: memref<1x2x1xf64, 1>, %arg1: memref<1x1xf64, 1>, %arg2: memref<2xf64, 1>, %arg3: memref<1xi32, 1>) attributes {enzymexla.memory_effects = ["read", "write", "allocate", "free"]} {
    %cst = arith.constant 1.000000e-04 : f64
    %cst_0 = arith.constant 0.000000e+00 : f64
    %cst_1 = arith.constant 1.000000e+00 : f64
    %cst_2 = arith.constant 2.800000e+02 : f64
    %cst_3 = arith.constant 2.000000e+00 : f64
    %0 = affine.load %arg3[0] {alignment = 4 : i64, invariant, ordering = 0 : i64, tbaa = [#tbaa_tag]} : memref<1xi32, 1>
    %1:2 = affine.for %arg4 = 0 to 2 iter_args(%arg5 = %cst, %arg6 = %cst_0) -> (f64, f64) {
      %8 = affine.load %arg2[%arg4] {alignment = 8 : i64, invariant, ordering = 0 : i64, tbaa = [#tbaa_tag]} : memref<2xf64, 1>
      %9 = arith.divf %cst_1, %8 {fastmathFlags = #llvm.fastmath<none>} : f64
      %10 = affine.for %arg7 = 0 to 2 iter_args(%arg8 = %cst) -> (f64) {
        %14 = affine.load %arg0[0, -%arg7 + 1, 0] : memref<1x2x1xf64, 1>
        %15 = arith.mulf %9, %14 {fastmathFlags = #llvm.fastmath<none>} : f64
        %16 = arith.subf %cst_1, %15 {fastmathFlags = #llvm.fastmath<none>} : f64
        %17 = arith.mulf %arg8, %16 {fastmathFlags = #llvm.fastmath<none>} : f64
        %18 = arith.subf %cst_1, %16 {fastmathFlags = #llvm.fastmath<none>} : f64
        %19 = arith.mulf %18, %cst_2 {fastmathFlags = #llvm.fastmath<none>} : f64
        %20 = arith.addf %17, %19 {fastmathFlags = #llvm.fastmath<none>} : f64
        affine.yield %20 : f64
      }
      %11 = arith.mulf %8, %10 {fastmathFlags = #llvm.fastmath<none>} : f64
      %12 = arith.addf %arg6, %11 {fastmathFlags = #llvm.fastmath<none>} : f64
      %13 = affine.if #set(%arg4) -> f64 {
        affine.yield %10 : f64
      } else {
        affine.yield %arg5 : f64
      }
      affine.yield %13, %12 : f64, f64
    }
    %2 = arith.index_cast %0 : i32 to index
    %3 = arith.index_cast %0 : i32 to index
    %4 = affine.load %arg1[symbol(%2) - 1, 0] : memref<1x1xf64, 1>
    %5 = arith.mulf %1#0, %cst_3 {fastmathFlags = #llvm.fastmath<none>} : f64
    %6 = arith.addf %1#1, %5 {fastmathFlags = #llvm.fastmath<none>} : f64
    %7 = arith.addf %4, %6 {fastmathFlags = #llvm.fastmath<none>} : f64
    affine.store %7, %arg1[symbol(%3) - 1, 0] : memref<1x1xf64, 1>
    return
  }
}

// CHECK:  func.func @main(%arg0: tensor<1x2x1xf64> {enzymexla.memory_effects = ["read", "write", "allocate", "free"], tf.aliasing_output = 0 : i32}, %arg1: tensor<1x1xf64> {enzymexla.memory_effects = ["read", "write", "allocate", "free"], tf.aliasing_output = 1 : i32}, %arg2: tensor<2xf64> {enzymexla.memory_effects = ["read", "write", "allocate", "free"], tf.aliasing_output = 2 : i32}, %arg3: tensor<1xi32> {enzymexla.memory_effects = ["read", "write", "allocate", "free"], tf.aliasing_output = 3 : i32}) -> (tensor<1x2x1xf64>, tensor<1x1xf64>, tensor<2xf64>, tensor<1xi32>) attributes {enzymexla.memory_effects = []} {
// CHECK-NEXT:    %c = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<-1> : tensor<1xi64>
// CHECK-NEXT:    %cst = stablehlo.constant dense<1.000000e-04> : tensor<f64>
// CHECK-NEXT:    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
// CHECK-NEXT:    %cst_2 = stablehlo.constant dense<2.800000e+02> : tensor<f64>
// CHECK-NEXT:    %cst_3 = stablehlo.constant dense<2.000000e+00> : tensor<f64>
// CHECK-NEXT:    %0 = stablehlo.slice %arg2 [0:1] : (tensor<2xf64>) -> tensor<1xf64>
// CHECK-NEXT:    %1 = stablehlo.reshape %0 : (tensor<1xf64>) -> tensor<f64>
// CHECK-NEXT:    %2 = stablehlo.divide %cst_1, %1 : tensor<f64>
// CHECK-NEXT:    %3 = stablehlo.slice %arg0 [0:1, 1:2, 0:1] : (tensor<1x2x1xf64>) -> tensor<1x1x1xf64>
// CHECK-NEXT:    %4 = stablehlo.reshape %3 : (tensor<1x1x1xf64>) -> tensor<f64>
// CHECK-NEXT:    %5 = stablehlo.multiply %2, %4 : tensor<f64>
// CHECK-NEXT:    %6 = stablehlo.subtract %cst_1, %5 : tensor<f64>
// CHECK-NEXT:    %7 = stablehlo.multiply %cst, %6 : tensor<f64>
// CHECK-NEXT:    %8 = stablehlo.subtract %cst_1, %6 : tensor<f64>
// CHECK-NEXT:    %9 = stablehlo.multiply %8, %cst_2 : tensor<f64>
// CHECK-NEXT:    %10 = stablehlo.add %7, %9 : tensor<f64>
// CHECK-NEXT:    %11 = stablehlo.slice %arg0 [0:1, 0:1, 0:1] : (tensor<1x2x1xf64>) -> tensor<1x1x1xf64>
// CHECK-NEXT:    %12 = stablehlo.reshape %11 : (tensor<1x1x1xf64>) -> tensor<f64>
// CHECK-NEXT:    %13 = stablehlo.multiply %2, %12 : tensor<f64>
// CHECK-NEXT:    %14 = stablehlo.subtract %cst_1, %13 : tensor<f64>
// CHECK-NEXT:    %15 = stablehlo.multiply %10, %14 : tensor<f64>
// CHECK-NEXT:    %16 = stablehlo.subtract %cst_1, %14 : tensor<f64>
// CHECK-NEXT:    %17 = stablehlo.multiply %16, %cst_2 : tensor<f64>
// CHECK-NEXT:    %18 = stablehlo.add %15, %17 : tensor<f64>
// CHECK-NEXT:    %19 = stablehlo.multiply %1, %18 : tensor<f64>
// CHECK-NEXT:    %20 = stablehlo.slice %arg2 [1:2] : (tensor<2xf64>) -> tensor<1xf64>
// CHECK-NEXT:    %21 = stablehlo.reshape %20 : (tensor<1xf64>) -> tensor<f64>
// CHECK-NEXT:    %22 = stablehlo.divide %cst_1, %21 : tensor<f64>
// CHECK-NEXT:    %23 = stablehlo.multiply %22, %4 : tensor<f64>
// CHECK-NEXT:    %24 = stablehlo.subtract %cst_1, %23 : tensor<f64>
// CHECK-NEXT:    %25 = stablehlo.multiply %cst, %24 : tensor<f64>
// CHECK-NEXT:    %26 = stablehlo.subtract %cst_1, %24 : tensor<f64>
// CHECK-NEXT:    %27 = stablehlo.multiply %26, %cst_2 : tensor<f64>
// CHECK-NEXT:    %28 = stablehlo.add %25, %27 : tensor<f64>
// CHECK-NEXT:    %29 = stablehlo.multiply %22, %12 : tensor<f64>
// CHECK-NEXT:    %30 = stablehlo.subtract %cst_1, %29 : tensor<f64>
// CHECK-NEXT:    %31 = stablehlo.multiply %28, %30 : tensor<f64>
// CHECK-NEXT:    %32 = stablehlo.subtract %cst_1, %30 : tensor<f64>
// CHECK-NEXT:    %33 = stablehlo.multiply %32, %cst_2 : tensor<f64>
// CHECK-NEXT:    %34 = stablehlo.add %31, %33 : tensor<f64>
// CHECK-NEXT:    %35 = stablehlo.multiply %21, %34 : tensor<f64>
// CHECK-NEXT:    %36 = stablehlo.add %19, %35 : tensor<f64>
// CHECK-NEXT:    %37 = stablehlo.convert %arg3 : (tensor<1xi32>) -> tensor<1xi64>
// CHECK-NEXT:    %38 = stablehlo.add %37, %c_0 : tensor<1xi64>
// CHECK-NEXT:    %39 = stablehlo.pad %38, %c, low = [0], high = [1], interior = [0] : (tensor<1xi64>, tensor<i64>) -> tensor<2xi64>
// CHECK-NEXT:    %40 = "stablehlo.gather"(%arg1, %39) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1], start_index_map = [0, 1]>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1>}> : (tensor<1x1xf64>, tensor<2xi64>) -> tensor<f64>
// CHECK-NEXT:    %41 = stablehlo.multiply %34, %cst_3 : tensor<f64>
// CHECK-NEXT:    %42 = stablehlo.add %36, %41 : tensor<f64>
// CHECK-NEXT:    %43 = stablehlo.add %40, %42 : tensor<f64>
// CHECK-NEXT:    %44 = "stablehlo.scatter"(%arg1, %39, %43) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1]>, unique_indices = true}> ({
// CHECK-NEXT:    ^bb0(%arg4: tensor<f64>, %arg5: tensor<f64>):
// CHECK-NEXT:      stablehlo.return %arg5 : tensor<f64>
// CHECK-NEXT:    }) : (tensor<1x1xf64>, tensor<2xi64>, tensor<f64>) -> tensor<1x1xf64>
// CHECK-NEXT:    return %arg0, %44, %arg2, %arg3 : tensor<1x2x1xf64>, tensor<1x1xf64>, tensor<2xf64>, tensor<1xi32>
// CHECK-NEXT:  }
