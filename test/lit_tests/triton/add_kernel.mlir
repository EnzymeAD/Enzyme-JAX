// RUN: enzymexlamlir-opt %s -canonicalize | FileCheck %s
// RUN: enzymexlamlir-opt %s -lower-triton | FileCheck %s --check-prefix=LOWER

module {
  // CHECK: enzymexla_tt_ext.module
  enzymexla_tt_ext.module @add_kernel_tt {
    builtin.module @add_kernel_inner {
      tt.func public @add_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
        %c1024_i32 = arith.constant 1024 : i32
        %0 = tt.get_program_id x : i32
        %1 = arith.muli %0, %c1024_i32 : i32
        %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
        %3 = tt.splat %1 : i32 -> tensor<1024xi32>
        %4 = arith.addi %3, %2 : tensor<1024xi32>
        %5 = tt.splat %arg3 : i32 -> tensor<1024xi32>
        %6 = arith.cmpi slt, %4, %5 : tensor<1024xi32>
        %7 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
        %8 = tt.addptr %7, %4 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
        %9 = tt.load %8, %6 : tensor<1024x!tt.ptr<f32>>
        %10 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
        %11 = tt.addptr %10, %4 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
        %12 = tt.load %11, %6 : tensor<1024x!tt.ptr<f32>>
        %13 = arith.addf %9, %12 : tensor<1024xf32>
        %14 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
        %15 = tt.addptr %14, %4 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
        tt.store %15, %13, %6 : tensor<1024x!tt.ptr<f32>>
        tt.return
      }
    }
  }
  func.func @main(%arg0: tensor<1024xf32>, %arg1: tensor<1024xf32>, %arg2: tensor<1024xf32>, %arg3: tensor<i32>) -> (tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>) {
    %c_0 = stablehlo.constant dense<1> : tensor<i64>
    %c_1 = stablehlo.constant dense<16> : tensor<i64>
    // CHECK: enzymexla_tt_ext.call
    %0:3 = enzymexla_tt_ext.call @add_kernel_tt::@add_kernel_inner::@add_kernel clusters in (%c_0, %c_0, %c_0) blocks in(%c_1, %c_0, %c_0) (%arg0, %arg1, %arg2, %arg3) {output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [1], operand_index = 1, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [2], operand_index = 2, operand_tuple_indices = []>]} : (tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<i32>) -> (tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>)
    return %0#0, %0#1, %0#2 : tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>
  }
}
