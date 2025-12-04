// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= argTys=enzyme_dup,enzyme_dup,enzyme_dup,enzyme_const retTys=enzyme_dup,enzyme_dup,enzyme_dup mode=ForwardMode" --canonicalize | FileCheck %s

module {
  enzymexla_tt_ext.module @add_kernel_tt {
    builtin.module @add_kernel_inner {
      tt.func public @add_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                 %arg3: i32 {tt.divisibility = 16 : i32},
                                 %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
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
    %0:3 = enzymexla_tt_ext.call @add_kernel_tt::@add_kernel_inner::@add_kernel clusters in (%c_0, %c_0, %c_0) blocks in(%c_1, %c_0, %c_0) (%arg0, %arg1, %arg3) {output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [1], operand_index = 1, operand_tuple_indices = []>]} : (tensor<1024xf32>, tensor<1024xf32>, tensor<i32>) -> (tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>)
    return %0#0, %0#1, %0#2 : tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>
  }
}

// CHECK:      tt.func private @fwddiffeadd_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg6: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
// CHECK-NEXT:        %c1024_i32 = arith.constant 1024 : i32
// CHECK-NEXT:        %[[v0:.+]] = tt.get_program_id x : i32
// CHECK-NEXT:        %[[v1:.+]] = arith.muli %[[v0]], %c1024_i32 : i32
// CHECK-NEXT:        %[[v2:.+]] = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
// CHECK-NEXT:        %[[v3:.+]] = tt.splat %[[v1]] : i32 -> tensor<1024xi32>
// CHECK-NEXT:        %[[v4:.+]] = arith.addi %[[v3]], %[[v2]] : tensor<1024xi32>
// CHECK-NEXT:        %[[v5:.+]] = tt.splat %arg4 : i32 -> tensor<1024xi32>
// CHECK-NEXT:        %[[v6:.+]] = arith.cmpi slt, %[[v4]], %[[v5]] : tensor<1024xi32>
// CHECK-NEXT:        %[[v7:.+]] = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
// CHECK-NEXT:        %[[v8:.+]] = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
// CHECK-NEXT:        %[[v9:.+]] = tt.addptr %[[v7]], %[[v4]] : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
// CHECK-NEXT:        %[[v10:.+]] = tt.addptr %[[v8]], %[[v4]] : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
// CHECK-NEXT:        %[[v11:.+]] = tt.load %[[v9]], %[[v6]] : tensor<1024x!tt.ptr<f32>>
// CHECK-NEXT:        %[[v12:.+]] = tt.load %[[v10]], %[[v6]] : tensor<1024x!tt.ptr<f32>>
// CHECK-NEXT:        %[[v13:.+]] = tt.splat %arg3 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
// CHECK-NEXT:        %[[v14:.+]] = tt.splat %arg2 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
// CHECK-NEXT:        %[[v15:.+]] = tt.addptr %[[v13]], %[[v4]] : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
// CHECK-NEXT:        %[[v16:.+]] = tt.addptr %[[v14]], %[[v4]] : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
// CHECK-NEXT:        %[[v17:.+]] = tt.load %[[v15]], %[[v6]] : tensor<1024x!tt.ptr<f32>>
// CHECK-NEXT:        %[[v18:.+]] = tt.load %[[v16]], %[[v6]] : tensor<1024x!tt.ptr<f32>>
// CHECK-NEXT:        %[[v19:.+]] = arith.addf %[[v11]], %[[v17]] : tensor<1024xf32>
// CHECK-NEXT:        %[[v20:.+]] = arith.addf %[[v12]], %[[v18]] : tensor<1024xf32>
// CHECK-NEXT:        %[[v21:.+]] = tt.splat %arg6 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
// CHECK-NEXT:        %[[v22:.+]] = tt.splat %arg5 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
// CHECK-NEXT:        %[[v23:.+]] = tt.addptr %[[v21]], %[[v4]] : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
// CHECK-NEXT:        %[[v24:.+]] = tt.addptr %[[v22]], %[[v4]] : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
// CHECK-NEXT:        tt.store %[[v23]], %[[v19]], %[[v6]] : tensor<1024x!tt.ptr<f32>>
// CHECK-NEXT:        tt.store %[[v24]], %[[v20]], %[[v6]] : tensor<1024x!tt.ptr<f32>>
// CHECK-NEXT:        tt.return
// CHECK-NEXT:      }

// CHECK:  func.func @main(%arg0: tensor<1024xf32>, %arg1: tensor<1024xf32>, %arg2: tensor<1024xf32>, %arg3: tensor<1024xf32>, %arg4: tensor<1024xf32>, %arg5: tensor<1024xf32>, %arg6: tensor<i32>) -> (tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>) {
// CHECK-NEXT:    %c = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<16> : tensor<i64>
// CHECK-NEXT:    %0:6 = enzymexla_tt_ext.call @add_kernel_tt::@add_kernel_inner::@fwddiffeadd_kernel clusters in(%c, %c, %c) blocks in(%c_0, %c, %c) (%arg0, %arg1, %arg2, %arg3, %arg6) {arg_attrs = [], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [1], operand_index = 1, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [2], operand_index = 2, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [3], operand_index = 3, operand_tuple_indices = []>], res_attrs = []} : (tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<i32>) -> (tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>)
// CHECK-NEXT:    return %0#0, %0#1, %0#2, %0#3, %0#4, %0#5 : tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>
// CHECK-NEXT:  }
