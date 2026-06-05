// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= argTys=enzyme_dup,enzyme_dup,enzyme_const,enzyme_const,enzyme_const,enzyme_const retTys=enzyme_dup,enzyme_dup mode=ForwardMode" --canonicalize | FileCheck %s

// Currently failing within scf.for.
module {
  // Checking softmax kernel. 
  enzymexla_tt_ext.module @softmax_kernel_tt {
    builtin.module @softmax_kernel_inner {
  tt.func public @softmax_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, 
                                  %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, 
                                  %arg2: i32 {tt.divisibility = 16 : i32}, 
                                  %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32, 
                                  %arg5: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0xFF800000> : tensor<64xf32>
    %0 = tt.get_program_id x : i32 
    %1 = tt.get_num_programs x : i32 
    scf.for %arg6 = %0 to %arg4 step %1  : i32 {
      %2 = arith.muli %arg6, %arg2 : i32 
      %3 = tt.addptr %arg1, %2 : !tt.ptr<f32>, i32 
      %4 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32> 
      %5 = tt.splat %3 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>> 
      %6 = tt.addptr %5, %4 : tensor<64x!tt.ptr<f32>>, tensor<64xi32> 
      %7 = tt.splat %arg5 : i32 -> tensor<64xi32> 
      %8 = arith.cmpi slt, %4, %7 : tensor<64xi32> 
      %9 = tt.load %6, %8, %cst : tensor<64x!tt.ptr<f32>> 
      %10 = "tt.reduce"(%9) <{axis = 0 : i32}> ({
      ^bb0(%arg7: f32 , %arg8: f32):
        %21 = arith.maxnumf %arg7, %arg8 : f32 
        tt.reduce.return %21 : f32 
      }) : (tensor<64xf32>) -> f32 
      %11 = tt.splat %10 : f32 -> tensor<64xf32> 
      %12 = arith.subf %9, %11 : tensor<64xf32> 
      %13 = math.exp %12 : tensor<64xf32> 
      %14 = "tt.reduce"(%13) <{axis = 0 : i32}> ({
      ^bb0(%arg7: f32, %arg8: f32):
        %21 = arith.addf %arg7, %arg8 : f32 
        tt.reduce.return %21 : f32 
      }) : (tensor<64xf32>) -> f32 
      %15 = tt.splat %14 : f32 -> tensor<64xf32> 
      %16 = arith.divf %13, %15 : tensor<64xf32> 
      %17 = arith.muli %arg6, %arg3 : i32 
      %18 = tt.addptr %arg0, %17 : !tt.ptr<f32>, i32 
      %19 = tt.splat %18 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>> 
      %20 = tt.addptr %19, %4 : tensor<64x!tt.ptr<f32>>, tensor<64xi32> 
      tt.store %20, %16, %8 : tensor<64x!tt.ptr<f32>> 
    } {tt.num_stages = 2 : i32} 
    tt.return 
    }
  }
 }
  func.func @main(%arg0: tensor<1024xf32>, 
                    %arg1: tensor<1024xf32>, 
                    %arg2: tensor<i32>,
                    %arg3: tensor<i32>,
                    %arg4: tensor<i32>,
                    %arg5: tensor<i32>) -> (tensor<1024xf32>, tensor<1024xf32>) {
    %c_0 = stablehlo.constant dense<1> : tensor<i64>
    %c_1 = stablehlo.constant dense<16> : tensor<i64>
    %0:2 = enzymexla_tt_ext.call @softmax_kernel_tt::@softmax_kernel_inner::@softmax_kernel clusters in (%c_0, %c_0, %c_0) blocks in(%c_1, %c_0, %c_0) (%arg0, %arg1, %arg2, %arg3, %arg4, %arg5) 
            {output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], 
            operand_index = 0, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [1], 
            operand_index = 1, operand_tuple_indices = []>]} : (tensor<1024xf32>, tensor<1024xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> (tensor<1024xf32>, tensor<1024xf32>)
    return %0#0, %0#1 : tensor<1024xf32>, tensor<1024xf32>
  }
}