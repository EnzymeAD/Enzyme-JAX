// RUN: enzymexlamlir-opt --drop-unsupported-attributes %s | FileCheck %s

module @reactant_JITFunc... attributes {mhlo.num_partitions = 1 : i64, mhlo.num_replicas = 1 : i64} {
  builtin.module @add_kernel_tt_module_e72661bb113efd0f {
    builtin.module @add_kernel_module_e72661bb113efd0f {
      // CHECK: tt.func public @add_kernel_call_e72661bb113efd0f(%arg0: !tt.ptr<f32> {llvm.nofree, llvm.readonly}, %arg1: !tt.ptr<f32> {llvm.nofree, llvm.readonly}, %arg2: !tt.ptr<f32> {llvm.nofree, llvm.writeonly}) attributes {noinline = false}
      tt.func public @add_kernel_call_e72661bb113efd0f(%arg0: !tt.ptr<f32> {enzymexla.memory_effects = ["read"], llvm.nofree, llvm.readonly}, %arg1: !tt.ptr<f32> {enzymexla.memory_effects = ["read"], llvm.nofree, llvm.readonly}, %arg2: !tt.ptr<f32> {enzymexla.memory_effects = ["write"], llvm.nofree, llvm.writeonly}) attributes {enzymexla.memory_effects = ["read", "write"], noinline = false} {
        %cst = arith.constant dense<1024> : tensor<64xi32>
        %c64_i32 = arith.constant 64 : i32
        %0 = tt.get_program_id x : i32
        %1 = arith.muli %0, %c64_i32 : i32
        %2 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
        %3 = tt.splat %1 : i32 -> tensor<64xi32>
        %4 = arith.addi %3, %2 : tensor<64xi32>
        %5 = arith.cmpi slt, %4, %cst : tensor<64xi32>
        %6 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
        %7 = tt.addptr %6, %4 : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
        %8 = tt.load %7, %5 : tensor<64x!tt.ptr<f32>>
        %9 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
        %10 = tt.addptr %9, %4 : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
        %11 = tt.load %10, %5 : tensor<64x!tt.ptr<f32>>
        %12 = arith.addf %8, %11 : tensor<64xf32>
        %13 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
        %14 = tt.addptr %13, %4 : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
        tt.store %14, %12, %5 : tensor<64x!tt.ptr<f32>>
        tt.return
      }
    }
  }
  // func.func @main(%arg0: tensor<1024xf32>, %arg1: tensor<1024xf32>, %arg2: tensor<1024xf32>) -> (tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>) {
  func.func @main(%arg0: tensor<1024xf32> {enzymexla.memory_effects = ["read", "write", "allocate", "free"]}, %arg1: tensor<1024xf32> {enzymexla.memory_effects = ["read", "write", "allocate", "free"]}, %arg2: tensor<1024xf32> {enzymexla.memory_effects = ["read", "write", "allocate", "free"]}) -> (tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>) attributes {enzymexla.memory_effects = ["read", "write", "allocate", "free"]} {
    %c = stablehlo.constant dense<64> : tensor<i64>
    %c_0 = stablehlo.constant dense<1> : tensor<i64>
    %c_1 = stablehlo.constant dense<16> : tensor<i64>
    return %arg0, %arg1, %arg2 : tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>
  }
}
