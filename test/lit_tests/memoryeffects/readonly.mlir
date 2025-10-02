// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(tt.func(mark-func-args-memory-effects))" %s | FileCheck %s

module {
  tt.func @main(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>) attributes {noinline = false} {
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

// CHECK: module {
// CHECK-NEXT:   tt.func @main(%arg0: !tt.ptr<f32> {enzymexla.memory_effects = ["read"], llvm.readonly}, %arg1: !tt.ptr<f32> {enzymexla.memory_effects = ["read"], llvm.readonly}, %arg2: !tt.ptr<f32> {enzymexla.memory_effects = ["write"], llvm.writeonly}) attributes {noinline = false} {
// CHECK-NEXT:     %cst = arith.constant dense<1024> : tensor<64xi32>
// CHECK-NEXT:     %c64_i32 = arith.constant 64 : i32
// CHECK-NEXT:     %0 = tt.get_program_id x : i32
// CHECK-NEXT:     %1 = arith.muli %0, %c64_i32 : i32
// CHECK-NEXT:     %2 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
// CHECK-NEXT:     %3 = tt.splat %1 : i32 -> tensor<64xi32>
// CHECK-NEXT:     %4 = arith.addi %3, %2 : tensor<64xi32>
// CHECK-NEXT:     %5 = arith.cmpi slt, %4, %cst : tensor<64xi32>
// CHECK-NEXT:     %6 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
// CHECK-NEXT:     %7 = tt.addptr %6, %4 : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
// CHECK-NEXT:     %8 = tt.load %7, %5 : tensor<64x!tt.ptr<f32>>
// CHECK-NEXT:     %9 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
// CHECK-NEXT:     %10 = tt.addptr %9, %4 : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
// CHECK-NEXT:     %11 = tt.load %10, %5 : tensor<64x!tt.ptr<f32>>
// CHECK-NEXT:     %12 = arith.addf %8, %11 : tensor<64xf32>
// CHECK-NEXT:     %13 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
// CHECK-NEXT:     %14 = tt.addptr %13, %4 : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
// CHECK-NEXT:     tt.store %14, %12, %5 : tensor<64x!tt.ptr<f32>>
// CHECK-NEXT:     tt.return
// CHECK-NEXT:   }
// CHECK-NEXT: }
