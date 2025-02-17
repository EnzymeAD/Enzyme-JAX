// RUN: enzymexlamlir-opt %s -split-input-file --pass-pipeline="builtin.module(llvm-to-memref-access)" | FileCheck %s

module {
  llvm.func internal ptx_kernelcc @single_block_kern(%arg0: !llvm.ptr<1>) {
    %0 = llvm.mlir.constant(63 : i32) : i32
    %1 = nvvm.read.ptx.sreg.tid.x : i32
    %4 = llvm.zext %1 : i32 to i64
    %5 = llvm.getelementptr inbounds %arg0[%4] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i64
    %6 = llvm.load %5 {alignment = 1 : i64} : !llvm.ptr<1> -> i64
    %7 = llvm.mul %6, %6 : i64
    llvm.store %7, %5 {alignment = 1 : i64} : i64, !llvm.ptr<1>
    llvm.return
  }
  func.func @main(%arg0: tensor<64xi64>) -> tensor<64xi64> {
    %c0 = stablehlo.constant dense<0> : tensor<i64>
    %c1 = stablehlo.constant dense<1> : tensor<i64>
    %c40 = stablehlo.constant dense<40> : tensor<i64>
    %0 = enzymexla.kernel_call @single_block_kern blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c40) shmem=%c0 (%arg0) {output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>]} : (tensor<64xi64>) -> tensor<64xi64>
    return %0 : tensor<64xi64>
  }
}

// CHECK-LABEL: func.func @single_block_kern
// CHECK-SAME:  %[[ARG:.*]]: memref<64xi64, 1>
// CHECK:       %[[ADDR:.*]] = "enzymexla.memref2pointer"(%[[ARG]]) : (memref<64xi64, 1>) -> !llvm.ptr<1>
// CHECK:       llvm.getelementptr {{.*}} %[[ADDR]]

// -----

module {
  llvm.func internal unnamed_addr fastcc @throw_boundserror_2676() attributes {dso_local, no_inline, sym_visibility = "private"} {
    llvm.unreachable
  }
  llvm.func internal ptx_kernelcc @simple_multi_blocks_kern(%arg0: !llvm.ptr<1>) {
    %0 = llvm.mlir.constant(63 : i32) : i32
    %1 = nvvm.read.ptx.sreg.tid.x : i32
    %2 = llvm.icmp "ugt" %1, %0 : i32
    llvm.cond_br %2, ^bb2, ^bb1
  ^bb1:  // pred: ^bb0
    %4 = llvm.zext %1 : i32 to i64
    %5 = llvm.getelementptr inbounds %arg0[%4] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i64
    %6 = llvm.load %5 {alignment = 1 : i64} : !llvm.ptr<1> -> i64
    %7 = llvm.mul %6, %6 : i64
    llvm.store %7, %5 {alignment = 1 : i64} : i64, !llvm.ptr<1>
    llvm.return
  ^bb2:  // pred: ^bb0
    llvm.call fastcc @throw_boundserror_2676() : () -> ()
    llvm.unreachable
  }
  func.func @main(%arg0: tensor<64xi64>) -> tensor<64xi64> {
    %c0 = stablehlo.constant dense<0> : tensor<i64>
    %c1 = stablehlo.constant dense<1> : tensor<i64>
    %c40 = stablehlo.constant dense<40> : tensor<i64>
    %0 = enzymexla.kernel_call @simple_multi_blocks_kern blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c40) shmem=%c0 (%arg0) {output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>]} : (tensor<64xi64>) -> tensor<64xi64>
    return %0 : tensor<64xi64>
  }
}

// CHECK-LABEL: func.func @simple_multi_blocks_kern
// CHECK-SAME:  %[[ARG:.*]]: memref<64xi64, 1>
// CHECK:       %[[ADDR:.*]] = "enzymexla.memref2pointer"(%[[ARG]]) : (memref<64xi64, 1>) -> !llvm.ptr<1>
// CHECK:       ^bb1:
// CHECK:       llvm.getelementptr {{.*}} %[[ADDR]]

// -----

module {
  llvm.func ptx_kernelcc @"multi_args_kern"(%arg0: !llvm.ptr<1> {llvm.noalias, llvm.nocapture, llvm.nofree}, %arg1: !llvm.ptr<1> {llvm.noalias, llvm.nocapture, llvm.nofree}, %arg2: !llvm.ptr<1> {llvm.noalias, llvm.nocapture, llvm.nofree, llvm.readonly}, %arg3: !llvm.ptr<1> {llvm.noalias, llvm.nocapture, llvm.nofree, llvm.readonly}) attributes {memory_effects = #llvm.memory_effects<other = none, argMem = readwrite, inaccessibleMem = none>, no_unwind, passthrough = ["mustprogress", "nofree", "norecurse", "nosync"], will_return} {
    %0 = llvm.mlir.constant(23 : i32) : i32
    %1 = llvm.mlir.constant(1152921504606846953 : i64) : i64
    %2 = llvm.mlir.constant(4 : i64) : i64
    %3 = llvm.mlir.constant(8 : i64) : i64
    %4 = llvm.mlir.constant(1 : i64) : i64
    %7 = llvm.mlir.constant(15 : i64) : i64
    %8 = llvm.mlir.constant(374 : i64) : i64
    %9 = llvm.mlir.constant(2244 : i64) : i64
    %10 = llvm.mlir.constant(1.000000e+01 : f64) : f64
    %11 = llvm.mlir.constant(2992 : i64) : i64
    %17 = nvvm.read.ptx.sreg.tid.x range <i32, 0, 256> : i32
    %18 = arith.extui %17 {nonNeg} : i32 to i64
    %19 = nvvm.read.ptx.sreg.ctaid.x range <i32, 0, 184> : i32
    %20 = arith.extui %19 {nonNeg} : i32 to i64
    %21 = arith.divui %19, %0 : i32
    %22 = arith.extui %21 {nonNeg} : i32 to i64
    %23 = arith.muli %22, %1 : i64
    %24 = arith.addi %23, %20 : i64
    %25 = arith.shrui %18, %2 : i64
    %26 = arith.andi %18, %3 : i64
    %27 = arith.addi %25, %4 : i64
    %28 = arith.shli %24, %2 : i64
    %29 = arith.shli %22, %2 : i64
    %30 = arith.addi %27, %29 : i64

    %58 = arith.andi %18, %7 : i64
    %59 = arith.addi %58, %4 : i64
    %60 = arith.addi %59, %28 : i64
    %61 = arith.muli %30, %8 : i64
    %62 = arith.addi %61, %9 : i64
    %63 = llvm.getelementptr inbounds %arg2[%60] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f64
    %64 = llvm.getelementptr inbounds %63[2806544] : (!llvm.ptr<1>) -> !llvm.ptr<1>, i8
    %65 = llvm.getelementptr inbounds %64[%62] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f64
    %66 = llvm.load %65 {alignment = 8 : i64} : !llvm.ptr<1> -> f64
    %67 = arith.mulf %66, %10 {fastmathFlags = #llvm.fastmath<none>} : f64
    %68 = llvm.getelementptr inbounds %arg0[%60] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f64
    %69 = arith.muli %30, %11 : i64
    %70 = llvm.getelementptr inbounds %68[%69] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i8
    %71 = llvm.getelementptr inbounds %70[18000] : (!llvm.ptr<1>) -> !llvm.ptr<1>, i8
    llvm.store %67, %71 {alignment = 8 : i64} : f64, !llvm.ptr<1>
    %72 = llvm.getelementptr inbounds %arg3[%60] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f64
    %73 = llvm.getelementptr inbounds %72[2827488] : (!llvm.ptr<1>) -> !llvm.ptr<1>, i8
    %74 = llvm.getelementptr inbounds %73[%62] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f64
    %75 = llvm.load %74 {alignment = 8 : i64} : !llvm.ptr<1> -> f64
    %76 = arith.mulf %75, %10 {fastmathFlags = #llvm.fastmath<none>} : f64
    %77 = llvm.getelementptr inbounds %arg1[%60] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f64
    %78 = llvm.getelementptr inbounds %77[%69] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i8
    %79 = llvm.getelementptr inbounds %78[18000] : (!llvm.ptr<1>) -> !llvm.ptr<1>, i8
    llvm.store %76, %79 {alignment = 8 : i64} : f64, !llvm.ptr<1>

    llvm.return
  }

  func.func @main(%arg8: tensor<1x134x374xf64>, %arg13: tensor<1x135x374xf64>, %arg14: tensor<114x134x374xf64>, %arg15: tensor<114x135x374xf64>) -> tensor<1x134x374xf64> {
    %c_3 = stablehlo.constant dense<184> : tensor<i64>
    %c_8 = stablehlo.constant dense<0> : tensor<i64>
    %c_9 = stablehlo.constant dense<256> : tensor<i64>
    %c_10 = stablehlo.constant dense<1> : tensor<i64>
    %13:2 = enzymexla.kernel_call @"multi_args_kern" blocks in(%c_3, %c_10, %c_10) threads in(%c_9, %c_10, %c_10) shmem = %c_8 (%arg8, %arg13, %arg14, %arg15) {output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [1], operand_index = 1, operand_tuple_indices = []>]} : (tensor<1x134x374xf64>, tensor<1x135x374xf64>, tensor<114x134x374xf64>, tensor<114x135x374xf64>) -> (tensor<1x134x374xf64>, tensor<1x135x374xf64>)
    return %13 : tensor<1x134x374xf64>
  }
}

// CHECK-LABEL: func.func @multi_args_kern
// CHECK-SAME:  %[[ARG0:.*]]: memref<1x134x374xf64, 1>
// CHECK-SAME:  %[[ARG1:.*]]: memref<1x135x374xf64, 1>
// CHECK-SAME:  %[[ARG2:.*]]: memref<114x134x374xf64, 1>
// CHECK-SAME:  %[[ARG3:.*]]: memref<114x135x374xf64, 1>
// CHECK-SAME:  CConv = #llvm.cconv<ptx_kernelcc>
// CHECK-SAME:  linkage = #llvm.linkage<external>
// CHECK-SAME:  memory_effects = #llvm.memory_effects<other = none, argMem = readwrite, inaccessibleMem = none>
// CHECK-SAME:  no_unwind
// CHECK-SAME:  passthrough = ["mustprogress", "nofree", "norecurse", "nosync"]
// CHECK-SAME:  unnamed_addr = 0 : i64
// CHECK-SAME:  visibility_ = 0 : i64
// CHECK-SAME:  will_return
// CHECK:       %[[ADDR0:.*]] = "enzymexla.memref2pointer"(%[[ARG0]])
// CHECK:       %[[ADDR1:.*]] = "enzymexla.memref2pointer"(%[[ARG1]])
// CHECK:       %[[ADDR2:.*]] = "enzymexla.memref2pointer"(%[[ARG2]])
// CHECK:       %[[ADDR3:.*]] = "enzymexla.memref2pointer"(%[[ARG3]])
// CHECK-DAG:   llvm.getelementptr {{.*}} %[[ADDR0]]
// CHECK-DAG:   llvm.getelementptr {{.*}} %[[ADDR1]]
// CHECK-DAG:   llvm.getelementptr {{.*}} %[[ADDR2]]
// CHECK-DAG:   llvm.getelementptr {{.*}} %[[ADDR3]]

// -----

module {
  llvm.func ptx_kernelcc @"loop_kern"(%arg0: !llvm.ptr<1> {llvm.noalias, llvm.nocapture, llvm.nofree}, %arg1: !llvm.ptr<1> {llvm.noalias, llvm.nocapture, llvm.nofree}, %arg2: !llvm.ptr<1> {llvm.noalias, llvm.nocapture, llvm.nofree, llvm.readonly}, %arg3: !llvm.ptr<1> {llvm.noalias, llvm.nocapture, llvm.nofree, llvm.readonly}) attributes {memory_effects = #llvm.memory_effects<other = none, argMem = readwrite, inaccessibleMem = none>, no_unwind, passthrough = ["mustprogress", "nofree", "norecurse", "nosync"], will_return} {
    %0 = llvm.mlir.constant(23 : i32) : i32
    %1 = llvm.mlir.constant(1152921504606846953 : i64) : i64
    %2 = llvm.mlir.constant(4 : i64) : i64
    %3 = llvm.mlir.constant(8 : i64) : i64
    %4 = llvm.mlir.constant(1 : i64) : i64
    %5 = llvm.mlir.constant(359 : i64) : i64
    %6 = llvm.mlir.constant(120 : i64) : i64
    %7 = llvm.mlir.constant(15 : i64) : i64
    %8 = llvm.mlir.constant(374 : i64) : i64
    %9 = llvm.mlir.constant(2244 : i64) : i64
    %10 = llvm.mlir.constant(1.000000e+01 : f64) : f64
    %11 = llvm.mlir.constant(2992 : i64) : i64
    %12 = llvm.mlir.constant(false) : i1
    %13 = llvm.mlir.constant(2 : i64) : i64
    %14 = llvm.mlir.constant(400928 : i64) : i64
    %15 = llvm.mlir.constant(403920 : i64) : i64
    %16 = llvm.mlir.constant(100 : i64) : i64
    %17 = nvvm.read.ptx.sreg.tid.x range <i32, 0, 256> : i32
    %18 = arith.extui %17 {nonNeg} : i32 to i64
    %19 = nvvm.read.ptx.sreg.ctaid.x range <i32, 0, 184> : i32
    %20 = arith.extui %19 {nonNeg} : i32 to i64
    %21 = arith.divui %19, %0 : i32
    %22 = arith.extui %21 {nonNeg} : i32 to i64
    %23 = arith.muli %22, %1 : i64
    %24 = arith.addi %23, %20 : i64
    %25 = arith.shrui %18, %2 : i64
    %26 = arith.andi %18, %3 : i64
    %27 = arith.addi %25, %4 : i64
    %28 = arith.shli %24, %2 : i64
    %29 = arith.shli %22, %2 : i64
    %30 = arith.addi %27, %29 : i64
    %31 = arith.ori %28, %26 {isDisjoint} : i64
    %32 = llvm.icmp "ugt" %31, %5 : i64
    %33 = llvm.icmp "ugt" %30, %6 : i64
    %34 = arith.ori %33, %32 : i1
    llvm.cond_br %34, ^bb2, ^bb3
  ^bb1(%35: i64):  // 2 preds: ^bb1, ^bb4
    %36 = llvm.load %71 {alignment = 8 : i64} : !llvm.ptr<1> -> f64
    %37 = llvm.getelementptr %arg2[%60] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f64
    %38 = llvm.getelementptr %37[48] : (!llvm.ptr<1>) -> !llvm.ptr<1>, i8
    %39 = llvm.getelementptr %38[%62] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f64
    %40 = arith.muli %35, %14 : i64
    %41 = llvm.getelementptr %39[%40] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i8
    %42 = llvm.getelementptr %41[2405568] : (!llvm.ptr<1>) -> !llvm.ptr<1>, i8
    %43 = llvm.load %42 {alignment = 8 : i64} : !llvm.ptr<1> -> f64
    %44 = arith.mulf %43, %10 {fastmathFlags = #llvm.fastmath<none>} : f64
    %45 = arith.addf %36, %44 {fastmathFlags = #llvm.fastmath<none>} : f64
    llvm.store %45, %71 {alignment = 8 : i64} : f64, !llvm.ptr<1>
    %46 = llvm.load %79 {alignment = 8 : i64} : !llvm.ptr<1> -> f64
    %47 = llvm.getelementptr %arg3[%60] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f64
    %48 = llvm.getelementptr %47[48] : (!llvm.ptr<1>) -> !llvm.ptr<1>, i8
    %49 = llvm.getelementptr %48[%62] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f64
    %50 = arith.muli %35, %15 : i64
    %51 = llvm.getelementptr %49[%50] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i8
    %52 = llvm.getelementptr %51[2423520] : (!llvm.ptr<1>) -> !llvm.ptr<1>, i8
    %53 = llvm.load %52 {alignment = 8 : i64} : !llvm.ptr<1> -> f64
    %54 = arith.mulf %53, %10 {fastmathFlags = #llvm.fastmath<none>} : f64
    %55 = arith.addf %46, %54 {fastmathFlags = #llvm.fastmath<none>} : f64
    llvm.store %55, %79 {alignment = 8 : i64} : f64, !llvm.ptr<1>
    %56 = llvm.icmp "eq" %35, %16 : i64
    %57 = arith.addi %35, %4 : i64
    llvm.cond_br %56, ^bb2, ^bb1(%57 : i64)
  ^bb2:  // 3 preds: ^bb0, ^bb1, ^bb3
    llvm.br ^bb5
  ^bb3:  // pred: ^bb0
    %58 = arith.andi %18, %7 : i64
    %59 = arith.addi %58, %4 : i64
    %60 = arith.addi %59, %28 : i64
    %61 = arith.muli %30, %8 : i64
    %62 = arith.addi %61, %9 : i64
    %63 = llvm.getelementptr inbounds %arg2[%60] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f64
    %64 = llvm.getelementptr inbounds %63[2806544] : (!llvm.ptr<1>) -> !llvm.ptr<1>, i8
    %65 = llvm.getelementptr inbounds %64[%62] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f64
    %66 = llvm.load %65 {alignment = 8 : i64} : !llvm.ptr<1> -> f64
    %67 = arith.mulf %66, %10 {fastmathFlags = #llvm.fastmath<none>} : f64
    %68 = llvm.getelementptr inbounds %arg0[%60] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f64
    %69 = arith.muli %30, %11 : i64
    %70 = llvm.getelementptr inbounds %68[%69] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i8
    %71 = llvm.getelementptr inbounds %70[18000] : (!llvm.ptr<1>) -> !llvm.ptr<1>, i8
    llvm.store %67, %71 {alignment = 8 : i64} : f64, !llvm.ptr<1>
    %72 = llvm.getelementptr inbounds %arg3[%60] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f64
    %73 = llvm.getelementptr inbounds %72[2827488] : (!llvm.ptr<1>) -> !llvm.ptr<1>, i8
    %74 = llvm.getelementptr inbounds %73[%62] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f64
    %75 = llvm.load %74 {alignment = 8 : i64} : !llvm.ptr<1> -> f64
    %76 = arith.mulf %75, %10 {fastmathFlags = #llvm.fastmath<none>} : f64
    %77 = llvm.getelementptr inbounds %arg1[%60] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f64
    %78 = llvm.getelementptr inbounds %77[%69] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i8
    %79 = llvm.getelementptr inbounds %78[18000] : (!llvm.ptr<1>) -> !llvm.ptr<1>, i8
    llvm.store %76, %79 {alignment = 8 : i64} : f64, !llvm.ptr<1>
    llvm.cond_br %12, ^bb2, ^bb4
  ^bb4:  // pred: ^bb3
    llvm.br ^bb1(%13 : i64)
  ^bb5:  // pred: ^bb2
    llvm.return
  }

  func.func @main(%arg8: tensor<1x134x374xf64>, %arg13: tensor<1x135x374xf64>, %arg14: tensor<114x134x374xf64>, %arg15: tensor<114x135x374xf64>) -> tensor<1x134x374xf64> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<1x135x374xf64>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<1x134x374xf64>
    %c = stablehlo.constant dense<18400> : tensor<i64>
    %c_1 = stablehlo.constant dense<216> : tensor<i64>
    %c_2 = stablehlo.constant dense<72> : tensor<i64>
    %c_3 = stablehlo.constant dense<184> : tensor<i64>
    %c_4 = stablehlo.constant dense<161> : tensor<i64>
    %c_5 = stablehlo.constant dense<135> : tensor<i64>
    %c_6 = stablehlo.constant dense<23> : tensor<i64>
    %c_7 = stablehlo.constant dense<134> : tensor<i64>
    %c_8 = stablehlo.constant dense<0> : tensor<i64>
    %c_9 = stablehlo.constant dense<256> : tensor<i64>
    %c_10 = stablehlo.constant dense<1> : tensor<i64>
    %c_11 = stablehlo.constant dense<2> : tensor<i64>
    %13:2 = enzymexla.kernel_call @"loop_kern" blocks in(%c_3, %c_10, %c_10) threads in(%c_9, %c_10, %c_10) shmem = %c_8 (%arg8, %arg13, %arg14, %arg15) {output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [1], operand_index = 1, operand_tuple_indices = []>]} : (tensor<1x134x374xf64>, tensor<1x135x374xf64>, tensor<114x134x374xf64>, tensor<114x135x374xf64>) -> (tensor<1x134x374xf64>, tensor<1x135x374xf64>)
    return %13 : tensor<1x134x374xf64>
  }
}

// CHECK-LABEL: func.func @loop_kern
// CHECK-SAME:  %[[ARG0:.*]]: memref<1x134x374xf64, 1>
// CHECK-SAME:  %[[ARG1:.*]]: memref<1x135x374xf64, 1>
// CHECK-SAME:  %[[ARG2:.*]]: memref<114x134x374xf64, 1>
// CHECK-SAME:  %[[ARG3:.*]]: memref<114x135x374xf64, 1>
// CHECK:       %[[ADDR0:.*]] = "enzymexla.memref2pointer"(%[[ARG0]])
// CHECK:       %[[ADDR1:.*]] = "enzymexla.memref2pointer"(%[[ARG1]])
// CHECK:       %[[ADDR2:.*]] = "enzymexla.memref2pointer"(%[[ARG2]])
// CHECK:       %[[ADDR3:.*]] = "enzymexla.memref2pointer"(%[[ARG3]])
// CHECK:       ^bb1{{.*}}pred
// CHECK:       llvm.getelementptr %[[ADDR2]]
// CHECK:       llvm.getelementptr %[[ADDR3]]
// CHECK:       ^bb3{{.*}}pred
// CHECK:       llvm.getelementptr {{.*}} %[[ADDR2]]
// CHECK:       llvm.getelementptr {{.*}} %[[ADDR0]]
// CHECK:       llvm.getelementptr {{.*}} %[[ADDR3]]
// CHECK:       llvm.getelementptr {{.*}} %[[ADDR1]]

// -----

module {
  llvm.func internal ptx_kernelcc @multi_callers_kern(%arg0: !llvm.ptr<1>) {
    %0 = llvm.mlir.constant(63 : i32) : i32
    %1 = nvvm.read.ptx.sreg.tid.x : i32
    %4 = llvm.zext %1 : i32 to i64
    %5 = llvm.getelementptr inbounds %arg0[%4] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i64
    %6 = llvm.load %5 {alignment = 1 : i64} : !llvm.ptr<1> -> i64
    %7 = llvm.mul %6, %6 : i64
    llvm.store %7, %5 {alignment = 1 : i64} : i64, !llvm.ptr<1>
    llvm.return
  }
  func.func @main(%arg0: tensor<64xi64>) -> tensor<64xi64> {
    %c0 = stablehlo.constant dense<0> : tensor<i64>
    %c1 = stablehlo.constant dense<1> : tensor<i64>
    %c40 = stablehlo.constant dense<40> : tensor<i64>
    %0 = enzymexla.kernel_call @multi_callers_kern blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c40) shmem=%c0 (%arg0) {output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>]} : (tensor<64xi64>) -> tensor<64xi64>
    %1 = enzymexla.kernel_call @multi_callers_kern blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c40) shmem=%c0 (%0) {output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>]} : (tensor<64xi64>) -> tensor<64xi64>
    return %1 : tensor<64xi64>
  }
}

// CHECK-LABEL: func.func @multi_callers_kern
// CHECK-SAME:  %[[ARG:.*]]: memref<64xi64, 1>
// CHECK:       %[[ADDR:.*]] = "enzymexla.memref2pointer"(%[[ARG]]) : (memref<64xi64, 1>) -> !llvm.ptr<1>
// CHECK:       llvm.getelementptr {{.*}} %[[ADDR]]

// -----

module {
  llvm.func internal ptx_kernelcc @multi_callers_different_elty_kern(%arg0: !llvm.ptr<1>) {
    %0 = llvm.mlir.constant(63 : i32) : i32
    %1 = nvvm.read.ptx.sreg.tid.x : i32
    %4 = llvm.zext %1 : i32 to i64
    %5 = llvm.getelementptr inbounds %arg0[%4] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i64
    %6 = llvm.load %5 {alignment = 1 : i64} : !llvm.ptr<1> -> i64
    %7 = llvm.mul %6, %6 : i64
    llvm.store %7, %5 {alignment = 1 : i64} : i64, !llvm.ptr<1>
    llvm.return
  }
  func.func @main(%arg0: tensor<64xi64>) -> tensor<64xi64> {
    %c0 = stablehlo.constant dense<0> : tensor<i64>
    %c1 = stablehlo.constant dense<1> : tensor<i64>
    %c40 = stablehlo.constant dense<40> : tensor<i64>
    %0 = enzymexla.kernel_call @multi_callers_different_elty_kern blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c40) shmem=%c0 (%arg0) {output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>]} : (tensor<64xi64>) -> tensor<64xi32>
    %1 = enzymexla.kernel_call @multi_callers_different_elty_kern blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c40) shmem=%c0 (%0) {output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>]} : (tensor<64xi32>) -> tensor<64xi64>
    return %1 : tensor<64xi64>
  }
}

// CHECK-LABEL: llvm.func internal ptx_kernelcc @multi_callers_different_elty_kern
// CHECK-SAME:  %[[ARG:.*]]: !llvm.ptr<1>
// CHECK-NOT:   "enzymexla.memref2pointer"
// CHECK:       llvm.getelementptr {{.*}} %[[ARG]]

// -----

module {
  llvm.func internal ptx_kernelcc @multi_callers_different_shape_kern(%arg0: !llvm.ptr<1>) {
    %0 = llvm.mlir.constant(63 : i32) : i32
    %1 = nvvm.read.ptx.sreg.tid.x : i32
    %4 = llvm.zext %1 : i32 to i64
    %5 = llvm.getelementptr inbounds %arg0[%4] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i64
    %6 = llvm.load %5 {alignment = 1 : i64} : !llvm.ptr<1> -> i64
    %7 = llvm.mul %6, %6 : i64
    llvm.store %7, %5 {alignment = 1 : i64} : i64, !llvm.ptr<1>
    llvm.return
  }
  func.func @main(%arg0: tensor<64xi64>) -> tensor<64xi64> {
    %c0 = stablehlo.constant dense<0> : tensor<i64>
    %c1 = stablehlo.constant dense<1> : tensor<i64>
    %c40 = stablehlo.constant dense<40> : tensor<i64>
    %0 = enzymexla.kernel_call @multi_callers_different_shape_kern blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c40) shmem=%c0 (%arg0) {output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>]} : (tensor<64xi64>) -> tensor<8x8xi64>
    %1 = enzymexla.kernel_call @multi_callers_different_shape_kern blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c40) shmem=%c0 (%0) {output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>]} : (tensor<8x8xi64>) -> tensor<64xi64>
    return %1 : tensor<64xi64>
  }
}

// CHECK-LABEL: func.func @multi_callers_different_shape_kern
// CHECK-SAME:  %[[ARG:.*]]: memref<?xi64, 1>
// CHECK:       %[[ADDR:.*]] = "enzymexla.memref2pointer"(%[[ARG]]) : (memref<?xi64, 1>) -> !llvm.ptr<1>
// CHECK:       llvm.getelementptr {{.*}} %[[ADDR]]


// -----

module {
  llvm.func internal @jitcall(%arg0: !llvm.ptr<1>) {
    %0 = llvm.mlir.constant(63 : i32) : i32
    %1 = nvvm.read.ptx.sreg.tid.x : i32
    %4 = llvm.zext %1 : i32 to i64
    %5 = llvm.getelementptr inbounds %arg0[%4] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i64
    %6 = llvm.load %5 {alignment = 1 : i64} : !llvm.ptr<1> -> i64
    %7 = llvm.mul %6, %6 : i64
    llvm.store %7, %5 {alignment = 1 : i64} : i64, !llvm.ptr<1>
    llvm.return
  }
  func.func @main(%arg0: tensor<8x8xi64>) -> tensor<8x8xi64> {
    %0 = enzymexla.jit_call @jitcall (%arg0) {output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>]} : (tensor<8x8xi64>) -> tensor<8x8xi64>
    return %0 : tensor<8x8xi64>
  }
}

// CHECK-LABEL: func.func @jitcall
// CHECK-SAME:  %[[ARG:.*]]: memref<8x8xi64, 1>
// CHECK:       %[[ADDR:.*]] = "enzymexla.memref2pointer"(%[[ARG]]) : (memref<8x8xi64, 1>) -> !llvm.ptr<1>
// CHECK:       llvm.getelementptr {{.*}} %[[ADDR]]

// -----

module {
  func.func @jitcall(%arg0: !llvm.ptr<1>) {
    %0 = llvm.mlir.constant(63 : i32) : i32
    %1 = nvvm.read.ptx.sreg.tid.x : i32
    %4 = llvm.zext %1 : i32 to i64
    %5 = llvm.getelementptr inbounds %arg0[%4] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i64
    %6 = llvm.load %5 {alignment = 1 : i64} : !llvm.ptr<1> -> i64
    %7 = llvm.mul %6, %6 : i64
    llvm.store %7, %5 {alignment = 1 : i64} : i64, !llvm.ptr<1>
    llvm.return
  }
  func.func @main(%arg0: tensor<8x8xi64>) -> tensor<8x8xi64> {
    %0 = enzymexla.jit_call @jitcall (%arg0) {output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>]} : (tensor<8x8xi64>) -> tensor<8x8xi64>
    return %0 : tensor<8x8xi64>
  }
}

// CHECK-LABEL: func.func @jitcall
// CHECK-SAME:  %[[ARG:.*]]: memref<8x8xi64, 1>
// CHECK:       %[[ADDR:.*]] = "enzymexla.memref2pointer"(%[[ARG]]) : (memref<8x8xi64, 1>) -> !llvm.ptr<1>
// CHECK:       llvm.getelementptr {{.*}} %[[ADDR]]
