// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(optimize-communication{dus_to_pad_manual_comp_comm=1 dus_to_pad_comm=0})" %s | FileCheck %s
// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(optimize-communication{dus_to_pad_manual_comp_comm=0 dus_to_pad_comm=1})" %s | FileCheck %s --check-prefix=PAD

sdy.mesh @mesh = <["x"=2, "y"=2, "z"=1]>

func.func @constantUpdate1D(%arg21: tensor<20x24x96xf64> {sdy.sharding = #sdy.sharding<@mesh, [{"z"}, {"y"}, {"x"}]>}) -> (tensor<20x24x96xf64> {sdy.sharding = #sdy.sharding<@mesh, [{"z"}, {"y"}, {"x"}]>}) {
    %c0 = sdy.constant dense<0> : tensor<i32>
    %274 = sdy.constant dense<8> : tensor<i32>
    %275 = sdy.constant dense<0.000000e+00> : tensor<20x24x80xf64>
    %305 = stablehlo.dynamic_update_slice %arg21, %275, %c0, %c0, %274 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<20x24x96xf64>, tensor<20x24x80xf64>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<20x24x96xf64>
    return %305 : tensor<20x24x96xf64>
}

// PAD: func.func @constantUpdate1D(%arg0: tensor<20x24x96xf64> {sdy.sharding = #sdy.sharding<@mesh, [{"z"}, {"y"}, {"x"}]>}) -> (tensor<20x24x96xf64> {sdy.sharding = #sdy.sharding<@mesh, [{"z"}, {"y"}, {"x"}]>}) {
// PAD-NEXT:     %[[cst1:.+]] = stablehlo.constant dense<1.000000e+00> : tensor<f64>
// PAD-NEXT:     %[[cst0:.+]] = stablehlo.constant {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z"}, {"y"}, {"x"}]>]>} dense<0.000000e+00> : tensor<20x24x80xf64>
// PAD-NEXT:     %[[p0:.+]] = stablehlo.pad %[[cst0]], %[[cst1]], low = [0, 0, 8], high = [0, 0, 8], interior = [0, 0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<20x24x80xf64>, tensor<f64>) -> tensor<20x24x96xf64>
// PAD-NEXT:     %[[m0:.+]] = stablehlo.multiply %arg0, %[[p0]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z"}, {"y"}, {"x"}]>]>} : tensor<20x24x96xf64>
// PAD-NEXT:     return %[[m0]] : tensor<20x24x96xf64>
// PAD-NEXT: }

// CHECK:  func.func @constantUpdate1D(%arg0: tensor<20x24x96xf64> {sdy.sharding = #sdy.sharding<@mesh, [{"z"}, {"y"}, {"x"}]>}) -> (tensor<20x24x96xf64> {sdy.sharding = #sdy.sharding<@mesh, [{"z"}, {"y"}, {"x"}]>}) {
// CHECK-NEXT:    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"z"}, {"y"}, {"x"}]>] out_shardings=[<@mesh, [{"z"}, {"y"}, {"x"}]>] manual_axes={"z", "y", "x"} (%arg1: tensor<20x12x48xf64>) {
// CHECK-NEXT:      %c = stablehlo.constant dense<40> : tensor<20x12x48xui32>
// CHECK-NEXT:      %c_0 = stablehlo.constant dense<8> : tensor<20x12x48xui32>
// CHECK-NEXT:      %c_1 = stablehlo.constant dense<1> : tensor<ui32>
// CHECK-NEXT:      %c_2 = stablehlo.constant dense<2> : tensor<ui32>
// CHECK-NEXT:      %c_3 = stablehlo.constant dense<0> : tensor<ui32>
// CHECK-NEXT:      %cst = stablehlo.constant dense<0.000000e+00> : tensor<20x12x48xf64>
// CHECK-NEXT:      %1 = stablehlo.partition_id : tensor<ui32>
// CHECK-NEXT:      %2 = stablehlo.divide %1, %c_2 : tensor<ui32>
// CHECK-NEXT:      %3 = stablehlo.compare  LE, %2, %c_3 : (tensor<ui32>, tensor<ui32>) -> tensor<i1>
// CHECK-NEXT:      %4 = stablehlo.compare  GE, %2, %c_1 : (tensor<ui32>, tensor<ui32>) -> tensor<i1>
// CHECK-NEXT:      %5 = stablehlo.or %3, %4 : tensor<i1>
// CHECK-NEXT:      %6 = "stablehlo.if"(%5) ({
// CHECK-NEXT:        %7 = stablehlo.iota dim = 2 : tensor<20x12x48xui32>
// CHECK-NEXT:        %8 = stablehlo.compare  LT, %7, %c_0 : (tensor<20x12x48xui32>, tensor<20x12x48xui32>) -> tensor<20x12x48xi1>
// CHECK-NEXT:        %9 = stablehlo.compare  EQ, %2, %c_3 : (tensor<ui32>, tensor<ui32>) -> tensor<i1>
// CHECK-NEXT:        %10 = stablehlo.broadcast_in_dim %9, dims = [] : (tensor<i1>) -> tensor<20x12x48xi1>
// CHECK-NEXT:        %11 = stablehlo.and %8, %10 : tensor<20x12x48xi1>
// CHECK-NEXT:        %12 = stablehlo.compare  GE, %7, %c : (tensor<20x12x48xui32>, tensor<20x12x48xui32>) -> tensor<20x12x48xi1>
// CHECK-NEXT:        %13 = stablehlo.compare  EQ, %2, %c_1 : (tensor<ui32>, tensor<ui32>) -> tensor<i1>
// CHECK-NEXT:        %14 = stablehlo.broadcast_in_dim %13, dims = [] : (tensor<i1>) -> tensor<20x12x48xi1>
// CHECK-NEXT:        %15 = stablehlo.and %12, %14 : tensor<20x12x48xi1>
// CHECK-NEXT:        %16 = stablehlo.or %11, %15 : tensor<20x12x48xi1>
// CHECK-NEXT:        %17 = stablehlo.select %16, %arg1, %cst : tensor<20x12x48xi1>, tensor<20x12x48xf64>
// CHECK-NEXT:        stablehlo.return %17 : tensor<20x12x48xf64>
// CHECK-NEXT:      }, {
// CHECK-NEXT:        stablehlo.return %cst : tensor<20x12x48xf64>
// CHECK-NEXT:      }) : (tensor<i1>) -> tensor<20x12x48xf64>
// CHECK-NEXT:      sdy.return %6 : tensor<20x12x48xf64>
// CHECK-NEXT:    } : (tensor<20x24x96xf64>) -> tensor<20x24x96xf64>
// CHECK-NEXT:    return %0 : tensor<20x24x96xf64>
// CHECK-NEXT:  }

func.func @constantUpdateOver(%arg21: tensor<20x24x96xf64> {sdy.sharding = #sdy.sharding<@mesh, [{"z"}, {"y"}, {"x"}]>}) -> (tensor<20x24x96xf64> {sdy.sharding = #sdy.sharding<@mesh, [{"z"}, {"y"}, {"x"}]>}) {
    %274 = sdy.constant dense<8> : tensor<i32>
    %275 = sdy.constant dense<0.000000e+00> : tensor<4x1x80xf64>
    %305 = stablehlo.dynamic_update_slice %arg21, %275, %274, %274, %274 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<20x24x96xf64>, tensor<4x1x80xf64>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<20x24x96xf64>
    return %305 : tensor<20x24x96xf64>
}

// PAD: func.func @constantUpdateOver(%arg0: tensor<20x24x96xf64> {sdy.sharding = #sdy.sharding<@mesh, [{"z"}, {"y"}, {"x"}]>}) -> (tensor<20x24x96xf64> {sdy.sharding = #sdy.sharding<@mesh, [{"z"}, {"y"}, {"x"}]>}) {
// PAD-DAG:     %[[cst0:.+]] = stablehlo.constant {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z"}, {"y"}, {"x"}]>]>} dense<0.000000e+00> : tensor<4x1x80xf64>
// PAD-DAG:     %[[cst1:.+]] = stablehlo.constant dense<1.000000e+00> : tensor<f64>
// PAD-NEXT:     %[[mask:.+]] = stablehlo.pad %[[cst0]], %[[cst1]], low = [8, 8, 8], high = [8, 15, 8], interior = [0, 0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<4x1x80xf64>, tensor<f64>) -> tensor<20x24x96xf64>
// PAD-NEXT:     %[[mul:.+]] = stablehlo.multiply %arg0, %[[mask]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z"}, {"y"}, {"x"}]>]>} : tensor<20x24x96xf64>
// PAD-NEXT:     return %[[mul]] : tensor<20x24x96xf64>
// PAD-NEXT: }

// CHECK:  func.func @constantUpdateOver(%arg0: tensor<20x24x96xf64> {sdy.sharding = #sdy.sharding<@mesh, [{"z"}, {"y"}, {"x"}]>}) -> (tensor<20x24x96xf64> {sdy.sharding = #sdy.sharding<@mesh, [{"z"}, {"y"}, {"x"}]>}) {
// CHECK-NEXT:    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"z"}, {"y"}, {"x"}]>] out_shardings=[<@mesh, [{"z"}, {"y"}, {"x"}]>] manual_axes={"z", "y", "x"} (%arg1: tensor<20x12x48xf64>) {
// CHECK-NEXT:      %c = stablehlo.constant dense<40> : tensor<20x12x48xui32>
// CHECK-NEXT:      %c_0 = stablehlo.constant dense<9> : tensor<20x12x48xui32>
// CHECK-NEXT:      %c_1 = stablehlo.constant dense<8> : tensor<20x12x48xui32>
// CHECK-NEXT:      %c_2 = stablehlo.constant dense<1> : tensor<ui32>
// CHECK-NEXT:      %c_3 = stablehlo.constant dense<2> : tensor<ui32>
// CHECK-NEXT:      %c_4 = stablehlo.constant dense<8> : tensor<ui32>
// CHECK-NEXT:      %c_5 = stablehlo.constant dense<0> : tensor<ui32>
// CHECK-NEXT:      %cst = stablehlo.constant dense<0.000000e+00> : tensor<20x12x48xf64>
// CHECK-NEXT:      %1 = stablehlo.partition_id : tensor<ui32>
// CHECK-NEXT:      %2 = stablehlo.dynamic_update_slice %arg1, %cst, %c_4, %c_5, %c_5 : (tensor<20x12x48xf64>, tensor<20x12x48xf64>, tensor<ui32>, tensor<ui32>, tensor<ui32>) -> tensor<20x12x48xf64>
// CHECK-NEXT:      %3 = stablehlo.remainder %1, %c_3 : tensor<ui32>
// CHECK-NEXT:      %4 = stablehlo.divide %1, %c_3 : tensor<ui32>
// CHECK-NEXT:      %5 = stablehlo.compare  LE, %3, %c_5 : (tensor<ui32>, tensor<ui32>) -> tensor<i1>
// CHECK-NEXT:      %6 = stablehlo.compare  LE, %4, %c_5 : (tensor<ui32>, tensor<ui32>) -> tensor<i1>
// CHECK-NEXT:      %7 = stablehlo.compare  GE, %3, %c_5 : (tensor<ui32>, tensor<ui32>) -> tensor<i1>
// CHECK-NEXT:      %8 = stablehlo.compare  GE, %4, %c_2 : (tensor<ui32>, tensor<ui32>) -> tensor<i1>
// CHECK-NEXT:      %9 = stablehlo.or %5, %7 : tensor<i1>
// CHECK-NEXT:      %10 = stablehlo.or %9, %6 : tensor<i1>
// CHECK-NEXT:      %11 = stablehlo.or %10, %8 : tensor<i1>
// CHECK-NEXT:      %12 = "stablehlo.if"(%11) ({
// CHECK-NEXT:        %13 = stablehlo.iota dim = 1 : tensor<20x12x48xui32>
// CHECK-NEXT:        %14 = stablehlo.compare  LT, %13, %c_1 : (tensor<20x12x48xui32>, tensor<20x12x48xui32>) -> tensor<20x12x48xi1>
// CHECK-NEXT:        %15 = stablehlo.compare  EQ, %3, %c_5 : (tensor<ui32>, tensor<ui32>) -> tensor<i1>
// CHECK-NEXT:        %16 = stablehlo.broadcast_in_dim %15, dims = [] : (tensor<i1>) -> tensor<20x12x48xi1>
// CHECK-NEXT:        %17 = stablehlo.and %14, %16 : tensor<20x12x48xi1>
// CHECK-NEXT:        %18 = stablehlo.compare  GE, %13, %c_0 : (tensor<20x12x48xui32>, tensor<20x12x48xui32>) -> tensor<20x12x48xi1>
// CHECK-NEXT:        %19 = stablehlo.compare  EQ, %3, %c_5 : (tensor<ui32>, tensor<ui32>) -> tensor<i1>
// CHECK-NEXT:        %20 = stablehlo.broadcast_in_dim %19, dims = [] : (tensor<i1>) -> tensor<20x12x48xi1>
// CHECK-NEXT:        %21 = stablehlo.and %18, %20 : tensor<20x12x48xi1>
// CHECK-NEXT:        %22 = stablehlo.compare  GT, %3, %c_5 : (tensor<ui32>, tensor<ui32>) -> tensor<i1>
// CHECK-NEXT:        %23 = stablehlo.broadcast_in_dim %22, dims = [] : (tensor<i1>) -> tensor<20x12x48xi1>
// CHECK-NEXT:        %24 = stablehlo.or %21, %23 : tensor<20x12x48xi1>
// CHECK-NEXT:        %25 = stablehlo.or %17, %24 : tensor<20x12x48xi1>
// CHECK-NEXT:        %26 = stablehlo.iota dim = 2 : tensor<20x12x48xui32>
// CHECK-NEXT:        %27 = stablehlo.compare  LT, %26, %c_1 : (tensor<20x12x48xui32>, tensor<20x12x48xui32>) -> tensor<20x12x48xi1>
// CHECK-NEXT:        %28 = stablehlo.compare  EQ, %4, %c_5 : (tensor<ui32>, tensor<ui32>) -> tensor<i1>
// CHECK-NEXT:        %29 = stablehlo.broadcast_in_dim %28, dims = [] : (tensor<i1>) -> tensor<20x12x48xi1>
// CHECK-NEXT:        %30 = stablehlo.and %27, %29 : tensor<20x12x48xi1>
// CHECK-NEXT:        %31 = stablehlo.compare  GE, %26, %c : (tensor<20x12x48xui32>, tensor<20x12x48xui32>) -> tensor<20x12x48xi1>
// CHECK-NEXT:        %32 = stablehlo.compare  EQ, %4, %c_2 : (tensor<ui32>, tensor<ui32>) -> tensor<i1>
// CHECK-NEXT:        %33 = stablehlo.broadcast_in_dim %32, dims = [] : (tensor<i1>) -> tensor<20x12x48xi1>
// CHECK-NEXT:        %34 = stablehlo.and %31, %33 : tensor<20x12x48xi1>
// CHECK-NEXT:        %35 = stablehlo.or %30, %34 : tensor<20x12x48xi1>
// CHECK-NEXT:        %36 = stablehlo.or %25, %35 : tensor<20x12x48xi1>
// CHECK-NEXT:        %37 = stablehlo.select %36, %arg1, %2 : tensor<20x12x48xi1>, tensor<20x12x48xf64>
// CHECK-NEXT:        stablehlo.return %37 : tensor<20x12x48xf64>
// CHECK-NEXT:      }, {
// CHECK-NEXT:        stablehlo.return %2 : tensor<20x12x48xf64>
// CHECK-NEXT:      }) : (tensor<i1>) -> tensor<20x12x48xf64>
// CHECK-NEXT:      sdy.return %12 : tensor<20x12x48xf64>
// CHECK-NEXT:    } : (tensor<20x24x96xf64>) -> tensor<20x24x96xf64>
// CHECK-NEXT:    return %0 : tensor<20x24x96xf64>
// CHECK-NEXT:  }

func.func @constantUpdate(%arg21: tensor<20x24x96xf64> {sdy.sharding = #sdy.sharding<@mesh, [{"z"}, {"y"}, {"x"}]>}) -> (tensor<20x24x96xf64> {sdy.sharding = #sdy.sharding<@mesh, [{"z"}, {"y"}, {"x"}]>}) {
    %274 = sdy.constant dense<8> : tensor<i32>
    %275 = sdy.constant dense<0.000000e+00> : tensor<4x8x80xf64>
    %305 = stablehlo.dynamic_update_slice %arg21, %275, %274, %274, %274 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<20x24x96xf64>, tensor<4x8x80xf64>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<20x24x96xf64>
    return %305 : tensor<20x24x96xf64>
}

// PAD: func.func @constantUpdate(%arg0: tensor<20x24x96xf64> {sdy.sharding = #sdy.sharding<@mesh, [{"z"}, {"y"}, {"x"}]>}) -> (tensor<20x24x96xf64> {sdy.sharding = #sdy.sharding<@mesh, [{"z"}, {"y"}, {"x"}]>}) {
// PAD-DAG:     %[[c0:.+]] = stablehlo.constant {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z"}, {"y"}, {"x"}]>]>} dense<0.000000e+00> : tensor<4x8x80xf64>
// PAD-DAG:     %[[cst1:.+]] = stablehlo.constant dense<1.000000e+00> : tensor<f64>
// PAD-NEXT:     %[[mask:.+]] = stablehlo.pad %[[c0]], %[[cst1]], low = [8, 8, 8], high = [8, 8, 8], interior = [0, 0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<4x8x80xf64>, tensor<f64>) -> tensor<20x24x96xf64>
// PAD-NEXT:     %[[mul:.+]] = stablehlo.multiply %arg0, %[[mask]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z"}, {"y"}, {"x"}]>]>} : tensor<20x24x96xf64>
// PAD-NEXT:     return %[[mul]] : tensor<20x24x96xf64>
// PAD-NEXT: }

// CHECK:  func.func @constantUpdate(%arg0: tensor<20x24x96xf64> {sdy.sharding = #sdy.sharding<@mesh, [{"z"}, {"y"}, {"x"}]>}) -> (tensor<20x24x96xf64> {sdy.sharding = #sdy.sharding<@mesh, [{"z"}, {"y"}, {"x"}]>}) {
// CHECK-NEXT:    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"z"}, {"y"}, {"x"}]>] out_shardings=[<@mesh, [{"z"}, {"y"}, {"x"}]>] manual_axes={"z", "y", "x"} (%arg1: tensor<20x12x48xf64>) {
// CHECK-NEXT:      %c = stablehlo.constant dense<40> : tensor<20x12x48xui32>
// CHECK-NEXT:      %c_0 = stablehlo.constant dense<4> : tensor<20x12x48xui32>
// CHECK-NEXT:      %c_1 = stablehlo.constant dense<8> : tensor<20x12x48xui32>
// CHECK-NEXT:      %c_2 = stablehlo.constant dense<1> : tensor<ui32>
// CHECK-NEXT:      %c_3 = stablehlo.constant dense<2> : tensor<ui32>
// CHECK-NEXT:      %c_4 = stablehlo.constant dense<8> : tensor<ui32>
// CHECK-NEXT:      %c_5 = stablehlo.constant dense<0> : tensor<ui32>
// CHECK-NEXT:      %cst = stablehlo.constant dense<0.000000e+00> : tensor<20x12x48xf64>
// CHECK-NEXT:      %1 = stablehlo.partition_id : tensor<ui32>
// CHECK-NEXT:      %2 = stablehlo.dynamic_update_slice %arg1, %cst, %c_4, %c_5, %c_5 : (tensor<20x12x48xf64>, tensor<20x12x48xf64>, tensor<ui32>, tensor<ui32>, tensor<ui32>) -> tensor<20x12x48xf64>
// CHECK-NEXT:      %3 = stablehlo.remainder %1, %c_3 : tensor<ui32>
// CHECK-NEXT:      %4 = stablehlo.divide %1, %c_3 : tensor<ui32>
// CHECK-NEXT:      %5 = stablehlo.compare  LE, %3, %c_5 : (tensor<ui32>, tensor<ui32>) -> tensor<i1>
// CHECK-NEXT:      %6 = stablehlo.compare  LE, %4, %c_5 : (tensor<ui32>, tensor<ui32>) -> tensor<i1>
// CHECK-NEXT:      %7 = stablehlo.compare  GE, %3, %c_2 : (tensor<ui32>, tensor<ui32>) -> tensor<i1>
// CHECK-NEXT:      %8 = stablehlo.compare  GE, %4, %c_2 : (tensor<ui32>, tensor<ui32>) -> tensor<i1>
// CHECK-NEXT:      %9 = stablehlo.or %5, %7 : tensor<i1>
// CHECK-NEXT:      %10 = stablehlo.or %9, %6 : tensor<i1>
// CHECK-NEXT:      %11 = stablehlo.or %10, %8 : tensor<i1>
// CHECK-NEXT:      %12 = "stablehlo.if"(%11) ({
// CHECK-NEXT:        %13 = stablehlo.iota dim = 1 : tensor<20x12x48xui32>
// CHECK-NEXT:        %14 = stablehlo.compare  LT, %13, %c_1 : (tensor<20x12x48xui32>, tensor<20x12x48xui32>) -> tensor<20x12x48xi1>
// CHECK-NEXT:        %15 = stablehlo.compare  EQ, %3, %c_5 : (tensor<ui32>, tensor<ui32>) -> tensor<i1>
// CHECK-NEXT:        %16 = stablehlo.broadcast_in_dim %15, dims = [] : (tensor<i1>) -> tensor<20x12x48xi1>
// CHECK-NEXT:        %17 = stablehlo.and %14, %16 : tensor<20x12x48xi1>
// CHECK-NEXT:        %18 = stablehlo.compare  GE, %13, %c_0 : (tensor<20x12x48xui32>, tensor<20x12x48xui32>) -> tensor<20x12x48xi1>
// CHECK-NEXT:        %19 = stablehlo.compare  EQ, %3, %c_2 : (tensor<ui32>, tensor<ui32>) -> tensor<i1>
// CHECK-NEXT:        %20 = stablehlo.broadcast_in_dim %19, dims = [] : (tensor<i1>) -> tensor<20x12x48xi1>
// CHECK-NEXT:        %21 = stablehlo.and %18, %20 : tensor<20x12x48xi1>
// CHECK-NEXT:        %22 = stablehlo.or %17, %21 : tensor<20x12x48xi1>
// CHECK-NEXT:        %23 = stablehlo.iota dim = 2 : tensor<20x12x48xui32>
// CHECK-NEXT:        %24 = stablehlo.compare  LT, %23, %c_1 : (tensor<20x12x48xui32>, tensor<20x12x48xui32>) -> tensor<20x12x48xi1>
// CHECK-NEXT:        %25 = stablehlo.compare  EQ, %4, %c_5 : (tensor<ui32>, tensor<ui32>) -> tensor<i1>
// CHECK-NEXT:        %26 = stablehlo.broadcast_in_dim %25, dims = [] : (tensor<i1>) -> tensor<20x12x48xi1>
// CHECK-NEXT:        %27 = stablehlo.and %24, %26 : tensor<20x12x48xi1>
// CHECK-NEXT:        %28 = stablehlo.compare  GE, %23, %c : (tensor<20x12x48xui32>, tensor<20x12x48xui32>) -> tensor<20x12x48xi1>
// CHECK-NEXT:        %29 = stablehlo.compare  EQ, %4, %c_2 : (tensor<ui32>, tensor<ui32>) -> tensor<i1>
// CHECK-NEXT:        %30 = stablehlo.broadcast_in_dim %29, dims = [] : (tensor<i1>) -> tensor<20x12x48xi1>
// CHECK-NEXT:        %31 = stablehlo.and %28, %30 : tensor<20x12x48xi1>
// CHECK-NEXT:        %32 = stablehlo.or %27, %31 : tensor<20x12x48xi1>
// CHECK-NEXT:        %33 = stablehlo.or %22, %32 : tensor<20x12x48xi1>
// CHECK-NEXT:        %34 = stablehlo.select %33, %arg1, %2 : tensor<20x12x48xi1>, tensor<20x12x48xf64>
// CHECK-NEXT:        stablehlo.return %34 : tensor<20x12x48xf64>
// CHECK-NEXT:      }, {
// CHECK-NEXT:        stablehlo.return %2 : tensor<20x12x48xf64>
// CHECK-NEXT:      }) : (tensor<i1>) -> tensor<20x12x48xf64>
// CHECK-NEXT:      sdy.return %12 : tensor<20x12x48xf64>
// CHECK-NEXT:    } : (tensor<20x24x96xf64>) -> tensor<20x24x96xf64>
// CHECK-NEXT:    return %0 : tensor<20x24x96xf64>
// CHECK-NEXT:  }

func.func @argUpdate1D(%arg21: tensor<20x24x96xf64> {sdy.sharding = #sdy.sharding<@mesh, [{"z"}, {"y"}, {"x"}]>}, %275 : tensor<20x24x80xf64> {sdy.sharding = #sdy.sharding<@mesh, [{"z"}, {"y"}, {"x"}]>}) -> (tensor<20x24x96xf64> {sdy.sharding = #sdy.sharding<@mesh, [{"z"}, {"y"}, {"x"}]>}) {
    %274 = sdy.constant dense<8> : tensor<i32>
    %c0 = sdy.constant dense<0> : tensor<i32>
    %305 = stablehlo.dynamic_update_slice %arg21, %275, %c0, %c0, %274 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<20x24x96xf64>, tensor<20x24x80xf64>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<20x24x96xf64>
    return %305 : tensor<20x24x96xf64>
}

// PAD: func.func @argUpdate1D(%arg0: tensor<20x24x96xf64> {sdy.sharding = #sdy.sharding<@mesh, [{"z"}, {"y"}, {"x"}]>}, %arg1: tensor<20x24x80xf64> {sdy.sharding = #sdy.sharding<@mesh, [{"z"}, {"y"}, {"x"}]>}) -> (tensor<20x24x96xf64> {sdy.sharding = #sdy.sharding<@mesh, [{"z"}, {"y"}, {"x"}]>}) {
// PAD-NEXT:     %cst = stablehlo.constant {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z"}, {"y"}, {"x"}]>]>} dense<0.000000e+00> : tensor<20x24x80xf64>
// PAD-NEXT:     %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// PAD-NEXT:     %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
// PAD-NEXT:     %0 = stablehlo.pad %arg1, %cst_0, low = [0, 0, 8], high = [0, 0, 8], interior = [0, 0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<20x24x80xf64>, tensor<f64>) -> tensor<20x24x96xf64>
// PAD-NEXT:     %1 = stablehlo.pad %cst, %cst_1, low = [0, 0, 8], high = [0, 0, 8], interior = [0, 0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<20x24x80xf64>, tensor<f64>) -> tensor<20x24x96xf64>
// PAD-NEXT:     %2 = stablehlo.multiply %arg0, %1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z"}, {"y"}, {"x"}]>]>} : tensor<20x24x96xf64>
// PAD-NEXT:     %3 = stablehlo.add %2, %0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z"}, {"y"}, {"x"}]>]>} : tensor<20x24x96xf64>
// PAD-NEXT:     return %3 : tensor<20x24x96xf64>
// PAD-NEXT: }

// CHECK:  func.func @argUpdate1D(%arg0: tensor<20x24x96xf64> {sdy.sharding = #sdy.sharding<@mesh, [{"z"}, {"y"}, {"x"}]>}, %arg1: tensor<20x24x80xf64> {sdy.sharding = #sdy.sharding<@mesh, [{"z"}, {"y"}, {"x"}]>}) -> (tensor<20x24x96xf64> {sdy.sharding = #sdy.sharding<@mesh, [{"z"}, {"y"}, {"x"}]>}) {
// CHECK-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:    %0 = stablehlo.pad %arg1, %cst, low = [0, 0, 8], high = [0, 0, 8], interior = [0, 0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<20x24x80xf64>, tensor<f64>) -> tensor<20x24x96xf64>
// CHECK-NEXT:    %1 = sdy.manual_computation(%arg0, %0) in_shardings=[<@mesh, [{"z"}, {"y"}, {"x"}]>, <@mesh, [{"z"}, {"y"}, {"x"}]>] out_shardings=[<@mesh, [{"z"}, {"y"}, {"x"}]>] manual_axes={"z", "y", "x"} (%arg2: tensor<20x12x48xf64>, %arg3: tensor<20x12x48xf64>) {
// CHECK-NEXT:      %c = stablehlo.constant dense<40> : tensor<20x12x48xui32>
// CHECK-NEXT:      %c_0 = stablehlo.constant dense<8> : tensor<20x12x48xui32>
// CHECK-NEXT:      %c_1 = stablehlo.constant dense<1> : tensor<ui32>
// CHECK-NEXT:      %c_2 = stablehlo.constant dense<2> : tensor<ui32>
// CHECK-NEXT:      %c_3 = stablehlo.constant dense<0> : tensor<ui32>
// CHECK-NEXT:      %2 = stablehlo.partition_id : tensor<ui32>
// CHECK-NEXT:      %3 = stablehlo.divide %2, %c_2 : tensor<ui32>
// CHECK-NEXT:      %4 = stablehlo.compare  LE, %3, %c_3 : (tensor<ui32>, tensor<ui32>) -> tensor<i1>
// CHECK-NEXT:      %5 = stablehlo.compare  GE, %3, %c_1 : (tensor<ui32>, tensor<ui32>) -> tensor<i1>
// CHECK-NEXT:      %6 = stablehlo.or %4, %5 : tensor<i1>
// CHECK-NEXT:      %7 = "stablehlo.if"(%6) ({
// CHECK-NEXT:        %8 = stablehlo.iota dim = 2 : tensor<20x12x48xui32>
// CHECK-NEXT:        %9 = stablehlo.compare  LT, %8, %c_0 : (tensor<20x12x48xui32>, tensor<20x12x48xui32>) -> tensor<20x12x48xi1>
// CHECK-NEXT:        %10 = stablehlo.compare  EQ, %3, %c_3 : (tensor<ui32>, tensor<ui32>) -> tensor<i1>
// CHECK-NEXT:        %11 = stablehlo.broadcast_in_dim %10, dims = [] : (tensor<i1>) -> tensor<20x12x48xi1>
// CHECK-NEXT:        %12 = stablehlo.and %9, %11 : tensor<20x12x48xi1>
// CHECK-NEXT:        %13 = stablehlo.compare  GE, %8, %c : (tensor<20x12x48xui32>, tensor<20x12x48xui32>) -> tensor<20x12x48xi1>
// CHECK-NEXT:        %14 = stablehlo.compare  EQ, %3, %c_1 : (tensor<ui32>, tensor<ui32>) -> tensor<i1>
// CHECK-NEXT:        %15 = stablehlo.broadcast_in_dim %14, dims = [] : (tensor<i1>) -> tensor<20x12x48xi1>
// CHECK-NEXT:        %16 = stablehlo.and %13, %15 : tensor<20x12x48xi1>
// CHECK-NEXT:        %17 = stablehlo.or %12, %16 : tensor<20x12x48xi1>
// CHECK-NEXT:        %18 = stablehlo.select %17, %arg2, %arg3 : tensor<20x12x48xi1>, tensor<20x12x48xf64>
// CHECK-NEXT:        stablehlo.return %18 : tensor<20x12x48xf64>
// CHECK-NEXT:      }, {
// CHECK-NEXT:        stablehlo.return %arg3 : tensor<20x12x48xf64>
// CHECK-NEXT:      }) : (tensor<i1>) -> tensor<20x12x48xf64>
// CHECK-NEXT:      sdy.return %7 : tensor<20x12x48xf64>
// CHECK-NEXT:    } : (tensor<20x24x96xf64>, tensor<20x24x96xf64>) -> tensor<20x24x96xf64>
// CHECK-NEXT:    return %1 : tensor<20x24x96xf64>
// CHECK-NEXT:  }

func.func @argUpdateOver(%arg21: tensor<20x24x96xf64> {sdy.sharding = #sdy.sharding<@mesh, [{"z"}, {"y"}, {"x"}]>}, %275 : tensor<4x1x80xf64> {sdy.sharding = #sdy.sharding<@mesh, [{"z"}, {"y"}, {"x"}]>}) -> (tensor<20x24x96xf64> {sdy.sharding = #sdy.sharding<@mesh, [{"z"}, {"y"}, {"x"}]>}) {
    %274 = sdy.constant dense<8> : tensor<i32>
    %305 = stablehlo.dynamic_update_slice %arg21, %275, %274, %274, %274 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<20x24x96xf64>, tensor<4x1x80xf64>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<20x24x96xf64>
    return %305 : tensor<20x24x96xf64>
}

// PAD: func.func @argUpdateOver(%arg0: tensor<20x24x96xf64> {sdy.sharding = #sdy.sharding<@mesh, [{"z"}, {"y"}, {"x"}]>}, %arg1: tensor<4x1x80xf64> {sdy.sharding = #sdy.sharding<@mesh, [{"z"}, {"y"}, {"x"}]>}) -> (tensor<20x24x96xf64> {sdy.sharding = #sdy.sharding<@mesh, [{"z"}, {"y"}, {"x"}]>}) {
// PAD-DAG:     %[[c0:.+]] = stablehlo.constant {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z"}, {"y"}, {"x"}]>]>} dense<0.000000e+00> : tensor<4x1x80xf64>
// PAD-DAG:     %[[cst0:.+]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// PAD-DAG:     %[[cst1:.+]] = stablehlo.constant dense<1.000000e+00> : tensor<f64>
// PAD-NEXT:     %0 = stablehlo.pad %arg1, %[[cst0]], low = [8, 8, 8], high = [8, 15, 8], interior = [0, 0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<4x1x80xf64>, tensor<f64>) -> tensor<20x24x96xf64>
// PAD-NEXT:     %1 = stablehlo.pad %[[c0]], %[[cst1]], low = [8, 8, 8], high = [8, 15, 8], interior = [0, 0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<4x1x80xf64>, tensor<f64>) -> tensor<20x24x96xf64>
// PAD-NEXT:     %2 = stablehlo.multiply %arg0, %1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z"}, {"y"}, {"x"}]>]>} : tensor<20x24x96xf64>
// PAD-NEXT:     %3 = stablehlo.add %2, %0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z"}, {"y"}, {"x"}]>]>} : tensor<20x24x96xf64>
// PAD-NEXT:     return %3 : tensor<20x24x96xf64>
// PAD-NEXT: }

// CHECK:  func.func @argUpdateOver(%arg0: tensor<20x24x96xf64> {sdy.sharding = #sdy.sharding<@mesh, [{"z"}, {"y"}, {"x"}]>}, %arg1: tensor<4x1x80xf64> {sdy.sharding = #sdy.sharding<@mesh, [{"z"}, {"y"}, {"x"}]>}) -> (tensor<20x24x96xf64> {sdy.sharding = #sdy.sharding<@mesh, [{"z"}, {"y"}, {"x"}]>}) {
// CHECK-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:    %0 = stablehlo.pad %arg1, %cst, low = [8, 8, 8], high = [8, 15, 8], interior = [0, 0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<4x1x80xf64>, tensor<f64>) -> tensor<20x24x96xf64>
// CHECK-NEXT:    %1 = sdy.manual_computation(%arg0, %0) in_shardings=[<@mesh, [{"z"}, {"y"}, {"x"}]>, <@mesh, [{"z"}, {"y"}, {"x"}]>] out_shardings=[<@mesh, [{"z"}, {"y"}, {"x"}]>] manual_axes={"z", "y", "x"} (%arg2: tensor<20x12x48xf64>, %arg3: tensor<20x12x48xf64>) {
// CHECK-NEXT:      %c = stablehlo.constant dense<40> : tensor<20x12x48xui32>
// CHECK-NEXT:      %c_0 = stablehlo.constant dense<9> : tensor<20x12x48xui32>
// CHECK-NEXT:      %c_1 = stablehlo.constant dense<8> : tensor<20x12x48xui32>
// CHECK-NEXT:      %c_2 = stablehlo.constant dense<1> : tensor<ui32>
// CHECK-NEXT:      %c_3 = stablehlo.constant dense<2> : tensor<ui32>
// CHECK-NEXT:      %c_4 = stablehlo.constant dense<8> : tensor<ui32>
// CHECK-NEXT:      %c_5 = stablehlo.constant dense<0> : tensor<ui32>
// CHECK-NEXT:      %2 = stablehlo.partition_id : tensor<ui32>
// CHECK-NEXT:      %3 = stablehlo.dynamic_update_slice %arg2, %arg3, %c_4, %c_5, %c_5 : (tensor<20x12x48xf64>, tensor<20x12x48xf64>, tensor<ui32>, tensor<ui32>, tensor<ui32>) -> tensor<20x12x48xf64>
// CHECK-NEXT:      %4 = stablehlo.remainder %2, %c_3 : tensor<ui32>
// CHECK-NEXT:      %5 = stablehlo.divide %2, %c_3 : tensor<ui32>
// CHECK-NEXT:      %6 = stablehlo.compare  LE, %4, %c_5 : (tensor<ui32>, tensor<ui32>) -> tensor<i1>
// CHECK-NEXT:      %7 = stablehlo.compare  LE, %5, %c_5 : (tensor<ui32>, tensor<ui32>) -> tensor<i1>
// CHECK-NEXT:      %8 = stablehlo.compare  GE, %4, %c_5 : (tensor<ui32>, tensor<ui32>) -> tensor<i1>
// CHECK-NEXT:      %9 = stablehlo.compare  GE, %5, %c_2 : (tensor<ui32>, tensor<ui32>) -> tensor<i1>
// CHECK-NEXT:      %10 = stablehlo.or %6, %8 : tensor<i1>
// CHECK-NEXT:      %11 = stablehlo.or %10, %7 : tensor<i1>
// CHECK-NEXT:      %12 = stablehlo.or %11, %9 : tensor<i1>
// CHECK-NEXT:      %13 = "stablehlo.if"(%12) ({
// CHECK-NEXT:        %14 = stablehlo.iota dim = 1 : tensor<20x12x48xui32>
// CHECK-NEXT:        %15 = stablehlo.compare  LT, %14, %c_1 : (tensor<20x12x48xui32>, tensor<20x12x48xui32>) -> tensor<20x12x48xi1>
// CHECK-NEXT:        %16 = stablehlo.compare  EQ, %4, %c_5 : (tensor<ui32>, tensor<ui32>) -> tensor<i1>
// CHECK-NEXT:        %17 = stablehlo.broadcast_in_dim %16, dims = [] : (tensor<i1>) -> tensor<20x12x48xi1>
// CHECK-NEXT:        %18 = stablehlo.and %15, %17 : tensor<20x12x48xi1>
// CHECK-NEXT:        %19 = stablehlo.compare  GE, %14, %c_0 : (tensor<20x12x48xui32>, tensor<20x12x48xui32>) -> tensor<20x12x48xi1>
// CHECK-NEXT:        %20 = stablehlo.compare  EQ, %4, %c_5 : (tensor<ui32>, tensor<ui32>) -> tensor<i1>
// CHECK-NEXT:        %21 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<i1>) -> tensor<20x12x48xi1>
// CHECK-NEXT:        %22 = stablehlo.and %19, %21 : tensor<20x12x48xi1>
// CHECK-NEXT:        %23 = stablehlo.compare  GT, %4, %c_5 : (tensor<ui32>, tensor<ui32>) -> tensor<i1>
// CHECK-NEXT:        %24 = stablehlo.broadcast_in_dim %23, dims = [] : (tensor<i1>) -> tensor<20x12x48xi1>
// CHECK-NEXT:        %25 = stablehlo.or %22, %24 : tensor<20x12x48xi1>
// CHECK-NEXT:        %26 = stablehlo.or %18, %25 : tensor<20x12x48xi1>
// CHECK-NEXT:        %27 = stablehlo.iota dim = 2 : tensor<20x12x48xui32>
// CHECK-NEXT:        %28 = stablehlo.compare  LT, %27, %c_1 : (tensor<20x12x48xui32>, tensor<20x12x48xui32>) -> tensor<20x12x48xi1>
// CHECK-NEXT:        %29 = stablehlo.compare  EQ, %5, %c_5 : (tensor<ui32>, tensor<ui32>) -> tensor<i1>
// CHECK-NEXT:        %30 = stablehlo.broadcast_in_dim %29, dims = [] : (tensor<i1>) -> tensor<20x12x48xi1>
// CHECK-NEXT:        %31 = stablehlo.and %28, %30 : tensor<20x12x48xi1>
// CHECK-NEXT:        %32 = stablehlo.compare  GE, %27, %c : (tensor<20x12x48xui32>, tensor<20x12x48xui32>) -> tensor<20x12x48xi1>
// CHECK-NEXT:        %33 = stablehlo.compare  EQ, %5, %c_2 : (tensor<ui32>, tensor<ui32>) -> tensor<i1>
// CHECK-NEXT:        %34 = stablehlo.broadcast_in_dim %33, dims = [] : (tensor<i1>) -> tensor<20x12x48xi1>
// CHECK-NEXT:        %35 = stablehlo.and %32, %34 : tensor<20x12x48xi1>
// CHECK-NEXT:        %36 = stablehlo.or %31, %35 : tensor<20x12x48xi1>
// CHECK-NEXT:        %37 = stablehlo.or %26, %36 : tensor<20x12x48xi1>
// CHECK-NEXT:        %38 = stablehlo.select %37, %arg2, %3 : tensor<20x12x48xi1>, tensor<20x12x48xf64>
// CHECK-NEXT:        stablehlo.return %38 : tensor<20x12x48xf64>
// CHECK-NEXT:      }, {
// CHECK-NEXT:        stablehlo.return %3 : tensor<20x12x48xf64>
// CHECK-NEXT:      }) : (tensor<i1>) -> tensor<20x12x48xf64>
// CHECK-NEXT:      sdy.return %13 : tensor<20x12x48xf64>
// CHECK-NEXT:    } : (tensor<20x24x96xf64>, tensor<20x24x96xf64>) -> tensor<20x24x96xf64>
// CHECK-NEXT:    return %1 : tensor<20x24x96xf64>
// CHECK-NEXT:  }

func.func @argUpdate(%arg21: tensor<20x24x96xf64> {sdy.sharding = #sdy.sharding<@mesh, [{"z"}, {"y"}, {"x"}]>}, %275 : tensor<4x8x80xf64> {sdy.sharding = #sdy.sharding<@mesh, [{"z"}, {"y"}, {"x"}]>}) -> (tensor<20x24x96xf64> {sdy.sharding = #sdy.sharding<@mesh, [{"z"}, {"y"}, {"x"}]>}) {
    %274 = sdy.constant dense<8> : tensor<i32>
    %305 = stablehlo.dynamic_update_slice %arg21, %275, %274, %274, %274 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<20x24x96xf64>, tensor<4x8x80xf64>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<20x24x96xf64>
    return %305 : tensor<20x24x96xf64>
}

// PAD: func.func @argUpdate(%arg0: tensor<20x24x96xf64> {sdy.sharding = #sdy.sharding<@mesh, [{"z"}, {"y"}, {"x"}]>}, %arg1: tensor<4x8x80xf64> {sdy.sharding = #sdy.sharding<@mesh, [{"z"}, {"y"}, {"x"}]>}) -> (tensor<20x24x96xf64> {sdy.sharding = #sdy.sharding<@mesh, [{"z"}, {"y"}, {"x"}]>}) {
// PAD-NEXT:     %cst = stablehlo.constant {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z"}, {"y"}, {"x"}]>]>} dense<0.000000e+00> : tensor<4x8x80xf64>
// PAD-NEXT:     %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// PAD-NEXT:     %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
// PAD-NEXT:     %0 = stablehlo.pad %arg1, %cst_0, low = [8, 8, 8], high = [8, 8, 8], interior = [0, 0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<4x8x80xf64>, tensor<f64>) -> tensor<20x24x96xf64>
// PAD-NEXT:     %1 = stablehlo.pad %cst, %cst_1, low = [8, 8, 8], high = [8, 8, 8], interior = [0, 0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<4x8x80xf64>, tensor<f64>) -> tensor<20x24x96xf64>
// PAD-NEXT:     %2 = stablehlo.multiply %arg0, %1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z"}, {"y"}, {"x"}]>]>} : tensor<20x24x96xf64>
// PAD-NEXT:     %3 = stablehlo.add %2, %0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z"}, {"y"}, {"x"}]>]>} : tensor<20x24x96xf64>
// PAD-NEXT:     return %3 : tensor<20x24x96xf64>
// PAD-NEXT: }

// CHECK:  func.func @argUpdate(%arg0: tensor<20x24x96xf64> {sdy.sharding = #sdy.sharding<@mesh, [{"z"}, {"y"}, {"x"}]>}, %arg1: tensor<4x8x80xf64> {sdy.sharding = #sdy.sharding<@mesh, [{"z"}, {"y"}, {"x"}]>}) -> (tensor<20x24x96xf64> {sdy.sharding = #sdy.sharding<@mesh, [{"z"}, {"y"}, {"x"}]>}) {
// CHECK-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:    %0 = stablehlo.pad %arg1, %cst, low = [8, 8, 8], high = [8, 8, 8], interior = [0, 0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<4x8x80xf64>, tensor<f64>) -> tensor<20x24x96xf64>
// CHECK-NEXT:    %1 = sdy.manual_computation(%arg0, %0) in_shardings=[<@mesh, [{"z"}, {"y"}, {"x"}]>, <@mesh, [{"z"}, {"y"}, {"x"}]>] out_shardings=[<@mesh, [{"z"}, {"y"}, {"x"}]>] manual_axes={"z", "y", "x"} (%arg2: tensor<20x12x48xf64>, %arg3: tensor<20x12x48xf64>) {
// CHECK-NEXT:      %c = stablehlo.constant dense<40> : tensor<20x12x48xui32>
// CHECK-NEXT:      %c_0 = stablehlo.constant dense<4> : tensor<20x12x48xui32>
// CHECK-NEXT:      %c_1 = stablehlo.constant dense<8> : tensor<20x12x48xui32>
// CHECK-NEXT:      %c_2 = stablehlo.constant dense<1> : tensor<ui32>
// CHECK-NEXT:      %c_3 = stablehlo.constant dense<2> : tensor<ui32>
// CHECK-NEXT:      %c_4 = stablehlo.constant dense<8> : tensor<ui32>
// CHECK-NEXT:      %c_5 = stablehlo.constant dense<0> : tensor<ui32>
// CHECK-NEXT:      %2 = stablehlo.partition_id : tensor<ui32>
// CHECK-NEXT:      %3 = stablehlo.dynamic_update_slice %arg2, %arg3, %c_4, %c_5, %c_5 : (tensor<20x12x48xf64>, tensor<20x12x48xf64>, tensor<ui32>, tensor<ui32>, tensor<ui32>) -> tensor<20x12x48xf64>
// CHECK-NEXT:      %4 = stablehlo.remainder %2, %c_3 : tensor<ui32>
// CHECK-NEXT:      %5 = stablehlo.divide %2, %c_3 : tensor<ui32>
// CHECK-NEXT:      %6 = stablehlo.compare  LE, %4, %c_5 : (tensor<ui32>, tensor<ui32>) -> tensor<i1>
// CHECK-NEXT:      %7 = stablehlo.compare  LE, %5, %c_5 : (tensor<ui32>, tensor<ui32>) -> tensor<i1>
// CHECK-NEXT:      %8 = stablehlo.compare  GE, %4, %c_2 : (tensor<ui32>, tensor<ui32>) -> tensor<i1>
// CHECK-NEXT:      %9 = stablehlo.compare  GE, %5, %c_2 : (tensor<ui32>, tensor<ui32>) -> tensor<i1>
// CHECK-NEXT:      %10 = stablehlo.or %6, %8 : tensor<i1>
// CHECK-NEXT:      %11 = stablehlo.or %10, %7 : tensor<i1>
// CHECK-NEXT:      %12 = stablehlo.or %11, %9 : tensor<i1>
// CHECK-NEXT:      %13 = "stablehlo.if"(%12) ({
// CHECK-NEXT:        %14 = stablehlo.iota dim = 1 : tensor<20x12x48xui32>
// CHECK-NEXT:        %15 = stablehlo.compare  LT, %14, %c_1 : (tensor<20x12x48xui32>, tensor<20x12x48xui32>) -> tensor<20x12x48xi1>
// CHECK-NEXT:        %16 = stablehlo.compare  EQ, %4, %c_5 : (tensor<ui32>, tensor<ui32>) -> tensor<i1>
// CHECK-NEXT:        %17 = stablehlo.broadcast_in_dim %16, dims = [] : (tensor<i1>) -> tensor<20x12x48xi1>
// CHECK-NEXT:        %18 = stablehlo.and %15, %17 : tensor<20x12x48xi1>
// CHECK-NEXT:        %19 = stablehlo.compare  GE, %14, %c_0 : (tensor<20x12x48xui32>, tensor<20x12x48xui32>) -> tensor<20x12x48xi1>
// CHECK-NEXT:        %20 = stablehlo.compare  EQ, %4, %c_2 : (tensor<ui32>, tensor<ui32>) -> tensor<i1>
// CHECK-NEXT:        %21 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<i1>) -> tensor<20x12x48xi1>
// CHECK-NEXT:        %22 = stablehlo.and %19, %21 : tensor<20x12x48xi1>
// CHECK-NEXT:        %23 = stablehlo.or %18, %22 : tensor<20x12x48xi1>
// CHECK-NEXT:        %24 = stablehlo.iota dim = 2 : tensor<20x12x48xui32>
// CHECK-NEXT:        %25 = stablehlo.compare  LT, %24, %c_1 : (tensor<20x12x48xui32>, tensor<20x12x48xui32>) -> tensor<20x12x48xi1>
// CHECK-NEXT:        %26 = stablehlo.compare  EQ, %5, %c_5 : (tensor<ui32>, tensor<ui32>) -> tensor<i1>
// CHECK-NEXT:        %27 = stablehlo.broadcast_in_dim %26, dims = [] : (tensor<i1>) -> tensor<20x12x48xi1>
// CHECK-NEXT:        %28 = stablehlo.and %25, %27 : tensor<20x12x48xi1>
// CHECK-NEXT:        %29 = stablehlo.compare  GE, %24, %c : (tensor<20x12x48xui32>, tensor<20x12x48xui32>) -> tensor<20x12x48xi1>
// CHECK-NEXT:        %30 = stablehlo.compare  EQ, %5, %c_2 : (tensor<ui32>, tensor<ui32>) -> tensor<i1>
// CHECK-NEXT:        %31 = stablehlo.broadcast_in_dim %30, dims = [] : (tensor<i1>) -> tensor<20x12x48xi1>
// CHECK-NEXT:        %32 = stablehlo.and %29, %31 : tensor<20x12x48xi1>
// CHECK-NEXT:        %33 = stablehlo.or %28, %32 : tensor<20x12x48xi1>
// CHECK-NEXT:        %34 = stablehlo.or %23, %33 : tensor<20x12x48xi1>
// CHECK-NEXT:        %35 = stablehlo.select %34, %arg2, %3 : tensor<20x12x48xi1>, tensor<20x12x48xf64>
// CHECK-NEXT:        stablehlo.return %35 : tensor<20x12x48xf64>
// CHECK-NEXT:      }, {
// CHECK-NEXT:        stablehlo.return %3 : tensor<20x12x48xf64>
// CHECK-NEXT:      }) : (tensor<i1>) -> tensor<20x12x48xf64>
// CHECK-NEXT:      sdy.return %13 : tensor<20x12x48xf64>
// CHECK-NEXT:    } : (tensor<20x24x96xf64>, tensor<20x24x96xf64>) -> tensor<20x24x96xf64>
// CHECK-NEXT:    return %1 : tensor<20x24x96xf64>
// CHECK-NEXT:  }
