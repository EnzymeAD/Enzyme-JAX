// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(optimize-communication{dus_to_pad_manual_comp_comm=1 dus_to_pad_comm=0})" %s | FileCheck %s
// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(optimize-communication{dus_to_pad_manual_comp_comm=0 dus_to_pad_comm=1})" %s | FileCheck %s --check-prefix=PAD

sdy.mesh @mesh = <["x"=2, "y"=1, "z"=1]>

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
// CHECK-NEXT:    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"z"}, {"y"}, {"x"}]>] out_shardings=[<@mesh, [{"z"}, {"y"}, {"x"}]>] manual_axes={"z", "y", "x"} (%arg1: tensor<20x24x48xf64>) {
// CHECK-NEXT:      %c = stablehlo.constant dense<40> : tensor<20x24x48xui32>
// CHECK-NEXT:      %c_0 = stablehlo.constant dense<8> : tensor<20x24x48xui32>
// CHECK-NEXT:      %c_1 = stablehlo.constant dense<1> : tensor<ui32>
// CHECK-NEXT:      %c_2 = stablehlo.constant dense<0> : tensor<ui32>
// CHECK-NEXT:      %cst = stablehlo.constant dense<0.000000e+00> : tensor<20x24x48xf64>
// CHECK-NEXT:      %1 = stablehlo.partition_id : tensor<ui32>
// CHECK-NEXT:      %2 = stablehlo.compare  LE, %1, %c_2 : (tensor<ui32>, tensor<ui32>) -> tensor<i1>
// CHECK-NEXT:      %3 = stablehlo.compare  GE, %1, %c_1 : (tensor<ui32>, tensor<ui32>) -> tensor<i1>
// CHECK-NEXT:      %4 = stablehlo.or %2, %3 : tensor<i1>
// CHECK-NEXT:      %5 = "stablehlo.if"(%4) ({
// CHECK-NEXT:        %6 = stablehlo.iota dim = 2 : tensor<20x24x48xui32>
// CHECK-NEXT:        %7 = stablehlo.compare  LT, %6, %c_0 : (tensor<20x24x48xui32>, tensor<20x24x48xui32>) -> tensor<20x24x48xi1>
// CHECK-NEXT:        %8 = stablehlo.compare  EQ, %1, %c_2 : (tensor<ui32>, tensor<ui32>) -> tensor<i1>
// CHECK-NEXT:        %9 = stablehlo.broadcast_in_dim %8, dims = [] : (tensor<i1>) -> tensor<20x24x48xi1>
// CHECK-NEXT:        %10 = stablehlo.and %7, %9 : tensor<20x24x48xi1>
// CHECK-NEXT:        %11 = stablehlo.compare  GE, %6, %c : (tensor<20x24x48xui32>, tensor<20x24x48xui32>) -> tensor<20x24x48xi1>
// CHECK-NEXT:        %12 = stablehlo.compare  EQ, %1, %c_1 : (tensor<ui32>, tensor<ui32>) -> tensor<i1>
// CHECK-NEXT:        %13 = stablehlo.broadcast_in_dim %12, dims = [] : (tensor<i1>) -> tensor<20x24x48xi1>
// CHECK-NEXT:        %14 = stablehlo.and %11, %13 : tensor<20x24x48xi1>
// CHECK-NEXT:        %15 = stablehlo.or %10, %14 : tensor<20x24x48xi1>
// CHECK-NEXT:        %16 = stablehlo.select %15, %arg1, %cst : tensor<20x24x48xi1>, tensor<20x24x48xf64>
// CHECK-NEXT:        stablehlo.return %16 : tensor<20x24x48xf64>
// CHECK-NEXT:      }, {
// CHECK-NEXT:        stablehlo.return %cst : tensor<20x24x48xf64>
// CHECK-NEXT:      }) : (tensor<i1>) -> tensor<20x24x48xf64>
// CHECK-NEXT:      sdy.return %5 : tensor<20x24x48xf64>
// CHECK-NEXT:    } : (tensor<20x24x96xf64>) -> tensor<20x24x96xf64>
// CHECK-NEXT:    return %0 : tensor<20x24x96xf64>
// CHECK-NEXT:  }

func.func @constantUpdate(%arg21: tensor<20x24x96xf64> {sdy.sharding = #sdy.sharding<@mesh, [{"z"}, {"y"}, {"x"}]>}) -> (tensor<20x24x96xf64> {sdy.sharding = #sdy.sharding<@mesh, [{"z"}, {"y"}, {"x"}]>}) {
    %274 = sdy.constant dense<8> : tensor<i32>
    %275 = sdy.constant dense<0.000000e+00> : tensor<4x1x80xf64>
    %305 = stablehlo.dynamic_update_slice %arg21, %275, %274, %274, %274 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<20x24x96xf64>, tensor<4x1x80xf64>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<20x24x96xf64>
    return %305 : tensor<20x24x96xf64>
}

// PAD: func.func @constantUpdate(%arg0: tensor<20x24x96xf64> {sdy.sharding = #sdy.sharding<@mesh, [{"z"}, {"y"}, {"x"}]>}) -> (tensor<20x24x96xf64> {sdy.sharding = #sdy.sharding<@mesh, [{"z"}, {"y"}, {"x"}]>}) {
// PAD-NEXT:     %[[cst1:.+]] = stablehlo.constant dense<1.000000e+00> : tensor<f64>
// PAD-NEXT:     %[[cst0:.+]] = stablehlo.constant {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z"}, {"y"}, {"x"}]>]>} dense<0.000000e+00> : tensor<4x1x80xf64>
// PAD-NEXT:     %[[mask:.+]] = stablehlo.pad %[[cst0]], %[[cst1]], low = [8, 8, 8], high = [8, 15, 8], interior = [0, 0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<4x1x80xf64>, tensor<f64>) -> tensor<20x24x96xf64>
// PAD-NEXT:     %[[mul:.+]] = stablehlo.multiply %arg0, %[[mask]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z"}, {"y"}, {"x"}]>]>} : tensor<20x24x96xf64>
// PAD-NEXT:     return %[[mul]] : tensor<20x24x96xf64>
// PAD-NEXT: }

// CHECK:  func.func @constantUpdate(%arg0: tensor<20x24x96xf64> {sdy.sharding = #sdy.sharding<@mesh, [{"z"}, {"y"}, {"x"}]>}) -> (tensor<20x24x96xf64> {sdy.sharding = #sdy.sharding<@mesh, [{"z"}, {"y"}, {"x"}]>}) {
// CHECK-NEXT:    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"z"}, {"y"}, {"x"}]>] out_shardings=[<@mesh, [{"z"}, {"y"}, {"x"}]>] manual_axes={"z", "y", "x"} (%arg1: tensor<20x24x48xf64>) {
// CHECK-NEXT:      %c = stablehlo.constant dense<40> : tensor<20x24x48xui32>
// CHECK-NEXT:      %c_0 = stablehlo.constant dense<8> : tensor<20x24x48xui32>
// CHECK-NEXT:      %c_1 = stablehlo.constant dense<1> : tensor<ui32>
// CHECK-NEXT:      %c_2 = stablehlo.constant dense<8> : tensor<ui32>
// CHECK-NEXT:      %c_3 = stablehlo.constant dense<0> : tensor<ui32>
// CHECK-NEXT:      %cst = stablehlo.constant dense<0.000000e+00> : tensor<20x24x48xf64>
// CHECK-NEXT:      %1 = stablehlo.partition_id : tensor<ui32>
// CHECK-NEXT:      %2 = stablehlo.dynamic_update_slice %arg1, %cst, %c_2, %c_2, %c_3 : (tensor<20x24x48xf64>, tensor<20x24x48xf64>, tensor<ui32>, tensor<ui32>, tensor<ui32>) -> tensor<20x24x48xf64>
// CHECK-NEXT:      %3 = stablehlo.compare  LE, %1, %c_3 : (tensor<ui32>, tensor<ui32>) -> tensor<i1>
// CHECK-NEXT:      %4 = stablehlo.compare  GE, %1, %c_1 : (tensor<ui32>, tensor<ui32>) -> tensor<i1>
// CHECK-NEXT:      %5 = stablehlo.or %3, %4 : tensor<i1>
// CHECK-NEXT:      %6 = "stablehlo.if"(%5) ({
// CHECK-NEXT:        %7 = stablehlo.iota dim = 2 : tensor<20x24x48xui32>
// CHECK-NEXT:        %8 = stablehlo.compare  LT, %7, %c_0 : (tensor<20x24x48xui32>, tensor<20x24x48xui32>) -> tensor<20x24x48xi1>
// CHECK-NEXT:        %9 = stablehlo.compare  EQ, %1, %c_3 : (tensor<ui32>, tensor<ui32>) -> tensor<i1>
// CHECK-NEXT:        %10 = stablehlo.broadcast_in_dim %9, dims = [] : (tensor<i1>) -> tensor<20x24x48xi1>
// CHECK-NEXT:        %11 = stablehlo.and %8, %10 : tensor<20x24x48xi1>
// CHECK-NEXT:        %12 = stablehlo.compare  GE, %7, %c : (tensor<20x24x48xui32>, tensor<20x24x48xui32>) -> tensor<20x24x48xi1>
// CHECK-NEXT:        %13 = stablehlo.compare  EQ, %1, %c_1 : (tensor<ui32>, tensor<ui32>) -> tensor<i1>
// CHECK-NEXT:        %14 = stablehlo.broadcast_in_dim %13, dims = [] : (tensor<i1>) -> tensor<20x24x48xi1>
// CHECK-NEXT:        %15 = stablehlo.and %12, %14 : tensor<20x24x48xi1>
// CHECK-NEXT:        %16 = stablehlo.or %11, %15 : tensor<20x24x48xi1>
// CHECK-NEXT:        %17 = stablehlo.select %16, %arg1, %2 : tensor<20x24x48xi1>, tensor<20x24x48xf64>
// CHECK-NEXT:        stablehlo.return %17 : tensor<20x24x48xf64>
// CHECK-NEXT:      }, {
// CHECK-NEXT:        stablehlo.return %2 : tensor<20x24x48xf64>
// CHECK-NEXT:      }) : (tensor<i1>) -> tensor<20x24x48xf64>
// CHECK-NEXT:      sdy.return %6 : tensor<20x24x48xf64>
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
// CHECK-NEXT:    %1 = sdy.manual_computation(%arg0, %0) in_shardings=[<@mesh, [{"z"}, {"y"}, {"x"}]>, <@mesh, [{"z"}, {"y"}, {"x"}]>] out_shardings=[<@mesh, [{"z"}, {"y"}, {"x"}]>] manual_axes={"z", "y", "x"} (%arg2: tensor<20x24x48xf64>, %arg3: tensor<20x24x48xf64>) {
// CHECK-NEXT:      %c = stablehlo.constant dense<40> : tensor<20x24x48xui32>
// CHECK-NEXT:      %c_0 = stablehlo.constant dense<8> : tensor<20x24x48xui32>
// CHECK-NEXT:      %c_1 = stablehlo.constant dense<1> : tensor<ui32>
// CHECK-NEXT:      %c_2 = stablehlo.constant dense<0> : tensor<ui32>
// CHECK-NEXT:      %2 = stablehlo.partition_id : tensor<ui32>
// CHECK-NEXT:      %3 = stablehlo.compare  LE, %2, %c_2 : (tensor<ui32>, tensor<ui32>) -> tensor<i1>
// CHECK-NEXT:      %4 = stablehlo.compare  GE, %2, %c_1 : (tensor<ui32>, tensor<ui32>) -> tensor<i1>
// CHECK-NEXT:      %5 = stablehlo.or %3, %4 : tensor<i1>
// CHECK-NEXT:      %6 = "stablehlo.if"(%5) ({
// CHECK-NEXT:        %7 = stablehlo.iota dim = 2 : tensor<20x24x48xui32>
// CHECK-NEXT:        %8 = stablehlo.compare  LT, %7, %c_0 : (tensor<20x24x48xui32>, tensor<20x24x48xui32>) -> tensor<20x24x48xi1>
// CHECK-NEXT:        %9 = stablehlo.compare  EQ, %2, %c_2 : (tensor<ui32>, tensor<ui32>) -> tensor<i1>
// CHECK-NEXT:        %10 = stablehlo.broadcast_in_dim %9, dims = [] : (tensor<i1>) -> tensor<20x24x48xi1>
// CHECK-NEXT:        %11 = stablehlo.and %8, %10 : tensor<20x24x48xi1>
// CHECK-NEXT:        %12 = stablehlo.compare  GE, %7, %c : (tensor<20x24x48xui32>, tensor<20x24x48xui32>) -> tensor<20x24x48xi1>
// CHECK-NEXT:        %13 = stablehlo.compare  EQ, %2, %c_1 : (tensor<ui32>, tensor<ui32>) -> tensor<i1>
// CHECK-NEXT:        %14 = stablehlo.broadcast_in_dim %13, dims = [] : (tensor<i1>) -> tensor<20x24x48xi1>
// CHECK-NEXT:        %15 = stablehlo.and %12, %14 : tensor<20x24x48xi1>
// CHECK-NEXT:        %16 = stablehlo.or %11, %15 : tensor<20x24x48xi1>
// CHECK-NEXT:        %17 = stablehlo.select %16, %arg2, %arg3 : tensor<20x24x48xi1>, tensor<20x24x48xf64>
// CHECK-NEXT:        stablehlo.return %17 : tensor<20x24x48xf64>
// CHECK-NEXT:      }, {
// CHECK-NEXT:        stablehlo.return %arg3 : tensor<20x24x48xf64>
// CHECK-NEXT:      }) : (tensor<i1>) -> tensor<20x24x48xf64>
// CHECK-NEXT:      sdy.return %6 : tensor<20x24x48xf64>
// CHECK-NEXT:    } : (tensor<20x24x96xf64>, tensor<20x24x96xf64>) -> tensor<20x24x96xf64>
// CHECK-NEXT:    return %1 : tensor<20x24x96xf64>
// CHECK-NEXT:  }

func.func @argUpdate(%arg21: tensor<20x24x96xf64> {sdy.sharding = #sdy.sharding<@mesh, [{"z"}, {"y"}, {"x"}]>}, %275 : tensor<4x1x80xf64> {sdy.sharding = #sdy.sharding<@mesh, [{"z"}, {"y"}, {"x"}]>}) -> (tensor<20x24x96xf64> {sdy.sharding = #sdy.sharding<@mesh, [{"z"}, {"y"}, {"x"}]>}) {
    %274 = sdy.constant dense<8> : tensor<i32>
    %305 = stablehlo.dynamic_update_slice %arg21, %275, %274, %274, %274 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<20x24x96xf64>, tensor<4x1x80xf64>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<20x24x96xf64>
    return %305 : tensor<20x24x96xf64>
}

// PAD: func.func @argUpdate(%arg0: tensor<20x24x96xf64> {sdy.sharding = #sdy.sharding<@mesh, [{"z"}, {"y"}, {"x"}]>}, %arg1: tensor<4x1x80xf64> {sdy.sharding = #sdy.sharding<@mesh, [{"z"}, {"y"}, {"x"}]>}) -> (tensor<20x24x96xf64> {sdy.sharding = #sdy.sharding<@mesh, [{"z"}, {"y"}, {"x"}]>}) {
// PAD-NEXT:     %cst = stablehlo.constant {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z"}, {"y"}, {"x"}]>]>} dense<0.000000e+00> : tensor<4x1x80xf64>
// PAD-NEXT:     %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// PAD-NEXT:     %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
// PAD-NEXT:     %0 = stablehlo.pad %arg1, %cst_0, low = [8, 8, 8], high = [8, 15, 8], interior = [0, 0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<4x1x80xf64>, tensor<f64>) -> tensor<20x24x96xf64>
// PAD-NEXT:     %1 = stablehlo.pad %cst, %cst_1, low = [8, 8, 8], high = [8, 15, 8], interior = [0, 0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<4x1x80xf64>, tensor<f64>) -> tensor<20x24x96xf64>
// PAD-NEXT:     %2 = stablehlo.multiply %arg0, %1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z"}, {"y"}, {"x"}]>]>} : tensor<20x24x96xf64>
// PAD-NEXT:     %3 = stablehlo.add %2, %0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z"}, {"y"}, {"x"}]>]>} : tensor<20x24x96xf64>
// PAD-NEXT:     return %3 : tensor<20x24x96xf64>
// PAD-NEXT: }

// CHECK:  func.func @argUpdate(%arg0: tensor<20x24x96xf64> {sdy.sharding = #sdy.sharding<@mesh, [{"z"}, {"y"}, {"x"}]>}, %arg1: tensor<4x1x80xf64> {sdy.sharding = #sdy.sharding<@mesh, [{"z"}, {"y"}, {"x"}]>}) -> (tensor<20x24x96xf64> {sdy.sharding = #sdy.sharding<@mesh, [{"z"}, {"y"}, {"x"}]>}) {
// CHECK-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:    %0 = stablehlo.pad %arg1, %cst, low = [8, 8, 8], high = [8, 15, 8], interior = [0, 0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<4x1x80xf64>, tensor<f64>) -> tensor<20x24x96xf64>
// CHECK-NEXT:    %1 = sdy.manual_computation(%arg0, %0) in_shardings=[<@mesh, [{"z"}, {"y"}, {"x"}]>, <@mesh, [{"z"}, {"y"}, {"x"}]>] out_shardings=[<@mesh, [{"z"}, {"y"}, {"x"}]>] manual_axes={"z", "y", "x"} (%arg2: tensor<20x24x48xf64>, %arg3: tensor<20x24x48xf64>) {
// CHECK-NEXT:      %c = stablehlo.constant dense<40> : tensor<20x24x48xui32>
// CHECK-NEXT:      %c_0 = stablehlo.constant dense<8> : tensor<20x24x48xui32>
// CHECK-NEXT:      %c_1 = stablehlo.constant dense<1> : tensor<ui32>
// CHECK-NEXT:      %c_2 = stablehlo.constant dense<8> : tensor<ui32>
// CHECK-NEXT:      %c_3 = stablehlo.constant dense<0> : tensor<ui32>
// CHECK-NEXT:      %2 = stablehlo.partition_id : tensor<ui32>
// CHECK-NEXT:      %3 = stablehlo.dynamic_update_slice %arg2, %arg3, %c_2, %c_2, %c_3 : (tensor<20x24x48xf64>, tensor<20x24x48xf64>, tensor<ui32>, tensor<ui32>, tensor<ui32>) -> tensor<20x24x48xf64>
// CHECK-NEXT:      %4 = stablehlo.compare  LE, %2, %c_3 : (tensor<ui32>, tensor<ui32>) -> tensor<i1>
// CHECK-NEXT:      %5 = stablehlo.compare  GE, %2, %c_1 : (tensor<ui32>, tensor<ui32>) -> tensor<i1>
// CHECK-NEXT:      %6 = stablehlo.or %4, %5 : tensor<i1>
// CHECK-NEXT:      %7 = "stablehlo.if"(%6) ({
// CHECK-NEXT:        %8 = stablehlo.iota dim = 2 : tensor<20x24x48xui32>
// CHECK-NEXT:        %9 = stablehlo.compare  LT, %8, %c_0 : (tensor<20x24x48xui32>, tensor<20x24x48xui32>) -> tensor<20x24x48xi1>
// CHECK-NEXT:        %10 = stablehlo.compare  EQ, %2, %c_3 : (tensor<ui32>, tensor<ui32>) -> tensor<i1>
// CHECK-NEXT:        %11 = stablehlo.broadcast_in_dim %10, dims = [] : (tensor<i1>) -> tensor<20x24x48xi1>
// CHECK-NEXT:        %12 = stablehlo.and %9, %11 : tensor<20x24x48xi1>
// CHECK-NEXT:        %13 = stablehlo.compare  GE, %8, %c : (tensor<20x24x48xui32>, tensor<20x24x48xui32>) -> tensor<20x24x48xi1>
// CHECK-NEXT:        %14 = stablehlo.compare  EQ, %2, %c_1 : (tensor<ui32>, tensor<ui32>) -> tensor<i1>
// CHECK-NEXT:        %15 = stablehlo.broadcast_in_dim %14, dims = [] : (tensor<i1>) -> tensor<20x24x48xi1>
// CHECK-NEXT:        %16 = stablehlo.and %13, %15 : tensor<20x24x48xi1>
// CHECK-NEXT:        %17 = stablehlo.or %12, %16 : tensor<20x24x48xi1>
// CHECK-NEXT:        %18 = stablehlo.select %17, %arg2, %3 : tensor<20x24x48xi1>, tensor<20x24x48xf64>
// CHECK-NEXT:        stablehlo.return %18 : tensor<20x24x48xf64>
// CHECK-NEXT:      }, {
// CHECK-NEXT:        stablehlo.return %3 : tensor<20x24x48xf64>
// CHECK-NEXT:      }) : (tensor<i1>) -> tensor<20x24x48xf64>
// CHECK-NEXT:      sdy.return %7 : tensor<20x24x48xf64>
// CHECK-NEXT:    } : (tensor<20x24x96xf64>, tensor<20x24x96xf64>) -> tensor<20x24x96xf64>
// CHECK-NEXT:    return %1 : tensor<20x24x96xf64>
// CHECK-NEXT:  }
