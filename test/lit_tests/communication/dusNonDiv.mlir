// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(optimize-communication{dus_to_pad_manual_comp_comm=1 dus_to_pad_comm=0})" %s | FileCheck %s
// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(optimize-communication{dus_to_pad_manual_comp_comm=0 dus_to_pad_comm=1})" %s | FileCheck %s --check-prefix=PAD

sdy.mesh @mesh = <["x"=2, "y"=1, "z"=1]>

func.func @constantUpdate1D(%arg21: tensor<20x24x97xf64> {sdy.sharding = #sdy.sharding<@mesh, [{"z"}, {"y"}, {"x"}]>}) -> (tensor<20x24x97xf64> {sdy.sharding = #sdy.sharding<@mesh, [{"z"}, {"y"}, {"x"}]>}) {
    %c0 = sdy.constant dense<0> : tensor<i32>
    %274 = sdy.constant dense<8> : tensor<i32>
    %275 = sdy.constant dense<0.000000e+00> : tensor<20x24x80xf64>
    %305 = stablehlo.dynamic_update_slice %arg21, %275, %c0, %c0, %274 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<20x24x97xf64>, tensor<20x24x80xf64>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<20x24x97xf64>
    return %305 : tensor<20x24x97xf64>
}

// PAD: func.func @constantUpdate1D(%arg0: tensor<20x24x97xf64> {sdy.sharding = #sdy.sharding<@mesh, [{"z"}, {"y"}, {"x"}]>}) -> (tensor<20x24x97xf64> {sdy.sharding = #sdy.sharding<@mesh, [{"z"}, {"y"}, {"x"}]>}) {
// PAD-NEXT:     %cst = stablehlo.constant {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z"}, {"y"}, {"x"}]>]>} dense<0.000000e+00> : tensor<20x24x80xf64>
// PAD-NEXT:     %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
// PAD-NEXT:     %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// PAD-NEXT:     %0 = sdy.constant {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z"}, {"y"}, {"x"}]>]>} dense<0.000000e+00> : tensor<20x24x80xf64>
// PAD-NEXT:     %1 = stablehlo.pad %0, %cst_1, low = [0, 0, 8], high = [0, 0, 9], interior = [0, 0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<20x24x80xf64>, tensor<f64>) -> tensor<20x24x97xf64>
// PAD-NEXT:     %2 = stablehlo.pad %cst, %cst_0, low = [0, 0, 8], high = [0, 0, 9], interior = [0, 0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<20x24x80xf64>, tensor<f64>) -> tensor<20x24x97xf64>
// PAD-NEXT:     %3 = stablehlo.multiply %arg0, %2 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z"}, {"y"}, {"x"}]>]>} : tensor<20x24x97xf64>
// PAD-NEXT:     %4 = stablehlo.add %3, %1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z"}, {"y"}, {"x"}]>]>} : tensor<20x24x97xf64>
// PAD-NEXT:     return %4 : tensor<20x24x97xf64>
// PAD-NEXT: }

// CHECK:  func.func @constantUpdate1D(%arg0: tensor<20x24x97xf64> {sdy.sharding = #sdy.sharding<@mesh, [{"z"}, {"y"}, {"x"}]>}) -> (tensor<20x24x97xf64> {sdy.sharding = #sdy.sharding<@mesh, [{"z"}, {"y"}, {"x"}]>}) {
// CHECK-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:    %0 = stablehlo.pad %arg0, %cst, low = [0, 0, 0], high = [0, 0, 1], interior = [0, 0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<20x24x97xf64>, tensor<f64>) -> tensor<20x24x98xf64>
// CHECK-NEXT:    %1 = sdy.manual_computation(%0) in_shardings=[<@mesh, [{"z"}, {"y"}, {"x"}]>] out_shardings=[<@mesh, [{"z"}, {"y"}, {"x"}]>] manual_axes={"z", "y", "x"} (%arg1: tensor<20x24x49xf64>) {
// CHECK-NEXT:      %c = stablehlo.constant dense<39> : tensor<20x24x49xui32>
// CHECK-NEXT:      %c_0 = stablehlo.constant dense<8> : tensor<20x24x49xui32>
// CHECK-NEXT:      %c_1 = stablehlo.constant dense<1> : tensor<ui32>
// CHECK-NEXT:      %c_2 = stablehlo.constant dense<0> : tensor<ui32>
// CHECK-NEXT:      %cst_3 = stablehlo.constant dense<0.000000e+00> : tensor<20x24x49xf64>
// CHECK-NEXT:      %3 = stablehlo.partition_id : tensor<ui32>
// CHECK-NEXT:      %4 = stablehlo.compare  LE, %3, %c_2 : (tensor<ui32>, tensor<ui32>) -> tensor<i1>
// CHECK-NEXT:      %5 = stablehlo.compare  GE, %3, %c_1 : (tensor<ui32>, tensor<ui32>) -> tensor<i1>
// CHECK-NEXT:      %6 = stablehlo.or %4, %5 : tensor<i1>
// CHECK-NEXT:      %7 = "stablehlo.if"(%6) ({
// CHECK-NEXT:        %8 = stablehlo.iota dim = 2 : tensor<20x24x49xui32>
// CHECK-NEXT:        %9 = stablehlo.compare  LT, %8, %c_0 : (tensor<20x24x49xui32>, tensor<20x24x49xui32>) -> tensor<20x24x49xi1>
// CHECK-NEXT:        %10 = stablehlo.compare  EQ, %3, %c_2 : (tensor<ui32>, tensor<ui32>) -> tensor<i1>
// CHECK-NEXT:        %11 = stablehlo.broadcast_in_dim %10, dims = [] : (tensor<i1>) -> tensor<20x24x49xi1>
// CHECK-NEXT:        %12 = stablehlo.and %9, %11 : tensor<20x24x49xi1>
// CHECK-NEXT:        %13 = stablehlo.compare  GE, %8, %c : (tensor<20x24x49xui32>, tensor<20x24x49xui32>) -> tensor<20x24x49xi1>
// CHECK-NEXT:        %14 = stablehlo.compare  EQ, %3, %c_1 : (tensor<ui32>, tensor<ui32>) -> tensor<i1>
// CHECK-NEXT:        %15 = stablehlo.broadcast_in_dim %14, dims = [] : (tensor<i1>) -> tensor<20x24x49xi1>
// CHECK-NEXT:        %16 = stablehlo.and %13, %15 : tensor<20x24x49xi1>
// CHECK-NEXT:        %17 = stablehlo.or %12, %16 : tensor<20x24x49xi1>
// CHECK-NEXT:        %18 = stablehlo.select %17, %arg1, %cst_3 : tensor<20x24x49xi1>, tensor<20x24x49xf64>
// CHECK-NEXT:        stablehlo.return %18 : tensor<20x24x49xf64>
// CHECK-NEXT:      }, {
// CHECK-NEXT:        stablehlo.return %cst_3 : tensor<20x24x49xf64>
// CHECK-NEXT:      }) : (tensor<i1>) -> tensor<20x24x49xf64>
// CHECK-NEXT:      sdy.return %7 : tensor<20x24x49xf64>
// CHECK-NEXT:    } : (tensor<20x24x98xf64>) -> tensor<20x24x98xf64>
// CHECK-NEXT:    %2 = stablehlo.slice %1 [0:20, 0:24, 0:97] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<20x24x98xf64>) -> tensor<20x24x97xf64>
// CHECK-NEXT:    return %2 : tensor<20x24x97xf64>
// CHECK-NEXT:  }


func.func @constantUpdate(%arg21: tensor<20x24x97xf64> {sdy.sharding = #sdy.sharding<@mesh, [{"z"}, {"y"}, {"x"}]>}) -> (tensor<20x24x97xf64> {sdy.sharding = #sdy.sharding<@mesh, [{"z"}, {"y"}, {"x"}]>}) {
    %274 = sdy.constant dense<8> : tensor<i32>
    %275 = sdy.constant dense<0.000000e+00> : tensor<4x1x80xf64>
    %305 = stablehlo.dynamic_update_slice %arg21, %275, %274, %274, %274 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<20x24x97xf64>, tensor<4x1x80xf64>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<20x24x97xf64>
    return %305 : tensor<20x24x97xf64>
}

// PAD: func.func @constantUpdate(%arg0: tensor<20x24x97xf64> {sdy.sharding = #sdy.sharding<@mesh, [{"z"}, {"y"}, {"x"}]>}) -> (tensor<20x24x97xf64> {sdy.sharding = #sdy.sharding<@mesh, [{"z"}, {"y"}, {"x"}]>}) {
// PAD-NEXT:     %cst = stablehlo.constant {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z"}, {"y"}, {"x"}]>]>} dense<0.000000e+00> : tensor<4x1x80xf64>
// PAD-NEXT:     %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
// PAD-NEXT:     %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// PAD-NEXT:     %0 = sdy.constant {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z"}, {"y"}, {"x"}]>]>} dense<0.000000e+00> : tensor<4x1x80xf64>
// PAD-NEXT:     %1 = stablehlo.pad %0, %cst_1, low = [8, 8, 8], high = [8, 15, 9], interior = [0, 0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<4x1x80xf64>, tensor<f64>) -> tensor<20x24x97xf64>
// PAD-NEXT:     %2 = stablehlo.pad %cst, %cst_0, low = [8, 8, 8], high = [8, 15, 9], interior = [0, 0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<4x1x80xf64>, tensor<f64>) -> tensor<20x24x97xf64>
// PAD-NEXT:     %3 = stablehlo.multiply %arg0, %2 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z"}, {"y"}, {"x"}]>]>} : tensor<20x24x97xf64>
// PAD-NEXT:     %4 = stablehlo.add %3, %1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z"}, {"y"}, {"x"}]>]>} : tensor<20x24x97xf64>
// PAD-NEXT:     return %4 : tensor<20x24x97xf64>
// PAD-NEXT:   }

// CHECK:  func.func @constantUpdate(%arg0: tensor<20x24x97xf64> {sdy.sharding = #sdy.sharding<@mesh, [{"z"}, {"y"}, {"x"}]>}) -> (tensor<20x24x97xf64> {sdy.sharding = #sdy.sharding<@mesh, [{"z"}, {"y"}, {"x"}]>}) {
// CHECK-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:    %0 = stablehlo.pad %arg0, %cst, low = [0, 0, 0], high = [0, 0, 1], interior = [0, 0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<20x24x97xf64>, tensor<f64>) -> tensor<20x24x98xf64>
// CHECK-NEXT:    %1 = sdy.manual_computation(%0) in_shardings=[<@mesh, [{"z"}, {"y"}, {"x"}]>] out_shardings=[<@mesh, [{"z"}, {"y"}, {"x"}]>] manual_axes={"z", "y", "x"} (%arg1: tensor<20x24x49xf64>) {
// CHECK-NEXT:      %c = stablehlo.constant dense<39> : tensor<20x24x49xui32>
// CHECK-NEXT:      %c_0 = stablehlo.constant dense<8> : tensor<20x24x49xui32>
// CHECK-NEXT:      %c_1 = stablehlo.constant dense<1> : tensor<ui32>
// CHECK-NEXT:      %c_2 = stablehlo.constant dense<8> : tensor<ui32>
// CHECK-NEXT:      %c_3 = stablehlo.constant dense<0> : tensor<ui32>
// CHECK-NEXT:      %cst_4 = stablehlo.constant dense<0.000000e+00> : tensor<20x24x49xf64>
// CHECK-NEXT:      %3 = stablehlo.partition_id : tensor<ui32>
// CHECK-NEXT:      %4 = stablehlo.dynamic_update_slice %arg1, %cst_4, %c_2, %c_2, %c_3 : (tensor<20x24x49xf64>, tensor<20x24x49xf64>, tensor<ui32>, tensor<ui32>, tensor<ui32>) -> tensor<20x24x49xf64>
// CHECK-NEXT:      %5 = stablehlo.compare  LE, %3, %c_3 : (tensor<ui32>, tensor<ui32>) -> tensor<i1>
// CHECK-NEXT:      %6 = stablehlo.compare  GE, %3, %c_1 : (tensor<ui32>, tensor<ui32>) -> tensor<i1>
// CHECK-NEXT:      %7 = stablehlo.or %5, %6 : tensor<i1>
// CHECK-NEXT:      %8 = "stablehlo.if"(%7) ({
// CHECK-NEXT:        %9 = stablehlo.iota dim = 2 : tensor<20x24x49xui32>
// CHECK-NEXT:        %10 = stablehlo.compare  LT, %9, %c_0 : (tensor<20x24x49xui32>, tensor<20x24x49xui32>) -> tensor<20x24x49xi1>
// CHECK-NEXT:        %11 = stablehlo.compare  EQ, %3, %c_3 : (tensor<ui32>, tensor<ui32>) -> tensor<i1>
// CHECK-NEXT:        %12 = stablehlo.broadcast_in_dim %11, dims = [] : (tensor<i1>) -> tensor<20x24x49xi1>
// CHECK-NEXT:        %13 = stablehlo.and %10, %12 : tensor<20x24x49xi1>
// CHECK-NEXT:        %14 = stablehlo.compare  GE, %9, %c : (tensor<20x24x49xui32>, tensor<20x24x49xui32>) -> tensor<20x24x49xi1>
// CHECK-NEXT:        %15 = stablehlo.compare  EQ, %3, %c_1 : (tensor<ui32>, tensor<ui32>) -> tensor<i1>
// CHECK-NEXT:        %16 = stablehlo.broadcast_in_dim %15, dims = [] : (tensor<i1>) -> tensor<20x24x49xi1>
// CHECK-NEXT:        %17 = stablehlo.and %14, %16 : tensor<20x24x49xi1>
// CHECK-NEXT:        %18 = stablehlo.or %13, %17 : tensor<20x24x49xi1>
// CHECK-NEXT:        %19 = stablehlo.select %18, %arg1, %4 : tensor<20x24x49xi1>, tensor<20x24x49xf64>
// CHECK-NEXT:        stablehlo.return %19 : tensor<20x24x49xf64>
// CHECK-NEXT:      }, {
// CHECK-NEXT:        stablehlo.return %4 : tensor<20x24x49xf64>
// CHECK-NEXT:      }) : (tensor<i1>) -> tensor<20x24x49xf64>
// CHECK-NEXT:      sdy.return %8 : tensor<20x24x49xf64>
// CHECK-NEXT:    } : (tensor<20x24x98xf64>) -> tensor<20x24x98xf64>
// CHECK-NEXT:    %2 = stablehlo.slice %1 [0:20, 0:24, 0:97] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<20x24x98xf64>) -> tensor<20x24x97xf64>
// CHECK-NEXT:    return %2 : tensor<20x24x97xf64>
// CHECK-NEXT:  }

func.func @argUpdate1D(%arg21: tensor<20x24x97xf64> {sdy.sharding = #sdy.sharding<@mesh, [{"z"}, {"y"}, {"x"}]>}, %275 : tensor<20x24x80xf64> {sdy.sharding = #sdy.sharding<@mesh, [{"z"}, {"y"}, {"x"}]>}) -> (tensor<20x24x97xf64> {sdy.sharding = #sdy.sharding<@mesh, [{"z"}, {"y"}, {"x"}]>}) {
    %274 = sdy.constant dense<8> : tensor<i32>
    %c0 = sdy.constant dense<0> : tensor<i32>
    %305 = stablehlo.dynamic_update_slice %arg21, %275, %c0, %c0, %274 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<20x24x97xf64>, tensor<20x24x80xf64>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<20x24x97xf64>
    return %305 : tensor<20x24x97xf64>
}

// PAD: func.func @argUpdate1D(%arg0: tensor<20x24x97xf64> {sdy.sharding = #sdy.sharding<@mesh, [{"z"}, {"y"}, {"x"}]>}, %arg1: tensor<20x24x80xf64> {sdy.sharding = #sdy.sharding<@mesh, [{"z"}, {"y"}, {"x"}]>}) -> (tensor<20x24x97xf64> {sdy.sharding = #sdy.sharding<@mesh, [{"z"}, {"y"}, {"x"}]>}) {
// PAD-NEXT:     %cst = stablehlo.constant {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z"}, {"y"}, {"x"}]>]>} dense<0.000000e+00> : tensor<20x24x80xf64>
// PAD-NEXT:     %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// PAD-NEXT:     %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
// PAD-NEXT:     %0 = stablehlo.pad %arg1, %cst_0, low = [0, 0, 8], high = [0, 0, 9], interior = [0, 0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<20x24x80xf64>, tensor<f64>) -> tensor<20x24x97xf64>
// PAD-NEXT:     %1 = stablehlo.pad %cst, %cst_1, low = [0, 0, 8], high = [0, 0, 9], interior = [0, 0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<20x24x80xf64>, tensor<f64>) -> tensor<20x24x97xf64>
// PAD-NEXT:     %2 = stablehlo.multiply %arg0, %1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z"}, {"y"}, {"x"}]>]>} : tensor<20x24x97xf64>
// PAD-NEXT:     %3 = stablehlo.add %2, %0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z"}, {"y"}, {"x"}]>]>} : tensor<20x24x97xf64>
// PAD-NEXT:     return %3 : tensor<20x24x97xf64>
// PAD-NEXT:   }

// CHECK:  func.func @argUpdate1D(%arg0: tensor<20x24x97xf64> {sdy.sharding = #sdy.sharding<@mesh, [{"z"}, {"y"}, {"x"}]>}, %arg1: tensor<20x24x80xf64> {sdy.sharding = #sdy.sharding<@mesh, [{"z"}, {"y"}, {"x"}]>}) -> (tensor<20x24x97xf64> {sdy.sharding = #sdy.sharding<@mesh, [{"z"}, {"y"}, {"x"}]>}) {
// CHECK-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:    %0 = stablehlo.pad %arg0, %cst, low = [0, 0, 0], high = [0, 0, 1], interior = [0, 0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<20x24x97xf64>, tensor<f64>) -> tensor<20x24x98xf64>
// CHECK-NEXT:    %1 = stablehlo.pad %arg1, %cst, low = [0, 0, 8], high = [0, 0, 10], interior = [0, 0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<20x24x80xf64>, tensor<f64>) -> tensor<20x24x98xf64>
// CHECK-NEXT:    %2 = sdy.manual_computation(%0, %1) in_shardings=[<@mesh, [{"z"}, {"y"}, {"x"}]>, <@mesh, [{"z"}, {"y"}, {"x"}]>] out_shardings=[<@mesh, [{"z"}, {"y"}, {"x"}]>] manual_axes={"z", "y", "x"} (%arg2: tensor<20x24x49xf64>, %arg3: tensor<20x24x49xf64>) {
// CHECK-NEXT:      %c = stablehlo.constant dense<39> : tensor<20x24x49xui32>
// CHECK-NEXT:      %c_0 = stablehlo.constant dense<8> : tensor<20x24x49xui32>
// CHECK-NEXT:      %c_1 = stablehlo.constant dense<1> : tensor<ui32>
// CHECK-NEXT:      %c_2 = stablehlo.constant dense<0> : tensor<ui32>
// CHECK-NEXT:      %4 = stablehlo.partition_id : tensor<ui32>
// CHECK-NEXT:      %5 = stablehlo.compare  LE, %4, %c_2 : (tensor<ui32>, tensor<ui32>) -> tensor<i1>
// CHECK-NEXT:      %6 = stablehlo.compare  GE, %4, %c_1 : (tensor<ui32>, tensor<ui32>) -> tensor<i1>
// CHECK-NEXT:      %7 = stablehlo.or %5, %6 : tensor<i1>
// CHECK-NEXT:      %8 = "stablehlo.if"(%7) ({
// CHECK-NEXT:        %9 = stablehlo.iota dim = 2 : tensor<20x24x49xui32>
// CHECK-NEXT:        %10 = stablehlo.compare  LT, %9, %c_0 : (tensor<20x24x49xui32>, tensor<20x24x49xui32>) -> tensor<20x24x49xi1>
// CHECK-NEXT:        %11 = stablehlo.compare  EQ, %4, %c_2 : (tensor<ui32>, tensor<ui32>) -> tensor<i1>
// CHECK-NEXT:        %12 = stablehlo.broadcast_in_dim %11, dims = [] : (tensor<i1>) -> tensor<20x24x49xi1>
// CHECK-NEXT:        %13 = stablehlo.and %10, %12 : tensor<20x24x49xi1>
// CHECK-NEXT:        %14 = stablehlo.compare  GE, %9, %c : (tensor<20x24x49xui32>, tensor<20x24x49xui32>) -> tensor<20x24x49xi1>
// CHECK-NEXT:        %15 = stablehlo.compare  EQ, %4, %c_1 : (tensor<ui32>, tensor<ui32>) -> tensor<i1>
// CHECK-NEXT:        %16 = stablehlo.broadcast_in_dim %15, dims = [] : (tensor<i1>) -> tensor<20x24x49xi1>
// CHECK-NEXT:        %17 = stablehlo.and %14, %16 : tensor<20x24x49xi1>
// CHECK-NEXT:        %18 = stablehlo.or %13, %17 : tensor<20x24x49xi1>
// CHECK-NEXT:        %19 = stablehlo.select %18, %arg2, %arg3 : tensor<20x24x49xi1>, tensor<20x24x49xf64>
// CHECK-NEXT:        stablehlo.return %19 : tensor<20x24x49xf64>
// CHECK-NEXT:      }, {
// CHECK-NEXT:        stablehlo.return %arg3 : tensor<20x24x49xf64>
// CHECK-NEXT:      }) : (tensor<i1>) -> tensor<20x24x49xf64>
// CHECK-NEXT:      sdy.return %8 : tensor<20x24x49xf64>
// CHECK-NEXT:    } : (tensor<20x24x98xf64>, tensor<20x24x98xf64>) -> tensor<20x24x98xf64>
// CHECK-NEXT:    %3 = stablehlo.slice %2 [0:20, 0:24, 0:97] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<20x24x98xf64>) -> tensor<20x24x97xf64>
// CHECK-NEXT:    return %3 : tensor<20x24x97xf64>
// CHECK-NEXT:  }


func.func @argUpdate(%arg21: tensor<20x24x97xf64> {sdy.sharding = #sdy.sharding<@mesh, [{"z"}, {"y"}, {"x"}]>}, %275 : tensor<4x1x80xf64> {sdy.sharding = #sdy.sharding<@mesh, [{"z"}, {"y"}, {"x"}]>}) -> (tensor<20x24x97xf64> {sdy.sharding = #sdy.sharding<@mesh, [{"z"}, {"y"}, {"x"}]>}) {
    %274 = sdy.constant dense<8> : tensor<i32>
    %305 = stablehlo.dynamic_update_slice %arg21, %275, %274, %274, %274 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<20x24x97xf64>, tensor<4x1x80xf64>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<20x24x97xf64>
    return %305 : tensor<20x24x97xf64>
}

// PAD: func.func @argUpdate(%arg0: tensor<20x24x97xf64> {sdy.sharding = #sdy.sharding<@mesh, [{"z"}, {"y"}, {"x"}]>}, %arg1: tensor<4x1x80xf64> {sdy.sharding = #sdy.sharding<@mesh, [{"z"}, {"y"}, {"x"}]>}) -> (tensor<20x24x97xf64> {sdy.sharding = #sdy.sharding<@mesh, [{"z"}, {"y"}, {"x"}]>}) {
// PAD-NEXT:     %cst = stablehlo.constant {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z"}, {"y"}, {"x"}]>]>} dense<0.000000e+00> : tensor<4x1x80xf64>
// PAD-NEXT:     %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// PAD-NEXT:     %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
// PAD-NEXT:     %0 = stablehlo.pad %arg1, %cst_0, low = [8, 8, 8], high = [8, 15, 9], interior = [0, 0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<4x1x80xf64>, tensor<f64>) -> tensor<20x24x97xf64>
// PAD-NEXT:     %1 = stablehlo.pad %cst, %cst_1, low = [8, 8, 8], high = [8, 15, 9], interior = [0, 0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<4x1x80xf64>, tensor<f64>) -> tensor<20x24x97xf64>
// PAD-NEXT:     %2 = stablehlo.multiply %arg0, %1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z"}, {"y"}, {"x"}]>]>} : tensor<20x24x97xf64>
// PAD-NEXT:     %3 = stablehlo.add %2, %0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z"}, {"y"}, {"x"}]>]>} : tensor<20x24x97xf64>
// PAD-NEXT:     return %3 : tensor<20x24x97xf64>
// PAD-NEXT:   }

// CHECK:  func.func @argUpdate(%arg0: tensor<20x24x97xf64> {sdy.sharding = #sdy.sharding<@mesh, [{"z"}, {"y"}, {"x"}]>}, %arg1: tensor<4x1x80xf64> {sdy.sharding = #sdy.sharding<@mesh, [{"z"}, {"y"}, {"x"}]>}) -> (tensor<20x24x97xf64> {sdy.sharding = #sdy.sharding<@mesh, [{"z"}, {"y"}, {"x"}]>}) {
// CHECK-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:    %0 = stablehlo.pad %arg0, %cst, low = [0, 0, 0], high = [0, 0, 1], interior = [0, 0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<20x24x97xf64>, tensor<f64>) -> tensor<20x24x98xf64>
// CHECK-NEXT:    %1 = stablehlo.pad %arg1, %cst, low = [8, 8, 8], high = [8, 15, 10], interior = [0, 0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<4x1x80xf64>, tensor<f64>) -> tensor<20x24x98xf64>
// CHECK-NEXT:    %2 = sdy.manual_computation(%0, %1) in_shardings=[<@mesh, [{"z"}, {"y"}, {"x"}]>, <@mesh, [{"z"}, {"y"}, {"x"}]>] out_shardings=[<@mesh, [{"z"}, {"y"}, {"x"}]>] manual_axes={"z", "y", "x"} (%arg2: tensor<20x24x49xf64>, %arg3: tensor<20x24x49xf64>) {
// CHECK-NEXT:      %c = stablehlo.constant dense<39> : tensor<20x24x49xui32>
// CHECK-NEXT:      %c_0 = stablehlo.constant dense<8> : tensor<20x24x49xui32>
// CHECK-NEXT:      %c_1 = stablehlo.constant dense<1> : tensor<ui32>
// CHECK-NEXT:      %c_2 = stablehlo.constant dense<8> : tensor<ui32>
// CHECK-NEXT:      %c_3 = stablehlo.constant dense<0> : tensor<ui32>
// CHECK-NEXT:      %4 = stablehlo.partition_id : tensor<ui32>
// CHECK-NEXT:      %5 = stablehlo.dynamic_update_slice %arg2, %arg3, %c_2, %c_2, %c_3 : (tensor<20x24x49xf64>, tensor<20x24x49xf64>, tensor<ui32>, tensor<ui32>, tensor<ui32>) -> tensor<20x24x49xf64>
// CHECK-NEXT:      %6 = stablehlo.compare  LE, %4, %c_3 : (tensor<ui32>, tensor<ui32>) -> tensor<i1>
// CHECK-NEXT:      %7 = stablehlo.compare  GE, %4, %c_1 : (tensor<ui32>, tensor<ui32>) -> tensor<i1>
// CHECK-NEXT:      %8 = stablehlo.or %6, %7 : tensor<i1>
// CHECK-NEXT:      %9 = "stablehlo.if"(%8) ({
// CHECK-NEXT:        %10 = stablehlo.iota dim = 2 : tensor<20x24x49xui32>
// CHECK-NEXT:        %11 = stablehlo.compare  LT, %10, %c_0 : (tensor<20x24x49xui32>, tensor<20x24x49xui32>) -> tensor<20x24x49xi1>
// CHECK-NEXT:        %12 = stablehlo.compare  EQ, %4, %c_3 : (tensor<ui32>, tensor<ui32>) -> tensor<i1>
// CHECK-NEXT:        %13 = stablehlo.broadcast_in_dim %12, dims = [] : (tensor<i1>) -> tensor<20x24x49xi1>
// CHECK-NEXT:        %14 = stablehlo.and %11, %13 : tensor<20x24x49xi1>
// CHECK-NEXT:        %15 = stablehlo.compare  GE, %10, %c : (tensor<20x24x49xui32>, tensor<20x24x49xui32>) -> tensor<20x24x49xi1>
// CHECK-NEXT:        %16 = stablehlo.compare  EQ, %4, %c_1 : (tensor<ui32>, tensor<ui32>) -> tensor<i1>
// CHECK-NEXT:        %17 = stablehlo.broadcast_in_dim %16, dims = [] : (tensor<i1>) -> tensor<20x24x49xi1>
// CHECK-NEXT:        %18 = stablehlo.and %15, %17 : tensor<20x24x49xi1>
// CHECK-NEXT:        %19 = stablehlo.or %14, %18 : tensor<20x24x49xi1>
// CHECK-NEXT:        %20 = stablehlo.select %19, %arg2, %5 : tensor<20x24x49xi1>, tensor<20x24x49xf64>
// CHECK-NEXT:        stablehlo.return %20 : tensor<20x24x49xf64>
// CHECK-NEXT:      }, {
// CHECK-NEXT:        stablehlo.return %5 : tensor<20x24x49xf64>
// CHECK-NEXT:      }) : (tensor<i1>) -> tensor<20x24x49xf64>
// CHECK-NEXT:      sdy.return %9 : tensor<20x24x49xf64>
// CHECK-NEXT:    } : (tensor<20x24x98xf64>, tensor<20x24x98xf64>) -> tensor<20x24x98xf64>
// CHECK-NEXT:    %3 = stablehlo.slice %2 [0:20, 0:24, 0:97] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<20x24x98xf64>) -> tensor<20x24x97xf64>
// CHECK-NEXT:    return %3 : tensor<20x24x97xf64>
// CHECK-NEXT:  }
