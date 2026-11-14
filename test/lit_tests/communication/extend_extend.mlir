// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(optimize-communication{extend_comm=0 extend_to_pad_comm=0 extend_to_pad_comm2=1},cse)" %s | FileCheck %s --check-prefix=SEL
// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(optimize-communication{extend_comm=0 extend_to_pad_comm=1 extend_to_pad_comm2=0},cse)" %s | FileCheck %s --check-prefix=PAD
// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(optimize-communication{extend_comm=1 extend_to_pad_comm=0 extend_to_pad_comm2=0},cse)" %s | FileCheck %s --check-prefix=COMM

sdy.mesh @mesh1 = <["z"=1, "y"=6, "x"=4]>
sdy.mesh @mesh2 = <["z"=2, "y"=6, "x"=4]>

// 1. Same-dimension Extend(Extend(x, 1, 1), 1, 1)
func.func @same_dim_repeat(%arg0: tensor<1x12x80xf64> {sdy.sharding = #sdy.sharding<@mesh1, [{"z"}, {"y"}, {"x"}]>}) -> (tensor<1x12x84xf64> {sdy.sharding = #sdy.sharding<@mesh1, [{"z"}, {"y"}, {"x"}]>}) {
  %0 = "enzymexla.extend"(%arg0) <{dimension = 2 : i64, lhs = 1 : i64, rhs = 1 : i64}> {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<1x12x80xf64>) -> tensor<1x12x82xf64>
  %1 = "enzymexla.extend"(%0) <{dimension = 2 : i64, lhs = 1 : i64, rhs = 1 : i64}> {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<1x12x82xf64>) -> tensor<1x12x84xf64>
  return %1 : tensor<1x12x84xf64>
}

// SEL-LABEL: func.func @same_dim_repeat(
// SEL-NOT: enzymexla.extend
// SEL: stablehlo.pad %arg0, %cst, low = [0, 0, 2], high = [0, 0, 2]
// SEL: stablehlo.pad %arg0, %cst, low = [0, 0, 1], high = [0, 0, 3]
// SEL: stablehlo.pad %arg0, %cst, low = [0, 0, 0], high = [0, 0, 4]
// SEL: stablehlo.pad %arg0, %cst, low = [0, 0, 3], high = [0, 0, 1]
// SEL: stablehlo.pad %arg0, %cst, low = [0, 0, 4], high = [0, 0, 0]
// SEL-NOT: stablehlo.pad %
// SEL: return

// PAD-LABEL: func.func @same_dim_repeat(
// PAD-NOT: enzymexla.extend
// PAD: %[[LSL:.+]] = stablehlo.slice %arg0 [0:1, 0:12, 0:1]
// PAD: %[[LB:.+]] = stablehlo.broadcast_in_dim %[[LSL]], dims = [0, 1, 2] {{.*}} : (tensor<1x12x1xf64>) -> tensor<1x12x2xf64>
// PAD: %[[LP:.+]] = stablehlo.pad %[[LB]], %cst, low = [0, 0, 0], high = [0, 0, 82]
// PAD: %[[MP:.+]] = stablehlo.pad %arg0, %cst, low = [0, 0, 2], high = [0, 0, 2]
// PAD: %[[ADD1:.+]] = stablehlo.add %[[LP]], %[[MP]]
// PAD: %[[RSL:.+]] = stablehlo.slice %arg0 [0:1, 0:12, 79:80]
// PAD: %[[RB:.+]] = stablehlo.broadcast_in_dim %[[RSL]], dims = [0, 1, 2] {{.*}} : (tensor<1x12x1xf64>) -> tensor<1x12x2xf64>
// PAD: %[[RP:.+]] = stablehlo.pad %[[RB]], %cst, low = [0, 0, 82], high = [0, 0, 0]
// PAD: %[[ADD2:.+]] = stablehlo.add %[[ADD1]], %[[RP]]
// PAD-NEXT: return %[[ADD2]]

// 2. Non-overlapping same-dim Extend(Extend(x, 0, 1), 1, 0) -> single Extend(x, 1, 1)
func.func @same_dim_non_overlapping(%arg0: tensor<1x12x80xf64> {sdy.sharding = #sdy.sharding<@mesh1, [{"z"}, {"y"}, {"x"}]>}) -> (tensor<1x12x82xf64> {sdy.sharding = #sdy.sharding<@mesh1, [{"z"}, {"y"}, {"x"}]>}) {
  %0 = "enzymexla.extend"(%arg0) <{dimension = 2 : i64, lhs = 0 : i64, rhs = 1 : i64}> {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<1x12x80xf64>) -> tensor<1x12x81xf64>
  %1 = "enzymexla.extend"(%0) <{dimension = 2 : i64, lhs = 1 : i64, rhs = 0 : i64}> {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<1x12x81xf64>) -> tensor<1x12x82xf64>
  return %1 : tensor<1x12x82xf64>
}

// COMM-LABEL: func.func @same_dim_non_overlapping(
// COMM-NOT: enzymexla.extend
// COMM: sdy.manual_computation(%arg0)
// COMM-NOT: sdy.manual_computation
// COMM: return

// 3. Different dimensions: unsharded inner dim (z=1) commuted outside sharded outer dim (x=4)
func.func @diff_dim_unsharded_inner(%arg0: tensor<10x12x80xf64> {sdy.sharding = #sdy.sharding<@mesh1, [{"z"}, {"y"}, {"x"}]>}) -> (tensor<14x12x82xf64> {sdy.sharding = #sdy.sharding<@mesh1, [{"z"}, {"y"}, {"x"}]>}) {
  %0 = "enzymexla.extend"(%arg0) <{dimension = 0 : i64, lhs = 2 : i64, rhs = 2 : i64}> {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<10x12x80xf64>) -> tensor<14x12x80xf64>
  %1 = "enzymexla.extend"(%0) <{dimension = 2 : i64, lhs = 1 : i64, rhs = 1 : i64}> {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<14x12x80xf64>) -> tensor<14x12x82xf64>
  return %1 : tensor<14x12x82xf64>
}

// SEL-LABEL: func.func @diff_dim_unsharded_inner(
// SEL: stablehlo.pad %arg0, %cst, low = [0, 0, 1], high = [0, 0, 1], interior = [0, 0, 0] {{.*}} : (tensor<10x12x80xf64>, tensor<f64>) -> tensor<10x12x82xf64>
// SEL: stablehlo.pad %arg0, %cst, low = [0, 0, 0], high = [0, 0, 2], interior = [0, 0, 0] {{.*}} : (tensor<10x12x80xf64>, tensor<f64>) -> tensor<10x12x82xf64>
// SEL: stablehlo.pad %arg0, %cst, low = [0, 0, 2], high = [0, 0, 0], interior = [0, 0, 0] {{.*}} : (tensor<10x12x80xf64>, tensor<f64>) -> tensor<10x12x82xf64>
// SEL: "enzymexla.extend"({{.*}}) <{dimension = 0 : i64, lhs = 2 : i64, rhs = 2 : i64}> {{.*}} : (tensor<10x12x82xf64>) -> tensor<14x12x82xf64>

// 4. Different dimensions under extend_comm=1: innermost non-dim pad happens once and manual_computations chain without intermediate slice/pad
func.func @diff_dim_manual_chain(%arg0: tensor<3x24x80xf64> {sdy.sharding = #sdy.sharding<@mesh2, [{"z"}, {"y"}, {"x"}]>}) -> (tensor<3x26x82xf64> {sdy.sharding = #sdy.sharding<@mesh2, [{"z"}, {"y"}, {"x"}]>}) {
  %0 = "enzymexla.extend"(%arg0) <{dimension = 2 : i64, lhs = 1 : i64, rhs = 1 : i64}> {sdy.sharding = #sdy.sharding_per_value<[<@mesh2, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<3x24x80xf64>) -> tensor<3x24x82xf64>
  %1 = "enzymexla.extend"(%0) <{dimension = 1 : i64, lhs = 1 : i64, rhs = 1 : i64}> {sdy.sharding = #sdy.sharding_per_value<[<@mesh2, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<3x24x82xf64>) -> tensor<3x26x82xf64>
  return %1 : tensor<3x26x82xf64>
}

// COMM-LABEL: func.func @diff_dim_manual_chain(
// COMM-NOT: enzymexla.extend
// COMM: %[[PAD0:.+]] = stablehlo.pad %arg0, %cst, low = [0, 0, 0], high = [1, 0, 0], interior = [0, 0, 0] : (tensor<3x24x80xf64>, tensor<f64>) -> tensor<4x24x80xf64>
// COMM-NEXT: %[[MAN1:.+]] = sdy.manual_computation(%[[PAD0]])
// COMM: } : (tensor<4x24x80xf64>) -> tensor<4x24x84xf64>
// COMM-NEXT: %[[MAN2:.+]] = sdy.manual_computation(%[[MAN1]])
// COMM: } : (tensor<4x24x84xf64>) -> tensor<4x30x84xf64>
// COMM-NEXT: %[[SL:.+]] = stablehlo.slice %[[MAN2]] [0:3, 2:28, 1:83] : (tensor<4x30x84xf64>) -> tensor<3x26x82xf64>
// COMM-NEXT: return %[[SL]] : tensor<3x26x82xf64>

// 5. Exact slice-pad cancellation in padWithUndefinedValueInDim
func.func @exact_slice_pad_cancel(%arg0: tensor<1x12x82xf64> {sdy.sharding = #sdy.sharding<@mesh1, [{"z"}, {"y"}, {"x"}]>}) -> (tensor<1x12x82xf64> {sdy.sharding = #sdy.sharding<@mesh1, [{"z"}, {"y"}, {"x"}]>}) {
  %0 = stablehlo.slice %arg0 [0:1, 0:12, 1:81] {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<1x12x82xf64>) -> tensor<1x12x80xf64>
  %1 = "enzymexla.extend"(%0) <{dimension = 2 : i64, lhs = 1 : i64, rhs = 1 : i64}> {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<1x12x80xf64>) -> tensor<1x12x82xf64>
  return %1 : tensor<1x12x82xf64>
}

// SEL-LABEL: func.func @exact_slice_pad_cancel(
// SEL-NOT: low = [0, 0, 1], high = [0, 0, 1]
// SEL: stablehlo.select {{.*}}, {{.*}}, %arg0
// SEL: return

// 6. Both dimensions sharded: commute so larger dimension (80->82) pads 1x12x80 before smaller dimension (12->16) inflates tensor by 33%
func.func @diff_dim_both_sharded_commute(%arg0: tensor<1x12x80xf64> {sdy.sharding = #sdy.sharding<@mesh1, [{"z"}, {"y"}, {"x"}]>}) -> (tensor<1x16x82xf64> {sdy.sharding = #sdy.sharding<@mesh1, [{"z"}, {"y"}, {"x"}]>}) {
  %0 = "enzymexla.extend"(%arg0) <{dimension = 1 : i64, lhs = 2 : i64, rhs = 2 : i64}> {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<1x12x80xf64>) -> tensor<1x16x80xf64>
  %1 = "enzymexla.extend"(%0) <{dimension = 2 : i64, lhs = 1 : i64, rhs = 1 : i64}> {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<1x16x80xf64>) -> tensor<1x16x82xf64>
  return %1 : tensor<1x16x82xf64>
}

// SEL-LABEL: func.func @diff_dim_both_sharded_commute(
// SEL: stablehlo.pad %arg0, %cst, low = [0, 0, 1], high = [0, 0, 1], interior = [0, 0, 0] {{.*}} : (tensor<1x12x80xf64>, tensor<f64>) -> tensor<1x12x82xf64>
// SEL: stablehlo.pad {{.*}}, %cst, low = [0, 2, 0], high = [0, 2, 0], interior = [0, 0, 0] {{.*}} : (tensor<1x12x82xf64>, tensor<f64>) -> tensor<1x16x82xf64>
// SEL: return
