// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

// CHECK-LABEL: func.func @widening_convert_concat_kept(
// CHECK-SAME:    %[[ARG0:.*]]: tensor<8xbf16>) -> tensor<8xf32> {
// CHECK-NEXT:    %[[S0:.*]] = stablehlo.slice %[[ARG0]] [0:4] : (tensor<8xbf16>) -> tensor<4xbf16>
// CHECK-NEXT:    %[[S1:.*]] = stablehlo.slice %[[ARG0]] [4:8] : (tensor<8xbf16>) -> tensor<4xbf16>
// CHECK-NEXT:    %[[NEG:.*]] = stablehlo.negate %[[S1]] : tensor<4xbf16>
// CHECK-NEXT:    %[[CAT:.*]] = stablehlo.concatenate %[[S0]], %[[NEG]], dim = 0 : (tensor<4xbf16>, tensor<4xbf16>) -> tensor<8xbf16>
// CHECK-NEXT:    %[[CVT:.*]] = stablehlo.convert %[[CAT]] : (tensor<8xbf16>) -> tensor<8xf32>
// CHECK-NEXT:    return %[[CVT]] : tensor<8xf32>
// CHECK-NEXT:  }
func.func @widening_convert_concat_kept(%arg0: tensor<8xbf16>) -> tensor<8xf32> {
  %0 = stablehlo.slice %arg0 [0:4] : (tensor<8xbf16>) -> tensor<4xbf16>
  %1 = stablehlo.slice %arg0 [4:8] : (tensor<8xbf16>) -> tensor<4xbf16>
  %2 = stablehlo.negate %1 : tensor<4xbf16>
  %3 = stablehlo.concatenate %0, %2, dim = 0 : (tensor<4xbf16>, tensor<4xbf16>) -> tensor<8xbf16>
  %4 = stablehlo.convert %3 : (tensor<8xbf16>) -> tensor<8xf32>
  return %4 : tensor<8xf32>
}

// CHECK-LABEL: func.func @concat_widening_converts_hoisted(
// CHECK-SAME:    %[[ARG0:.*]]: tensor<8xbf16>) -> tensor<8xf32> {
// CHECK-NEXT:    %[[S0:.*]] = stablehlo.slice %[[ARG0]] [0:4] : (tensor<8xbf16>) -> tensor<4xbf16>
// CHECK-NEXT:    %[[S1:.*]] = stablehlo.slice %[[ARG0]] [4:8] : (tensor<8xbf16>) -> tensor<4xbf16>
// CHECK-NEXT:    %[[NEG:.*]] = stablehlo.negate %[[S1]] : tensor<4xbf16>
// CHECK-NEXT:    %[[CAT:.*]] = stablehlo.concatenate %[[S0]], %[[NEG]], dim = 0 : (tensor<4xbf16>, tensor<4xbf16>) -> tensor<8xbf16>
// CHECK-NEXT:    %[[CVT:.*]] = stablehlo.convert %[[CAT]] : (tensor<8xbf16>) -> tensor<8xf32>
// CHECK-NEXT:    return %[[CVT]] : tensor<8xf32>
// CHECK-NEXT:  }
func.func @concat_widening_converts_hoisted(%arg0: tensor<8xbf16>) -> tensor<8xf32> {
  %0 = stablehlo.slice %arg0 [0:4] : (tensor<8xbf16>) -> tensor<4xbf16>
  %1 = stablehlo.slice %arg0 [4:8] : (tensor<8xbf16>) -> tensor<4xbf16>
  %2 = stablehlo.negate %1 : tensor<4xbf16>
  %3 = stablehlo.convert %0 : (tensor<4xbf16>) -> tensor<4xf32>
  %4 = stablehlo.convert %2 : (tensor<4xbf16>) -> tensor<4xf32>
  %5 = stablehlo.concatenate %3, %4, dim = 0 : (tensor<4xf32>, tensor<4xf32>) -> tensor<8xf32>
  return %5 : tensor<8xf32>
}

// CHECK-LABEL: func.func @narrowing_convert_concat_pushed(
// CHECK-SAME:    %[[A:.*]]: tensor<4xf32>, %[[B:.*]]: tensor<4xf32>) -> tensor<8xbf16> {
// CHECK-NEXT:    %[[CA:.*]] = stablehlo.convert %[[A]] : (tensor<4xf32>) -> tensor<4xbf16>
// CHECK-NEXT:    %[[CB:.*]] = stablehlo.convert %[[B]] : (tensor<4xf32>) -> tensor<4xbf16>
// CHECK-NEXT:    %[[CAT:.*]] = stablehlo.concatenate %[[CA]], %[[CB]], dim = 0 : (tensor<4xbf16>, tensor<4xbf16>) -> tensor<8xbf16>
// CHECK-NEXT:    return %[[CAT]] : tensor<8xbf16>
// CHECK-NEXT:  }
func.func @narrowing_convert_concat_pushed(%a: tensor<4xf32>, %b: tensor<4xf32>) -> tensor<8xbf16> {
  %0 = stablehlo.concatenate %a, %b, dim = 0 : (tensor<4xf32>, tensor<4xf32>) -> tensor<8xf32>
  %1 = stablehlo.convert %0 : (tensor<8xf32>) -> tensor<8xbf16>
  return %1 : tensor<8xbf16>
}

// CHECK-LABEL: func.func @concat_mixed_source_converts_untouched(
// CHECK-SAME:    %[[A:.*]]: tensor<4xi8>, %[[B:.*]]: tensor<4xbf16>) -> tensor<8xf32> {
// CHECK-NEXT:    %[[CA:.*]] = stablehlo.convert %[[A]] : (tensor<4xi8>) -> tensor<4xf32>
// CHECK-NEXT:    %[[CB:.*]] = stablehlo.convert %[[B]] : (tensor<4xbf16>) -> tensor<4xf32>
// CHECK-NEXT:    %[[CAT:.*]] = stablehlo.concatenate %[[CA]], %[[CB]], dim = 0 : (tensor<4xf32>, tensor<4xf32>) -> tensor<8xf32>
// CHECK-NEXT:    return %[[CAT]] : tensor<8xf32>
// CHECK-NEXT:  }
func.func @concat_mixed_source_converts_untouched(%a: tensor<4xi8>, %b: tensor<4xbf16>) -> tensor<8xf32> {
  %0 = stablehlo.convert %a : (tensor<4xi8>) -> tensor<4xf32>
  %1 = stablehlo.convert %b : (tensor<4xbf16>) -> tensor<4xf32>
  %2 = stablehlo.concatenate %0, %1, dim = 0 : (tensor<4xf32>, tensor<4xf32>) -> tensor<8xf32>
  return %2 : tensor<8xf32>
}

// CHECK-LABEL: func.func @broadcast_widening_convert_reordered(
// CHECK-SAME:    %[[ARG:.*]]: tensor<32xbf16>) -> tensor<128x12x32xf32> {
// CHECK-NEXT:    %[[BCAST:.*]] = stablehlo.broadcast_in_dim %[[ARG]], dims = [2] : (tensor<32xbf16>) -> tensor<128x12x32xbf16>
// CHECK-NEXT:    %[[CVT:.*]] = stablehlo.convert %[[BCAST]] : (tensor<128x12x32xbf16>) -> tensor<128x12x32xf32>
// CHECK-NEXT:    return %[[CVT]] : tensor<128x12x32xf32>
// CHECK-NEXT:  }
func.func @broadcast_widening_convert_reordered(%arg: tensor<32xbf16>) -> tensor<128x12x32xf32> {
  %0 = stablehlo.convert %arg : (tensor<32xbf16>) -> tensor<32xf32>
  %1 = stablehlo.broadcast_in_dim %0, dims = [2] : (tensor<32xf32>) -> tensor<128x12x32xf32>
  return %1 : tensor<128x12x32xf32>
}

// CHECK-LABEL: func.func @broadcast_narrowing_convert_kept(
// CHECK-SAME:    %[[ARG:.*]]: tensor<32xf32>) -> tensor<128x12x32xbf16> {
// CHECK-NEXT:    %[[CVT:.*]] = stablehlo.convert %[[ARG]] : (tensor<32xf32>) -> tensor<32xbf16>
// CHECK-NEXT:    %[[BCAST:.*]] = stablehlo.broadcast_in_dim %[[CVT]], dims = [2] : (tensor<32xbf16>) -> tensor<128x12x32xbf16>
// CHECK-NEXT:    return %[[BCAST]] : tensor<128x12x32xbf16>
// CHECK-NEXT:  }
func.func @broadcast_narrowing_convert_kept(%arg: tensor<32xf32>) -> tensor<128x12x32xbf16> {
  %0 = stablehlo.convert %arg : (tensor<32xf32>) -> tensor<32xbf16>
  %1 = stablehlo.broadcast_in_dim %0, dims = [2] : (tensor<32xbf16>) -> tensor<128x12x32xbf16>
  return %1 : tensor<128x12x32xbf16>
}

// CHECK-LABEL: func.func @broadcast_scalar_widening_convert_kept(
// CHECK-SAME:    %[[ARG:.*]]: tensor<bf16>) -> tensor<16xf32> {
// CHECK-NEXT:    %[[CVT:.*]] = stablehlo.convert %[[ARG]] : (tensor<bf16>) -> tensor<f32>
// CHECK-NEXT:    %[[BCAST:.*]] = stablehlo.broadcast_in_dim %[[CVT]], dims = [] : (tensor<f32>) -> tensor<16xf32>
// CHECK-NEXT:    return %[[BCAST]] : tensor<16xf32>
// CHECK-NEXT:  }
func.func @broadcast_scalar_widening_convert_kept(%arg: tensor<bf16>) -> tensor<16xf32> {
  %0 = stablehlo.convert %arg : (tensor<bf16>) -> tensor<f32>
  %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f32>) -> tensor<16xf32>
  return %1 : tensor<16xf32>
}
