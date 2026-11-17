// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=slice_of_updatewithoutcorners" --transform-interpreter --enzyme-hlo-remove-transform %s --split-input-file | FileCheck %s


func.func @t1(%79: tensor<6x6130x12272xf64>, %336: tensor<6x6130x12272xf64>) -> tensor<4x6128x12272xf64> {
  %338 = "enzymexla.update_without_corners"(%79, %336) <{dimensionX = 0 : i64, dimensionY = 1 : i64, x1 = 1 : i64, x2 = 5 : i64, y1 = 1 : i64, y2 = 6129 : i64}> : (tensor<6x6130x12272xf64>, tensor<6x6130x12272xf64>) -> tensor<6x6130x12272xf64>
  %389 = stablehlo.slice %338 [1:5, 1:6129, 0:12272] : (tensor<6x6130x12272xf64>) -> tensor<4x6128x12272xf64>
  return %389 : tensor<4x6128x12272xf64>
}

// CHECK:  func.func @t1(%arg0: tensor<6x6130x12272xf64>, %arg1: tensor<6x6130x12272xf64>) -> tensor<4x6128x12272xf64> {
// CHECK-NEXT:    %0 = stablehlo.slice %arg1 [1:5, 1:6129, 0:12272] : (tensor<6x6130x12272xf64>) -> tensor<4x6128x12272xf64>
// CHECK-NEXT:    return %0 : tensor<4x6128x12272xf64>
// CHECK-NEXT:  }

func.func @t2(%79: tensor<6x6130x12272xf64>, %336: tensor<6x6130x12272xf64>) -> tensor<1x1x12272xf64> {
  %338 = "enzymexla.update_without_corners"(%79, %336) <{dimensionX = 0 : i64, dimensionY = 1 : i64, x1 = 1 : i64, x2 = 5 : i64, y1 = 1 : i64, y2 = 6129 : i64}> : (tensor<6x6130x12272xf64>, tensor<6x6130x12272xf64>) -> tensor<6x6130x12272xf64>
  %1288 = stablehlo.slice %338 [5:6, 0:1, 0:12272] : (tensor<6x6130x12272xf64>) -> tensor<1x1x12272xf64>
  return %1288 : tensor<1x1x12272xf64>
}

// CHECK:  func.func @t2(%arg0: tensor<6x6130x12272xf64>, %arg1: tensor<6x6130x12272xf64>) -> tensor<1x1x12272xf64> {
// CHECK-NEXT:    %0 = stablehlo.slice %arg0 [5:6, 0:1, 0:12272] : (tensor<6x6130x12272xf64>) -> tensor<1x1x12272xf64>
// CHECK-NEXT:    return %0 : tensor<1x1x12272xf64>
// CHECK-NEXT:  }

func.func @t3(%79: tensor<6x6130x12272xf64>, %336: tensor<6x6130x12272xf64>) -> tensor<1x1x12272xf64> {
  %338 = "enzymexla.update_without_corners"(%79, %336) <{dimensionX = 0 : i64, dimensionY = 1 : i64, x1 = 1 : i64, x2 = 5 : i64, y1 = 1 : i64, y2 = 6129 : i64}> : (tensor<6x6130x12272xf64>, tensor<6x6130x12272xf64>) -> tensor<6x6130x12272xf64>
  %1289 = stablehlo.slice %338 [5:6, 6129:6130, 0:12272] : (tensor<6x6130x12272xf64>) -> tensor<1x1x12272xf64>
  return %1289 : tensor<1x1x12272xf64>
}

// CHECK:  func.func @t3(%arg0: tensor<6x6130x12272xf64>, %arg1: tensor<6x6130x12272xf64>) -> tensor<1x1x12272xf64> {
// CHECK-NEXT:    %0 = stablehlo.slice %arg0 [5:6, 6129:6130, 0:12272] : (tensor<6x6130x12272xf64>) -> tensor<1x1x12272xf64>
// CHECK-NEXT:    return %0 : tensor<1x1x12272xf64>
// CHECK-NEXT:  }

// -----

// Non-corner edge slices (spanning full X with interior Y, or interior X with full Y)
// never touch corners and must redirect to %arg1 (update), including strided slices.
func.func @edge_non_corner_and_strided(%old: tensor<6x6130x8xf64>, %upd: tensor<6x6130x8xf64>) -> (tensor<6x6128x8xf64>, tensor<2x6130x8xf64>, tensor<2x2x8xf64>) {
  %u = "enzymexla.update_without_corners"(%old, %upd) <{dimensionX = 0 : i64, dimensionY = 1 : i64, x1 = 1 : i64, x2 = 5 : i64, y1 = 1 : i64, y2 = 6129 : i64}> : (tensor<6x6130x8xf64>, tensor<6x6130x8xf64>) -> tensor<6x6130x8xf64>
  %s1 = stablehlo.slice %u [0:6, 1:6129, 0:8] : (tensor<6x6130x8xf64>) -> tensor<6x6128x8xf64>
  %s2 = stablehlo.slice %u [1:5:2, 0:6130, 0:8] : (tensor<6x6130x8xf64>) -> tensor<2x6130x8xf64>
  %s3 = stablehlo.slice %u [0:6:5, 0:6130:6129, 0:8] : (tensor<6x6130x8xf64>) -> tensor<2x2x8xf64>
  return %s1, %s2, %s3 : tensor<6x6128x8xf64>, tensor<2x6130x8xf64>, tensor<2x2x8xf64>
}

// CHECK-LABEL: func.func @edge_non_corner_and_strided
// CHECK-NEXT:    %0 = stablehlo.slice %arg1 [0:6, 1:6129, 0:8]
// CHECK-NEXT:    %1 = stablehlo.slice %arg1 [1:5:2, 0:6130, 0:8]
// CHECK-NEXT:    %2 = stablehlo.slice %arg0 [0:6:5, 0:6130:6129, 0:8]
// CHECK-NEXT:    return %0, %1, %2

// -----

// Single-use UpdateWithoutCorners sliced along a full border row splits into 1D concat.
func.func @single_use_border_row_split(%old: tensor<6x6130x8xf64>, %upd: tensor<6x6130x8xf64>) -> tensor<6x1x8xf64> {
  %u = "enzymexla.update_without_corners"(%old, %upd) <{dimensionX = 0 : i64, dimensionY = 1 : i64, x1 = 1 : i64, x2 = 5 : i64, y1 = 1 : i64, y2 = 6129 : i64}> : (tensor<6x6130x8xf64>, tensor<6x6130x8xf64>) -> tensor<6x6130x8xf64>
  %s = stablehlo.slice %u [0:6, 0:1, 0:8] : (tensor<6x6130x8xf64>) -> tensor<6x1x8xf64>
  return %s : tensor<6x1x8xf64>
}

// CHECK-LABEL: func.func @single_use_border_row_split
// CHECK-NEXT:    %0 = stablehlo.slice %arg0 [0:1, 0:1, 0:8]
// CHECK-NEXT:    %1 = stablehlo.slice %arg1 [1:5, 0:1, 0:8]
// CHECK-NEXT:    %2 = stablehlo.slice %arg0 [5:6, 0:1, 0:8]
// CHECK-NEXT:    %3 = stablehlo.concatenate %0, %1, %2, dim = 0
// CHECK-NEXT:    return %3
