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
