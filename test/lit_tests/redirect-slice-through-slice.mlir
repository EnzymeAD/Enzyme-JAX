// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=redirect_slice_through_slice" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

func.func @main(%1227: tensor<20x1536x3056xf32>, %1231: tensor<1x1520x3056xf32>, %1232: tensor<1x1520x3056xf32>) -> (tensor<4x1520x3056xf32>, tensor<20x1536x3056xf32>, tensor<3x1520x3056xf32>, tensor<3x1520x3056xf32>) {
    %c_299 = stablehlo.constant dense<12> : tensor<i32>
    %c_300 = stablehlo.constant dense<7> : tensor<i32>
    %c_302 = stablehlo.constant dense<8> : tensor<i32>
    %c_365 = stablehlo.constant dense<0> : tensor<i32>
    %1237 = stablehlo.dynamic_update_slice %1227, %1231, %c_300, %c_302, %c_365 : (tensor<20x1536x3056xf32>, tensor<1x1520x3056xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<20x1536x3056xf32>
    %1238 = stablehlo.slice %1237 [7:11, 8:1528, 0:3056] : (tensor<20x1536x3056xf32>) -> tensor<4x1520x3056xf32>
    %1245 = stablehlo.dynamic_update_slice %1237, %1232, %c_299, %c_302, %c_365 : (tensor<20x1536x3056xf32>, tensor<1x1520x3056xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<20x1536x3056xf32>
    %3556 = stablehlo.slice %1237 [7:10, 8:1528, 0:3056] : (tensor<20x1536x3056xf32>) -> tensor<3x1520x3056xf32>
    %3557 = stablehlo.slice %1245 [10:13, 8:1528, 0:3056] : (tensor<20x1536x3056xf32>) -> tensor<3x1520x3056xf32>
    return %1238, %1245, %3556, %3557 : tensor<4x1520x3056xf32>, tensor<20x1536x3056xf32>, tensor<3x1520x3056xf32>, tensor<3x1520x3056xf32>
}

// CHECK: func.func @main(%arg0: tensor<20x1536x3056xf32>, %arg1: tensor<1x1520x3056xf32>, %arg2: tensor<1x1520x3056xf32>) -> (tensor<4x1520x3056xf32>, tensor<20x1536x3056xf32>, tensor<3x1520x3056xf32>, tensor<3x1520x3056xf32>) {
// CHECK-NEXT:     %c = stablehlo.constant dense<12> : tensor<i32>
// CHECK-NEXT:     %c_0 = stablehlo.constant dense<7> : tensor<i32>
// CHECK-NEXT:     %c_1 = stablehlo.constant dense<8> : tensor<i32>
// CHECK-NEXT:     %c_2 = stablehlo.constant dense<0> : tensor<i32>
// CHECK-NEXT:     %0 = stablehlo.dynamic_update_slice %arg0, %arg1, %c_0, %c_1, %c_2 : (tensor<20x1536x3056xf32>, tensor<1x1520x3056xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<20x1536x3056xf32>
// CHECK-NEXT:     %1 = stablehlo.slice %0 [7:11, 8:1528, 0:3056] : (tensor<20x1536x3056xf32>) -> tensor<4x1520x3056xf32>
// CHECK-NEXT:     %2 = stablehlo.dynamic_update_slice %0, %arg2, %c, %c_1, %c_2 : (tensor<20x1536x3056xf32>, tensor<1x1520x3056xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<20x1536x3056xf32>
// CHECK-NEXT:     %3 = stablehlo.slice %1 [0:3, 0:1520, 0:3056] : (tensor<4x1520x3056xf32>) -> tensor<3x1520x3056xf32>
// CHECK-NEXT:     %4 = stablehlo.slice %2 [10:13, 8:1528, 0:3056] : (tensor<20x1536x3056xf32>) -> tensor<3x1520x3056xf32>
// CHECK-NEXT:     return %1, %2, %3, %4 : tensor<4x1520x3056xf32>, tensor<20x1536x3056xf32>, tensor<3x1520x3056xf32>, tensor<3x1520x3056xf32>
// CHECK-NEXT: }

func.func @main2(%arg6: tensor<1536xf32>) -> (tensor<1520xf32>,tensor<1520xf32>,tensor<1520xf32>,tensor<1534xf32>,tensor<1534xf32>,tensor<1519xf32>,tensor<1519xf32>,tensor<1519xf32>,tensor<1514xf32>,tensor<1514xf32>) {
    %0 = stablehlo.slice %arg6 [7:1527] : (tensor<1536xf32>) -> tensor<1520xf32>
    %1 = stablehlo.slice %arg6 [8:1528] : (tensor<1536xf32>) -> tensor<1520xf32>
    %2 = stablehlo.slice %arg6 [9:1529] : (tensor<1536xf32>) -> tensor<1520xf32>
    %17 = stablehlo.slice %arg6 [1:1535] : (tensor<1536xf32>) -> tensor<1534xf32>
    %18 = stablehlo.slice %arg6 [2:1536] : (tensor<1536xf32>) -> tensor<1534xf32>
    %465 = stablehlo.slice %arg6 [9:1528] : (tensor<1536xf32>) -> tensor<1519xf32>
    %677 = stablehlo.slice %arg6 [11:1530] : (tensor<1536xf32>) -> tensor<1519xf32>
    %679 = stablehlo.slice %arg6 [7:1526] : (tensor<1536xf32>) -> tensor<1519xf32>
    %695 = stablehlo.slice %arg6 [14:1528] : (tensor<1536xf32>) -> tensor<1514xf32>
    %707 = stablehlo.slice %arg6 [9:1523] : (tensor<1536xf32>) -> tensor<1514xf32>
    return %0, %1, %2, %17, %18, %465, %677, %679, %695, %707 : tensor<1520xf32>, tensor<1520xf32>, tensor<1520xf32>, tensor<1534xf32>, tensor<1534xf32>, tensor<1519xf32>, tensor<1519xf32>, tensor<1519xf32>, tensor<1514xf32>, tensor<1514xf32>
}

// CHECK: func.func @main2(%arg0: tensor<1536xf32>) -> (tensor<1520xf32>, tensor<1520xf32>, tensor<1520xf32>, tensor<1534xf32>, tensor<1534xf32>, tensor<1519xf32>, tensor<1519xf32>, tensor<1519xf32>, tensor<1514xf32>, tensor<1514xf32>) {
// CHECK-NEXT:    %0 = stablehlo.slice %arg0 [7:1527] : (tensor<1536xf32>) -> tensor<1520xf32>
// CHECK-NEXT:    %1 = stablehlo.slice %arg0 [8:1528] : (tensor<1536xf32>) -> tensor<1520xf32>
// CHECK-NEXT:    %2 = stablehlo.slice %arg0 [9:1529] : (tensor<1536xf32>) -> tensor<1520xf32>
// CHECK-NEXT:    %3 = stablehlo.slice %arg0 [1:1535] : (tensor<1536xf32>) -> tensor<1534xf32>
// CHECK-NEXT:    %4 = stablehlo.slice %arg0 [2:1536] : (tensor<1536xf32>) -> tensor<1534xf32>
// CHECK-NEXT:    %5 = stablehlo.slice %4 [7:1526] : (tensor<1534xf32>) -> tensor<1519xf32>
// CHECK-NEXT:    %6 = stablehlo.slice %4 [9:1528] : (tensor<1534xf32>) -> tensor<1519xf32>
// CHECK-NEXT:    %7 = stablehlo.slice %4 [5:1524] : (tensor<1534xf32>) -> tensor<1519xf32>
// CHECK-NEXT:    %8 = stablehlo.slice %6 [3:1517] : (tensor<1519xf32>) -> tensor<1514xf32>
// CHECK-NEXT:    %9 = stablehlo.slice %7 [2:1516] : (tensor<1519xf32>) -> tensor<1514xf32>
// CHECK-NEXT:    return %0, %1, %2, %3, %4, %5, %6, %7, %8, %9 : tensor<1520xf32>, tensor<1520xf32>, tensor<1520xf32>, tensor<1534xf32>, tensor<1534xf32>, tensor<1519xf32>, tensor<1519xf32>, tensor<1519xf32>, tensor<1514xf32>, tensor<1514xf32>
// CHECK-NEXT: }