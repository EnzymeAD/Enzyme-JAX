// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=dusdus_to_duspad" --transform-interpreter --enzyme-hlo-remove-transform -allow-unregistered-dialect --split-input-file %s | FileCheck %s

func.func @f(%iterArg_170 : tensor<20x6144x12272xf64>) -> (tensor<20x6144x12272xf64>, tensor<4x6126x12272xf64>, tensor<4x6124x12272xf64>, tensor<4x6125x12272xf64>, tensor<4x6123x12272xf64>, tensor<4x6129x12272xf64>, tensor<4x6129x12272xf64>, tensor<4x6128x12272xf64>) {

%c_162 = stablehlo.constant dense<6136> : tensor<i32>
%c_163 = stablehlo.constant dense<8> : tensor<i32>
%c_165 = stablehlo.constant dense<0> : tensor<i32>
%cst_164 = stablehlo.constant dense<0.000000e+00> : tensor<4x1x12272xf64>

%494 = stablehlo.dynamic_update_slice %iterArg_170, %cst_164, %c_163, %c_163, %c_165 : (tensor<20x6144x12272xf64>, tensor<4x1x12272xf64>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<20x6144x12272xf64>

  %633 = stablehlo.slice %494 [8:12, 8:6134, 0:12272] : (tensor<20x6144x12272xf64>) -> tensor<4x6126x12272xf64>
  %752 = stablehlo.slice %494 [8:12, 8:6132, 0:12272] : (tensor<20x6144x12272xf64>) -> tensor<4x6124x12272xf64>
  %2052 = stablehlo.slice %494 [8:12, 8:6133, 0:12272] : (tensor<20x6144x12272xf64>) -> tensor<4x6125x12272xf64>
  %2144 = stablehlo.slice %494 [8:12, 8:6131, 0:12272] : (tensor<20x6144x12272xf64>) -> tensor<4x6123x12272xf64>
  %2495 = stablehlo.slice %494 [8:12, 6:6135, 0:12272] : (tensor<20x6144x12272xf64>) -> tensor<4x6129x12272xf64>
  %2497 = stablehlo.slice %494 [8:12, 7:6136, 0:12272] : (tensor<20x6144x12272xf64>) -> tensor<4x6129x12272xf64>

%497 = stablehlo.dynamic_update_slice %494, %cst_164, %c_163, %c_162, %c_165 : (tensor<20x6144x12272xf64>, tensor<4x1x12272xf64>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<20x6144x12272xf64>

%496 = stablehlo.slice %494 [8:12, 8:6136, 0:12272] : (tensor<20x6144x12272xf64>) -> tensor<4x6128x12272xf64>

  func.return %497,  %633, %752, %2052, %2144, %2495, %2497, %496 : tensor<20x6144x12272xf64>, tensor<4x6126x12272xf64>, tensor<4x6124x12272xf64>, tensor<4x6125x12272xf64>, tensor<4x6123x12272xf64>, tensor<4x6129x12272xf64>, tensor<4x6129x12272xf64>, tensor<4x6128x12272xf64>
}

// CHECK:  func.func @f(%arg0: tensor<20x6144x12272xf64>) -> (tensor<20x6144x12272xf64>, tensor<4x6126x12272xf64>, tensor<4x6124x12272xf64>, tensor<4x6125x12272xf64>, tensor<4x6123x12272xf64>, tensor<4x6129x12272xf64>, tensor<4x6129x12272xf64>, tensor<4x6128x12272xf64>) {
// CHECK-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:    %c = stablehlo.constant dense<8> : tensor<i32>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<0> : tensor<i32>
// CHECK-NEXT:    %0 = stablehlo.slice %arg0 [8:12, 9:6136, 0:12272] : (tensor<20x6144x12272xf64>) -> tensor<4x6127x12272xf64>
// CHECK-NEXT:    %1 = stablehlo.pad %0, %cst, low = [0, 1, 0], high = [0, 1, 0], interior = [0, 0, 0] : (tensor<4x6127x12272xf64>, tensor<f64>) -> tensor<4x6129x12272xf64>
// CHECK-NEXT:    %2 = stablehlo.dynamic_update_slice %arg0, %1, %c, %c, %c_0 : (tensor<20x6144x12272xf64>, tensor<4x6129x12272xf64>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<20x6144x12272xf64>
// CHECK-NEXT:    %3 = stablehlo.slice %2 [8:12, 8:6134, 0:12272] : (tensor<20x6144x12272xf64>) -> tensor<4x6126x12272xf64>
// CHECK-NEXT:    %4 = stablehlo.slice %2 [8:12, 8:6132, 0:12272] : (tensor<20x6144x12272xf64>) -> tensor<4x6124x12272xf64>
// CHECK-NEXT:    %5 = stablehlo.slice %2 [8:12, 8:6133, 0:12272] : (tensor<20x6144x12272xf64>) -> tensor<4x6125x12272xf64>
// CHECK-NEXT:    %6 = stablehlo.slice %2 [8:12, 8:6131, 0:12272] : (tensor<20x6144x12272xf64>) -> tensor<4x6123x12272xf64>
// CHECK-NEXT:    %7 = stablehlo.slice %2 [8:12, 6:6135, 0:12272] : (tensor<20x6144x12272xf64>) -> tensor<4x6129x12272xf64>
// CHECK-NEXT:    %8 = stablehlo.slice %2 [8:12, 7:6136, 0:12272] : (tensor<20x6144x12272xf64>) -> tensor<4x6129x12272xf64>
// CHECK-NEXT:    %9 = stablehlo.slice %2 [8:12, 8:6136, 0:12272] : (tensor<20x6144x12272xf64>) -> tensor<4x6128x12272xf64>
// CHECK-NEXT:    return %2, %3, %4, %5, %6, %7, %8, %9 : tensor<20x6144x12272xf64>, tensor<4x6126x12272xf64>, tensor<4x6124x12272xf64>, tensor<4x6125x12272xf64>, tensor<4x6123x12272xf64>, tensor<4x6129x12272xf64>, tensor<4x6129x12272xf64>, tensor<4x6128x12272xf64>
// CHECK-NEXT:  }


