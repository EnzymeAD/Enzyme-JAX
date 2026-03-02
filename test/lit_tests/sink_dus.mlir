// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=sink_dus" --transform-interpreter --enzyme-hlo-remove-transform -allow-unregistered-dialect %s | FileCheck %s

module {

  func.func @"loop!"(%arg38 : tensor<i64>, %5: tensor<6x3057x6128xf64>) -> tensor<6x3057x6128xf64> {
    %c = stablehlo.constant dense<8> : tensor<i32>
    %c_0 = stablehlo.constant dense<7> : tensor<i32>
    %c_1 = stablehlo.constant dense<3057> : tensor<i32>
    %c_2 = stablehlo.constant dense<5> : tensor<i32>
    %c_3 = stablehlo.constant dense<3056> : tensor<i32>
    %c_4 = stablehlo.constant dense<0> : tensor<i32>
    %c_5 = stablehlo.constant dense<1> : tensor<i32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<4x1x6128xf64>
    %c_6 = stablehlo.constant dense<0> : tensor<i64>
    %c_7 = stablehlo.constant dense<1> : tensor<i64>
    %11:2 = stablehlo.while(%iterArg = %c_6, %iterArg_9 = %5) : tensor<i64>, tensor<6x3057x6128xf64>
    cond {
      %31 = stablehlo.compare  LT, %iterArg, %arg38 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %31 : tensor<i1>
    } do {
      %31 = stablehlo.add %iterArg, %c_7 : tensor<i64>
      
      %35 = stablehlo.dynamic_update_slice %iterArg_9, %cst, %c_5, %c_3, %c_4 : (tensor<6x3057x6128xf64>, tensor<4x1x6128xf64>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<6x3057x6128xf64>
      
      %a33 = stablehlo.slice %35 [4:5, 0:1, 0:6128] : (tensor<6x3057x6128xf64>) -> tensor<1x1x6128xf64>
      "test.use"(%a33) : (tensor<1x1x6128xf64>) -> ()

      %33 = stablehlo.slice %35 [4:5, 0:3056, 0:6128] : (tensor<6x3057x6128xf64>) -> tensor<1x3056x6128xf64>
      // was originally 0:3056, changed for better test
      %34 = stablehlo.slice %35 [1:2, 1:3057, 0:6128] : (tensor<6x3057x6128xf64>) -> tensor<1x3056x6128xf64>
      
      %43 = stablehlo.dynamic_update_slice %35, %34, %c_4, %c_4, %c_4 : (tensor<6x3057x6128xf64>, tensor<1x3056x6128xf64>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<6x3057x6128xf64>
      %47 = stablehlo.dynamic_update_slice %43, %33, %c_2, %c_4, %c_4 : (tensor<6x3057x6128xf64>, tensor<1x3056x6128xf64>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<6x3057x6128xf64>
     
      // 1:5, 0:1, 0:6128 
      %32 = stablehlo.dynamic_update_slice %47, %cst, %c_5, %c_4, %c_4 : (tensor<6x3057x6128xf64>, tensor<4x1x6128xf64>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<6x3057x6128xf64>
      
      stablehlo.return %31, %32 : tensor<i64>, tensor<6x3057x6128xf64>
    }
    stablehlo.return %11#1 : tensor<6x3057x6128xf64>
  }
}

// CHECK:  func.func @"loop!"(%arg0: tensor<i64>, %arg1: tensor<6x3057x6128xf64>) -> tensor<6x3057x6128xf64> {
// CHECK-NEXT:    %c = stablehlo.constant dense<5> : tensor<i32>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<3056> : tensor<i32>
// CHECK-NEXT:    %c_1 = stablehlo.constant dense<0> : tensor<i32>
// CHECK-NEXT:    %c_2 = stablehlo.constant dense<1> : tensor<i32>
// CHECK-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<4x1x6128xf64>
// CHECK-NEXT:    %c_3 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %c_4 = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:    %0:2 = stablehlo.while(%iterArg = %c_3, %iterArg_5 = %arg1) : tensor<i64>, tensor<6x3057x6128xf64>
// CHECK-NEXT:    cond {
// CHECK-NEXT:      %4 = stablehlo.compare  LT, %iterArg, %arg0 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:      stablehlo.return %4 : tensor<i1>
// CHECK-NEXT:    } do {
// CHECK-NEXT:      %4 = stablehlo.compare  EQ, %iterArg, %c_3 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:      %5 = stablehlo.add %iterArg, %c_4 : tensor<i64>
// CHECK-NEXT:      %6 = stablehlo.dynamic_update_slice %iterArg_5, %cst, %c_2, %c_0, %c_1 : (tensor<6x3057x6128xf64>, tensor<4x1x6128xf64>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<6x3057x6128xf64>
// CHECK-NEXT:      %7 = stablehlo.slice %cst [3:4, 0:1, 0:6128] : (tensor<4x1x6128xf64>) -> tensor<1x1x6128xf64>
// CHECK-NEXT:      %8 = stablehlo.slice %6 [4:5, 0:1, 0:6128] : (tensor<6x3057x6128xf64>) -> tensor<1x1x6128xf64>
// CHECK-NEXT:      %9 = stablehlo.select %4, %8, %7 : tensor<i1>, tensor<1x1x6128xf64>
// CHECK-NEXT:      "test.use"(%9) : (tensor<1x1x6128xf64>) -> ()
// CHECK-NEXT:      %10 = stablehlo.dynamic_update_slice %6, %cst, %c_2, %c_1, %c_1 : (tensor<6x3057x6128xf64>, tensor<4x1x6128xf64>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<6x3057x6128xf64>
// CHECK-NEXT:      %11 = stablehlo.slice %10 [4:5, 0:3056, 0:6128] : (tensor<6x3057x6128xf64>) -> tensor<1x3056x6128xf64>
// CHECK-NEXT:      %12 = stablehlo.slice %6 [4:5, 0:3056, 0:6128] : (tensor<6x3057x6128xf64>) -> tensor<1x3056x6128xf64>
// CHECK-NEXT:      %13 = stablehlo.select %4, %12, %11 : tensor<i1>, tensor<1x3056x6128xf64>
// CHECK-NEXT:      %14 = stablehlo.slice %6 [1:2, 1:3057, 0:6128] : (tensor<6x3057x6128xf64>) -> tensor<1x3056x6128xf64>
// CHECK-NEXT:      %15 = stablehlo.dynamic_update_slice %6, %14, %c_1, %c_1, %c_1 : (tensor<6x3057x6128xf64>, tensor<1x3056x6128xf64>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<6x3057x6128xf64>
// CHECK-NEXT:      %16 = stablehlo.dynamic_update_slice %15, %13, %c, %c_1, %c_1 : (tensor<6x3057x6128xf64>, tensor<1x3056x6128xf64>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<6x3057x6128xf64>
// CHECK-NEXT:      stablehlo.return %5, %16 : tensor<i64>, tensor<6x3057x6128xf64>
// CHECK-NEXT:    }
// CHECK-NEXT:    %1 = stablehlo.compare  LT, %c_3, %arg0 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:    %2 = stablehlo.dynamic_update_slice %0#1, %cst, %c_2, %c_1, %c_1 : (tensor<6x3057x6128xf64>, tensor<4x1x6128xf64>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<6x3057x6128xf64>
// CHECK-NEXT:    %3 = stablehlo.select %1, %2, %0#1 : tensor<i1>, tensor<6x3057x6128xf64>
// CHECK-NEXT:    stablehlo.return %3 : tensor<6x3057x6128xf64>
// CHECK-NEXT:  }

