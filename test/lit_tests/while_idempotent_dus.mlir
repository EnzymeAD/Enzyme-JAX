// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

// CHECK-LABEL: func @idempotent_dus
// CHECK-SAME: (%[[ARG0:.+]]: tensor<20x6144x12288xf64>, %[[ARG1:.+]]: tensor<20x6144x12272xf64>, %[[ARG2:.+]]: tensor<i64>)
func.func @idempotent_dus(%arg20: tensor<20x6144x12288xf64>,
  %arg37 : tensor<20x6144x12272xf64>,
  %arg38 : tensor<i64>) -> (tensor<i64>, tensor<20x6144x12272xf64>) {


  %67 = stablehlo.slice %arg20 [11:12, 8:6136, 8:12280] : (tensor<20x6144x12288xf64>) -> tensor<1x6128x12272xf64>

  %70 = stablehlo.slice %arg20 [8:9, 8:6136, 8:12280] : (tensor<20x6144x12288xf64>) -> tensor<1x6128x12272xf64>

  %89 = stablehlo.slice %arg20 [8:12, 6135:6136, 8:12280] : (tensor<20x6144x12288xf64>) -> tensor<4x1x12272xf64> 

  %92 = stablehlo.slice %arg20 [8:12, 8:9, 8:12280] : (tensor<20x6144x12288xf64>) -> tensor<4x1x12272xf64>


  %c_160 = stablehlo.constant dense<12> : tensor<i32> 
  %c_161 = stablehlo.constant dense<7> : tensor<i32> 
  %c_7 = stablehlo.constant dense<7> : tensor<i64> 

  %c_162 = stablehlo.constant dense<6136> : tensor<i32> 
  %c_163 = stablehlo.constant dense<8> : tensor<i32> 
  %c_165 = stablehlo.constant dense<0> : tensor<i32> 
  %c_167 = stablehlo.constant dense<0> : tensor<i64> 

  // CHECK:  %c = stablehlo.constant {enzymexla.non_negative = [#enzymexla<guaranteed GUARANTEED>]} dense<0> : tensor<i64>
  // CHECK-NEXT:  %c_0 = stablehlo.constant dense<0> : tensor<i32>
  // CHECK-NEXT:  %c_1 = stablehlo.constant dense<8> : tensor<i32>
  // CHECK-NEXT:  %c_2 = stablehlo.constant dense<6136> : tensor<i32>
  // CHECK-NEXT:  %c_3 = stablehlo.constant dense<7> : tensor<i32>
  // CHECK-NEXT:  %c_4 = stablehlo.constant dense<12> : tensor<i32>
  // CHECK-NEXT:  %0 = stablehlo.slice %arg0 [11:12, 8:6136, 8:12280] : (tensor<20x6144x12288xf64>) -> tensor<1x6128x12272xf64>
  // CHECK-NEXT:  %1 = stablehlo.slice %arg0 [8:9, 8:6136, 8:12280] : (tensor<20x6144x12288xf64>) -> tensor<1x6128x12272xf64>
  // CHECK-NEXT:  %2 = stablehlo.slice %arg0 [8:12, 6135:6136, 8:12280] : (tensor<20x6144x12288xf64>) -> tensor<4x1x12272xf64>
  // CHECK-NEXT:  %3 = stablehlo.slice %arg0 [8:12, 8:9, 8:12280] : (tensor<20x6144x12288xf64>) -> tensor<4x1x12272xf64>
  // CHECK-NEXT:  %4 = stablehlo.dynamic_update_slice %arg1, %1, %c_3, %c_1, %c_0 : (tensor<20x6144x12272xf64>, tensor<1x6128x12272xf64>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<20x6144x12272xf64>
  // CHECK-NEXT:  %5 = stablehlo.dynamic_update_slice %4, %0, %c_4, %c_1, %c_0 : (tensor<20x6144x12272xf64>, tensor<1x6128x12272xf64>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<20x6144x12272xf64>
  // CHECK-NEXT:  %6 = stablehlo.dynamic_update_slice %5, %3, %c_1, %c_3, %c_0 : (tensor<20x6144x12272xf64>, tensor<4x1x12272xf64>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<20x6144x12272xf64>
  // CHECK-NEXT:  %7 = stablehlo.dynamic_update_slice %6, %2, %c_1, %c_2, %c_0 : (tensor<20x6144x12272xf64>, tensor<4x1x12272xf64>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<20x6144x12272xf64>
  // CHECK-NEXT:  %8 = stablehlo.compare  EQ, %arg2, %c : (tensor<i64>, tensor<i64>) -> tensor<i1>
  // CHECK-NEXT:  %9 = stablehlo.select %8, %arg1, %7 : tensor<i1>, tensor<20x6144x12272xf64>
  // CHECK-NEXT:  return %arg2, %9 : tensor<i64>, tensor<20x6144x12272xf64>
  
  %456:2 = stablehlo.while(%iterArg = %c_167, %iterArg_169 = %arg37) : tensor<i64>, tensor<20x6144x12272xf64> attributes {enzyme.disable_mincut}
  cond {
    %492 = stablehlo.compare  LT, %iterArg, %arg38 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %492 : tensor<i1>
  } do {
    %493 = stablehlo.add %iterArg, %c_7 : tensor<i64>
    %500 = stablehlo.dynamic_update_slice %iterArg_169, %70, %c_161, %c_163, %c_165 : (tensor<20x6144x12272xf64>, tensor<1x6128x12272xf64>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<20x6144x12272xf64>


    %505 = stablehlo.dynamic_update_slice %500, %67, %c_160, %c_163, %c_165 : (tensor<20x6144x12272xf64>, tensor<1x6128x12272xf64>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<20x6144x12272xf64>

    %522 = stablehlo.dynamic_update_slice %505, %92, %c_163, %c_161, %c_165 : (tensor<20x6144x12272xf64>, tensor<4x1x12272xf64>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<20x6144x12272xf64>


    %529 = stablehlo.dynamic_update_slice %522, %89, %c_163, %c_162, %c_165 : (tensor<20x6144x12272xf64>, tensor<4x1x12272xf64>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<20x6144x12272xf64>

    stablehlo.return %493, %529 : tensor<i64>, tensor<20x6144x12272xf64>
  }
  return %456#0, %456#1 : tensor<i64>, tensor<20x6144x12272xf64>
}
