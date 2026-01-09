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

  // CHECK: %[[START:.+]] = stablehlo.constant dense<0> : tensor<i64>
  // CHECK: %[[COND:.+]] = stablehlo.compare  LT, %[[START]], %[[ARG2]] : (tensor<i64>, tensor<i64>) -> tensor<i1>
  // CHECK: %[[IF:.+]]:2 = "stablehlo.if"(%4) ({
  // CHECK:     %[[TRIP_COUNT:.+]] = stablehlo.divide %[[ARG2]], %c_3 : tensor<i64>
  // CHECK:     %[[DUS1:.+]] = stablehlo.dynamic_update_slice %[[ARG1]]
  // CHECK:     %[[DUS2:.+]] = stablehlo.dynamic_update_slice %[[DUS1]]
  // CHECK:     %[[DUS3:.+]] = stablehlo.dynamic_update_slice %[[DUS2]]
  // CHECK:     %[[DUS4:.+]] = stablehlo.dynamic_update_slice %[[DUS3]]
  // CHECK:     stablehlo.return %[[TRIP_COUNT]], %[[DUS4]] : tensor<i64>, tensor<20x6144x12272xf64>
  // CHECK:   }, {
  // CHECK:     stablehlo.return %[[START]], %[[ARG1]] : tensor<i64>, tensor<20x6144x12272xf64>
  // CHECK:   }) : (tensor<i1>) -> (tensor<i64>, tensor<20x6144x12272xf64>)
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
