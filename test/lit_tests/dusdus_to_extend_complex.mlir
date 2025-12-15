// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt | FileCheck %s

module {
  func.func @f(%arg0: tensor<20x1536x3072xf32>, %arg1: tensor<20x1536x3072xf32>) -> (tensor<20x1536x3072xf32>) {
    %c = stablehlo.constant dense<1528> : tensor<i32>
    %c_0 = stablehlo.constant dense<8> : tensor<i32>
    %c_1 = stablehlo.constant dense<7> : tensor<i32>
  
    %0 = stablehlo.slice %arg1 [8:12, 8:1528, 8:3064] : (tensor<20x1536x3072xf32>) -> tensor<4x1520x3056xf32>
  
    %1 = "enzymexla.extend"(%0) <{dimension = 0 : i64, lhs = 1 : i64, rhs = 1 : i64}> : (tensor<4x1520x3056xf32>) -> tensor<6x1520x3056xf32>

    // %2 := %arg0[7:13, 8:1528, 8:3064] = %1 = extend(%0, lhs=1, rhs=1) = extend([8:12, 8:1528, 8:3064], lhs=1, rhs=1, dim=0)
    %2 = stablehlo.dynamic_update_slice %arg0, %1, %c_1, %c_0, %c_0 : (tensor<20x1536x3072xf32>, tensor<6x1520x3056xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<20x1536x3072xf32>
    %3 = stablehlo.slice %arg1 [8:12, 8:9, 8:3064] : (tensor<20x1536x3072xf32>) -> tensor<4x1x3056xf32>
    %4 = stablehlo.slice %arg1 [8:12, 1527:1528, 8:3064] : (tensor<20x1536x3072xf32>) -> tensor<4x1x3056xf32>

    // %5 := %2[8:12, 7:8, 8:3064] = %3 = %arg1 [8:12, 8:9, 8:3064]
    %5 = stablehlo.dynamic_update_slice %2, %3, %c_0, %c_1, %c_0 : (tensor<20x1536x3072xf32>, tensor<4x1x3056xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<20x1536x3072xf32>

    // %6 := %5[8:12, 1528:1529, 8:3064] = %4 = %arg1 [8:12, 1527:1528, 8:3064]
    %6 = stablehlo.dynamic_update_slice %5, %4, %c_0, %c, %c_0 : (tensor<20x1536x3072xf32>, tensor<4x1x3056xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<20x1536x3072xf32>

    return %6 : tensor<20x1536x3072xf32>
  }
}

// CHECK-LABEL: func.func @f
// CHECK: %[[SLICE:.*]] = stablehlo.slice %arg1 [8:12, 9:1527, 8:3064]
// CHECK: %[[EXTEND:.*]] = enzymexla.extend(%[[SLICE]]) <{dimension = 1 : i64, lhs = 1 : i64, rhs = 1 : i64}>
// CHECK: stablehlo.dynamic_update_slice %{{.*}}, %[[EXTEND]]
