// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=dusdus_to_dusextend" --transform-interpreter --enzyme-hlo-remove-transform -allow-unregistered-dialect --split-input-file %s | FileCheck %s

func.func @f(%iterArg_169 : tensor<20x1536x1520xf64>) -> (tensor<20x1536x1520xf64>, tensor<4x1520x1520xf64>, tensor<4x1520x1520xf64>) {
 


%c_160 = stablehlo.constant dense<12> : tensor<i32>
%c_161 = stablehlo.constant dense<7> : tensor<i32>
%c_162 = stablehlo.constant dense<1528> : tensor<i32>
%c_163 = stablehlo.constant dense<8> : tensor<i32>

%c_165 = stablehlo.constant dense<0> : tensor<i32>
 
 %305 = stablehlo.slice %iterArg_169 [8:9, 8:1528, 0:1520] : (tensor<20x1536x1520xf64>) -> tensor<1x1520x1520xf64>
 %311 = stablehlo.dynamic_update_slice %iterArg_169, %305, %c_161, %c_163, %c_165 : (tensor<20x1536x1520xf64>, tensor<1x1520x1520xf64>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<20x1536x1520xf64>

  %308 = stablehlo.slice %iterArg_169 [11:12, 8:1528, 0:1520] : (tensor<20x1536x1520xf64>) -> tensor<1x1520x1520xf64>
  %325 = stablehlo.dynamic_update_slice %311, %308, %c_160, %c_163, %c_165 : (tensor<20x1536x1520xf64>, tensor<1x1520x1520xf64>, tensor<i32>, tensor<i32>, tensor<i32>)-> tensor<20x1536x1520xf64>
 
  %312 = stablehlo.slice %311 [7:11, 8:1528, 0:1520] : (tensor<20x1536x1520xf64>) -> tensor<4x1520x1520xf64>
  %1189 = stablehlo.slice %311 [6:10, 8:1528, 0:1520] : (tensor<20x1536x1520xf64>) -> tensor<4x1520x1520xf64>
  func.return %325, %312, %1189 : tensor<20x1536x1520xf64>, tensor<4x1520x1520xf64>, tensor<4x1520x1520xf64>
}

// CHECK:  func.func @f(%arg0: tensor<20x1536x1520xf64>) -> (tensor<20x1536x1520xf64>, tensor<4x1520x1520xf64>, tensor<4x1520x1520xf64>) {
// CHECK-NEXT:    %c = stablehlo.constant dense<7> : tensor<i32>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<8> : tensor<i32>
// CHECK-NEXT:    %c_1 = stablehlo.constant dense<0> : tensor<i32>
// CHECK-NEXT:    %0 = stablehlo.slice %arg0 [8:12, 8:1528, 0:1520] : (tensor<20x1536x1520xf64>) -> tensor<4x1520x1520xf64>
// CHECK-NEXT:    %1 = "enzymexla.extend"(%0) <{dimension = 0 : i64, lhs = 1 : i64, rhs = 1 : i64}> : (tensor<4x1520x1520xf64>) -> tensor<6x1520x1520xf64>
// CHECK-NEXT:    %2 = stablehlo.dynamic_update_slice %arg0, %1, %c, %c_0, %c_1 : (tensor<20x1536x1520xf64>, tensor<6x1520x1520xf64>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<20x1536x1520xf64>
// CHECK-NEXT:    %3 = stablehlo.slice %2 [7:11, 8:1528, 0:1520] : (tensor<20x1536x1520xf64>) -> tensor<4x1520x1520xf64>
// CHECK-NEXT:    %4 = stablehlo.slice %2 [6:10, 8:1528, 0:1520] : (tensor<20x1536x1520xf64>) -> tensor<4x1520x1520xf64>
// CHECK-NEXT:    return %2, %3, %4 : tensor<20x1536x1520xf64>, tensor<4x1520x1520xf64>, tensor<4x1520x1520xf64>
// CHECK-NEXT:  }

func.func @f2(%iterArg_169 : tensor<20x1536x1520xf64>) -> (tensor<20x1536x1520xf64>) {
%c_160 = stablehlo.constant dense<12> : tensor<i32>
%c_161 = stablehlo.constant dense<7> : tensor<i32>
%c_162 = stablehlo.constant dense<1528> : tensor<i32>
%c_163 = stablehlo.constant dense<8> : tensor<i32>

%c_165 = stablehlo.constant dense<0> : tensor<i32>
 
 %305 = stablehlo.slice %iterArg_169 [8:9, 8:1528, 0:1520] : (tensor<20x1536x1520xf64>) -> tensor<1x1520x1520xf64>
 %308 = stablehlo.slice %iterArg_169 [11:12, 8:1528, 0:1520] : (tensor<20x1536x1520xf64>) -> tensor<1x1520x1520xf64>
 
  %311 = stablehlo.dynamic_update_slice %iterArg_169, %308, %c_160, %c_163, %c_165 : (tensor<20x1536x1520xf64>, tensor<1x1520x1520xf64>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<20x1536x1520xf64>

  %325 = stablehlo.dynamic_update_slice %311, %305, %c_161, %c_163, %c_165 : (tensor<20x1536x1520xf64>, tensor<1x1520x1520xf64>, tensor<i32>, tensor<i32>, tensor<i32>)-> tensor<20x1536x1520xf64>
 
  func.return %325 : tensor<20x1536x1520xf64>
}

// CHECK:  func.func @f2(%arg0: tensor<20x1536x1520xf64>) -> tensor<20x1536x1520xf64> {
// CHECK-NEXT:    %c = stablehlo.constant dense<7> : tensor<i32>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<8> : tensor<i32>
// CHECK-NEXT:    %c_1 = stablehlo.constant dense<0> : tensor<i32>
// CHECK-NEXT:    %0 = stablehlo.slice %arg0 [8:12, 8:1528, 0:1520] : (tensor<20x1536x1520xf64>) -> tensor<4x1520x1520xf64>
// CHECK-NEXT:    %1 = "enzymexla.extend"(%0) <{dimension = 0 : i64, lhs = 1 : i64, rhs = 1 : i64}> : (tensor<4x1520x1520xf64>) -> tensor<6x1520x1520xf64>
// CHECK-NEXT:    %2 = stablehlo.dynamic_update_slice %arg0, %1, %c, %c_0, %c_1 : (tensor<20x1536x1520xf64>, tensor<6x1520x1520xf64>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<20x1536x1520xf64>
// CHECK-NEXT:    return %2 : tensor<20x1536x1520xf64>
// CHECK-NEXT:  }

