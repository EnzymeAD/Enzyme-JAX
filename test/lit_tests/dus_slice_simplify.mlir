// RUN: enzymexlamlir-opt %s --split-input-file --enzyme-hlo-generate-td="patterns=dus_slice_simplify" --transform-interpreter --enzyme-hlo-remove-transform | FileCheck %s

module {
  func.func @foo(%in: tensor<528x2048x2048xf64>, %update: tensor<1x2032x2032xf64>) -> (tensor<528x2048x2032xf64>, tensor<512x2032x2032xf64>) {
    %c_22 = stablehlo.constant dense<2040> : tensor<i32>
    %c_20 = stablehlo.constant dense<520> : tensor<i32>
    %c_23 = stablehlo.constant dense<8> : tensor<i32>
    %dus = stablehlo.dynamic_update_slice %in, %update, %c_20, %c_23, %c_23 : (tensor<528x2048x2048xf64>, tensor<1x2032x2032xf64>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<528x2048x2048xf64>
    %r1 = stablehlo.slice %dus [0:528, 0:2048, 8:2040] : (tensor<528x2048x2048xf64>) -> tensor<528x2048x2032xf64>
    %r2 = stablehlo.slice %dus [9:521, 8:2040, 8:2040] : (tensor<528x2048x2048xf64>) -> tensor<512x2032x2032xf64>
    return %r1, %r2 : tensor<528x2048x2032xf64>, tensor<512x2032x2032xf64>
  }
}

// CHECK-LABEL:   func.func @foo(
// CHECK-SAME:                   %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: tensor<528x2048x2048xf64>,
// CHECK-SAME:                   %[[VAL_1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: tensor<1x2032x2032xf64>) -> (tensor<528x2048x2032xf64>, tensor<512x2032x2032xf64>) {
// CHECK:           %[[VAL_2:.*]] = stablehlo.constant dense<0> : tensor<i32>
// CHECK:           %[[C8:.+]] = stablehlo.constant dense<8> : tensor<i32>
// CHECK:           %[[C520:.+]] = stablehlo.constant dense<520> : tensor<i32>
// CHECK:           %[[VAL_3:.*]] = stablehlo.slice %[[VAL_0]] [0:528, 0:2048, 8:2040] : (tensor<528x2048x2048xf64>) -> tensor<528x2048x2032xf64>
// CHECK:           %[[VAL_4:.*]] = stablehlo.slice %[[VAL_1]] [0:1, 0:2032, 0:2032] : (tensor<1x2032x2032xf64>) -> tensor<1x2032x2032xf64>
// CHECK:           %[[VAL_5:.*]] = stablehlo.dynamic_update_slice %[[VAL_3]], %[[VAL_4]], %[[C520]], %[[C8]], %[[VAL_2]] : (tensor<528x2048x2032xf64>, tensor<1x2032x2032xf64>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<528x2048x2032xf64>
// CHECK:           %[[VAL_6:.*]] = stablehlo.slice %[[VAL_5]] [9:521, 8:2040, 0:2032] : (tensor<528x2048x2032xf64>) -> tensor<512x2032x2032xf64>
// CHECK:           %[[VAL_7:.*]] = stablehlo.slice %[[VAL_5]] [0:528, 0:2048, 0:2032] : (tensor<528x2048x2032xf64>) -> tensor<528x2048x2032xf64>
// CHECK:           return %[[VAL_7]], %[[VAL_6]] : tensor<528x2048x2032xf64>, tensor<512x2032x2032xf64>
// CHECK:         }

// -----

module {
  func.func @main(%arg15: tensor<1x48x48xf64>, %32: tensor<1x32x32xf64>) -> tensor<1x14x48xf64> {
    %c_257 = stablehlo.constant dense<0> : tensor<i32>
    %c_258 = stablehlo.constant dense<8> : tensor<i32>
    %c_259 = stablehlo.constant dense<0> : tensor<i32>

    %35 = stablehlo.dynamic_update_slice %arg15, %32, %c_257, %c_258, %c_259 : (tensor<1x48x48xf64>, tensor<1x32x32xf64>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x48x48xf64>
    %slice = stablehlo.slice %35 [0:1, 34:48, 0:48] : (tensor<1x48x48xf64>) -> tensor<1x14x48xf64>
    return %slice : tensor<1x14x48xf64> 
  }
}

// CHECK:  func.func @main(%arg0: tensor<1x48x48xf64>, %arg1: tensor<1x32x32xf64>) -> tensor<1x14x48xf64> {
// CHECK-NEXT:    %c = stablehlo.constant dense<0> : tensor<i32>
// CHECK-NEXT:    %0 = stablehlo.slice %arg0 [0:1, 34:48, 0:48] : (tensor<1x48x48xf64>) -> tensor<1x14x48xf64>
// CHECK-NEXT:    %1 = stablehlo.slice %arg1 [0:1, 26:32, 0:32] : (tensor<1x32x32xf64>) -> tensor<1x6x32xf64>
// CHECK-NEXT:    %2 = stablehlo.dynamic_update_slice %0, %1, %c, %c, %c : (tensor<1x14x48xf64>, tensor<1x6x32xf64>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x14x48xf64>
// CHECK-NEXT:    %3 = stablehlo.slice %2 [0:1, 0:14, 0:48] : (tensor<1x14x48xf64>) -> tensor<1x14x48xf64>
// CHECK-NEXT:    return %3 : tensor<1x14x48xf64>
// CHECK-NEXT:  }

// -----

module {
  func.func @main(%arg15: tensor<1x48x48xf64>, %32: tensor<1x32x32xf64>) -> tensor<1x42x48xf64> {
    %c_257 = stablehlo.constant dense<0> : tensor<i32>
    %c_258 = stablehlo.constant dense<8> : tensor<i32>
    %c_259 = stablehlo.constant dense<0> : tensor<i32>

    %35 = stablehlo.dynamic_update_slice %arg15, %32, %c_257, %c_258, %c_259 : (tensor<1x48x48xf64>, tensor<1x32x32xf64>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x48x48xf64>
    %slice = stablehlo.slice %35 [0:1, 6:48, 0:48] : (tensor<1x48x48xf64>) -> tensor<1x42x48xf64>
    return %slice : tensor<1x42x48xf64> 
  }
}

// CHECK:  func.func @main(%arg0: tensor<1x48x48xf64>, %arg1: tensor<1x32x32xf64>) -> tensor<1x42x48xf64> {
// CHECK-NEXT:    %c = stablehlo.constant dense<2> : tensor<i32>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<0> : tensor<i32>
// CHECK-NEXT:    %0 = stablehlo.slice %arg0 [0:1, 6:48, 0:48] : (tensor<1x48x48xf64>) -> tensor<1x42x48xf64>
// CHECK-NEXT:    %1 = stablehlo.slice %arg1 [0:1, 0:32, 0:32] : (tensor<1x32x32xf64>) -> tensor<1x32x32xf64>
// CHECK-NEXT:    %2 = stablehlo.dynamic_update_slice %0, %1, %c_0, %c, %c_0 : (tensor<1x42x48xf64>, tensor<1x32x32xf64>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x42x48xf64>
// CHECK-NEXT:    %3 = stablehlo.slice %2 [0:1, 0:42, 0:48] : (tensor<1x42x48xf64>) -> tensor<1x42x48xf64>
// CHECK-NEXT:    return %3 : tensor<1x42x48xf64>
// CHECK-NEXT:  }
