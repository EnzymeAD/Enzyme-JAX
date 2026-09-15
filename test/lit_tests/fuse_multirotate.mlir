// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=reduce_unused_multirotate" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

func.func @main(%arg0: tensor<4x1520x1520xf64>) -> (tensor<4x1520x1520xf64>, tensor<4x1520x1520xf64>) {
    %a:3 = "enzymexla.multi_rotate"(%arg0) <{dimension = 2 : i32, left_amount = 2 : i32, right_amount = 0 : i32}> : (tensor<4x1520x1520xf64>) -> (tensor<4x1520x1520xf64>, tensor<4x1520x1520xf64>, tensor<4x1520x1520xf64>)
    %b:4 = "enzymexla.multi_rotate"(%arg0) <{dimension = 2 : i32, left_amount = 3 : i32, right_amount = 0 : i32}> : (tensor<4x1520x1520xf64>) -> (tensor<4x1520x1520xf64>, tensor<4x1520x1520xf64>, tensor<4x1520x1520xf64>, tensor<4x1520x1520xf64>)
    return %a#0, %a#1 : tensor<4x1520x1520xf64>, tensor<4x1520x1520xf64>
}

// CHECK:  func.func @main(%arg0: tensor<4x1520x1520xf64>) -> (tensor<4x1520x1520xf64>, tensor<4x1520x1520xf64>) {
// CHECK-NEXT:    %0:2 = "enzymexla.multi_rotate"(%arg0) <{dimension = 2 : i32, left_amount = 2 : i32, right_amount = -1 : i32}> : (tensor<4x1520x1520xf64>) -> (tensor<4x1520x1520xf64>, tensor<4x1520x1520xf64>)
// CHECK-NEXT:    return %0#0, %0#1 : tensor<4x1520x1520xf64>, tensor<4x1520x1520xf64>
// CHECK-NEXT:  }
