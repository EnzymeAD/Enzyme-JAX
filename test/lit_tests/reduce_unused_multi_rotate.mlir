// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=reduce_unused_multirotate" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

// CHECK:      func.func @multirotate(%arg0: tensor<20x24x80xf64>) -> (tensor<20x24x80xf64>, tensor<20x24x80xf64>) {
// CHECK-NEXT:   %0 = "enzymexla.rotate"(%arg0) <{amount = -1 : si32, dimension = 2 : si32}> : (tensor<20x24x80xf64>) -> tensor<20x24x80xf64>
// CHECK-NEXT:   %1:3 = "enzymexla.multi_rotate"(%arg0) <{dimension = 2 : si32, left_amount = 1 : si32, right_amount = 1 : si32}> : (tensor<20x24x80xf64>) -> (tensor<20x24x80xf64>, tensor<20x24x80xf64>, tensor<20x24x80xf64>)
// CHECK-NEXT:   return %arg0, %0 : tensor<20x24x80xf64>, tensor<20x24x80xf64>


func.func @multirotate(%arg0: tensor<20x24x80xf64>) -> (tensor<20x24x80xf64>, tensor<20x24x80xf64>, tensor<20x24x80xf64>, tensor<20x24x80xf64>) {
    %a_0, %a_1, %a_2, %a_3, %a_4 = "enzymexla.multi_rotate"(%arg0) <{dimension = 2 : si32, left_amount = 2 : si32, right_amount = 2 : si32}> : (tensor<20x24x80xf64>) -> (tensor<20x24x80xf64>, tensor<20x24x80xf64>, tensor<20x24x80xf64>, tensor<20x24x80xf64>, tensor<20x24x80xf64>)
    %b_0, %b_1, %b_2, %b_3, %b_4 = "enzymexla.multi_rotate"(%arg0) <{dimension = 2 : si32, left_amount = 2 : si32, right_amount = 2 : si32}> : (tensor<20x24x80xf64>) -> (tensor<20x24x80xf64>, tensor<20x24x80xf64>, tensor<20x24x80xf64>, tensor<20x24x80xf64>, tensor<20x24x80xf64>)
    %c_0, %c_1, %c_2, %c_3, %c_4 = "enzymexla.multi_rotate"(%arg0) <{dimension = 2 : si32, left_amount = 2 : si32, right_amount = 2 : si32}> : (tensor<20x24x80xf64>) -> (tensor<20x24x80xf64>, tensor<20x24x80xf64>, tensor<20x24x80xf64>, tensor<20x24x80xf64>, tensor<20x24x80xf64>)
    return %a_2, %b_3, %c_1, %c_3 : tensor<20x24x80xf64>, tensor<20x24x80xf64>, tensor<20x24x80xf64>, tensor<20x24x80xf64>
}
