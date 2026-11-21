// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=reduce_unused_multirotate" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

// CHECK:      func.func @multirotate(%arg0: tensor<20x24x80xf64>) -> (tensor<20x24x80xf64>, tensor<20x24x80xf64>, tensor<20x24x80xf64>, tensor<20x24x80xf64>) {
// CHECK-NEXT:   %0 = "enzymexla.rotate"(%arg0) <{amount = 79 : i32, dimension = 2 : i32}> : (tensor<20x24x80xf64>) -> tensor<20x24x80xf64>
// CHECK-NEXT:   %1:3 = "enzymexla.multi_rotate"(%arg0) <{dimension = 2 : i32, left_amount = 1 : i32, right_amount = 1 : i32}> : (tensor<20x24x80xf64>) -> (tensor<20x24x80xf64>, tensor<20x24x80xf64>, tensor<20x24x80xf64>)
// CHECK-NEXT:   return %arg0, %0, %1#0, %1#2 : tensor<20x24x80xf64>, tensor<20x24x80xf64>, tensor<20x24x80xf64>, tensor<20x24x80xf64>


func.func @multirotate(%arg0: tensor<20x24x80xf64>) -> (tensor<20x24x80xf64>, tensor<20x24x80xf64>, tensor<20x24x80xf64>, tensor<20x24x80xf64>) {
    %a_0, %a_1, %a_2, %a_3, %a_4 = "enzymexla.multi_rotate"(%arg0) <{dimension = 2 : i32, left_amount = 2 : i32, right_amount = 2 : i32}> : (tensor<20x24x80xf64>) -> (tensor<20x24x80xf64>, tensor<20x24x80xf64>, tensor<20x24x80xf64>, tensor<20x24x80xf64>, tensor<20x24x80xf64>)
    %b_0, %b_1, %b_2, %b_3, %b_4 = "enzymexla.multi_rotate"(%arg0) <{dimension = 2 : i32, left_amount = 2 : i32, right_amount = 2 : i32}> : (tensor<20x24x80xf64>) -> (tensor<20x24x80xf64>, tensor<20x24x80xf64>, tensor<20x24x80xf64>, tensor<20x24x80xf64>, tensor<20x24x80xf64>)
    %c_0, %c_1, %c_2, %c_3, %c_4 = "enzymexla.multi_rotate"(%arg0) <{dimension = 2 : i32, left_amount = 2 : i32, right_amount = 2 : i32}> : (tensor<20x24x80xf64>) -> (tensor<20x24x80xf64>, tensor<20x24x80xf64>, tensor<20x24x80xf64>, tensor<20x24x80xf64>, tensor<20x24x80xf64>)
    return %a_2, %b_3, %c_1, %c_3 : tensor<20x24x80xf64>, tensor<20x24x80xf64>, tensor<20x24x80xf64>, tensor<20x24x80xf64>
}

// CHECK-LABEL: func.func @one_sided_unused(
// CHECK-NEXT:    %[[L:.*]]:4 = "enzymexla.multi_rotate"(%arg0) <{dimension = 0 : i32, left_amount = 3 : i32, right_amount = 0 : i32}>
// CHECK-NEXT:    %[[R:.*]]:4 = "enzymexla.multi_rotate"(%arg0) <{dimension = 0 : i32, left_amount = 0 : i32, right_amount = 3 : i32}>
// CHECK-NEXT:    return %[[L]]#0, %[[L]]#1, %[[R]]#2, %[[R]]#3
func.func @one_sided_unused(%arg0: tensor<20xf64>) -> (tensor<20xf64>, tensor<20xf64>, tensor<20xf64>, tensor<20xf64>) {
    %a:5 = "enzymexla.multi_rotate"(%arg0) <{dimension = 0 : i32, left_amount = 3 : i32, right_amount = 1 : i32}> : (tensor<20xf64>) -> (tensor<20xf64>, tensor<20xf64>, tensor<20xf64>, tensor<20xf64>, tensor<20xf64>)
    %b:5 = "enzymexla.multi_rotate"(%arg0) <{dimension = 0 : i32, left_amount = 1 : i32, right_amount = 3 : i32}> : (tensor<20xf64>) -> (tensor<20xf64>, tensor<20xf64>, tensor<20xf64>, tensor<20xf64>, tensor<20xf64>)
    return %a#0, %a#1, %b#3, %b#4 : tensor<20xf64>, tensor<20xf64>, tensor<20xf64>, tensor<20xf64>
}

sdy.mesh @mesh = <["x"=2]>
// CHECK-LABEL: func.func @sharded_reduce_unused(
// CHECK-NEXT:    %[[ROT:.*]] = "enzymexla.rotate"(%arg0) <{amount = 1 : i32, dimension = 0 : i32}> {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>}
// CHECK-NEXT:    %[[MR:.*]]:3 = "enzymexla.multi_rotate"(%arg0) <{dimension = 0 : i32, left_amount = 1 : i32, right_amount = 1 : i32}> {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>, <@mesh, [{"x"}]>, <@mesh, [{"x"}]>]>}
// CHECK-NEXT:    return %arg0, %[[ROT]], %[[MR]]#0, %[[MR]]#2
func.func @sharded_reduce_unused(%arg0: tensor<20xf64> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) -> (tensor<20xf64>, tensor<20xf64>, tensor<20xf64>, tensor<20xf64>) {
    %n:3 = "enzymexla.multi_rotate"(%arg0) <{dimension = 0 : i32, left_amount = 1 : i32, right_amount = 1 : i32}> {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>, <@mesh, [{"x"}]>, <@mesh, [{"x"}]>]>} : (tensor<20xf64>) -> (tensor<20xf64>, tensor<20xf64>, tensor<20xf64>)
    %a:3 = "enzymexla.multi_rotate"(%arg0) <{dimension = 0 : i32, left_amount = 1 : i32, right_amount = 1 : i32}> {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>, <@mesh, [{"x"}]>, <@mesh, [{"x"}]>]>} : (tensor<20xf64>) -> (tensor<20xf64>, tensor<20xf64>, tensor<20xf64>)
    %b:5 = "enzymexla.multi_rotate"(%arg0) <{dimension = 0 : i32, left_amount = 2 : i32, right_amount = 2 : i32}> {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>, <@mesh, [{"x"}]>, <@mesh, [{"x"}]>, <@mesh, [{"x"}]>, <@mesh, [{"x"}]>]>} : (tensor<20xf64>) -> (tensor<20xf64>, tensor<20xf64>, tensor<20xf64>, tensor<20xf64>, tensor<20xf64>)
    return %n#1, %a#0, %b#1, %b#3 : tensor<20xf64>, tensor<20xf64>, tensor<20xf64>, tensor<20xf64>
}
