// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=recognize_multi_pad" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

func.func @main(%arg0: tensor<1519x3056xf64>) -> (tensor<1520x3056xf64>, tensor<1520x3056xf64>) {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %0 = stablehlo.pad %arg0, %cst, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<1519x3056xf64>, tensor<f64>) -> tensor<1520x3056xf64>
    %1 = stablehlo.pad %arg0, %cst, low = [1, 0], high = [0, 0], interior = [0, 0] : (tensor<1519x3056xf64>, tensor<f64>) -> tensor<1520x3056xf64>
    return %1, %0 : tensor<1520x3056xf64>, tensor<1520x3056xf64>
}

// CHECK-LABEL: func.func @main
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:     %[[MP:.*]]:2 = "enzymexla.multi_pad"(%arg0, %cst) <{amount = 1 : i64, dimension = 0 : i32}> : (tensor<1519x3056xf64>, tensor<f64>) -> (tensor<1520x3056xf64>, tensor<1520x3056xf64>)
// CHECK-NEXT:     return %[[MP]]#0, %[[MP]]#1 : tensor<1520x3056xf64>, tensor<1520x3056xf64>
