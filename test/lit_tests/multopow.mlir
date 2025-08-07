// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=chained_multiply_to_power;power_multiply_to_power;add_const_prop;common_associative_commutative_op_reorder" --transform-interpreter --enzyme-hlo-remove-transform --canonicalize --cse --enzyme-hlo-generate-td="patterns=chained_multiply_to_power;power_multiply_to_power;add_const_prop;common_associative_commutative_op_reorder" --enzyme-hlo-remove-transform --canonicalize | FileCheck %s

func.func @main1(%arg0: tensor<f64>) -> tensor<f64> {
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<f64>
    %1 = stablehlo.multiply %0, %arg0 : tensor<f64>
    %2 = stablehlo.multiply %1, %arg0 : tensor<f64>
    %3 = stablehlo.multiply %2, %arg0 : tensor<f64>
    return %3 : tensor<f64>
}

// CHECK: func.func @main1(%arg0: tensor<f64>) -> tensor<f64> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<5.000000e+00> : tensor<f64>
// CHECK-NEXT:     %0 = stablehlo.power %arg0, %cst : tensor<f64>
// CHECK-NEXT:     return %0 : tensor<f64>
// CHECK-NEXT: }

func.func @main2(%arg0: tensor<f64>) -> tensor<f64> {
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<f64>
    %1 = stablehlo.multiply %0, %arg0 : tensor<f64>
    %2 = stablehlo.multiply %1, %arg0 : tensor<f64>
    %3 = stablehlo.multiply %2, %arg0 : tensor<f64>
    %4 = stablehlo.multiply %arg0, %0 : tensor<f64>
    %5 = stablehlo.multiply %4, %3 : tensor<f64>
    return %5 : tensor<f64>
}

// CHECK: func.func @main2(%arg0: tensor<f64>) -> tensor<f64> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<8.000000e+00> : tensor<f64>
// CHECK-NEXT:     %0 = stablehlo.power %arg0, %cst : tensor<f64>
// CHECK-NEXT:     return %0 : tensor<f64>
// CHECK-NEXT: }

func.func @main3(%arg0: tensor<f64>) -> tensor<f64> {
    %cst = stablehlo.constant dense<3.000000e+00> : tensor<f64>
    %cst_2 = stablehlo.constant dense<2.000000e+00> : tensor<f64>
    %0 = stablehlo.power %arg0, %cst : tensor<f64>
    %1 = stablehlo.power %arg0, %cst_2 : tensor<f64>
    %2 = stablehlo.multiply %1, %0 : tensor<f64>
    return %2 : tensor<f64>
}

// CHECK: func.func @main3(%arg0: tensor<f64>) -> tensor<f64> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<5.000000e+00> : tensor<f64>
// CHECK-NEXT:     %0 = stablehlo.power %arg0, %cst : tensor<f64>
// CHECK-NEXT:     return %0 : tensor<f64>
// CHECK-NEXT: }
