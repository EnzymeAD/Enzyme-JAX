// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(enzyme-hlo-generate-td{patterns=log_simplify;log_const_prop},transform-interpreter,enzyme-hlo-remove-transform)" | FileCheck %s

func.func @main1(%arg0: tensor<f64>) -> tensor<f64> {
    %0 = stablehlo.exponential %arg0 : tensor<f64>
    %1 = stablehlo.log %0 : tensor<f64>
    return %1 : tensor<f64>
}

// CHECK: func.func @main1(%arg0: tensor<f64>) -> tensor<f64> {
// CHECK-NEXT:     return %arg0 : tensor<f64>
// CHECK-NEXT: }

func.func @main2(%arg0: tensor<f64>) -> tensor<f64> {
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<f64>
    %1 = stablehlo.multiply %0, %0 : tensor<f64>
    %2 = stablehlo.log %1 : tensor<f64>
    return %2 : tensor<f64>
}

// CHECK: func.func @main2(%arg0: tensor<f64>) -> tensor<f64> {
// CHECK-NEXT:    %cst = stablehlo.constant dense<2.000000e+00> : tensor<f64>
// CHECK-NEXT:    %0 = stablehlo.log %arg0 : tensor<f64>
// CHECK-NEXT:    %1 = stablehlo.multiply %cst, %0 : tensor<f64>
// CHECK-NEXT:    %2 = stablehlo.multiply %cst, %1 : tensor<f64>
// CHECK-NEXT:    return %2 : tensor<f64>
// CHECK-NEXT:  }

func.func @main3(%arg0: tensor<f64>) -> tensor<f64> {
    %cst = stablehlo.constant dense<4.000000e+00> : tensor<f64>
    %0 = stablehlo.multiply %arg0, %cst : tensor<f64>
    %1 = stablehlo.log %0 : tensor<f64>
    return %1 : tensor<f64>
}

// CHECK: func.func @main3(%arg0: tensor<f64>) -> tensor<f64> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<1.3862943611198906> : tensor<f64>
// CHECK-NEXT:     %0 = stablehlo.log %arg0 : tensor<f64>
// CHECK-NEXT:     %1 = stablehlo.add %0, %cst : tensor<f64>
// CHECK-NEXT:     return %1 : tensor<f64>
// CHECK-NEXT: }

func.func @main4(%arg0: tensor<f64>) -> tensor<f64> {
    %cst = stablehlo.constant dense<4.000000e+00> : tensor<f64>
    %0 = stablehlo.multiply %cst, %arg0 : tensor<f64>
    %1 = stablehlo.log %0 : tensor<f64>
    return %1 : tensor<f64>
}

// CHECK: func.func @main4(%arg0: tensor<f64>) -> tensor<f64> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<1.3862943611198906> : tensor<f64>
// CHECK-NEXT:     %0 = stablehlo.log %arg0 : tensor<f64>
// CHECK-NEXT:     %1 = stablehlo.add %cst, %0 : tensor<f64>
// CHECK-NEXT:     return %1 : tensor<f64>
// CHECK-NEXT: }

func.func @main5(%arg0: tensor<f64>) -> tensor<f64> {
    %0 = stablehlo.add %arg0, %arg0 : tensor<f64>
    %1 = stablehlo.log %0 : tensor<f64>
    return %1 : tensor<f64>
}

// CHECK: func.func @main5(%arg0: tensor<f64>) -> tensor<f64> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.69314718055994529> : tensor<f64>
// CHECK-NEXT:     %0 = stablehlo.log %arg0 : tensor<f64>
// CHECK-NEXT:     %1 = stablehlo.add %cst, %0 : tensor<f64>
// CHECK-NEXT:     return %1 : tensor<f64>
// CHECK-NEXT: }

func.func @main6(%arg0: tensor<f64>) -> tensor<f64> {
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<f64>
    %1 = stablehlo.log %0 : tensor<f64>
    return %1 : tensor<f64>
}

// CHECK: func.func @main6(%arg0: tensor<f64>) -> tensor<f64> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<2.000000e+00> : tensor<f64>
// CHECK-NEXT:     %0 = stablehlo.log %arg0 : tensor<f64>
// CHECK-NEXT:     %1 = stablehlo.multiply %cst, %0 : tensor<f64>
// CHECK-NEXT:     return %1 : tensor<f64>
// CHECK-NEXT: }

func.func @main7(%arg0: tensor<f64>) -> tensor<f64> {
    %cst = stablehlo.constant dense<4.000000e+00> : tensor<f64>
    %0 = stablehlo.divide %arg0, %cst : tensor<f64>
    %1 = stablehlo.log %0 : tensor<f64>
    return %1 : tensor<f64>
}

// CHECK: func.func @main7(%arg0: tensor<f64>) -> tensor<f64> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<1.3862943611198906> : tensor<f64>
// CHECK-NEXT:     %0 = stablehlo.log %arg0 : tensor<f64>
// CHECK-NEXT:     %1 = stablehlo.subtract %0, %cst : tensor<f64>
// CHECK-NEXT:     return %1 : tensor<f64>
// CHECK-NEXT: }

func.func @main8(%arg0: tensor<f64>) -> tensor<f64> {
    %cst = stablehlo.constant dense<4.000000e+00> : tensor<f64>
    %0 = stablehlo.divide %cst, %arg0 : tensor<f64>
    %1 = stablehlo.log %0 : tensor<f64>
    return %1 : tensor<f64>
}

// CHECK: func.func @main8(%arg0: tensor<f64>) -> tensor<f64> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<1.3862943611198906> : tensor<f64>
// CHECK-NEXT:     %0 = stablehlo.log %arg0 : tensor<f64>
// CHECK-NEXT:     %1 = stablehlo.subtract %cst, %0 : tensor<f64>
// CHECK-NEXT:     return %1 : tensor<f64>
// CHECK-NEXT: }

func.func @main9(%arg0: tensor<f64>) -> tensor<f64> {
    %0 = stablehlo.sqrt %arg0 : tensor<f64>
    %1 = stablehlo.log %0 : tensor<f64>
    return %1 : tensor<f64>
}

// CHECK: func.func @main9(%arg0: tensor<f64>) -> tensor<f64> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<2.000000e+00> : tensor<f64>
// CHECK-NEXT:     %0 = stablehlo.log %arg0 : tensor<f64>
// CHECK-NEXT:     %1 = stablehlo.divide %0, %cst : tensor<f64>
// CHECK-NEXT:     return %1 : tensor<f64>
// CHECK-NEXT: }

func.func @main10(%arg0: tensor<f64>) -> tensor<f64> {
    %0 = stablehlo.rsqrt %arg0 : tensor<f64>
    %1 = stablehlo.log %0 : tensor<f64>
    return %1 : tensor<f64>
}

// CHECK: func.func @main10(%arg0: tensor<f64>) -> tensor<f64> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<-2.000000e+00> : tensor<f64>
// CHECK-NEXT:     %0 = stablehlo.log %arg0 : tensor<f64>
// CHECK-NEXT:     %1 = stablehlo.divide %0, %cst : tensor<f64>
// CHECK-NEXT:     return %1 : tensor<f64>
// CHECK-NEXT: }

func.func @main11(%arg0: tensor<f64>) -> tensor<f64> {
    %0 = stablehlo.cbrt %arg0 : tensor<f64>
    %1 = stablehlo.log %0 : tensor<f64>
    return %1 : tensor<f64>
}

// CHECK: func.func @main11(%arg0: tensor<f64>) -> tensor<f64> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<3.000000e+00> : tensor<f64>
// CHECK-NEXT:     %0 = stablehlo.log %arg0 : tensor<f64>
// CHECK-NEXT:     %1 = stablehlo.divide %0, %cst : tensor<f64>
// CHECK-NEXT:     return %1 : tensor<f64>
// CHECK-NEXT: }
