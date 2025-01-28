// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

func.func @case1(%arg0: tensor<f64>) -> tensor<f64> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %0 = stablehlo.compare  LT, %arg0, %cst : (tensor<f64>, tensor<f64>) -> tensor<i1>
    %1 = stablehlo.select %0, %cst, %arg0 : tensor<i1>, tensor<f64>
    return %1 : tensor<f64>
}

// CHECK:  func.func @case1(%arg0: tensor<f64>) -> tensor<f64> {
// CHECK-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:    %0 = stablehlo.maximum %cst, %arg0 : tensor<f64>
// CHECK-NEXT:    return %0 : tensor<f64>
// CHECK-NEXT:  }

func.func @case2(%arg0: tensor<f64>) -> tensor<f64> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %0 = stablehlo.compare LE, %arg0, %cst : (tensor<f64>, tensor<f64>) -> tensor<i1>
    %1 = stablehlo.select %0, %cst, %arg0 : tensor<i1>, tensor<f64>
    return %1 : tensor<f64>
}

// CHECK:  func.func @case2(%arg0: tensor<f64>) -> tensor<f64> {
// CHECK-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:    %0 = stablehlo.maximum %cst, %arg0 : tensor<f64>
// CHECK-NEXT:    return %0 : tensor<f64>
// CHECK-NEXT:  }

func.func @case3(%arg0: tensor<f64>) -> tensor<f64> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %0 = stablehlo.compare LT, %arg0, %cst : (tensor<f64>, tensor<f64>) -> tensor<i1>
    %1 = stablehlo.select %0, %arg0, %cst : tensor<i1>, tensor<f64>
    return %1 : tensor<f64>
}

// CHECK:  func.func @case3(%arg0: tensor<f64>) -> tensor<f64> {
// CHECK-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:    %0 = stablehlo.minimum %arg0, %cst : tensor<f64>
// CHECK-NEXT:    return %0 : tensor<f64>
// CHECK-NEXT:  }

func.func @case4(%arg0: tensor<f64>) -> tensor<f64> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %0 = stablehlo.compare LE, %arg0, %cst : (tensor<f64>, tensor<f64>) -> tensor<i1>
    %1 = stablehlo.select %0, %arg0, %cst : tensor<i1>, tensor<f64>
    return %1 : tensor<f64>
}

// CHECK:  func.func @case4(%arg0: tensor<f64>) -> tensor<f64> {
// CHECK-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:    %0 = stablehlo.minimum %arg0, %cst : tensor<f64>
// CHECK-NEXT:    return %0 : tensor<f64>
// CHECK-NEXT:  }

func.func @case5(%arg0: tensor<f64>) -> tensor<f64> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %0 = stablehlo.compare GT, %arg0, %cst : (tensor<f64>, tensor<f64>) -> tensor<i1>
    %1 = stablehlo.select %0, %cst, %arg0 : tensor<i1>, tensor<f64>
    return %1 : tensor<f64>
}

// CHECK:  func.func @case5(%arg0: tensor<f64>) -> tensor<f64> {
// CHECK-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:    %0 = stablehlo.minimum %cst, %arg0 : tensor<f64>
// CHECK-NEXT:    return %0 : tensor<f64>
// CHECK-NEXT:  }

func.func @case6(%arg0: tensor<f64>) -> tensor<f64> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %0 = stablehlo.compare GE, %arg0, %cst : (tensor<f64>, tensor<f64>) -> tensor<i1>
    %1 = stablehlo.select %0, %cst, %arg0 : tensor<i1>, tensor<f64>
    return %1 : tensor<f64>
}

// CHECK:  func.func @case6(%arg0: tensor<f64>) -> tensor<f64> {
// CHECK-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:    %0 = stablehlo.minimum %cst, %arg0 : tensor<f64>
// CHECK-NEXT:    return %0 : tensor<f64>
// CHECK-NEXT:  }

func.func @case7(%arg0: tensor<f64>) -> tensor<f64> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %0 = stablehlo.compare GT, %arg0, %cst : (tensor<f64>, tensor<f64>) -> tensor<i1>
    %1 = stablehlo.select %0, %arg0, %cst : tensor<i1>, tensor<f64>
    return %1 : tensor<f64>
}

// CHECK:  func.func @case7(%arg0: tensor<f64>) -> tensor<f64> {
// CHECK-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:    %0 = stablehlo.maximum %arg0, %cst : tensor<f64>
// CHECK-NEXT:    return %0 : tensor<f64>
// CHECK-NEXT:  }

func.func @case8(%arg0: tensor<f64>) -> tensor<f64> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %0 = stablehlo.compare GE, %arg0, %cst : (tensor<f64>, tensor<f64>) -> tensor<i1>
    %1 = stablehlo.select %0, %arg0, %cst : tensor<i1>, tensor<f64>
    return %1 : tensor<f64>
}

// CHECK:  func.func @case8(%arg0: tensor<f64>) -> tensor<f64> {
// CHECK-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:    %0 = stablehlo.maximum %arg0, %cst : tensor<f64>
// CHECK-NEXT:    return %0 : tensor<f64>
// CHECK-NEXT:  }
