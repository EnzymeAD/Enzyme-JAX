// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

// (x * 2) / 4 == x * (2 / 4)
func.func @muldivconst(%arg0: tensor<f64>) -> tensor<f64> {
  // CHECK-LABEL: func @muldivconst
  // CHECK:      %[[CST:.*]] = stablehlo.constant dense<5.000000e-01> : tensor<f64>
  // CHECK-NEXT: %[[RES:.*]] = stablehlo.multiply %arg0, %[[CST]] : tensor<f64>
  // CHECK-NEXT: return %[[RES]] : tensor<f64>
  %cst = stablehlo.constant dense<2.0> : tensor<f64>
  %cst_0 = stablehlo.constant dense<4.0> : tensor<f64>
  %0 = stablehlo.multiply %arg0, %cst : tensor<f64>
  %1 = stablehlo.divide %0, %cst_0 : tensor<f64>
  return %1 : tensor<f64>
}

// (x * 2) * 4 == x * (4 * 2)
func.func @mulmulconst(%arg0: tensor<f64>) -> tensor<f64> {
  // CHECK-LABEL: func @mulmulconst
  // CHECK:      %[[CST:.*]] = stablehlo.constant dense<8.000000e+00> : tensor<f64>
  // CHECK-NEXT: %[[RES:.*]] = stablehlo.multiply %arg0, %[[CST]] : tensor<f64>
  // CHECK-NEXT: return %[[RES]] : tensor<f64>
  %cst = stablehlo.constant dense<2.0> : tensor<f64>
  %cst_0 = stablehlo.constant dense<4.0> : tensor<f64>
  %0 = stablehlo.multiply %arg0, %cst : tensor<f64>
  %1 = stablehlo.multiply %0, %cst_0 : tensor<f64>
  return %1 : tensor<f64>
}

// (x / 2) * 4 == x * (4 / 2)
func.func @divmulconst(%arg0: tensor<f64>) -> tensor<f64> {
  // CHECK-LABEL: func @divmulconst
  // CHECK:      %[[CST:.*]] = stablehlo.constant dense<2.000000e+00> : tensor<f64>
  // CHECK-NEXT: %[[RES:.*]] = stablehlo.multiply %arg0, %[[CST]] : tensor<f64>
  // CHECK-NEXT: return %[[RES]] : tensor<f64>
  %cst = stablehlo.constant dense<2.0> : tensor<f64>
  %cst_0 = stablehlo.constant dense<4.0> : tensor<f64>
  %0 = stablehlo.divide %arg0, %cst : tensor<f64>
  %1 = stablehlo.multiply %0, %cst_0 : tensor<f64>
  return %1 : tensor<f64>
}

// (x / 2) / 4 == x / (2 * 4)
func.func @divdivconst(%arg0: tensor<f64>) -> tensor<f64> {
  // CHECK-LABEL: func @divdivconst
  // CHECK:      %[[CST:.*]] = stablehlo.constant dense<1.250000e-01> : tensor<f64>
  // CHECK-NEXT: %[[RES:.*]] = stablehlo.multiply %arg0, %[[CST]] : tensor<f64>
  // CHECK-NEXT: return %[[RES]] : tensor<f64>
  %cst = stablehlo.constant dense<2.0> : tensor<f64>
  %cst_0 = stablehlo.constant dense<4.0> : tensor<f64>
  %0 = stablehlo.divide %arg0, %cst : tensor<f64>
  %1 = stablehlo.divide %0, %cst_0 : tensor<f64>
  return %1 : tensor<f64>
}

// (x + 2) + 4 == x + (2 + 4)
func.func @addaddconst(%arg0: tensor<f64>) -> tensor<f64> {
  // CHECK-LABEL: func @addaddconst
  // CHECK:      %[[CST:.*]] = stablehlo.constant dense<6.000000e+00> : tensor<f64>
  // CHECK-NEXT: %[[RES:.*]] = stablehlo.add %arg0, %[[CST]] : tensor<f64>
  // CHECK-NEXT: return %[[RES]] : tensor<f64>
  %cst = stablehlo.constant dense<2.0> : tensor<f64>
  %cst_0 = stablehlo.constant dense<4.0> : tensor<f64>
  %0 = stablehlo.add %arg0, %cst : tensor<f64>
  %1 = stablehlo.add %0, %cst_0 : tensor<f64>
  return %1 : tensor<f64>
}

// (x - 2) + 4 == x + (4 - 2)
func.func @addsubconst(%arg0: tensor<f64>) -> tensor<f64> {
  // CHECK-LABEL: func @addsubconst
  // CHECK:      %[[CST:.*]] = stablehlo.constant dense<2.000000e+00> : tensor<f64>
  // CHECK-NEXT: %[[RES:.*]] = stablehlo.add %arg0, %[[CST]] : tensor<f64>
  // CHECK-NEXT: return %[[RES]] : tensor<f64>
  %cst = stablehlo.constant dense<2.0> : tensor<f64>
  %cst_0 = stablehlo.constant dense<4.0> : tensor<f64>
  %0 = stablehlo.subtract %arg0, %cst : tensor<f64>
  %1 = stablehlo.add %0, %cst_0 : tensor<f64>
  return %1 : tensor<f64>
}

// (x - 2) - 4 == x - (2 + 4)
func.func @subsubconst(%arg0: tensor<f64>) -> tensor<f64> {
  // CHECK-LABEL: func @subsubconst
  // CHECK:      %[[CST:.*]] = stablehlo.constant dense<6.000000e+00> : tensor<f64>
  // CHECK-NEXT: %[[RES:.*]] = stablehlo.subtract %arg0, %[[CST]] : tensor<f64>
  // CHECK-NEXT: return %[[RES]] : tensor<f64>
  %cst = stablehlo.constant dense<2.0> : tensor<f64>
  %cst_0 = stablehlo.constant dense<4.0> : tensor<f64>
  %0 = stablehlo.subtract %arg0, %cst : tensor<f64>
  %1 = stablehlo.subtract %0, %cst_0 : tensor<f64>
  return %1 : tensor<f64>
}

// (x + 2) - 4 == x + (2 - 4)
func.func @subaddconst(%arg0: tensor<f64>) -> tensor<f64> {
  // CHECK-LABEL: func @subaddconst
  // CHECK:      %[[CST:.*]] = stablehlo.constant dense<-2.000000e+00> : tensor<f64>
  // CHECK-NEXT: %[[RES:.*]] = stablehlo.add %arg0, %[[CST]] : tensor<f64>
  // CHECK-NEXT: return %[[RES]] : tensor<f64>
  %cst = stablehlo.constant dense<2.0> : tensor<f64>
  %cst_0 = stablehlo.constant dense<4.0> : tensor<f64>
  %0 = stablehlo.add %arg0, %cst : tensor<f64>
  %1 = stablehlo.subtract %0, %cst_0 : tensor<f64>
  return %1 : tensor<f64>
}


func.func @add_chain(%arg0: tensor<3xi64>) -> tensor<3xi64> {
  // CHECK-LABEL: func @add_chain
  // CHECK:       %[[CST:.*]] = stablehlo.constant dense<3> : tensor<3xi64>
  // CHECK-NEXT:  %[[RES:.*]] = stablehlo.add %arg0, %[[CST]] : tensor<3xi64>
  // CHECK-NEXT:  return %[[RES]] : tensor<3xi64>
  %c = stablehlo.constant dense<1> : tensor<3xi64>
  %0 = stablehlo.add %arg0, %c : tensor<3xi64>
  %1 = stablehlo.add %0, %c : tensor<3xi64>
  %2 = stablehlo.add %1, %c : tensor<3xi64>
  return %2 : tensor<3xi64>
}

func.func @mul_chain(%arg0: tensor<3xi64>) -> tensor<3xi64> {
  // CHECK-LABEL: func @mul_chain
  // CHECK:       %[[CST:.*]] = stablehlo.constant dense<8> : tensor<3xi64>
  // CHECK-NEXT:  %[[RES:.*]] = stablehlo.multiply %arg0, %[[CST]] : tensor<3xi64>
  // CHECK-NEXT:  return %[[RES]] : tensor<3xi64>
  %c = stablehlo.constant dense<2> : tensor<3xi64>
  %0 = stablehlo.multiply %arg0, %c : tensor<3xi64>
  %1 = stablehlo.multiply %0, %c : tensor<3xi64>
  %2 = stablehlo.multiply %1, %c : tensor<3xi64>
  return %2 : tensor<3xi64>
}