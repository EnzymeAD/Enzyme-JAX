// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=not_compare" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

func.func @not_lt(%a: tensor<4xf32>, %b: tensor<4xf32>) -> tensor<4xi1> {
  %0 = stablehlo.compare LT, %a, %b, FLOAT : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
  %1 = stablehlo.not %0 : tensor<4xi1>
  return %1 : tensor<4xi1>
}

// CHECK:  func.func @not_lt(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xi1> {
// CHECK-NEXT:    %0 = stablehlo.compare LT, %arg0, %arg1, FLOAT : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
// CHECK-NEXT:    %1 = stablehlo.not %0 : tensor<4xi1>
// CHECK-NEXT:    return %1 : tensor<4xi1>
// CHECK-NEXT:  }

func.func @not_le(%a: tensor<4xf32>, %b: tensor<4xf32>) -> tensor<4xi1> {
  %0 = stablehlo.compare LE, %a, %b, FLOAT : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
  %1 = stablehlo.not %0 : tensor<4xi1>
  return %1 : tensor<4xi1>
}

// CHECK:  func.func @not_le(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xi1> {
// CHECK-NEXT:    %0 = stablehlo.compare LE, %arg0, %arg1, FLOAT : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
// CHECK-NEXT:    %1 = stablehlo.not %0 : tensor<4xi1>
// CHECK-NEXT:    return %1 : tensor<4xi1>
// CHECK-NEXT:  }

func.func @not_gt(%a: tensor<4xf32>, %b: tensor<4xf32>) -> tensor<4xi1> {
  %0 = stablehlo.compare GT, %a, %b, FLOAT : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
  %1 = stablehlo.not %0 : tensor<4xi1>
  return %1 : tensor<4xi1>
}

// CHECK:  func.func @not_gt(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xi1> {
// CHECK-NEXT:    %0 = stablehlo.compare GT, %arg0, %arg1, FLOAT : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
// CHECK-NEXT:    %1 = stablehlo.not %0 : tensor<4xi1>
// CHECK-NEXT:    return %1 : tensor<4xi1>
// CHECK-NEXT:  }

func.func @not_ge(%a: tensor<4xf32>, %b: tensor<4xf32>) -> tensor<4xi1> {
  %0 = stablehlo.compare GE, %a, %b, FLOAT : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
  %1 = stablehlo.not %0 : tensor<4xi1>
  return %1 : tensor<4xi1>
}

// CHECK:  func.func @not_ge(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xi1> {
// CHECK-NEXT:    %0 = stablehlo.compare GE, %arg0, %arg1, FLOAT : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
// CHECK-NEXT:    %1 = stablehlo.not %0 : tensor<4xi1>
// CHECK-NEXT:    return %1 : tensor<4xi1>
// CHECK-NEXT:  }

func.func @not_lt_notype(%a: tensor<4xf32>, %b: tensor<4xf32>) -> tensor<4xi1> {
  %0 = stablehlo.compare LT, %a, %b : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
  %1 = stablehlo.not %0 : tensor<4xi1>
  return %1 : tensor<4xi1>
}

// CHECK:  func.func @not_lt_notype(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xi1> {
// CHECK-NEXT:    %0 = stablehlo.compare LT, %arg0, %arg1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
// CHECK-NEXT:    %1 = stablehlo.not %0 : tensor<4xi1>
// CHECK-NEXT:    return %1 : tensor<4xi1>
// CHECK-NEXT:  }

func.func @not_eq(%a: tensor<4xf32>, %b: tensor<4xf32>) -> tensor<4xi1> {
  %0 = stablehlo.compare EQ, %a, %b, FLOAT : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
  %1 = stablehlo.not %0 : tensor<4xi1>
  return %1 : tensor<4xi1>
}

// CHECK:  func.func @not_eq(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xi1> {
// CHECK-NEXT:    %0 = stablehlo.compare NE, %arg0, %arg1, FLOAT : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
// CHECK-NEXT:    return %0 : tensor<4xi1>
// CHECK-NEXT:  }

func.func @not_ne(%a: tensor<4xf32>, %b: tensor<4xf32>) -> tensor<4xi1> {
  %0 = stablehlo.compare NE, %a, %b, FLOAT : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
  %1 = stablehlo.not %0 : tensor<4xi1>
  return %1 : tensor<4xi1>
}

// CHECK:  func.func @not_ne(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xi1> {
// CHECK-NEXT:    %0 = stablehlo.compare EQ, %arg0, %arg1, FLOAT : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
// CHECK-NEXT:    return %0 : tensor<4xi1>
// CHECK-NEXT:  }

func.func @not_lt_totalorder(%a: tensor<4xf32>, %b: tensor<4xf32>) -> tensor<4xi1> {
  %0 = stablehlo.compare LT, %a, %b, TOTALORDER : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
  %1 = stablehlo.not %0 : tensor<4xi1>
  return %1 : tensor<4xi1>
}

// CHECK:  func.func @not_lt_totalorder(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xi1> {
// CHECK-NEXT:    %0 = stablehlo.compare GE, %arg0, %arg1, TOTALORDER : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
// CHECK-NEXT:    return %0 : tensor<4xi1>
// CHECK-NEXT:  }
