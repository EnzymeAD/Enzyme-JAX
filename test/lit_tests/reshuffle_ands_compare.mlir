// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td=patterns=reshuffle_ands_compares --transform-interpreter --enzyme-hlo-remove-transform | FileCheck %s

module {
  func.func @main(%3241: tensor<4x6128x12272xi64>) -> tensor<4x6128x12272xi1> {
    %c_304 = stablehlo.constant dense<6124> : tensor<4x6128x12272xi64>
    %c_306 = stablehlo.constant dense<2122> : tensor<4x6128x12272xi64>
    %c_308 = stablehlo.constant dense<6122> : tensor<4x6128x12272xi64>
    %3249 = stablehlo.compare  LE, %3241, %c_304 : (tensor<4x6128x12272xi64>, tensor<4x6128x12272xi64>) -> tensor<4x6128x12272xi1>
    %3253 = stablehlo.compare  GE, %3241, %c_308 : (tensor<4x6128x12272xi64>, tensor<4x6128x12272xi64>) -> tensor<4x6128x12272xi1>
    %3255 = stablehlo.and %3249, %3253 : tensor<4x6128x12272xi1>
    %3256 = stablehlo.compare  LE, %3241, %c_306 : (tensor<4x6128x12272xi64>, tensor<4x6128x12272xi64>) -> tensor<4x6128x12272xi1>
    %3261 = stablehlo.and %3255, %3256 : tensor<4x6128x12272xi1>
    return %3261 : tensor<4x6128x12272xi1>
  }
}

// CHECK:  func.func @main(%[[ARG:.+]]: tensor<4x6128x12272xi64>) -> tensor<4x6128x12272xi1> {
// CHECK-NEXT:    %[[C304:.+]] = stablehlo.constant dense<6124> : tensor<4x6128x12272xi64>
// CHECK-NEXT:    %[[C306:.+]] = stablehlo.constant dense<2122> : tensor<4x6128x12272xi64>
// CHECK-NEXT:    %[[C308:.+]] = stablehlo.constant dense<6122> : tensor<4x6128x12272xi64>
// CHECK-NEXT:    %[[COND2:.+]] = stablehlo.compare  GE, %[[ARG]], %[[C308]] : (tensor<4x6128x12272xi64>, tensor<4x6128x12272xi64>) -> tensor<4x6128x12272xi1>
// CHECK-NEXT:    %[[MIN:.+]] = stablehlo.minimum %[[C306]], %[[C304]] : tensor<4x6128x12272xi64>
// CHECK-NEXT:    %[[COND1:.+]] = stablehlo.compare LE, %[[ARG]], %[[MIN]] : (tensor<4x6128x12272xi64>, tensor<4x6128x12272xi64>) -> tensor<4x6128x12272xi1>
// CHECK-NEXT:    %[[AND:.+]] = stablehlo.and %[[COND1]], %[[COND2]] : tensor<4x6128x12272xi1>
// CHECK-NEXT:    return %[[AND]] : tensor<4x6128x12272xi1>
// CHECK-NEXT:  }
