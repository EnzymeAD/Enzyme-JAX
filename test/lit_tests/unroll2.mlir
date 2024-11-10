// RUN: enzymexlamlir-opt %s --enzyme-hlo-unroll | FileCheck %s

module {
  func.func @main(%arg0: tensor<i64>) -> tensor<i64> {
    %c = stablehlo.constant dense<10> : tensor<i64>
    %c_0 = stablehlo.constant dense<1> : tensor<i64>
    %0:2 = stablehlo.while(%iterArg = %c_0, %iterArg_1 = %arg0) : tensor<i64>, tensor<i64>
     cond {
      %1 = stablehlo.compare  LE, %iterArg, %c : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %1 : tensor<i1>
    } do {
      %1 = stablehlo.add %iterArg_1, %iterArg : tensor<i64>
      %2 = stablehlo.add %iterArg, %c_0 : tensor<i64>
      stablehlo.return %2, %1 : tensor<i64>, tensor<i64>
    }
    return %0#1 : tensor<i64>
  }
}

// CHECK:  func.func @main(%arg0: tensor<i64>) -> tensor<i64> {
// CHECK-NEXT:    %c = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:    %0 = stablehlo.add %arg0, %c : tensor<i64>
// CHECK-NEXT:    %1 = stablehlo.add %c, %c : tensor<i64>
// CHECK-NEXT:    %2 = stablehlo.add %0, %1 : tensor<i64>
// CHECK-NEXT:    %3 = stablehlo.add %1, %c : tensor<i64>
// CHECK-NEXT:    %4 = stablehlo.add %2, %3 : tensor<i64>
// CHECK-NEXT:    %5 = stablehlo.add %3, %c : tensor<i64>
// CHECK-NEXT:    %6 = stablehlo.add %4, %5 : tensor<i64>
// CHECK-NEXT:    %7 = stablehlo.add %5, %c : tensor<i64>
// CHECK-NEXT:    %8 = stablehlo.add %6, %7 : tensor<i64>
// CHECK-NEXT:    %9 = stablehlo.add %7, %c : tensor<i64>
// CHECK-NEXT:    %10 = stablehlo.add %8, %9 : tensor<i64>
// CHECK-NEXT:    %11 = stablehlo.add %9, %c : tensor<i64>
// CHECK-NEXT:    %12 = stablehlo.add %10, %11 : tensor<i64>
// CHECK-NEXT:    %13 = stablehlo.add %11, %c : tensor<i64>
// CHECK-NEXT:    %14 = stablehlo.add %12, %13 : tensor<i64>
// CHECK-NEXT:    %15 = stablehlo.add %13, %c : tensor<i64>
// CHECK-NEXT:    %16 = stablehlo.add %14, %15 : tensor<i64>
// CHECK-NEXT:    %17 = stablehlo.add %15, %c : tensor<i64>
// CHECK-NEXT:    %18 = stablehlo.add %16, %17 : tensor<i64>
// CHECK-NEXT:    return %18 : tensor<i64>
// CHECK-NEXT:  }
