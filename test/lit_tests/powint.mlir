// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

module {
  func.func @main() -> (tensor<i64>) {
    %c = stablehlo.constant dense<2> : tensor<i64>
    %c_0 = stablehlo.constant dense<4> : tensor<i64>
    %1 = stablehlo.power %c, %c_0 : tensor<i64>
    return %1 : tensor<i64>
  }
}

// CHECK:  func.func @main() -> tensor<i64> {
// CHECK-NEXT:    %c = stablehlo.constant dense<16> : tensor<i64>
// CHECK-NEXT:    return %c : tensor<i64>
// CHECK-NEXT:  }

module {
  func.func @main() -> (tensor<i64>) {
    %c = stablehlo.constant dense<6> : tensor<i64>
    %c_0 = stablehlo.constant dense<0> : tensor<i64>
    %1 = stablehlo.power %c, %c_0 : tensor<i64>
    return %1 : tensor<i64>
  }
}

// CHECK:  func.func @main() -> tensor<i64> {
// CHECK-NEXT:    %c = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:    return %c : tensor<i64>
// CHECK-NEXT:  }

module {
  func.func @main() -> (tensor<i64>) {
    %c = stablehlo.constant dense<6> : tensor<i64>
    %c_0 = stablehlo.constant dense<-1> : tensor<i64>
    %1 = stablehlo.power %c, %c_0 : tensor<i64>
    return %1 : tensor<i64>
  }
}

// CHECK:  func.func @main() -> tensor<i64> {
// CHECK-NEXT:    %c = stablehlo.constant dense<6> : tensor<i64>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<-1> : tensor<i64>
// CHECK-NEXT:    %0 = stablehlo.power %c, %c_0 : tensor<i64>
// CHECK-NEXT:    return %0 : tensor<i64>
// CHECK-NEXT:  }
