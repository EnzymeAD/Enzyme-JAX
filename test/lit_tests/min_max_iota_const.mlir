// RUN: enzymexlamlir-opt %s -enzyme-hlo-opt="max_constant_expansion=0" | FileCheck %s

module {
  func.func @test_iota_max() -> tensor<4xi64> {
    %iota = stablehlo.iota dim = 0 : tensor<4xi64>
    %c_0 = stablehlo.constant dense<-1> : tensor<4xi64>
    %1 = stablehlo.maximum %iota, %c_0 : tensor<4xi64>
    return %1 : tensor<4xi64>
  }
  // CHECK:  func.func @test_iota_max() -> tensor<4xi64> {
  // CHECK-NEXT:    %[[IOTA:.+]] = stablehlo.iota dim = 0 : tensor<4xi64>
  // CHECK-NEXT:    return %[[IOTA]] : tensor<4xi64>

  func.func @test_iota_max_cst() -> tensor<4xi64> {
    %iota = stablehlo.iota dim = 0 : tensor<4xi64>
    %c_0 = stablehlo.constant dense<5> : tensor<4xi64>
    %1 = stablehlo.maximum %iota, %c_0 : tensor<4xi64>
    return %1 : tensor<4xi64>
  }
  // CHECK:  func.func @test_iota_max_cst() -> tensor<4xi64> {
  // CHECK-NEXT:    %[[C:.+]] = stablehlo.constant dense<5> : tensor<4xi64>
  // CHECK-NEXT:    return %[[C]] : tensor<4xi64>

  func.func @test_iota_min() -> tensor<4xi64> {
    %iota = stablehlo.iota dim = 0 : tensor<4xi64>
    %c_0 = stablehlo.constant dense<5> : tensor<4xi64>
    %1 = stablehlo.minimum %iota, %c_0 : tensor<4xi64>
    return %1 : tensor<4xi64>
  }
  // CHECK:  func.func @test_iota_min() -> tensor<4xi64> {
  // CHECK-NEXT:    %[[IOTA:.+]] = stablehlo.iota dim = 0 : tensor<4xi64>
  // CHECK-NEXT:    return %[[IOTA]] : tensor<4xi64>

  func.func @test_iota_min_cst() -> tensor<4xi64> {
    %iota = stablehlo.iota dim = 0 : tensor<4xi64>
    %c_0 = stablehlo.constant dense<-2> : tensor<4xi64>
    %1 = stablehlo.minimum %iota, %c_0 : tensor<4xi64>
    return %1 : tensor<4xi64>
  }
  // CHECK:  func.func @test_iota_min_cst() -> tensor<4xi64> {
  // CHECK-NEXT:    %[[C:.+]] = stablehlo.constant dense<-2> : tensor<4xi64>
  // CHECK-NEXT:    return %[[C]] : tensor<4xi64>

  func.func @test_iota_clamp() -> tensor<4xi64> {
    %c = stablehlo.constant dense<5> : tensor<4xi64>
    %c_0 = stablehlo.constant dense<-1> : tensor<4xi64>
    %iota = stablehlo.iota dim = 0 : tensor<4xi64>
    %1 = stablehlo.clamp %c_0, %iota, %c : tensor<4xi64>
    return %1 : tensor<4xi64>
  }
  // CHECK:  func.func @test_iota_clamp() -> tensor<4xi64> {
  // CHECK-NEXT:    %[[IOTA:.+]] = stablehlo.iota dim = 0 : tensor<4xi64>
  // CHECK-NEXT:    return %[[IOTA]] : tensor<4xi64>
  // CHECK-NEXT: }

  func.func @test_iota_clamp_min() -> tensor<4xi64> {
    %c = stablehlo.constant dense<10> : tensor<4xi64>
    %c_0 = stablehlo.constant dense<6> : tensor<4xi64>
    %iota = stablehlo.iota dim = 0 : tensor<4xi64>
    %1 = stablehlo.clamp %c_0, %iota, %c : tensor<4xi64>
    return %1 : tensor<4xi64>
  }
  // CHECK:  func.func @test_iota_clamp_min() -> tensor<4xi64> {
  // CHECK-NEXT:    %[[C:.+]] = stablehlo.constant dense<6> : tensor<4xi64>
  // CHECK-NEXT:    return %[[C]] : tensor<4xi64>
  // CHECK-NEXT: }

  func.func @test_iota_clamp_max() -> tensor<4xi64> {
    %c = stablehlo.constant dense<-1> : tensor<4xi64>
    %c_0 = stablehlo.constant dense<-6> : tensor<4xi64>
    %iota = stablehlo.iota dim = 0 : tensor<4xi64>
    %1 = stablehlo.clamp %c_0, %iota, %c : tensor<4xi64>
    return %1 : tensor<4xi64>
  }
  // CHECK:  func.func @test_iota_clamp_max() -> tensor<4xi64> {
  // CHECK-NEXT:    %[[C:.+]] = stablehlo.constant dense<-1> : tensor<4xi64>
  // CHECK-NEXT:    return %[[C]] : tensor<4xi64>
  // CHECK-NEXT: }

  func.func @test_iota_max_float() -> tensor<4xf32> {
    %iota = stablehlo.iota dim = 0 : tensor<4xf32>
    %c_0 = stablehlo.constant dense<-1.0> : tensor<4xf32>
    %1 = stablehlo.maximum %iota, %c_0 : tensor<4xf32>
    return %1 : tensor<4xf32>
  }
  // CHECK:  func.func @test_iota_max_float() -> tensor<4xf32> {
  // CHECK-NEXT:    %[[IOTA:.+]] = stablehlo.iota dim = 0 : tensor<4xf32>
  // CHECK-NEXT:    return %[[IOTA]] : tensor<4xf32>

  func.func @test_iota_max_float_cst() -> tensor<4xf32> {
    %iota = stablehlo.iota dim = 0 : tensor<4xf32>
    %c_0 = stablehlo.constant dense<5.0> : tensor<4xf32>
    %1 = stablehlo.maximum %iota, %c_0 : tensor<4xf32>
    return %1 : tensor<4xf32>
  }
  // CHECK:  func.func @test_iota_max_float_cst() -> tensor<4xf32> {
  // CHECK-NEXT:    %[[C:.+]] = stablehlo.constant dense<5.{{.*}}> : tensor<4xf32>
  // CHECK-NEXT:    return %[[C]] : tensor<4xf32>

  func.func @test_iota_min_float() -> tensor<4xf32> {
    %iota = stablehlo.iota dim = 0 : tensor<4xf32>
    %c_0 = stablehlo.constant dense<5.5> : tensor<4xf32>
    %1 = stablehlo.minimum %iota, %c_0 : tensor<4xf32>
    return %1 : tensor<4xf32>
  }
  // CHECK:  func.func @test_iota_min_float() -> tensor<4xf32> {
  // CHECK-NEXT:    %[[IOTA:.+]] = stablehlo.iota dim = 0 : tensor<4xf32>
  // CHECK-NEXT:    return %[[IOTA]] : tensor<4xf32>

  func.func @test_iota_min_float_cst() -> tensor<4xf32> {
    %iota = stablehlo.iota dim = 0 : tensor<4xf32>
    %c_0 = stablehlo.constant dense<-2.0> : tensor<4xf32>
    %1 = stablehlo.minimum %iota, %c_0 : tensor<4xf32>
    return %1 : tensor<4xf32>
  }
  // CHECK:  func.func @test_iota_min_float_cst() -> tensor<4xf32> {
  // CHECK-NEXT:    %[[C:.+]] = stablehlo.constant dense<-2.{{.*}}> : tensor<4xf32>
  // CHECK-NEXT:    return %[[C]] : tensor<4xf32>

  func.func @test_iota_clamp_float() -> tensor<4xf32> {
    %c = stablehlo.constant dense<5.5> : tensor<4xf32>
    %c_0 = stablehlo.constant dense<-1.5> : tensor<4xf32>
    %iota = stablehlo.iota dim = 0 : tensor<4xf32>
    %1 = stablehlo.clamp %c_0, %iota, %c : tensor<4xf32>
    return %1 : tensor<4xf32>
  }
  // CHECK:  func.func @test_iota_clamp_float() -> tensor<4xf32> {
  // CHECK-NEXT:    %[[IOTA:.+]] = stablehlo.iota dim = 0 : tensor<4xf32>
  // CHECK-NEXT:    return %[[IOTA]] : tensor<4xf32>
  // CHECK-NEXT: }

  func.func @test_iota_clamp_min_float() -> tensor<4xf32> {
    %c = stablehlo.constant dense<10.0> : tensor<4xf32>
    %c_0 = stablehlo.constant dense<6.0> : tensor<4xf32>
    %iota = stablehlo.iota dim = 0 : tensor<4xf32>
    %1 = stablehlo.clamp %c_0, %iota, %c : tensor<4xf32>
    return %1 : tensor<4xf32>
  }
  // CHECK:  func.func @test_iota_clamp_min_float() -> tensor<4xf32> {
  // CHECK-NEXT:    %[[C:.+]] = stablehlo.constant dense<6.{{.*}}> : tensor<4xf32>
  // CHECK-NEXT:    return %[[C]] : tensor<4xf32>
  // CHECK-NEXT: }

  func.func @test_iota_clamp_max_float() -> tensor<4xf32> {
    %c = stablehlo.constant dense<-1.0> : tensor<4xf32>
    %c_0 = stablehlo.constant dense<-6.0> : tensor<4xf32>
    %iota = stablehlo.iota dim = 0 : tensor<4xf32>
    %1 = stablehlo.clamp %c_0, %iota, %c : tensor<4xf32>
    return %1 : tensor<4xf32>
  }
  // CHECK:  func.func @test_iota_clamp_max_float() -> tensor<4xf32> {
  // CHECK-NEXT:    %[[C:.+]] = stablehlo.constant dense<-1.{{.*}}> : tensor<4xf32>
  // CHECK-NEXT:    return %[[C]] : tensor<4xf32>
  // CHECK-NEXT: }
}
