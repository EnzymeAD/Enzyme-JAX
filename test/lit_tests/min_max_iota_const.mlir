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

  // 1200 - i for i in [0, 1200) ranges over [1, 1200], so the max with 0 is a no-op.
  func.func @test_const_minus_iota_max() -> tensor<1200xi64> {
    %c = stablehlo.constant dense<1200> : tensor<1200xi64>
    %c_0 = stablehlo.constant dense<0> : tensor<1200xi64>
    %iota = stablehlo.iota dim = 0 : tensor<1200xi64>
    %0 = stablehlo.subtract %c, %iota : tensor<1200xi64>
    %1 = stablehlo.maximum %0, %c_0 : tensor<1200xi64>
    return %1 : tensor<1200xi64>
  }
  // CHECK:  func.func @test_const_minus_iota_max() -> tensor<1200xi64> {
  // CHECK-NEXT:    %[[C:.+]] = stablehlo.constant dense<1200> : tensor<1200xi64>
  // CHECK-NEXT:    %[[IOTA:.+]] = stablehlo.iota dim = 0 : tensor<1200xi64>
  // CHECK-NEXT:    %[[SUB:.+]] = stablehlo.subtract %[[C]], %[[IOTA]] : tensor<1200xi64>
  // CHECK-NEXT:    return %[[SUB]] : tensor<1200xi64>
  // CHECK-NEXT: }

  // 3 - i for i in [0, 4) ranges over [0, 3], so the min with 3 is the constant.
  func.func @test_const_minus_iota_min_cst() -> tensor<4xi64> {
    %c = stablehlo.constant dense<3> : tensor<4xi64>
    %iota = stablehlo.iota dim = 0 : tensor<4xi64>
    %0 = stablehlo.subtract %c, %iota : tensor<4xi64>
    %1 = stablehlo.minimum %0, %c : tensor<4xi64>
    return %1 : tensor<4xi64>
  }
  // CHECK:  func.func @test_const_minus_iota_min_cst() -> tensor<4xi64> {
  // CHECK-NEXT:    %[[C:.+]] = stablehlo.constant dense<3> : tensor<4xi64>
  // CHECK-NEXT:    %[[IOTA:.+]] = stablehlo.iota dim = 0 : tensor<4xi64>
  // CHECK-NEXT:    %[[SUB:.+]] = stablehlo.subtract %[[C]], %[[IOTA]] : tensor<4xi64>
  // CHECK-NEXT:    return %[[SUB]] : tensor<4xi64>
  // CHECK-NEXT: }

  // -i for i in [0, 4) ranges over [-3, 0], so the max with 0 is 0.
  func.func @test_neg_iota_max_cst() -> tensor<4xi64> {
    %c = stablehlo.constant dense<0> : tensor<4xi64>
    %iota = stablehlo.iota dim = 0 : tensor<4xi64>
    %0 = stablehlo.negate %iota : tensor<4xi64>
    %1 = stablehlo.maximum %0, %c : tensor<4xi64>
    return %1 : tensor<4xi64>
  }
  // CHECK:  func.func @test_neg_iota_max_cst() -> tensor<4xi64> {
  // CHECK-NEXT:    %[[C:.+]] = stablehlo.constant dense<0> : tensor<4xi64>
  // CHECK-NEXT:    return %[[C]] : tensor<4xi64>
  // CHECK-NEXT: }
}
