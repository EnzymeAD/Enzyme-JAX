// RUN: enzymexlamlir-opt %s -raise-affine-to-stablehlo | FileCheck %s

func.func @test_if_masking(%arg0: memref<10xf32>, %arg1: memref<10xi1>) {
  affine.for %i = 0 to 10 {
    %cond = affine.load %arg1[%i] : memref<10xi1>
    scf.if %cond {
      %v = affine.load %arg0[%i] : memref<10xf32>
      %v2 = arith.addf %v, %v : f32
      affine.store %v2, %arg0[%i] : memref<10xf32>
    }
  }
  return
}

func.func @test_nested_if_masking(%arg0: memref<10xf32>, %arg1: memref<10xi1>, %arg2: memref<10xi1>) {
  affine.for %i = 0 to 10 {
    %cond1 = affine.load %arg1[%i] : memref<10xi1>
    scf.if %cond1 {
      %cond2 = affine.load %arg2[%i] : memref<10xi1>
      scf.if %cond2 {
        %v = affine.load %arg0[%i] : memref<10xf32>
        %v2 = arith.addf %v, %v : f32
        affine.store %v2, %arg0[%i] : memref<10xf32>
      }
    }
  }
  return
}

func.func @test_if_else_masking(%arg0: memref<10xf32>, %arg1: memref<10xi1>) {
  affine.for %i = 0 to 10 {
    %cond = affine.load %arg1[%i] : memref<10xi1>
    scf.if %cond {
      %v = affine.load %arg0[%i] : memref<10xf32>
      %v2 = arith.addf %v, %v : f32
      affine.store %v2, %arg0[%i] : memref<10xf32>
    } else {
      %v = affine.load %arg0[%i] : memref<10xf32>
      %v2 = arith.mulf %v, %v : f32
      affine.store %v2, %arg0[%i] : memref<10xf32>
    }
  }
  return
}

func.func @test_affine_if_masking(%arg0: memref<10xf32>) {
  affine.for %i = 0 to 10 {
    affine.if affine_set<(d0) : (d0 - 5 >= 0)>(%i) {
      %v = affine.load %arg0[%i] : memref<10xf32>
      %v2 = arith.addf %v, %v : f32
      affine.store %v2, %arg0[%i] : memref<10xf32>
    }
  }
  return
}

// CHECK:    func.func private @test_affine_if_masking_raised(%[[a1:.+]]: tensor<10xf32>) -> tensor<10xf32> {
// CHECK-NEXT:    %[[a2:.+]] = stablehlo.iota dim = 0 : tensor<10xi64>
// CHECK-NEXT:    %[[a3:.+]] = stablehlo.constant dense<0> : tensor<10xi64>
// CHECK-NEXT:    %[[a4:.+]] = stablehlo.add %[[a2]], %[[a3]] : tensor<10xi64>
// CHECK-NEXT:    %[[a5:.+]] = stablehlo.constant dense<1> : tensor<10xi64>
// CHECK-NEXT:    %[[a6:.+]] = stablehlo.multiply %[[a4]], %[[a5]] : tensor<10xi64>
// CHECK-NEXT:    %[[a7:.+]] = stablehlo.constant dense<-5> : tensor<i64>
// CHECK-NEXT:    %[[a8:.+]] = stablehlo.broadcast_in_dim %[[a7]], dims = [] : (tensor<i64>) -> tensor<10xi64>
// CHECK-NEXT:    %[[a9:.+]] = stablehlo.add %[[a6]], %[[a8]] : tensor<10xi64>
// CHECK-NEXT:    %[[a10:.+]] = stablehlo.constant dense<0> : tensor<10xi64>
// CHECK-NEXT:    %[[a11:.+]] = stablehlo.compare GE, %[[a9]], %[[a10]] : (tensor<10xi64>, tensor<10xi64>) -> tensor<10xi1>
// CHECK-NEXT:    %[[a12:.+]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %[[a13:.+]] = arith.addf %[[a1]], %[[a1]] : tensor<10xf32>
// CHECK-NEXT:    %[[a14:.+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %[[a15:.+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %[[a16:.+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %[[a17:.+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %[[a18:.+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %[[a19:.+]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %[[a20:.+]] = stablehlo.select %[[a11]], %[[a13]], %[[a1]] : tensor<10xi1>, tensor<10xf32>
// CHECK-NEXT:    %[[a21:.+]] = stablehlo.dynamic_update_slice %[[a1]], %[[a20]], %[[a19]] : (tensor<10xf32>, tensor<10xf32>, tensor<i64>) -> tensor<10xf32>
// CHECK-NEXT:    return %[[a21]] : tensor<10xf32>
// CHECK-NEXT:  }
// CHECK-NEXT:  func.func private @test_if_else_masking_raised(%[[a1]]: tensor<10xf32>, %[[a22:.+]]: tensor<10xi1>) -> (tensor<10xf32>, tensor<10xi1>) {
// CHECK-NEXT:    %0 = stablehlo.iota dim = 0 : tensor<10xi64>
// CHECK-NEXT:    %c = stablehlo.constant dense<0> : tensor<10xi64>
// CHECK-NEXT:    %1 = stablehlo.add %0, %c : tensor<10xi64>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<1> : tensor<10xi64>
// CHECK-NEXT:    %2 = stablehlo.multiply %1, %c_0 : tensor<10xi64>
// CHECK-NEXT:    %c_1 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %c_2 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %3 = arith.addf %arg0, %arg0 : tensor<10xf32>
// CHECK-NEXT:    %c_3 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_4 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_5 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_6 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_7 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_8 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %4 = stablehlo.select %arg1, %3, %arg0 : tensor<10xi1>, tensor<10xf32>
// CHECK-NEXT:    %5 = stablehlo.dynamic_update_slice %arg0, %4, %c_8 : (tensor<10xf32>, tensor<10xf32>, tensor<i64>) -> tensor<10xf32>
// CHECK-NEXT:    %6 = stablehlo.not %arg1 : tensor<10xi1>
// CHECK-NEXT:    %c_9 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %7 = arith.mulf %5, %5 : tensor<10xf32>
// CHECK-NEXT:    %c_10 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_11 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_12 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_13 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_14 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_15 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %8 = stablehlo.select %6, %7, %5 : tensor<10xi1>, tensor<10xf32>
// CHECK-NEXT:    %9 = stablehlo.dynamic_update_slice %5, %8, %c_15 : (tensor<10xf32>, tensor<10xf32>, tensor<i64>) -> tensor<10xf32>
// CHECK-NEXT:    return %9, %arg1 : tensor<10xf32>, tensor<10xi1>
// CHECK-NEXT:  }
// CHECK-NEXT:  func.func private @test_nested_if_masking_raised(%[[a1]]: tensor<10xf32>, %[[a22]]: tensor<10xi1>, %[[a42:.+]]: tensor<10xi1>) -> (tensor<10xf32>, tensor<10xi1>, tensor<10xi1>) {
// CHECK-NEXT:    %[[a2]] = stablehlo.iota dim = 0 : tensor<10xi64>
// CHECK-NEXT:    %[[a3]] = stablehlo.constant dense<0> : tensor<10xi64>
// CHECK-NEXT:    %[[a4]] = stablehlo.add %[[a2]], %[[a3]] : tensor<10xi64>
// CHECK-NEXT:    %[[a5]] = stablehlo.constant dense<1> : tensor<10xi64>
// CHECK-NEXT:    %[[a6]] = stablehlo.multiply %[[a4]], %[[a5]] : tensor<10xi64>
// CHECK-NEXT:    %[[a7]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %[[a10]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %[[a8]] = stablehlo.and %[[a22]], %[[a42]] : tensor<10xi1>
// CHECK-NEXT:    %[[a12]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %[[a9]] = arith.addf %[[a1]], %[[a1]] : tensor<10xf32>
// CHECK-NEXT:    %[[a14]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %[[a15]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %[[a16]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %[[a17]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %[[a18]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %[[a19]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %[[a11]] = stablehlo.select %[[a8]], %[[a9]], %[[a1]] : tensor<10xi1>, tensor<10xf32>
// CHECK-NEXT:    %[[a13]] = stablehlo.dynamic_update_slice %[[a1]], %[[a11]], %[[a19]] : (tensor<10xf32>, tensor<10xf32>, tensor<i64>) -> tensor<10xf32>
// CHECK-NEXT:    return %[[a13]], %[[a22]], %[[a42]] : tensor<10xf32>, tensor<10xi1>, tensor<10xi1>
// CHECK-NEXT:  }
// CHECK-NEXT:  func.func private @test_if_masking_raised(%[[a1]]: tensor<10xf32>, %[[a22]]: tensor<10xi1>) -> (tensor<10xf32>, tensor<10xi1>) {
// CHECK-NEXT:    %[[a2]] = stablehlo.iota dim = 0 : tensor<10xi64>
// CHECK-NEXT:    %[[a3]] = stablehlo.constant dense<0> : tensor<10xi64>
// CHECK-NEXT:    %[[a4]] = stablehlo.add %[[a2]], %[[a3]] : tensor<10xi64>
// CHECK-NEXT:    %[[a5]] = stablehlo.constant dense<1> : tensor<10xi64>
// CHECK-NEXT:    %[[a6]] = stablehlo.multiply %[[a4]], %[[a5]] : tensor<10xi64>
// CHECK-NEXT:    %[[a7]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %[[a10]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %[[a8]] = arith.addf %[[a1]], %[[a1]] : tensor<10xf32>
// CHECK-NEXT:    %[[a12]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %[[a14]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %[[a15]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %[[a16]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %[[a17]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %[[a18]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %[[a9]] = stablehlo.select %[[a22]], %[[a8]], %[[a1]] : tensor<10xi1>, tensor<10xf32>
// CHECK-NEXT:    %[[a11]] = stablehlo.dynamic_update_slice %[[a1]], %[[a9]], %[[a18]] : (tensor<10xf32>, tensor<10xf32>, tensor<i64>) -> tensor<10xf32>
// CHECK-NEXT:    return %[[a11]], %[[a22]] : tensor<10xf32>, tensor<10xi1>
// CHECK-NEXT:  }
