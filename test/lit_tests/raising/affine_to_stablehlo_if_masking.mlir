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
// CHECK-NEXT:    %[[a3]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %[[a5]] = stablehlo.constant dense<10> : tensor<i64>
// CHECK-NEXT:    %[[a7]] = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:    %[[a2]]:3 = stablehlo.while(%[[a23:.+]] = %[[a3]], %[[a24:.+]] = %[[a1]], %[[a25:.+]] = %[[a22]]) : tensor<i64>, tensor<10xf32>, tensor<10xi1>
// CHECK-NEXT:    cond {
// CHECK-NEXT:      %[[a4]] = stablehlo.compare LT, %[[a23]], %[[a5]] : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:      stablehlo.return %[[a4]] : tensor<i1>
// CHECK-NEXT:    } do {
// CHECK-NEXT:      %[[a4]] = stablehlo.dynamic_slice %[[a25]], %[[a23]], sizes = [1] : (tensor<10xi1>, tensor<i64>) -> tensor<1xi1>
// CHECK-NEXT:      %[[a6]] = stablehlo.reshape %[[a4]] : (tensor<1xi1>) -> tensor<i1>
// CHECK-NEXT:      %[[a8]] = stablehlo.dynamic_slice %[[a24]], %[[a23]], sizes = [1] : (tensor<10xf32>, tensor<i64>) -> tensor<1xf32>
// CHECK-NEXT:      %[[a9]] = stablehlo.reshape %[[a8]] : (tensor<1xf32>) -> tensor<f32>
// CHECK-NEXT:      %[[a11]] = arith.addf %[[a9]], %[[a9]] : tensor<f32>
// CHECK-NEXT:      %[[a14]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:      %[[a15]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:      %[[a16]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:      %[[a17]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:      %[[a18]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:      %[[a13]] = stablehlo.broadcast_in_dim %[[a11]], dims = [] : (tensor<f32>) -> tensor<1xf32>
// CHECK-NEXT:      %[[a20]] = stablehlo.dynamic_slice %[[a24]], %[[a23]], sizes = [1] : (tensor<10xf32>, tensor<i64>) -> tensor<1xf32>
// CHECK-NEXT:      %[[a21]] = stablehlo.broadcast_in_dim %[[a6]], dims = [] : (tensor<i1>) -> tensor<1xi1>
// CHECK-NEXT:      %[[a26:.+]] = stablehlo.select %[[a21]], %[[a13]], %[[a20]] : tensor<1xi1>, tensor<1xf32>
// CHECK-NEXT:      %[[a27:.+]] = stablehlo.dynamic_update_slice %[[a24]], %[[a26]], %[[a23]] : (tensor<10xf32>, tensor<1xf32>, tensor<i64>) -> tensor<10xf32>
// CHECK-NEXT:      %[[a28:.+]] = stablehlo.not %[[a6]] : tensor<i1>
// CHECK-NEXT:      %[[a29:.+]] = stablehlo.dynamic_slice %[[a27]], %[[a23]], sizes = [1] : (tensor<10xf32>, tensor<i64>) -> tensor<1xf32>
// CHECK-NEXT:      %[[a30:.+]] = stablehlo.reshape %[[a29]] : (tensor<1xf32>) -> tensor<f32>
// CHECK-NEXT:      %[[a31:.+]] = arith.mulf %[[a30]], %[[a30]] : tensor<f32>
// CHECK-NEXT:      %[[a19]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:      %[[a32:.+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:      %[[a33:.+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:      %[[a34:.+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:      %[[a35:.+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:      %[[a36:.+]] = stablehlo.broadcast_in_dim %[[a31]], dims = [] : (tensor<f32>) -> tensor<1xf32>
// CHECK-NEXT:      %[[a37:.+]] = stablehlo.dynamic_slice %[[a27]], %[[a23]], sizes = [1] : (tensor<10xf32>, tensor<i64>) -> tensor<1xf32>
// CHECK-NEXT:      %[[a38:.+]] = stablehlo.broadcast_in_dim %[[a28]], dims = [] : (tensor<i1>) -> tensor<1xi1>
// CHECK-NEXT:      %[[a39:.+]] = stablehlo.select %[[a38]], %[[a36]], %[[a37]] : tensor<1xi1>, tensor<1xf32>
// CHECK-NEXT:      %[[a40:.+]] = stablehlo.dynamic_update_slice %[[a27]], %[[a39]], %[[a23]] : (tensor<10xf32>, tensor<1xf32>, tensor<i64>) -> tensor<10xf32>
// CHECK-NEXT:      %[[a41:.+]] = stablehlo.add %[[a23]], %[[a7]] : tensor<i64>
// CHECK-NEXT:      stablehlo.return %[[a41]], %[[a40]], %[[a25]] : tensor<i64>, tensor<10xf32>, tensor<10xi1>
// CHECK-NEXT:    }
// CHECK-NEXT:    return %[[a2]]#1, %[[a2]]#2 : tensor<10xf32>, tensor<10xi1>
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
