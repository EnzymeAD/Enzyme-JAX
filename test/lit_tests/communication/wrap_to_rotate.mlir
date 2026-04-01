// RUN: enzymexlamlir-opt %s --optimize-communication="wrap_to_rotate=1 wrap_to_pad_comm=0" | FileCheck %s

module {
  func.func @test_wrap_to_rotate_symmetric(%arg0: tensor<10xf32>) -> tensor<14xf32> {
    %wrap = "enzymexla.wrap"(%arg0) <{dimension = 0 : i64, lhs = 2 : i64, rhs = 2 : i64}> : (tensor<10xf32>) -> tensor<14xf32>
    return %wrap : tensor<14xf32>
  }

  func.func @test_wrap_to_rotate_asymmetric(%arg0: tensor<10xf32>) -> tensor<15xf32> {
    %wrap = "enzymexla.wrap"(%arg0) <{dimension = 0 : i64, lhs = 3 : i64, rhs = 2 : i64}> : (tensor<10xf32>) -> tensor<15xf32>
    return %wrap : tensor<15xf32>
  }
}

// CHECK: func.func @test_wrap_to_rotate_symmetric(%arg0: tensor<10xf32>) -> tensor<14xf32> {
// CHECK-NEXT:     %c = stablehlo.constant dense<12> : tensor<14xi32>
// CHECK-NEXT:     %c_0 = stablehlo.constant dense<2> : tensor<14xi32>
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %0 = stablehlo.pad %arg0, %cst, low = [2], high = [2], interior = [0] : (tensor<10xf32>, tensor<f32>) -> tensor<14xf32>
// CHECK-NEXT:     %1 = "enzymexla.rotate"(%0) <{amount = 4 : i32, dimension = 0 : i32}> : (tensor<14xf32>) -> tensor<14xf32>
// CHECK-NEXT:     %2 = "enzymexla.rotate"(%0) <{amount = 10 : i32, dimension = 0 : i32}> : (tensor<14xf32>) -> tensor<14xf32>
// CHECK-NEXT:     %3 = stablehlo.iota dim = 0 : tensor<14xi32>
// CHECK-NEXT:     %4 = stablehlo.compare  LT, %3, %c_0 : (tensor<14xi32>, tensor<14xi32>) -> tensor<14xi1>
// CHECK-NEXT:     %5 = stablehlo.compare  LT, %3, %c : (tensor<14xi32>, tensor<14xi32>) -> tensor<14xi1>
// CHECK-NEXT:     %6 = stablehlo.select %4, %2, %0 : tensor<14xi1>, tensor<14xf32>
// CHECK-NEXT:     %7 = stablehlo.select %5, %6, %1 : tensor<14xi1>, tensor<14xf32>
// CHECK-NEXT:     return %7 : tensor<14xf32>
// CHECK-NEXT:   }

// CHECK:  func.func @test_wrap_to_rotate_asymmetric(%arg0: tensor<10xf32>) -> tensor<15xf32> {
// CHECK-NEXT:     %c = stablehlo.constant dense<13> : tensor<15xi32>
// CHECK-NEXT:     %c_0 = stablehlo.constant dense<3> : tensor<15xi32>
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %0 = stablehlo.pad %arg0, %cst, low = [3], high = [2], interior = [0] : (tensor<10xf32>, tensor<f32>) -> tensor<15xf32>
// CHECK-NEXT:     %1 = "enzymexla.rotate"(%0) <{amount = 5 : i32, dimension = 0 : i32}> : (tensor<15xf32>) -> tensor<15xf32>
// CHECK-NEXT:     %2 = "enzymexla.rotate"(%0) <{amount = 10 : i32, dimension = 0 : i32}> : (tensor<15xf32>) -> tensor<15xf32>
// CHECK-NEXT:     %3 = stablehlo.iota dim = 0 : tensor<15xi32>
// CHECK-NEXT:     %4 = stablehlo.compare  LT, %3, %c_0 : (tensor<15xi32>, tensor<15xi32>) -> tensor<15xi1>
// CHECK-NEXT:     %5 = stablehlo.compare  LT, %3, %c : (tensor<15xi32>, tensor<15xi32>) -> tensor<15xi1>
// CHECK-NEXT:     %6 = stablehlo.select %4, %2, %0 : tensor<15xi1>, tensor<15xf32>
// CHECK-NEXT:     %7 = stablehlo.select %5, %6, %1 : tensor<15xi1>, tensor<15xf32>
// CHECK-NEXT:     return %7 : tensor<15xf32>
// CHECK-NEXT: }

module {
  sdy.mesh @mesh = <["x"=12]>
  func.func @wrap(%arg0: tensor<2304xf64> {enzymexla.memory_effects = [], sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) -> (tensor<2308xf64> {sdy.sharding = #sdy.sharding<@mesh, [{"x":(1)4}]>}, tensor<2304xf64> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) attributes {enzymexla.memory_effects = []} {
    %0 = "enzymexla.wrap"(%arg0) <{dimension = 0 : i64, lhs = 2 : i64, rhs = 2 : i64}> {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : (tensor<2304xf64>) -> tensor<2308xf64>
    return %0, %arg0 : tensor<2308xf64>, tensor<2304xf64>
  }
}
