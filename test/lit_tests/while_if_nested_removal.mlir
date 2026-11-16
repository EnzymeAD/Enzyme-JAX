// RUN: enzymexlamlir-opt %s --remove-unnecessary-enzyme-ops --canonicalize | FileCheck %s

module {
  func.func @main(%arg0: tensor<10xf32>) -> tensor<10xf32> {
    %c0 = stablehlo.constant dense<0> : tensor<i64>
    %c1 = stablehlo.constant dense<1> : tensor<i64>
    %c10 = stablehlo.constant dense<10> : tensor<i64>

    // Gradient slots keyed by the while loop's init values.
    %g0 = "enzyme.init"() : () -> !enzyme.Gradient<tensor<10xf32>>
    %g1 = "enzyme.init"() : () -> !enzyme.Gradient<tensor<i64>>
    "enzyme.set"(%g0, %arg0) : (!enzyme.Gradient<tensor<10xf32>>, tensor<10xf32>) -> ()
    "enzyme.set"(%g1, %c0) : (!enzyme.Gradient<tensor<i64>>, tensor<i64>) -> ()

    %0:2 = stablehlo.while(%iterArg = %arg0, %iterArg_0 = %c0) : tensor<10xf32>, tensor<i64>
     cond {
      %1 = stablehlo.compare LT, %iterArg_0, %c10 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %1 : tensor<i1>
    } do {
      %pred = stablehlo.compare LT, %iterArg_0, %c1 : (tensor<i64>, tensor<i64>) -> tensor<i1>

      %2 = "stablehlo.if"(%pred) ({
        %v0 = "enzyme.get"(%g0) : (!enzyme.Gradient<tensor<10xf32>>) -> tensor<10xf32>
        %v1 = stablehlo.add %v0, %iterArg : tensor<10xf32>
        "enzyme.set"(%g0, %v1) : (!enzyme.Gradient<tensor<10xf32>>, tensor<10xf32>) -> ()
        stablehlo.return %v1 : tensor<10xf32>
      }, {
        %v0 = "enzyme.get"(%g0) : (!enzyme.Gradient<tensor<10xf32>>) -> tensor<10xf32>
        stablehlo.return %v0 : tensor<10xf32>
      }) : (tensor<i1>) -> tensor<10xf32>

      %3 = stablehlo.add %iterArg_0, %c1 : tensor<i64>
      "enzyme.set"(%g1, %3) : (!enzyme.Gradient<tensor<i64>>, tensor<i64>) -> ()

      stablehlo.return %2, %3 : tensor<10xf32>, tensor<i64>
    }

    return %0#0 : tensor<10xf32>
  }
}

// CHECK:  func.func @main(%arg0: tensor<10xf32>) -> tensor<10xf32> {
// CHECK-NEXT:      %c = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:      %c_0 = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:      %c_1 = stablehlo.constant dense<10> : tensor<i64>
// CHECK-NEXT:      %0:3 = stablehlo.while(%iterArg = %arg0, %iterArg_2 = %c, %iterArg_3 = %arg0) : tensor<10xf32>, tensor<i64>, tensor<10xf32>
// CHECK-NEXT:      cond {
// CHECK-NEXT:        %1 = stablehlo.compare LT, %iterArg_2, %c_1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:        stablehlo.return %1 : tensor<i1>
// CHECK-NEXT:      } do {
// CHECK-NEXT:        %1 = stablehlo.compare LT, %iterArg_2, %c_0 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:        %2:2 = "stablehlo.if"(%1) ({
// CHECK-NEXT:          %4 = stablehlo.add %iterArg_3, %iterArg : tensor<10xf32>
// CHECK-NEXT:          stablehlo.return %4, %4 : tensor<10xf32>, tensor<10xf32>
// CHECK-NEXT:        }, {
// CHECK-NEXT:          stablehlo.return %iterArg_3, %iterArg_3 : tensor<10xf32>, tensor<10xf32>
// CHECK-NEXT:        }) : (tensor<i1>) -> (tensor<10xf32>, tensor<10xf32>)
// CHECK-NEXT:        %3 = stablehlo.add %iterArg_2, %c_0 : tensor<i64>
// CHECK-NEXT:        stablehlo.return %2#0, %3, %2#1 : tensor<10xf32>, tensor<i64>, tensor<10xf32>
// CHECK-NEXT:      }
// CHECK-NEXT:      return %0#0 : tensor<10xf32>
// CHECK-NEXT:    }
