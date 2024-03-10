// RUN: enzymexlamlir-opt --enzyme-hlo-unroll %s | FileCheck %s

module {

  func.func @main(%a : tensor<2x2xf32>) -> tensor<2x2xf32> {

    %start = stablehlo.constant dense<0> : tensor<i32>
    
    %lim = stablehlo.constant dense<5> : tensor<i32>

    %step = stablehlo.constant dense<1> : tensor<i32>

    %w:2 = stablehlo.while(%iterArg = %a, %iterArg_0 = %start) : tensor<2x2xf32>, tensor<i32>
     cond {
      %9737 = stablehlo.compare  LT, %iterArg_0, %lim,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %9737 : tensor<i1>
    } do {
      %next = stablehlo.add %iterArg, %iterArg : tensor<2x2xf32>
       %ni = stablehlo.add %iterArg_0, %step : tensor<i32>
      stablehlo.return %next, %ni : tensor<2x2xf32>, tensor<i32>
    }
    return %w#0 : tensor<2x2xf32>
  }
}

// CHECK:   func.func @main(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
// CHECK-NEXT:     %0 = stablehlo.add %arg0, %arg0 : tensor<2x2xf32>
// CHECK-NEXT:     %1 = stablehlo.add %0, %0 : tensor<2x2xf32>
// CHECK-NEXT:     %2 = stablehlo.add %1, %1 : tensor<2x2xf32>
// CHECK-NEXT:     %3 = stablehlo.add %2, %2 : tensor<2x2xf32>
// CHECK-NEXT:     %4 = stablehlo.add %3, %3 : tensor<2x2xf32>
// CHECK-NEXT:     return %4 : tensor<2x2xf32>
// CHECK-NEXT:   }
