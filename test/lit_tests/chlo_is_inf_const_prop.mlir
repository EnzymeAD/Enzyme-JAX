// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(enzyme-hlo-opt)" | FileCheck %s

module {
  func.func @chlo_prop_mix() -> tensor<2xi1> {  
    %arg = stablehlo.constant dense<[1.000000e+00,0xFF800000]> : tensor<2xf32>
    %result =  chlo.is_inf %arg : tensor<2xf32> -> tensor<2xi1> 
    func.return %result : tensor<2xi1>
  }
  
  func.func @chlo_prop_false_splat() -> tensor<2xi1> {  
    %arg = stablehlo.constant dense<2.000000e+00> : tensor<2xf32>
    %result = chlo.is_inf %arg : tensor<2xf32> -> tensor<2xi1>
    func.return %result : tensor<2xi1>
  } 

  func.func @chlo_prop_true_splat() -> tensor<2xi1> {  
    %arg = stablehlo.constant dense<0x7F800000> : tensor<2xf32>
    %result = chlo.is_inf %arg : tensor<2xf32> -> tensor<2xi1>
    func.return %result : tensor<2xi1>
  }
}

// CHECK: func.func @chlo_prop_mix() -> tensor<2xi1> {
// CHECK-NEXT:   %c = stablehlo.constant dense<[false, true]> : tensor<2xi1>
// CHECK-NEXT:   return %c : tensor<2xi1>
// CHECK-NEXT: }

// CHECK: func.func @chlo_prop_false_splat() -> tensor<2xi1> {
// CHECK-NEXT:   %c = stablehlo.constant dense<false> : tensor<2xi1>
// CHECK-NEXT:   return %c : tensor<2xi1>
// CHECK-NEXT: }

// CHECK: func.func @chlo_prop_true_splat() -> tensor<2xi1> {
// CHECK-NEXT:   %c = stablehlo.constant dense<true> : tensor<2xi1>
// CHECK-NEXT:   return %c : tensor<2xi1>
// CHECK-NEXT: }
