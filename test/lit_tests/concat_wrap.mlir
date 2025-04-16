// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=concat_slice;recognize_wrap;slice_slice" --transform-interpreter --enzyme-hlo-remove-transform -allow-unregistered-dialect %s | FileCheck %s

module @"reactant_loop!" {

  func.func @main(%90: tensor<1x4096x4096xf64>) -> (tensor<1x4096x4096xf64>) {
      %91 = stablehlo.slice %90 [0:1, 0:4096, 8:4088] : (tensor<1x4096x4096xf64>) -> tensor<1x4096x4080xf64>
      %92 = "enzymexla.wrap"(%91) <{dimension = 2 : i64, lhs = 1 : i64, rhs = 8 : i64}> : (tensor<1x4096x4080xf64>) -> tensor<1x4096x4089xf64>
      %93 = stablehlo.slice %90 [0:1, 0:4096, 4080:4087] : (tensor<1x4096x4096xf64>) -> tensor<1x4096x7xf64>
      %94 = stablehlo.concatenate %93, %92, dim = 2 : (tensor<1x4096x7xf64>, tensor<1x4096x4089xf64>) -> tensor<1x4096x4096xf64>
      stablehlo.return %94 :  tensor<1x4096x4096xf64>
  }
}

// CHECK:  func.func @main(%arg0: tensor<1x4096x4096xf64>) -> tensor<1x4096x4096xf64> {
// CHECK-NEXT:    %0 = stablehlo.slice %arg0 [0:1, 0:4096, 8:4088] : (tensor<1x4096x4096xf64>) -> tensor<1x4096x4080xf64>
// CHECK-NEXT:    %1 = "enzymexla.wrap"(%0) <{dimension = 2 : i64, lhs = 8 : i64, rhs = 8 : i64}> : (tensor<1x4096x4080xf64>) -> tensor<1x4096x4096xf64>
// CHECK-NEXT:    stablehlo.return %1 : tensor<1x4096x4096xf64>
// CHECK-NEXT:  }