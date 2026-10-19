// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt --split-input-file | FileCheck %s
// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=reduce_or_and" --transform-interpreter --enzyme-hlo-remove-transform --split-input-file | FileCheck %s --check-prefix=TD
// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=reduce_or_and" --transform-interpreter --enzyme-hlo-remove-transform --split-input-file %s | FileCheck %s --check-prefix=TD

// A one-hot lane mask or-reduced over the lanes is true; and-reduced it is
// false.
func.func @onehot(%a: tensor<16xf64>, %b: tensor<16xf64>) -> (tensor<16xf64>, tensor<16xf64>) {
  %init = stablehlo.constant dense<false> : tensor<i1>
  %top = stablehlo.constant dense<true> : tensor<i1>
  %mask = stablehlo.constant dense<[true, false, false, false]> : tensor<4xi1>
  %any = stablehlo.reduce(%mask init: %init) applies stablehlo.or across dimensions = [0] : (tensor<4xi1>, tensor<i1>) -> tensor<i1>
  %all = stablehlo.reduce(%mask init: %top) applies stablehlo.and across dimensions = [0] : (tensor<4xi1>, tensor<i1>) -> tensor<i1>
  %r0 = stablehlo.select %any, %a, %b : tensor<i1>, tensor<16xf64>
  %r1 = stablehlo.select %all, %a, %b : tensor<i1>, tensor<16xf64>
  return %r0, %r1 : tensor<16xf64>, tensor<16xf64>
}



// CHECK:    func.func @onehot(%arg0: tensor<16xf64>, %arg1: tensor<16xf64>) -> (tensor<16xf64>, tensor<16xf64>) {
// CHECK-NEXT:    return %arg0, %arg1 : tensor<16xf64>, tensor<16xf64>
// CHECK-NEXT:  }

// TD:  module {
// TD-NEXT:  func.func @onehot(%arg0: tensor<16xf64>, %arg1: tensor<16xf64>) -> (tensor<16xf64>, tensor<16xf64>) {
// TD-NEXT:    %c = stablehlo.constant dense<true> : tensor<i1>
// TD-NEXT:    %c_0 = stablehlo.constant dense<false> : tensor<i1>
// TD-NEXT:    %0 = stablehlo.select %c, %arg0, %arg1 : tensor<i1>, tensor<16xf64>
// TD-NEXT:    %1 = stablehlo.select %c_0, %arg0, %arg1 : tensor<i1>, tensor<16xf64>
// TD-NEXT:    return %0, %1 : tensor<16xf64>, tensor<16xf64>
// TD-NEXT:  }

// -----

// Reducing one axis of a 2-D mask keeps the other: rows [true,false] and
// [false,false] or-reduce over the columns to [true, false].
func.func @rows() -> tensor<2xi1> {
  %init = stablehlo.constant dense<false> : tensor<i1>
  %mask = stablehlo.constant dense<[[true, false, false], [false, false, false]]> : tensor<2x3xi1>
  %r = stablehlo.reduce(%mask init: %init) applies stablehlo.or across dimensions = [1] : (tensor<2x3xi1>, tensor<i1>) -> tensor<2xi1>
  return %r : tensor<2xi1>
}



// CHECK:    func.func @rows() -> tensor<2xi1> {
// CHECK-NEXT:    %c = stablehlo.constant dense<[true, false]> : tensor<2xi1>
// CHECK-NEXT:    return %c : tensor<2xi1>
// CHECK-NEXT:  }

// TD:  module {
// TD-NEXT:  func.func @rows() -> tensor<2xi1> {
// TD-NEXT:    %c = stablehlo.constant dense<[true, false]> : tensor<2xi1>
// TD-NEXT:    return %c : tensor<2xi1>
// TD-NEXT:  }

// -----

// A splat mask folds through the init: all-false or-reduced with a true
// init is true; and-reduced with a true init it is false.
func.func @splat() -> (tensor<i1>, tensor<i1>) {
  %top = stablehlo.constant dense<true> : tensor<i1>
  %mask = stablehlo.constant dense<false> : tensor<32xi1>
  %r0 = stablehlo.reduce(%mask init: %top) applies stablehlo.or across dimensions = [0] : (tensor<32xi1>, tensor<i1>) -> tensor<i1>
  %r1 = stablehlo.reduce(%mask init: %top) applies stablehlo.and across dimensions = [0] : (tensor<32xi1>, tensor<i1>) -> tensor<i1>
  return %r0, %r1 : tensor<i1>, tensor<i1>
}



// CHECK:    func.func @splat() -> (tensor<i1>, tensor<i1>) {
// CHECK-NEXT:    %c = stablehlo.constant dense<true> : tensor<i1>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<false> : tensor<i1>
// CHECK-NEXT:    return %c, %c_0 : tensor<i1>, tensor<i1>
// CHECK-NEXT:  }

// TD:  module {
// TD-NEXT:  func.func @splat() -> (tensor<i1>, tensor<i1>) {
// TD-NEXT:    %c = stablehlo.constant dense<true> : tensor<i1>
// TD-NEXT:    %c_0 = stablehlo.constant dense<false> : tensor<i1>
// TD-NEXT:    return %c, %c_0 : tensor<i1>, tensor<i1>
// TD-NEXT:  }

// -----

// A splat mask folds to a splat without materializing its elements, so
// its size does not matter.
func.func @hugesplat() -> (tensor<i1>, tensor<4096xi1>) {
  %f = stablehlo.constant dense<false> : tensor<i1>
  %mask = stablehlo.constant dense<true> : tensor<1073741824xi1>
  %r0 = stablehlo.reduce(%mask init: %f) applies stablehlo.or across dimensions = [0] : (tensor<1073741824xi1>, tensor<i1>) -> tensor<i1>
  %t = stablehlo.constant dense<true> : tensor<i1>
  %mask2 = stablehlo.constant dense<true> : tensor<4096x1048576xi1>
  %r1 = stablehlo.reduce(%mask2 init: %t) applies stablehlo.and across dimensions = [1] : (tensor<4096x1048576xi1>, tensor<i1>) -> tensor<4096xi1>
  return %r0, %r1 : tensor<i1>, tensor<4096xi1>
}



// CHECK:    func.func @hugesplat() -> (tensor<i1>, tensor<4096xi1>) {
// CHECK-NEXT:    %c = stablehlo.constant dense<true> : tensor<i1>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<true> : tensor<4096xi1>
// CHECK-NEXT:    return %c, %c_0 : tensor<i1>, tensor<4096xi1>
// CHECK-NEXT:  }

// TD:  module {
// TD-NEXT:  func.func @hugesplat() -> (tensor<i1>, tensor<4096xi1>) {
// TD-NEXT:    %c = stablehlo.constant dense<true> : tensor<i1>
// TD-NEXT:    %c_0 = stablehlo.constant dense<true> : tensor<4096xi1>
// TD-NEXT:    return %c, %c_0 : tensor<i1>, tensor<4096xi1>
// TD-NEXT:  }
