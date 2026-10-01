// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=reduce_const_prop" --transform-interpreter --enzyme-hlo-remove-transform | FileCheck %s

// A loop bound the raiser makes of the largest per-lane trip count, once the
// lanes' counts are all the same constant.
func.func @splat_max() -> tensor<i32> {
  %c = stablehlo.constant dense<1> : tensor<24xi32>
  %init = stablehlo.constant dense<0> : tensor<i32>
  %0 = stablehlo.reduce(%c init: %init) applies stablehlo.maximum across dimensions = [0] : (tensor<24xi32>, tensor<i32>) -> tensor<i32>
  return %0 : tensor<i32>
}

// CHECK:  func.func @splat_max() -> tensor<i32> {
// CHECK-NEXT:    %c = stablehlo.constant dense<1> : tensor<i32>
// CHECK-NEXT:    return %c : tensor<i32>
// CHECK-NEXT:  }

func.func @sum_rows() -> tensor<2xi64> {
  %c = stablehlo.constant dense<[[1, 2, 3], [4, 5, 6]]> : tensor<2x3xi64>
  %init = stablehlo.constant dense<10> : tensor<i64>
  %0 = stablehlo.reduce(%c init: %init) applies stablehlo.add across dimensions = [1] : (tensor<2x3xi64>, tensor<i64>) -> tensor<2xi64>
  return %0 : tensor<2xi64>
}

// CHECK:  func.func @sum_rows() -> tensor<2xi64> {
// CHECK-NEXT:    %c = stablehlo.constant dense<[16, 25]> : tensor<2xi64>
// CHECK-NEXT:    return %c : tensor<2xi64>
// CHECK-NEXT:  }

func.func @max_cols() -> tensor<3xf32> {
  %c = stablehlo.constant dense<[[1.0, 5.0, 3.0], [4.0, 2.0, 6.0]]> : tensor<2x3xf32>
  %init = stablehlo.constant dense<0xFF800000> : tensor<f32>
  %0 = stablehlo.reduce(%c init: %init) applies stablehlo.maximum across dimensions = [0] : (tensor<2x3xf32>, tensor<f32>) -> tensor<3xf32>
  return %0 : tensor<3xf32>
}

// CHECK:  func.func @max_cols() -> tensor<3xf32> {
// CHECK-NEXT:    %cst = stablehlo.constant dense<[4.000000e+00, 5.000000e+00, 6.000000e+00]> : tensor<3xf32>
// CHECK-NEXT:    return %cst : tensor<3xf32>
// CHECK-NEXT:  }

// A body that is not one of the folded ones stays.
func.func @other_body() -> tensor<i32> {
  %c = stablehlo.constant dense<1> : tensor<4xi32>
  %init = stablehlo.constant dense<0> : tensor<i32>
  %0 = stablehlo.reduce(%c init: %init) across dimensions = [0] : (tensor<4xi32>, tensor<i32>) -> tensor<i32>
   reducer(%a: tensor<i32>, %b: tensor<i32>) {
    %s = stablehlo.subtract %a, %b : tensor<i32>
    stablehlo.return %s : tensor<i32>
  }
  return %0 : tensor<i32>
}

// CHECK:  func.func @other_body() -> tensor<i32> {
// CHECK:    stablehlo.reduce

// A splat input folds in closed form, whatever its size: a sum is the value
// times the number of elements reduced (2621441 is 0x4A200004), a product its
// power.
func.func @splat_sum() -> tensor<4xf32> {
  %c = stablehlo.constant dense<2.500000e+00> : tensor<4x1048576xf32>
  %init = stablehlo.constant dense<1.000000e+00> : tensor<f32>
  %0 = stablehlo.reduce(%c init: %init) applies stablehlo.add across dimensions = [1] : (tensor<4x1048576xf32>, tensor<f32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// CHECK:  func.func @splat_sum() -> tensor<4xf32> {
// CHECK-NEXT:    %cst = stablehlo.constant dense<0x4A200004> : tensor<4xf32>
// CHECK-NEXT:    return %cst : tensor<4xf32>
// CHECK-NEXT:  }

func.func @splat_product() -> tensor<i32> {
  %c = stablehlo.constant dense<3> : tensor<1048576x2xi32>
  %init = stablehlo.constant dense<1> : tensor<i32>
  %0 = stablehlo.reduce(%c init: %init) applies stablehlo.multiply across dimensions = [0, 1] : (tensor<1048576x2xi32>, tensor<i32>) -> tensor<i32>
  return %0 : tensor<i32>
}

// 3^2097152 mod 2^32
// CHECK:  func.func @splat_product() -> tensor<i32> {
// CHECK-NEXT:    %c = stablehlo.constant dense<511705089> : tensor<i32>
// CHECK-NEXT:    return %c : tensor<i32>
// CHECK-NEXT:  }

func.func @splat_sum_i8() -> tensor<i8> {
  %c = stablehlo.constant dense<3> : tensor<100xi8>
  %init = stablehlo.constant dense<7> : tensor<i8>
  %0 = stablehlo.reduce(%c init: %init) applies stablehlo.add across dimensions = [0] : (tensor<100xi8>, tensor<i8>) -> tensor<i8>
  return %0 : tensor<i8>
}

// 7 + 300 wraps to 51
// CHECK:  func.func @splat_sum_i8() -> tensor<i8> {
// CHECK-NEXT:    %c = stablehlo.constant dense<51> : tensor<i8>
// CHECK-NEXT:    return %c : tensor<i8>
// CHECK-NEXT:  }
