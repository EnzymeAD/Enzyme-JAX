// RUN: enzymexlamlir-opt %s --enzyme --canonicalize --remove-unnecessary-enzyme-ops --arith-raise --enzyme-hlo-opt | FileCheck %s
// RUN: enzymexlamlir-opt %s --enzyme --canonicalize --remove-unnecessary-enzyme-ops --arith-raise --enzyme-hlo-opt | stablehlo-translate - --interpret

module {
  func.func @transpose(%x: tensor<2x3xf32>) -> tensor<3x2xf32> {
    %0 = stablehlo.transpose %x, dims = [1, 0] : (tensor<2x3xf32>) -> tensor<3x2xf32>
    return %0 : tensor<3x2xf32>
  }

  func.func @reshape(%x: tensor<2x3xf32>) -> tensor<6xf32> {
    %0 = stablehlo.reshape %x : (tensor<2x3xf32>) -> tensor<6xf32>
    return %0 : tensor<6xf32>
  }

  func.func @matmul(%x: tensor<2x2xf32>, %y: tensor<2x2xf32>) -> tensor<2x2xf32> {
    %0 = stablehlo.dot_general %x, %y, contracting_dims = [1] x [0] : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    return %0 : tensor<2x2xf32>
  }

  func.func @bcast(%x: tensor<2x1xf32>) -> tensor<2x3xf32> {
    %0 = stablehlo.broadcast_in_dim %x, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x3xf32>
    return %0 : tensor<2x3xf32>
  }

  func.func @concat(%x: tensor<2x2xf32>, %y: tensor<1x2xf32>) -> tensor<3x2xf32> {
    %0 = stablehlo.concatenate %x, %y, dim = 0 : (tensor<2x2xf32>, tensor<1x2xf32>) -> tensor<3x2xf32>
    return %0 : tensor<3x2xf32>
  }

  func.func @main() {
    // 1. Transpose
    %x_t = stablehlo.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf32>
    %dr_t = stablehlo.constant dense<[[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], [[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]]]> : tensor<2x3x2xf32>
    %dx_t = enzyme.autodiff @transpose(%x_t, %dr_t) <{
      activity = [#enzyme.activity<enzyme_active>],
      ret_activity = [#enzyme.activity<enzyme_activenoneed>],
      width = 2 : i64
    }> : (tensor<2x3xf32>, tensor<2x3x2xf32>) -> tensor<2x2x3xf32>
    %expected_dx_t = stablehlo.constant dense<[[[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]], [[10.0, 30.0, 50.0], [20.0, 40.0, 60.0]]]> : tensor<2x2x3xf32>
    check.expect_almost_eq %dx_t, %expected_dx_t : tensor<2x2x3xf32>

    // 2. Reshape
    %dr_r = stablehlo.constant dense<[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [10.0, 20.0, 30.0, 40.0, 50.0, 6.0e+01]]> : tensor<2x6xf32>
    %dx_r = enzyme.autodiff @reshape(%x_t, %dr_r) <{
      activity = [#enzyme.activity<enzyme_active>],
      ret_activity = [#enzyme.activity<enzyme_activenoneed>],
      width = 2 : i64
    }> : (tensor<2x3xf32>, tensor<2x6xf32>) -> tensor<2x2x3xf32>
    %expected_dx_r = stablehlo.constant dense<[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]]]> : tensor<2x2x3xf32>
    check.expect_almost_eq %dx_r, %expected_dx_r : tensor<2x2x3xf32>

    // 3. DotGeneral
    %x_m = stablehlo.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>
    %y_m = stablehlo.constant dense<[[5.0, 6.0], [7.0, 8.0]]> : tensor<2x2xf32>
    %dr_m = stablehlo.constant dense<[[[1.0, 0.0], [0.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]]> : tensor<2x2x2xf32>
    %d_m:2 = enzyme.autodiff @matmul(%x_m, %y_m, %dr_m) <{
      activity = [#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_active>],
      ret_activity = [#enzyme.activity<enzyme_activenoneed>],
      width = 2 : i64
    }> : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2x2xf32>) -> (tensor<2x2x2xf32>, tensor<2x2x2xf32>)
    %expected_dx_m = stablehlo.constant dense<[[[5.0, 7.0], [6.0, 8.0]], [[11.0, 15.0], [11.0, 15.0]]]> : tensor<2x2x2xf32>
    %expected_dy_m = stablehlo.constant dense<[[[1.0, 3.0], [2.0, 4.0]], [[4.0, 4.0], [6.0, 6.0]]]> : tensor<2x2x2xf32>
    check.expect_almost_eq %d_m#0, %expected_dx_m : tensor<2x2x2xf32>
    check.expect_almost_eq %d_m#1, %expected_dy_m : tensor<2x2x2xf32>

    // 4. BroadcastInDim
    %x_b = stablehlo.constant dense<[[1.0], [2.0]]> : tensor<2x1xf32>
    %dr_b = stablehlo.constant dense<[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]]]> : tensor<2x2x3xf32>
    %dx_b = enzyme.autodiff @bcast(%x_b, %dr_b) <{
      activity = [#enzyme.activity<enzyme_active>],
      ret_activity = [#enzyme.activity<enzyme_activenoneed>],
      width = 2 : i64
    }> : (tensor<2x1xf32>, tensor<2x2x3xf32>) -> tensor<2x2x1xf32>
    %expected_dx_b = stablehlo.constant dense<[[[6.0], [15.0]], [[60.0], [150.0]]]> : tensor<2x2x1xf32>
    check.expect_almost_eq %dx_b, %expected_dx_b : tensor<2x2x1xf32>

    // 5. Concatenate
    %x_c = stablehlo.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>
    %y_c = stablehlo.constant dense<[[5.0, 6.0]]> : tensor<1x2xf32>
    %dr_c = stablehlo.constant dense<[[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], [[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]]]> : tensor<2x3x2xf32>
    %d_c:2 = enzyme.autodiff @concat(%x_c, %y_c, %dr_c) <{
      activity = [#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_active>],
      ret_activity = [#enzyme.activity<enzyme_activenoneed>],
      width = 2 : i64
    }> : (tensor<2x2xf32>, tensor<1x2xf32>, tensor<2x3x2xf32>) -> (tensor<2x2x2xf32>, tensor<2x1x2xf32>)
    %expected_dx_c = stablehlo.constant dense<[[[1.0, 2.0], [3.0, 4.0]], [[10.0, 20.0], [30.0, 40.0]]]> : tensor<2x2x2xf32>
    %expected_dy_c = stablehlo.constant dense<[[[5.0, 6.0]], [[50.0, 60.0]]]> : tensor<2x1x2xf32>
    check.expect_almost_eq %d_c#0, %expected_dx_c : tensor<2x2x2xf32>
    check.expect_almost_eq %d_c#1, %expected_dy_c : tensor<2x1x2xf32>

    func.return
  }
}

// CHECK-LABEL: func.func private @diffe2transpose(%arg0: tensor<2x3xf32>, %arg1: tensor<2x3x2xf32>) -> tensor<2x2x3xf32> {
// CHECK-NEXT:    %0 = stablehlo.transpose %arg1, dims = [0, 2, 1] : (tensor<2x3x2xf32>) -> tensor<2x2x3xf32>
// CHECK-NEXT:    return %0 : tensor<2x2x3xf32>
// CHECK-NEXT:  }

// CHECK-LABEL: func.func private @diffe2reshape(%arg0: tensor<2x3xf32>, %arg1: tensor<2x6xf32>) -> tensor<2x2x3xf32> {
// CHECK-NEXT:    %0 = stablehlo.reshape %arg1 : (tensor<2x6xf32>) -> tensor<2x2x3xf32>
// CHECK-NEXT:    return %0 : tensor<2x2x3xf32>
// CHECK-NEXT:  }

// CHECK-LABEL: func.func private @diffe2matmul(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>, %arg2: tensor<2x2x2xf32>) -> (tensor<2x2x2xf32>, tensor<2x2x2xf32>) {
// CHECK-NEXT:    %0 = stablehlo.dot_general %arg2, %arg1, contracting_dims = [2] x [1] : (tensor<2x2x2xf32>, tensor<2x2xf32>) -> tensor<2x2x2xf32>
// CHECK-NEXT:    %1 = stablehlo.dot_general %arg0, %arg2, contracting_dims = [0] x [1] : (tensor<2x2xf32>, tensor<2x2x2xf32>) -> tensor<2x2x2xf32>
// CHECK-NEXT:    %2 = stablehlo.transpose %1, dims = [1, 0, 2] : (tensor<2x2x2xf32>) -> tensor<2x2x2xf32>
// CHECK-NEXT:    return %0, %2 : tensor<2x2x2xf32>, tensor<2x2x2xf32>
// CHECK-NEXT:  }

// CHECK-LABEL: func.func private @diffe2bcast(%arg0: tensor<2x1xf32>, %arg1: tensor<2x2x3xf32>) -> tensor<2x2x1xf32> {
// CHECK-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:    %0 = stablehlo.reduce(%arg1 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<2x2x3xf32>, tensor<f32>) -> tensor<2x2xf32>
// CHECK-NEXT:    %1 = stablehlo.reshape %0 : (tensor<2x2xf32>) -> tensor<2x2x1xf32>
// CHECK-NEXT:    return %1 : tensor<2x2x1xf32>
// CHECK-NEXT:  }

// CHECK-LABEL: func.func private @diffe2concat(%arg0: tensor<2x2xf32>, %arg1: tensor<1x2xf32>, %arg2: tensor<2x3x2xf32>) -> (tensor<2x2x2xf32>, tensor<2x1x2xf32>) {
// CHECK-NEXT:    %0 = stablehlo.slice %arg2 [0:2, 0:2, 0:2] : (tensor<2x3x2xf32>) -> tensor<2x2x2xf32>
// CHECK-NEXT:    %1 = stablehlo.slice %arg2 [0:2, 2:3, 0:2] : (tensor<2x3x2xf32>) -> tensor<2x1x2xf32>
// CHECK-NEXT:    return %0, %1 : tensor<2x2x2xf32>, tensor<2x1x2xf32>
// CHECK-NEXT:  }
