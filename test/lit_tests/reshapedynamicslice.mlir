// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=reshape_dynamic_slice(1)" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

module {
  // Test case where reshape adds a unit dimension at the beginning.

  func.func @reshape_slice_add_unit_dim_front(%arg0: tensor<300x2100xf64>) -> tensor<1x256x2048xf64> {
    %c6 = stablehlo.constant dense<6> : tensor<i64>
    %c5 = stablehlo.constant dense<5> : tensor<i64>
    %0 = stablehlo.dynamic_slice %arg0, %c6, %c5, sizes = [256, 2048] : (tensor<300x2100xf64>, tensor<i64>, tensor<i64>) -> tensor<256x2048xf64>
    %1 = stablehlo.reshape %0 : (tensor<256x2048xf64>) -> tensor<1x256x2048xf64>
    return %1 : tensor<1x256x2048xf64>
  }

  // CHECK: func.func @reshape_slice_add_unit_dim_front(%arg0: tensor<300x2100xf64>) -> tensor<1x256x2048xf64> {
  // CHECK-NEXT:   %c = stablehlo.constant dense<6> : tensor<i64>
  // CHECK-NEXT:   %c_0 = stablehlo.constant dense<5> : tensor<i64>
  // CHECK-NEXT:   %c_1 = stablehlo.constant dense<0> : tensor<i64>
  // CHECK-NEXT:   %0 = stablehlo.reshape %arg0 : (tensor<300x2100xf64>) -> tensor<1x300x2100xf64>
  // CHECK-NEXT:   %1 = stablehlo.dynamic_slice %0, %c_1, %c, %c_0, sizes = [1, 256, 2048] : (tensor<1x300x2100xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x256x2048xf64>
  // CHECK-NEXT:   return %1 : tensor<1x256x2048xf64>
  // CHECK-NEXT: }

  // Test case where reshape adds a unit dimension in the middle.

  func.func @reshape_slice_add_unit_dim_middle(%arg0: tensor<300x2100xf64>) -> tensor<256x1x2048xf64> {
    %c6 = stablehlo.constant dense<6> : tensor<i64>
    %c5 = stablehlo.constant dense<5> : tensor<i64>
    %0 = stablehlo.dynamic_slice %arg0, %c6, %c5, sizes = [256, 2048] : (tensor<300x2100xf64>, tensor<i64>, tensor<i64>) -> tensor<256x2048xf64>
    %1 = stablehlo.reshape %0 : (tensor<256x2048xf64>) -> tensor<256x1x2048xf64>
    return %1 : tensor<256x1x2048xf64>
  }

  // CHECK: func.func @reshape_slice_add_unit_dim_middle(%arg0: tensor<300x2100xf64>) -> tensor<256x1x2048xf64> {
  // CHECK-NEXT:   %c = stablehlo.constant dense<6> : tensor<i64>
  // CHECK-NEXT:   %c_0 = stablehlo.constant dense<5> : tensor<i64>
  // CHECK-NEXT:   %c_1 = stablehlo.constant dense<0> : tensor<i64>
  // CHECK-NEXT:   %0 = stablehlo.reshape %arg0 : (tensor<300x2100xf64>) -> tensor<300x1x2100xf64>
  // CHECK-NEXT:   %1 = stablehlo.dynamic_slice %0, %c, %c_1, %c_0, sizes = [256, 1, 2048] : (tensor<300x1x2100xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<256x1x2048xf64>
  // CHECK-NEXT:   return %1 : tensor<256x1x2048xf64>
  // CHECK-NEXT: }

  // Test case where reshape adds a unit dimension at the end.

  func.func @reshape_slice_add_unit_dim_end(%arg0: tensor<300x2100xf64>) -> tensor<256x2048x1xf64> {
    %c6 = stablehlo.constant dense<6> : tensor<i64>
    %c5 = stablehlo.constant dense<5> : tensor<i64>
    %0 = stablehlo.dynamic_slice %arg0, %c6, %c5, sizes = [256, 2048] : (tensor<300x2100xf64>, tensor<i64>, tensor<i64>) -> tensor<256x2048xf64>
    %1 = stablehlo.reshape %0 : (tensor<256x2048xf64>) -> tensor<256x2048x1xf64>
    return %1 : tensor<256x2048x1xf64>
  }

  // CHECK: func.func @reshape_slice_add_unit_dim_end(%arg0: tensor<300x2100xf64>) -> tensor<256x2048x1xf64> {
  // CHECK-NEXT:   %c = stablehlo.constant dense<6> : tensor<i64>
  // CHECK-NEXT:   %c_0 = stablehlo.constant dense<5> : tensor<i64>
  // CHECK-NEXT:   %c_1 = stablehlo.constant dense<0> : tensor<i64>
  // CHECK-NEXT:   %0 = stablehlo.reshape %arg0 : (tensor<300x2100xf64>) -> tensor<300x2100x1xf64>
  // CHECK-NEXT:   %1 = stablehlo.dynamic_slice %0, %c, %c_0, %c_1, sizes = [256, 2048, 1] : (tensor<300x2100x1xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<256x2048x1xf64>
  // CHECK-NEXT:   return %1 : tensor<256x2048x1xf64>
  // CHECK-NEXT: }

  // Test case where reshape removes one unit dimension and adds two others.

  func.func @reshape_slice_remove_add_unit_dims(%arg0: tensor<1x300x1x2100x1xf64>) -> tensor<1x256x2048x1x1x1xf64> {
    %c0 = stablehlo.constant dense<0> : tensor<i64>
    %c6 = stablehlo.constant dense<6> : tensor<i64>
    %c5 = stablehlo.constant dense<5> : tensor<i64>
    // Slice retains rank 5, modifying non-unit dims
    %0 = stablehlo.dynamic_slice %arg0, %c0, %c6, %c0, %c5, %c0, sizes = [1, 256, 1, 2048, 1] : (tensor<1x300x1x2100x1xf64>, tensor<i64>, tensor<i64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x256x1x2048x1xf64>
    // Reshape removes dim 2 (size 1) and adds two unit dims at the end (rank 5 -> rank 6)
    %1 = stablehlo.reshape %0 : (tensor<1x256x1x2048x1xf64>) -> tensor<1x256x2048x1x1x1xf64>
    return %1 : tensor<1x256x2048x1x1x1xf64>
  }

  // CHECK: func.func @reshape_slice_remove_add_unit_dims(%arg0: tensor<1x300x1x2100x1xf64>) -> tensor<1x256x2048x1x1x1xf64> {
  // CHECK-NEXT:   %c = stablehlo.constant dense<0> : tensor<i64>
  // CHECK-NEXT:   %c_0 = stablehlo.constant dense<6> : tensor<i64>
  // CHECK-NEXT:   %c_1 = stablehlo.constant dense<5> : tensor<i64>
  // CHECK-NEXT:   %0 = stablehlo.reshape %arg0 : (tensor<1x300x1x2100x1xf64>) -> tensor<1x300x2100x1x1x1xf64>
  // CHECK-NEXT:   %1 = stablehlo.dynamic_slice %0, %c, %c_0, %c_1, %c, %c, %c, sizes = [1, 256, 2048, 1, 1, 1] : (tensor<1x300x2100x1x1x1xf64>, tensor<i64>, tensor<i64>, tensor<i64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x256x2048x1x1x1xf64>
  // CHECK-NEXT:   return %1 : tensor<1x256x2048x1x1x1xf64>
  // CHECK-NEXT: }

  // Test that no transformation is applied when the slice reduces a dimension to one itself.

  func.func @reshape_slice_no_transformation(%arg0: tensor<142x271x2062xf64>) -> tensor<268x2060xf64> {
    %c8 = stablehlo.constant dense<8> : tensor<i64>
    %c2 = stablehlo.constant dense<2> : tensor<i64>
    %c1 = stablehlo.constant dense<1> : tensor<i64>
    %139 = stablehlo.dynamic_slice %arg0, %c8, %c2, %c1, sizes = [1, 268, 2060] {mhlo.sharding = "{devices=[1,2,2,2]<=[2,2,2]T(1,0,2) last_tile_dim_replicate}"} : (tensor<142x271x2062xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x268x2060xf64>
    %140 = stablehlo.reshape %139 {mhlo.sharding = "{devices=[2,2,2]<=[2,2,2]T(1,0,2) last_tile_dim_replicate}"} : (tensor<1x268x2060xf64>) -> tensor<268x2060xf64>
    return %140 : tensor<268x2060xf64>
  }

  // CHECK: func.func @reshape_slice_no_transformation(%arg0: tensor<142x271x2062xf64>) -> tensor<268x2060xf64> {
  // CHECK-NEXT:   %c = stablehlo.constant dense<8> : tensor<i64>
  // CHECK-NEXT:   %c_0 = stablehlo.constant dense<2> : tensor<i64>
  // CHECK-NEXT:   %c_1 = stablehlo.constant dense<1> : tensor<i64>
  // CHECK-NEXT:   %0 = stablehlo.dynamic_slice %arg0, %c, %c_0, %c_1, sizes = [1, 268, 2060] {mhlo.sharding = "{devices=[1,2,2,2]<=[2,2,2]T(1,0,2) last_tile_dim_replicate}"} : (tensor<142x271x2062xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x268x2060xf64>
  // CHECK-NEXT:   %1 = stablehlo.reshape %0 {mhlo.sharding = "{devices=[2,2,2]<=[2,2,2]T(1,0,2) last_tile_dim_replicate}"} : (tensor<1x268x2060xf64>) -> tensor<268x2060xf64>
  // CHECK-NEXT:   return %1 : tensor<268x2060xf64>
  // CHECK-NEXT: }

  func.func @reshape_slice_remove_first(%arg0: tensor<1x2048x2048xf64>) -> tensor<1x2032xf64> {
    %c0 = stablehlo.constant dense<0> : tensor<i64>
    %c8 = stablehlo.constant dense<8> : tensor<i64>
    %0 = stablehlo.dynamic_slice %arg0, %c0, %c8, %c8, sizes = [1, 1, 2032] : (tensor<1x2048x2048xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x1x2032xf64>
    %86 = stablehlo.reshape %0 : (tensor<1x1x2032xf64>) -> tensor<1x2032xf64>
    return %86 : tensor<1x2032xf64>
  }

  // CHECK: func.func @reshape_slice_remove_first(%arg0: tensor<1x2048x2048xf64>) -> tensor<1x2032xf64> {
  // CHECK-NEXT:   %c = stablehlo.constant dense<8> : tensor<i64>
  // CHECK-NEXT:   %0 = stablehlo.reshape %arg0 : (tensor<1x2048x2048xf64>) -> tensor<2048x2048xf64>
  // CHECK-NEXT:   %1 = stablehlo.dynamic_slice %0, %c, %c, sizes = [1, 2032] : (tensor<2048x2048xf64>, tensor<i64>, tensor<i64>) -> tensor<1x2032xf64>
  // CHECK-NEXT:   return %1 : tensor<1x2032xf64>
  // CHECK-NEXT: }

}

module {
  func.func @main(%arg0: tensor<5x4x3xf32>, %arg1: tensor<3x1x4x1x5xf32>) -> tensor<5x4x3xf32> {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %c_0 = stablehlo.constant dense<1> : tensor<i32>
    %c_1 = stablehlo.constant dense<0> : tensor<i64>
    %c_2 = stablehlo.constant dense<1> : tensor<i64>
    %c_3 = stablehlo.constant dense<4> : tensor<i64>
    %0 = stablehlo.transpose %arg1, dims = [4, 1, 2, 3, 0] : (tensor<3x1x4x1x5xf32>) -> tensor<5x1x4x1x3xf32>
    %1:2 = stablehlo.while(%iterArg = %c_1, %iterArg_3 = %arg0) : tensor<i64>, tensor<5x4x3xf32>
    cond {
      %2 = stablehlo.compare  LT, %iterArg, %c_3 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %2 : tensor<i1>
    } do {
      %2 = stablehlo.add %c_2, %iterArg : tensor<i64>
      %3 = stablehlo.convert %2 : (tensor<i64>) -> tensor<i32>
      %4 = stablehlo.subtract %3, %c_0 : tensor<i32>
      %5 = stablehlo.dynamic_slice %0, %c, %c, %4, %c, %c, sizes = [5, 1, 1, 1, 3] : (tensor<5x1x4x1x3xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<5x1x1x1x3xf32>
      %6 = stablehlo.reshape %5 : (tensor<5x1x1x1x3xf32>) -> tensor<5x1x3xf32>
      %7 = stablehlo.dynamic_update_slice %iterArg_3, %6, %c, %4, %c : (tensor<5x4x3xf32>, tensor<5x1x3xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<5x4x3xf32>
      // CHECK: %5 = stablehlo.reshape %0 : (tensor<5x1x4x1x3xf32>) -> tensor<5x4x3xf32>
      // CHECK-NEXT: %6 = stablehlo.dynamic_slice %5, %c, %4, %c, sizes = [5, 1, 3] : (tensor<5x4x3xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<5x1x3xf32>
      // CHECK-NEXT: %7 = stablehlo.dynamic_update_slice %iterArg_4, %6, %c, %4, %c : (tensor<5x4x3xf32>, tensor<5x1x3xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<5x4x3xf32>
      stablehlo.return %2, %7 : tensor<i64>, tensor<5x4x3xf32>
    }
    return %1#1 : tensor<5x4x3xf32>
  }
}
