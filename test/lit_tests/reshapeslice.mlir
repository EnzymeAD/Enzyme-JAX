// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=reshape_slice(1)" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

module {
  // Test case where reshape adds a unit dimension at the beginning.

  // CHECK-LABEL: func.func @reshape_slice_add_unit_dim_front(%arg0: tensor<300x2100xf64>) -> tensor<1x256x2048xf64> {
  // CHECK-NEXT:    %0 = stablehlo.reshape %arg0 : (tensor<300x2100xf64>) -> tensor<1x300x2100xf64>
  // CHECK-NEXT:    %1 = stablehlo.slice %0 [0:1, 6:262, 5:2053] : (tensor<1x300x2100xf64>) -> tensor<1x256x2048xf64>
  // CHECK-NEXT:    return %1 : tensor<1x256x2048xf64>
  // CHECK-NEXT:  }
  func.func @reshape_slice_add_unit_dim_front(%arg0: tensor<300x2100xf64>) -> tensor<1x256x2048xf64> {
    %0 = stablehlo.slice %arg0 [6:262, 5:2053] : (tensor<300x2100xf64>) -> tensor<256x2048xf64>
    %1 = stablehlo.reshape %0 : (tensor<256x2048xf64>) -> tensor<1x256x2048xf64>
    return %1 : tensor<1x256x2048xf64>
  }

  // Test case where reshape adds a unit dimension in the middle.

  // CHECK-LABEL: func.func @reshape_slice_add_unit_dim_middle(%arg0: tensor<300x2100xf64>) -> tensor<256x1x2048xf64> {
  // CHECK-NEXT:    %0 = stablehlo.reshape %arg0 : (tensor<300x2100xf64>) -> tensor<300x1x2100xf64>
  // CHECK-NEXT:    %1 = stablehlo.slice %0 [6:262, 0:1, 5:2053] : (tensor<300x1x2100xf64>) -> tensor<256x1x2048xf64>
  // CHECK-NEXT:    return %1 : tensor<256x1x2048xf64>
  // CHECK-NEXT:  }
  func.func @reshape_slice_add_unit_dim_middle(%arg0: tensor<300x2100xf64>) -> tensor<256x1x2048xf64> {
    %0 = stablehlo.slice %arg0 [6:262, 5:2053] : (tensor<300x2100xf64>) -> tensor<256x2048xf64>
    %1 = stablehlo.reshape %0 : (tensor<256x2048xf64>) -> tensor<256x1x2048xf64>
    return %1 : tensor<256x1x2048xf64>
  }


  // Test case where reshape adds a unit dimension at the end.

  // CHECK-LABEL: func.func @reshape_slice_add_unit_dim_end(%arg0: tensor<300x2100xf64>) -> tensor<256x2048x1xf64> {
  // CHECK-NEXT:    %0 = stablehlo.reshape %arg0 : (tensor<300x2100xf64>) -> tensor<300x2100x1xf64>
  // CHECK-NEXT:    %1 = stablehlo.slice %0 [6:262, 5:2053, 0:1] : (tensor<300x2100x1xf64>) -> tensor<256x2048x1xf64>
  // CHECK-NEXT:    return %1 : tensor<256x2048x1xf64>
  // CHECK-NEXT:  }
  func.func @reshape_slice_add_unit_dim_end(%arg0: tensor<300x2100xf64>) -> tensor<256x2048x1xf64> {
    %0 = stablehlo.slice %arg0 [6:262, 5:2053] : (tensor<300x2100xf64>) -> tensor<256x2048xf64>
    %1 = stablehlo.reshape %0 : (tensor<256x2048xf64>) -> tensor<256x2048x1xf64>
    return %1 : tensor<256x2048x1xf64>
  }

  // Test case where reshape removes one unit dimension and adds two others.
  // CHECK-LABEL: func.func @reshape_slice_remove_add_unit_dims(%arg0: tensor<1x300x1x2100x1xf64>) -> tensor<1x256x2048x1x1x1xf64> {
  // CHECK-NEXT:    %0 = stablehlo.reshape %arg0 : (tensor<1x300x1x2100x1xf64>) -> tensor<1x300x2100x1x1x1xf64>
  // CHECK-NEXT:    %1 = stablehlo.slice %0 [0:1, 6:262, 5:2053, 0:1, 0:1, 0:1] : (tensor<1x300x2100x1x1x1xf64>) -> tensor<1x256x2048x1x1x1xf64>
  // CHECK-NEXT:    return %1 : tensor<1x256x2048x1x1x1xf64>
  // CHECK-NEXT:  }

  func.func @reshape_slice_remove_add_unit_dims(%arg0: tensor<1x300x1x2100x1xf64>) -> tensor<1x256x2048x1x1x1xf64> {
    // Slice retains rank 5, modifying non-unit dims
    %0 = stablehlo.slice %arg0 [0:1, 6:262, 0:1, 5:2053, 0:1] : (tensor<1x300x1x2100x1xf64>) -> tensor<1x256x1x2048x1xf64>
    // Reshape removes dim 2 (size 1) and adds two unit dims at the end (rank 5 -> rank 6)
    %1 = stablehlo.reshape %0 : (tensor<1x256x1x2048x1xf64>) -> tensor<1x256x2048x1x1x1xf64>
    return %1 : tensor<1x256x2048x1x1x1xf64>
  }


  // Test that no transformation is applied when the slice reduces a dimension to one itself.

  // CHECK-LABEL: func.func @reshape_slice_no_transformation(%arg0: tensor<142x271x2062xf64>) -> tensor<268x2060xf64> {
  // CHECK-NEXT:    %0 = stablehlo.slice %arg0 [8:9, 2:270, 1:2061] {mhlo.sharding = "{devices=[1,2,2,2]<=[2,2,2]T(1,0,2) last_tile_dim_replicate}"} : (tensor<142x271x2062xf64>) -> tensor<1x268x2060xf64>
  // CHECK-NEXT:    %1 = stablehlo.reshape %0 {mhlo.sharding = "{devices=[2,2,2]<=[2,2,2]T(1,0,2) last_tile_dim_replicate}"} : (tensor<1x268x2060xf64>) -> tensor<268x2060xf64>
  // CHECK-NEXT:    return %1 : tensor<268x2060xf64>
  // CHECK-NEXT:  }

  func.func @reshape_slice_no_transformation(%arg0: tensor<142x271x2062xf64>) -> tensor<268x2060xf64> {
    %139 = stablehlo.slice %arg0 [8:9, 2:270, 1:2061] {mhlo.sharding = "{devices=[1,2,2,2]<=[2,2,2]T(1,0,2) last_tile_dim_replicate}"} : (tensor<142x271x2062xf64>) -> tensor<1x268x2060xf64>
    %140 = stablehlo.reshape %139 {mhlo.sharding = "{devices=[2,2,2]<=[2,2,2]T(1,0,2) last_tile_dim_replicate}"} : (tensor<1x268x2060xf64>) -> tensor<268x2060xf64>
    return %140 : tensor<268x2060xf64>
  }

}
