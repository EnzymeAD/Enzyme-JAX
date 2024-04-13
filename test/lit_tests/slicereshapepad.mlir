// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=slice_reshape_pad" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s


func.func @r(%4: tensor<1x30x4x42xf32>, %pv : tensor<f32>) -> tensor<1x1x20x4x42xf32> {
  %5 = stablehlo.pad %4, %pv, low = [0, 0, 0, 0], high = [0, 70, 0, 0], interior = [0, 0, 0, 0] : (tensor<1x30x4x42xf32>, tensor<f32>) -> tensor<1x100x4x42xf32>
  %rs = stablehlo.reshape %5 : (tensor<1x100x4x42xf32>) -> tensor<1x1x100x4x42xf32> 
  %add = stablehlo.slice %rs [0:1, 0:1, 60:80, 0:4, 0:42] : (tensor<1x1x100x4x42xf32>) -> tensor<1x1x20x4x42xf32> 
  return %add : tensor<1x1x20x4x42xf32>
}

func.func @r2(%4: tensor<1x30x4x42xf32>, %pv : tensor<f32>) -> tensor<1x1x20x4x42xf32> {
  %5 = stablehlo.pad %4, %pv, low = [0, 0, 0, 0], high = [0, 70, 0, 0], interior = [0, 0, 0, 0] : (tensor<1x30x4x42xf32>, tensor<f32>) -> tensor<1x100x4x42xf32>
  %rs = stablehlo.reshape %5 : (tensor<1x100x4x42xf32>) -> tensor<1x1x100x4x42xf32> 
  %add = stablehlo.slice %rs [0:1, 0:1, 0:20, 0:4, 0:42] : (tensor<1x1x100x4x42xf32>) -> tensor<1x1x20x4x42xf32> 
  return %add : tensor<1x1x20x4x42xf32>
}

func.func @r3(%4: tensor<1x30x4x42xf32>, %pv : tensor<f32>) -> tensor<1x1x20x4x42xf32> {
  %5 = stablehlo.pad %4, %pv, low = [0, 0, 0, 0], high = [0, 70, 0, 0], interior = [0, 0, 0, 0] : (tensor<1x30x4x42xf32>, tensor<f32>) -> tensor<1x100x4x42xf32>
  %rs = stablehlo.reshape %5 : (tensor<1x100x4x42xf32>) -> tensor<1x1x100x4x42xf32> 
  %add = stablehlo.slice %rs [0:1, 0:1, 20:40, 0:4, 0:42] : (tensor<1x1x100x4x42xf32>) -> tensor<1x1x20x4x42xf32> 
  return %add : tensor<1x1x20x4x42xf32>
}

// CHECK:  func.func @r(%arg0: tensor<1x30x4x42xf32>, %arg1: tensor<f32>) -> tensor<1x1x20x4x42xf32> {
// CHECK-NEXT:    %0 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<f32>) -> tensor<1x1x20x4x42xf32>
// CHECK-NEXT:    return %0 : tensor<1x1x20x4x42xf32>
// CHECK-NEXT:  }
// CHECK:  func.func @r2(%arg0: tensor<1x30x4x42xf32>, %arg1: tensor<f32>) -> tensor<1x1x20x4x42xf32> {
// CHECK-NEXT:    %0 = stablehlo.slice %arg0 [0:1, 0:20, 0:4, 0:42] : (tensor<1x30x4x42xf32>) -> tensor<1x20x4x42xf32>
// CHECK-NEXT:    %1 = stablehlo.reshape %0 : (tensor<1x20x4x42xf32>) -> tensor<1x1x20x4x42xf32>
// CHECK-NEXT:    return %1 : tensor<1x1x20x4x42xf32>
// CHECK-NEXT:  }
// CHECK:  func.func @r3(%arg0: tensor<1x30x4x42xf32>, %arg1: tensor<f32>) -> tensor<1x1x20x4x42xf32> {
// CHECK-NEXT:    %0 = stablehlo.pad %arg0, %arg1, low = [0, 0, 0, 0], high = [0, 70, 0, 0], interior = [0, 0, 0, 0] : (tensor<1x30x4x42xf32>, tensor<f32>) -> tensor<1x100x4x42xf32>
// CHECK-NEXT:    %1 = stablehlo.reshape %0 : (tensor<1x100x4x42xf32>) -> tensor<1x1x100x4x42xf32>
// CHECK-NEXT:    %2 = stablehlo.slice %1 [0:1, 0:1, 20:40, 0:4, 0:42] : (tensor<1x1x100x4x42xf32>) -> tensor<1x1x20x4x42xf32>
// CHECK-NEXT:    return %2 : tensor<1x1x20x4x42xf32>
// CHECK-NEXT:  }
