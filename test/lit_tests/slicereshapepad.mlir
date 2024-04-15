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

func.func @r12(%4: tensor<1x30x4x42xf32>, %pv : tensor<f32>) -> (tensor<1x1x20x4x42xf32>, tensor<1x1x20x4x42xf32>) {
  %5 = stablehlo.pad %4, %pv, low = [0, 0, 0, 0], high = [0, 70, 0, 0], interior = [0, 0, 0, 0] : (tensor<1x30x4x42xf32>, tensor<f32>) -> tensor<1x100x4x42xf32>
  %rs = stablehlo.reshape %5 : (tensor<1x100x4x42xf32>) -> tensor<1x1x100x4x42xf32> 
  %add = stablehlo.slice %rs [0:1, 0:1, 0:20, 0:4, 0:42] : (tensor<1x1x100x4x42xf32>) -> tensor<1x1x20x4x42xf32> 
  %add2 = stablehlo.slice %rs [0:1, 0:1, 60:80, 0:4, 0:42] : (tensor<1x1x100x4x42xf32>) -> tensor<1x1x20x4x42xf32> 
  return %add, %add2 : tensor<1x1x20x4x42xf32>, tensor<1x1x20x4x42xf32>
}

func.func @r3(%4: tensor<1x30x4x42xf32>, %pv : tensor<f32>) -> tensor<1x1x20x4x42xf32> {
  %5 = stablehlo.pad %4, %pv, low = [0, 0, 0, 0], high = [0, 70, 0, 0], interior = [0, 0, 0, 0] : (tensor<1x30x4x42xf32>, tensor<f32>) -> tensor<1x100x4x42xf32>
  %rs = stablehlo.reshape %5 : (tensor<1x100x4x42xf32>) -> tensor<1x1x100x4x42xf32> 
  %add = stablehlo.slice %rs [0:1, 0:1, 20:40, 0:4, 0:42] : (tensor<1x1x100x4x42xf32>) -> tensor<1x1x20x4x42xf32> 
  return %add : tensor<1x1x20x4x42xf32>
}

func.func @r4(%1069: tensor<1x3072x4x256xbf16>, %pv : tensor<bf16>) -> tensor<1x1x2048x4x256xbf16> {
    %1070 = stablehlo.pad %1069, %pv, low = [0, 0, 0, 0], high = [0, 5120, 0, 0], interior = [0, 0, 0, 0] : (tensor<1x3072x4x256xbf16>, tensor<bf16>) -> tensor<1x8192x4x256xbf16>
    %1071 = stablehlo.reshape %1070 : (tensor<1x8192x4x256xbf16>) -> tensor<1x1x8192x4x256xbf16> 
    %1130 = stablehlo.slice %1071 [0:1, 0:1, 2048:4096, 0:4, 0:256] : (tensor<1x1x8192x4x256xbf16>) -> tensor<1x1x2048x4x256xbf16>
    return %1130 : tensor<1x1x2048x4x256xbf16>
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

// CHECK:  func.func @r12(%arg0: tensor<1x30x4x42xf32>, %arg1: tensor<f32>) -> (tensor<1x1x20x4x42xf32>, tensor<1x1x20x4x42xf32>) {
// CHECK-NEXT:    %0 = stablehlo.slice %arg0 [0:1, 0:20, 0:4, 0:42] : (tensor<1x30x4x42xf32>) -> tensor<1x20x4x42xf32>
// CHECK-NEXT:    %1 = stablehlo.reshape %0 : (tensor<1x20x4x42xf32>) -> tensor<1x1x20x4x42xf32>
// CHECK-NEXT:    %2 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<f32>) -> tensor<1x1x20x4x42xf32>
// CHECK-NEXT:    return %1, %2 : tensor<1x1x20x4x42xf32>, tensor<1x1x20x4x42xf32>
// CHECK-NEXT:  }

// CHECK:  func.func @r3(%arg0: tensor<1x30x4x42xf32>, %arg1: tensor<f32>) -> tensor<1x1x20x4x42xf32> {
// CHECK-NEXT:    %0 = stablehlo.slice %arg0 [0:1, 20:30, 0:4, 0:42] : (tensor<1x30x4x42xf32>) -> tensor<1x10x4x42xf32>
// CHECK-NEXT:    %1 = stablehlo.pad %0, %arg1, low = [0, 0, 0, 0], high = [0, 10, 0, 0], interior = [0, 0, 0, 0] : (tensor<1x10x4x42xf32>, tensor<f32>) -> tensor<1x20x4x42xf32>
// CHECK-NEXT:    %2 = stablehlo.reshape %1 : (tensor<1x20x4x42xf32>) -> tensor<1x1x20x4x42xf32>
// CHECK-NEXT:    return %2 : tensor<1x1x20x4x42xf32>
// CHECK-NEXT:  }
// CHECK:  func.func @r4(%arg0: tensor<1x3072x4x256xbf16>, %arg1: tensor<bf16>) -> tensor<1x1x2048x4x256xbf16> {
// CHECK-NEXT:    %0 = stablehlo.slice %arg0 [0:1, 2048:3072, 0:4, 0:256] : (tensor<1x3072x4x256xbf16>) -> tensor<1x1024x4x256xbf16>
// CHECK-NEXT:    %1 = stablehlo.pad %0, %arg1, low = [0, 0, 0, 0], high = [0, 1024, 0, 0], interior = [0, 0, 0, 0] : (tensor<1x1024x4x256xbf16>, tensor<bf16>) -> tensor<1x2048x4x256xbf16>
// CHECK-NEXT:    %2 = stablehlo.reshape %1 : (tensor<1x2048x4x256xbf16>) -> tensor<1x1x2048x4x256xbf16>
// CHECK-NEXT:    return %2 : tensor<1x1x2048x4x256xbf16>
// CHECK-NEXT:  }
