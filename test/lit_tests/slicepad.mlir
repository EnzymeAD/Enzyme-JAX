// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

func.func @t1(%4: tensor<100xf32>, %pv : tensor<f32>) -> tensor<100xf32> {
  %5 = stablehlo.pad %4, %pv, low = [100], high = [100], interior = [0] : (tensor<100xf32>, tensor<f32>) -> tensor<300xf32>
  %add = stablehlo.slice %5 [0:100] : (tensor<300xf32>) ->tensor<100xf32>
  return %add : tensor<100xf32>
}

func.func @t2(%4: tensor<100xf32>, %pv : tensor<f32>) -> tensor<100xf32> {
  %5 = stablehlo.pad %4, %pv, low = [100], high = [100], interior = [0] : (tensor<100xf32>, tensor<f32>) -> tensor<300xf32>
  %add = stablehlo.slice %5 [50:150] : (tensor<300xf32>) ->tensor<100xf32>
  return %add : tensor<100xf32>
}

func.func @t3(%4: tensor<100xf32>, %pv : tensor<f32>) -> tensor<100xf32> {
  %5 = stablehlo.pad %4, %pv, low = [100], high = [100], interior = [0] : (tensor<100xf32>, tensor<f32>) -> tensor<300xf32>
  %add = stablehlo.slice %5 [100:200] : (tensor<300xf32>) ->tensor<100xf32>
  return %add : tensor<100xf32>
}

func.func @t4(%4: tensor<100xf32>, %pv : tensor<f32>) -> tensor<100xf32> {
  %5 = stablehlo.pad %4, %pv, low = [100], high = [100], interior = [0] : (tensor<100xf32>, tensor<f32>) -> tensor<300xf32>
  %add = stablehlo.slice %5 [150:250] : (tensor<300xf32>) ->tensor<100xf32>
  return %add : tensor<100xf32>
}

func.func @t5(%4: tensor<100xf32>, %pv : tensor<f32>) -> tensor<100xf32> {
  %5 = stablehlo.pad %4, %pv, low = [100], high = [100], interior = [0] : (tensor<100xf32>, tensor<f32>) -> tensor<300xf32>
  %add = stablehlo.slice %5 [200:300] : (tensor<300xf32>) ->tensor<100xf32>
  return %add : tensor<100xf32>
}

// CHECK:  func.func @t1(%arg0: tensor<100xf32>, %arg1: tensor<f32>) -> tensor<100xf32> {
// CHECK-NEXT:    %0 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<f32>) -> tensor<100xf32>
// CHECK-NEXT:    return %0 : tensor<100xf32>
// CHECK-NEXT:  }
// CHECK:  func.func @t2(%arg0: tensor<100xf32>, %arg1: tensor<f32>) -> tensor<100xf32> {
// CHECK-NEXT:    %0 = stablehlo.slice %arg0 [0:50] : (tensor<100xf32>) -> tensor<50xf32>
// CHECK-NEXT:    %1 = stablehlo.pad %0, %arg1, low = [50], high = [0], interior = [0] : (tensor<50xf32>, tensor<f32>) -> tensor<100xf32>
// CHECK-NEXT:    return %1 : tensor<100xf32>
// CHECK-NEXT:  }
// CHECK:  func.func @t3(%arg0: tensor<100xf32>, %arg1: tensor<f32>) -> tensor<100xf32> {
// CHECK-NEXT:    return %arg0 : tensor<100xf32>
// CHECK-NEXT:  }
// CHECK:  func.func @t4(%arg0: tensor<100xf32>, %arg1: tensor<f32>) -> tensor<100xf32> {
// CHECK-NEXT:    %0 = stablehlo.slice %arg0 [50:100] : (tensor<100xf32>) -> tensor<50xf32>
// CHECK-NEXT:    %1 = stablehlo.pad %0, %arg1, low = [0], high = [50], interior = [0] : (tensor<50xf32>, tensor<f32>) -> tensor<100xf32>
// CHECK-NEXT:    return %1 : tensor<100xf32>
// CHECK-NEXT:  }
// CHECK:  func.func @t5(%arg0: tensor<100xf32>, %arg1: tensor<f32>) -> tensor<100xf32> {
// CHECK-NEXT:    %0 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<f32>) -> tensor<100xf32>
// CHECK-NEXT:    return %0 : tensor<100xf32>
// CHECK-NEXT:  }
