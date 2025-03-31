// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=reshape_elementwise" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

module {
  func.func @main(%a : tensor<100x200x300xbf16>, %b: tensor<100x200x300xbf16>) -> tensor<20000x300xbf16> {
    %1909 = stablehlo.subtract %a, %b : tensor<100x200x300xbf16>
    %1910 = stablehlo.reshape %1909 : (tensor<100x200x300xbf16>) -> tensor<20000x300xbf16> 
    return %1910 : tensor<20000x300xbf16> 
  }
  func.func @main2(%a : tensor<100x200x300xbf16> ) -> tensor<20000x300xf32> {
    %1909 = stablehlo.convert %a : (tensor<100x200x300xbf16>) -> tensor<100x200x300xf32>
    %1910 = stablehlo.reshape %1909 : (tensor<100x200x300xf32>) -> tensor<20000x300xf32>
    return %1910 : tensor<20000x300xf32>
  }
}

// CHECK:  func.func @main(%arg0: tensor<100x200x300xbf16>, %arg1: tensor<100x200x300xbf16>) -> tensor<20000x300xbf16> {
// CHECK-NEXT:    %0 = stablehlo.reshape %arg0 : (tensor<100x200x300xbf16>) -> tensor<20000x300xbf16>
// CHECK-NEXT:    %1 = stablehlo.reshape %arg1 : (tensor<100x200x300xbf16>) -> tensor<20000x300xbf16>
// CHECK-NEXT:    %2 = stablehlo.subtract %0, %1 : tensor<20000x300xbf16>
// CHECK-NEXT:    return %2 : tensor<20000x300xbf16>
// CHECK-NEXT:  }
// CHECK:  func.func @main2(%arg0: tensor<100x200x300xbf16>) -> tensor<20000x300xf32> {
// CHECK-NEXT:    %0 = stablehlo.reshape %arg0 : (tensor<100x200x300xbf16>) -> tensor<20000x300xbf16>
// CHECK-NEXT:    %1 = stablehlo.convert %0 : (tensor<20000x300xbf16>) -> tensor<20000x300xf32>
// CHECK-NEXT:    return %1 : tensor<20000x300xf32>
// CHECK-NEXT:  }
