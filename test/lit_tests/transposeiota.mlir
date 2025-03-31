// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(enzyme-hlo-opt{passses=65536})" %s | FileCheck %s

module {
  func.func @main() -> tensor<300x150x200xbf16> {
    %1909 = stablehlo.iota dim = 0 : tensor<150x200x300xbf16>
    %1910 = stablehlo.transpose %1909, dims = [2, 0, 1] : (tensor<150x200x300xbf16>) -> tensor<300x150x200xbf16> 
    return %1910 : tensor<300x150x200xbf16> 
  }
}

// CHECK:  func.func @main() -> tensor<300x150x200xbf16> {
// CHECK-NEXT:    %0 = stablehlo.iota dim = 1 : tensor<300x150x200xbf16>
// CHECK-NEXT:    return %0 : tensor<300x150x200xbf16>
// CHECK-NEXT:  }