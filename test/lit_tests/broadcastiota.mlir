// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

module {
  func.func @main() -> tensor<4x6128x12272xi64> {
    %899 = stablehlo.iota dim = 0 {mhlo.sharding = "{devices=[4,4]<=[4,4]T(1,0) last_tile_dim_replicate}"} : tensor<6128xi64>
    %1486 = stablehlo.broadcast_in_dim %899, dims = [1] {mhlo.sharding = "{devices=[1,4,4]<=[4,4]T(1,0)}"} : (tensor<6128xi64>) -> tensor<4x6128x12272xi64>
    return %1486 : tensor<4x6128x12272xi64>
  }
}

// CHECK:  func.func @main() -> tensor<4x6128x12272xi64> {
// CHECK-NEXT:    %0 = stablehlo.iota dim = 1 : tensor<4x6128x12272xi64>
// CHECK-NEXT:    return %0 : tensor<4x6128x12272xi64>
// CHECK-NEXT:  }
