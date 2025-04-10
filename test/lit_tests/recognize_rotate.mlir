// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=recognize_rotate" --transform-interpreter --enzyme-hlo-remove-transform -allow-unregistered-dialect %s

module @"reactant_loop!" {
  func.func @main(%arg23: tensor<20x24x96xf64>) -> (tensor<4x8x80xf64>) {
      %11781 = stablehlo.slice %arg23 [8:12, 8:16, 10:88] : (tensor<20x24x96xf64>) -> tensor<4x8x78xf64>
      %11782 = stablehlo.slice %arg23 [8:12, 8:16, 8:10] : (tensor<20x24x96xf64>) -> tensor<4x8x2xf64>
      %11783 = stablehlo.concatenate %11781, %11782, dim = 2 : (tensor<4x8x78xf64>, tensor<4x8x2xf64>) -> tensor<4x8x80xf64>
      stablehlo.return %11783 :  tensor<4x8x80xf64>
  }

  
  sdy.mesh @mesh = <["x"=2, "y"=4]>
  func.func @rotate(%arg0: tensor<12x1024xi64> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x"}]>}) -> (tensor<12x1024xi64> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x"}]>}) {
    %0 = stablehlo.slice %arg0 [0:12, 100:1024] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y", ?}, {"x", ?}]>]>} : (tensor<12x1024xi64>) -> tensor<12x924xi64>
    %1 = stablehlo.slice %arg0 [0:12, 0:100] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y", ?}, {"x", ?}]>]>} : (tensor<12x1024xi64>) -> tensor<12x100xi64>
    %2 = stablehlo.concatenate %0, %1, dim = 1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y", ?}, {"x", ?}]>]>} : (tensor<12x924xi64>, tensor<12x100xi64>) -> tensor<12x1024xi64>
    return %2 : tensor<12x1024xi64>
  }
}

// CHECK:  func.func @main(%arg0: tensor<20x24x96xf64>) -> tensor<4x8x80xf64> {
// CHECK-NEXT:    %0 = stablehlo.slice %arg0 [8:12, 8:16, 8:88] : (tensor<20x24x96xf64>) -> tensor<4x8x80xf64>
// CHECK-NEXT:    %1 = "enzymexla.rotate"(%0) <{amount = 2 : si32, dimension = 2 : si32}> : (tensor<4x8x80xf64>) -> tensor<4x8x80xf64>
// CHECK-NEXT:    %2 = stablehlo.concatenate %1, dim = 2 : (tensor<4x8x80xf64>) -> tensor<4x8x80xf64>
// CHECK-NEXT:    stablehlo.return %2 : tensor<4x8x80xf64>
// CHECK-NEXT:  }

// CHECK:  func.func @rotate(%arg0: tensor<12x1024xi64> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x"}]>}) -> (tensor<12x1024xi64> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x"}]>}) {
// CHECK-NEXT:    %0 = "enzymexla.rotate"(%arg0) <{amount = 100 : si32, dimension = 1 : si32}> {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y", ?}, {"x", ?}]>]>} : (tensor<12x1024xi64>) -> tensor<12x1024xi64>
// CHECK-NEXT:    return %0 : tensor<12x1024xi64>
// CHECK-NEXT:  }