// RUN: enzymexlamlir-opt --print-shardy-function-names -o /dev/null %s | FileCheck %s

module {
  sdy.mesh @mesh = <["x"=2]>
  sdy.mesh @mesh_ab = <["a"=4, "b"=2]>
  sdy.mesh @mesh_xyz = <["x"=2, "y"=2, "z"=2]>

  func.func @plain(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    return %arg0 : tensor<4xf32>
  }

  func.func @sharded_arg(%arg0: tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) -> tensor<4xf32> {
    return %arg0 : tensor<4xf32>
  }

  func.func @sharded_arg_2(%arg0: tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) -> tensor<4xf32> {
    return %arg0 : tensor<4xf32>
  }

  func.func @body_only(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    %0 = stablehlo.add %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>] >} : tensor<4xf32>
    return %0 : tensor<4xf32>
  }

  func.func @multi_mesh_signature(
      %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_ab, [{"a"}, {"b"}]>})
      -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"x"}, {"z", "y"}]>}) {
    return %arg0 : tensor<8x8xf32>
  }

  func.func @multi_mesh_body(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
    %0 = stablehlo.add %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_ab, [{"a"}, {"b"}]>]>} : tensor<8x8xf32>
    %1 = stablehlo.negate %0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{"x"}, {"z", "y"}]>]>} : tensor<8x8xf32>
    return %1 : tensor<8x8xf32>
  }

  func.func @multi_mesh_mixed(
      %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>})
      -> tensor<8x8xf32> {
    %0 = stablehlo.add %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_ab, [{"a"}, {"b"}]>]>} : tensor<8x8xf32>
    %1 = stablehlo.negate %0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{"x"}, {"z", "y"}]>]>} : tensor<8x8xf32>
    return %1 : tensor<8x8xf32>
  }

  func.func @anonymous_mesh_signature(
      %arg0: tensor<4xf32> {sdy.sharding = #sdy.sharding<mesh<["u"=2]>, [{"u"}]>})
      -> (tensor<4xf32> {sdy.sharding = #sdy.sharding<mesh<["u"=2]>, [{"u"}]>}) {
    return %arg0 : tensor<4xf32>
  }

  func.func @named_and_anonymous_same_mesh(
      %arg0: tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>})
      -> tensor<4xf32> {
    %0 = stablehlo.add %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<mesh<["x"=2]>, [{"x"}]>]>} : tensor<4xf32>
    return %0 : tensor<4xf32>
  }
}

// CHECK: sharded_arg: #sdy.mesh<["x"=2]>
// CHECK: sharded_arg_2: #sdy.mesh<["x"=2]>
// CHECK: body_only: #sdy.mesh<["x"=2]>
// CHECK: multi_mesh_signature: #sdy.mesh<["a"=4, "b"=2]>, #sdy.mesh<["x"=2, "y"=2, "z"=2]>
// CHECK: multi_mesh_body: #sdy.mesh<["a"=4, "b"=2]>, #sdy.mesh<["x"=2, "y"=2, "z"=2]>
// CHECK: multi_mesh_mixed: #sdy.mesh<["x"=2]>, #sdy.mesh<["a"=4, "b"=2]>, #sdy.mesh<["x"=2, "y"=2, "z"=2]>
// CHECK: anonymous_mesh_signature: #sdy.mesh<["u"=2]>
// CHECK: named_and_anonymous_same_mesh: #sdy.mesh<["x"=2]>
// CHECK-NOT: plain
