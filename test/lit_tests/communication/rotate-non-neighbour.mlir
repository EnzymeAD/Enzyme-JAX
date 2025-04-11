// RUN: enzymexlamlir-opt --optimize-communication %s | FileCheck %s

// TODO we can support this case if neede
// XFAIL: *
// CHECK:  sdy.manual_computation

module {
  sdy.mesh @mesh = <["x"=4, "y"=1, "z"=1]>
  func.func @right_to_left(%arg: tensor<4x8x100xf64>) -> (tensor<4x8x100xf64>) {
    %res = "enzymexla.rotate"(%arg) <{amount = 30 : si32, dimension = 2 : si32}> {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z", ?}, {"y", ?}, {"x", ?}]>]>} : (tensor<4x8x100xf64>) -> tensor<4x8x100xf64>
    func.return %res : tensor<4x8x100xf64>
  }
}
