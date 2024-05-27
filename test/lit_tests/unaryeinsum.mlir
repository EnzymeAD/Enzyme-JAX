// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

module {
  func.func @main(%x : tensor<2x3xf32>) -> tensor<2xf32> {
    %y = "stablehlo.unary_einsum"(%x) {einsum_config = "ab->a"} : (tensor<2x3xf32>) -> tensor<2xf32>
    func.return %y : tensor<2xf32>
  }

  func.func @rdiffmain(%x : tensor<2x3xf32>, %dr : tensor<2xf32>) -> tensor<2xf32> {
    %r = enzyme.autodiff @main(%x, %dx) { activity=[#enzyme<activity enzime_dup>], ret_activity=[#enzyme<activity enzyme_dupnoneed>] } : (tensor<2x3xf32>, tensor<2xf32>) -> tensor<2xf32>
    func.return %r : tensor<2xf32>
  }
}
