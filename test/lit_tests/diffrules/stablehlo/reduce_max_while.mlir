// RUN: enzymexlamlir-opt %s --enzyme --canonicalize --remove-unnecessary-enzyme-ops --arith-raise --enzyme-hlo-opt | stablehlo-translate - --interpret
// RUN: enzymexlamlir-opt %s --enzyme --canonicalize --remove-unnecessary-enzyme-ops --enzyme-simplify-math --arith-raise --canonicalize | FileCheck %s

// Reverse-mode reduce maximum inside a while must not use the forward input
// or result from the other loop body.
//
// Each iteration subtracts the column max, so after the first iteration the
// max of every column is 0 and the primal is x - colmax(x). The gradient is
// seed with the column sum of seed removed at each column's argmax.

module {
  func.func @in_loop(%x: tensor<4x8xf32>) -> tensor<4x8xf32> {
    %c0 = stablehlo.constant dense<0> : tensor<i64>
    %c1 = stablehlo.constant dense<1> : tensor<i64>
    %c5 = stablehlo.constant dense<5> : tensor<i64>
    %neg_inf = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %r:2 = stablehlo.while(%i = %c0, %acc = %x) : tensor<i64>, tensor<4x8xf32>
     cond {
      %cmp = stablehlo.compare  LT, %i, %c5 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %cmp : tensor<i1>
    } do {
      %m = stablehlo.reduce(%acc init: %neg_inf) applies stablehlo.maximum across dimensions = [0] : (tensor<4x8xf32>, tensor<f32>) -> tensor<8xf32>
      %b = stablehlo.broadcast_in_dim %m, dims = [1] : (tensor<8xf32>) -> tensor<4x8xf32>
      %next = stablehlo.subtract %acc, %b : tensor<4x8xf32>
      %inext = stablehlo.add %i, %c1 : tensor<i64>
      stablehlo.return %inext, %next : tensor<i64>, tensor<4x8xf32>
    }
    return %r#1 : tensor<4x8xf32>
  }

  func.func @main() {
    %x = stablehlo.constant dense<[[1.0, 5.0, -2.0, 3.0, 0.0, 7.0, 2.0, -1.0],
                                   [4.0, 2.0, 1.0, -3.0, 6.0, 1.0, -2.0, 3.0],
                                   [-1.0, 3.0, 4.0, 2.0, 1.0, -4.0, 5.0, 0.0],
                                   [2.0, -2.0, 0.0, 1.0, 3.0, 2.0, 1.0, 4.0]]> : tensor<4x8xf32>
    %seed = stablehlo.constant dense<[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                                      [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0],
                                      [2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0],
                                      [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]]> : tensor<4x8xf32>

    %res:2 = enzyme.autodiff @in_loop(%x, %seed) {
      activity = [#enzyme<activity enzyme_active>],
      ret_activity = [#enzyme<activity enzyme_active>]
    } : (tensor<4x8xf32>, tensor<4x8xf32>) -> (tensor<4x8xf32>, tensor<4x8xf32>)

    check.expect_eq_const %res#0, dense<[[-3.0, 0.0, -6.0, 0.0, -6.0, 0.0, -3.0, -5.0],
                                         [0.0, -3.0, -3.0, -6.0, 0.0, -6.0, -7.0, -1.0],
                                         [-5.0, -2.0, 0.0, -1.0, -5.0, -11.0, 0.0, -4.0],
                                         [-2.0, -7.0, -4.0, -2.0, -3.0, -5.0, -4.0, 0.0]]> : tensor<4x8xf32>
    check.expect_eq_const %res#1, dense<[[1.0, -1.5, 3.0, -1.5, 5.0, -1.5, 7.0, 8.0],
                                         [-3.5, 1.0, -1.0, 1.0, -7.5, 1.0, -1.0, 1.0],
                                         [2.0, 0.0, -2.5, 0.0, 2.0, 0.0, -6.5, 0.0],
                                         [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, -9.0]]> : tensor<4x8xf32>
    func.return
  }
}

// CHECK-LABEL: func.func private @diffein_loop(
// CHECK:         stablehlo.while
// CHECK:           stablehlo.reduce({{.*}}) applies stablehlo.maximum
// CHECK:         stablehlo.while
// CHECK:           stablehlo.dynamic_slice {{.*}}sizes
// CHECK:           %[[CMP:.+]] = stablehlo.compare {{ *}}EQ, %{{.+}}, %{{.+}} : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xi1>
// CHECK:           stablehlo.select %[[CMP]],
