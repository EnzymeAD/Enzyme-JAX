// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=enzyme_hlo_unroll(4)" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

// The outer loop's own block is a handful of ops, but it wraps a loop whose
// body is 130 ops: unrolling it copies that loop four times, well past the
// budget of 4 x 128 ops. The body is measured with its nested regions, so
// neither loop unrolls (the inner one is itself over budget: 4 x 130).

module {
  func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    %c = stablehlo.constant dense<0> : tensor<i64>
    %c_0 = stablehlo.constant dense<4> : tensor<i64>
    %c_1 = stablehlo.constant dense<1> : tensor<i64>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<4xf32>
    %0:2 = stablehlo.while(%iterArg = %c, %iterArg_2 = %arg0) : tensor<i64>, tensor<4xf32>
    cond {
      %1 = stablehlo.compare  LT, %iterArg, %c_0 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %1 : tensor<i1>
    } do {
      %1:2 = stablehlo.while(%iterArg_i = %c, %iterArg_i1 = %iterArg_2) : tensor<i64>, tensor<4xf32>
      cond {
        %3 = stablehlo.compare  LT, %iterArg_i, %c_0 : (tensor<i64>, tensor<i64>) -> tensor<i1>
        stablehlo.return %3 : tensor<i1>
      } do {
        %b0 = stablehlo.add %iterArg_i1, %cst : tensor<4xf32>
        %b1 = stablehlo.add %b0, %cst : tensor<4xf32>
        %b2 = stablehlo.add %b1, %cst : tensor<4xf32>
        %b3 = stablehlo.add %b2, %cst : tensor<4xf32>
        %b4 = stablehlo.add %b3, %cst : tensor<4xf32>
        %b5 = stablehlo.add %b4, %cst : tensor<4xf32>
        %b6 = stablehlo.add %b5, %cst : tensor<4xf32>
        %b7 = stablehlo.add %b6, %cst : tensor<4xf32>
        %b8 = stablehlo.add %b7, %cst : tensor<4xf32>
        %b9 = stablehlo.add %b8, %cst : tensor<4xf32>
        %b10 = stablehlo.add %b9, %cst : tensor<4xf32>
        %b11 = stablehlo.add %b10, %cst : tensor<4xf32>
        %b12 = stablehlo.add %b11, %cst : tensor<4xf32>
        %b13 = stablehlo.add %b12, %cst : tensor<4xf32>
        %b14 = stablehlo.add %b13, %cst : tensor<4xf32>
        %b15 = stablehlo.add %b14, %cst : tensor<4xf32>
        %b16 = stablehlo.add %b15, %cst : tensor<4xf32>
        %b17 = stablehlo.add %b16, %cst : tensor<4xf32>
        %b18 = stablehlo.add %b17, %cst : tensor<4xf32>
        %b19 = stablehlo.add %b18, %cst : tensor<4xf32>
        %b20 = stablehlo.add %b19, %cst : tensor<4xf32>
        %b21 = stablehlo.add %b20, %cst : tensor<4xf32>
        %b22 = stablehlo.add %b21, %cst : tensor<4xf32>
        %b23 = stablehlo.add %b22, %cst : tensor<4xf32>
        %b24 = stablehlo.add %b23, %cst : tensor<4xf32>
        %b25 = stablehlo.add %b24, %cst : tensor<4xf32>
        %b26 = stablehlo.add %b25, %cst : tensor<4xf32>
        %b27 = stablehlo.add %b26, %cst : tensor<4xf32>
        %b28 = stablehlo.add %b27, %cst : tensor<4xf32>
        %b29 = stablehlo.add %b28, %cst : tensor<4xf32>
        %b30 = stablehlo.add %b29, %cst : tensor<4xf32>
        %b31 = stablehlo.add %b30, %cst : tensor<4xf32>
        %b32 = stablehlo.add %b31, %cst : tensor<4xf32>
        %b33 = stablehlo.add %b32, %cst : tensor<4xf32>
        %b34 = stablehlo.add %b33, %cst : tensor<4xf32>
        %b35 = stablehlo.add %b34, %cst : tensor<4xf32>
        %b36 = stablehlo.add %b35, %cst : tensor<4xf32>
        %b37 = stablehlo.add %b36, %cst : tensor<4xf32>
        %b38 = stablehlo.add %b37, %cst : tensor<4xf32>
        %b39 = stablehlo.add %b38, %cst : tensor<4xf32>
        %b40 = stablehlo.add %b39, %cst : tensor<4xf32>
        %b41 = stablehlo.add %b40, %cst : tensor<4xf32>
        %b42 = stablehlo.add %b41, %cst : tensor<4xf32>
        %b43 = stablehlo.add %b42, %cst : tensor<4xf32>
        %b44 = stablehlo.add %b43, %cst : tensor<4xf32>
        %b45 = stablehlo.add %b44, %cst : tensor<4xf32>
        %b46 = stablehlo.add %b45, %cst : tensor<4xf32>
        %b47 = stablehlo.add %b46, %cst : tensor<4xf32>
        %b48 = stablehlo.add %b47, %cst : tensor<4xf32>
        %b49 = stablehlo.add %b48, %cst : tensor<4xf32>
        %b50 = stablehlo.add %b49, %cst : tensor<4xf32>
        %b51 = stablehlo.add %b50, %cst : tensor<4xf32>
        %b52 = stablehlo.add %b51, %cst : tensor<4xf32>
        %b53 = stablehlo.add %b52, %cst : tensor<4xf32>
        %b54 = stablehlo.add %b53, %cst : tensor<4xf32>
        %b55 = stablehlo.add %b54, %cst : tensor<4xf32>
        %b56 = stablehlo.add %b55, %cst : tensor<4xf32>
        %b57 = stablehlo.add %b56, %cst : tensor<4xf32>
        %b58 = stablehlo.add %b57, %cst : tensor<4xf32>
        %b59 = stablehlo.add %b58, %cst : tensor<4xf32>
        %b60 = stablehlo.add %b59, %cst : tensor<4xf32>
        %b61 = stablehlo.add %b60, %cst : tensor<4xf32>
        %b62 = stablehlo.add %b61, %cst : tensor<4xf32>
        %b63 = stablehlo.add %b62, %cst : tensor<4xf32>
        %b64 = stablehlo.add %b63, %cst : tensor<4xf32>
        %b65 = stablehlo.add %b64, %cst : tensor<4xf32>
        %b66 = stablehlo.add %b65, %cst : tensor<4xf32>
        %b67 = stablehlo.add %b66, %cst : tensor<4xf32>
        %b68 = stablehlo.add %b67, %cst : tensor<4xf32>
        %b69 = stablehlo.add %b68, %cst : tensor<4xf32>
        %b70 = stablehlo.add %b69, %cst : tensor<4xf32>
        %b71 = stablehlo.add %b70, %cst : tensor<4xf32>
        %b72 = stablehlo.add %b71, %cst : tensor<4xf32>
        %b73 = stablehlo.add %b72, %cst : tensor<4xf32>
        %b74 = stablehlo.add %b73, %cst : tensor<4xf32>
        %b75 = stablehlo.add %b74, %cst : tensor<4xf32>
        %b76 = stablehlo.add %b75, %cst : tensor<4xf32>
        %b77 = stablehlo.add %b76, %cst : tensor<4xf32>
        %b78 = stablehlo.add %b77, %cst : tensor<4xf32>
        %b79 = stablehlo.add %b78, %cst : tensor<4xf32>
        %b80 = stablehlo.add %b79, %cst : tensor<4xf32>
        %b81 = stablehlo.add %b80, %cst : tensor<4xf32>
        %b82 = stablehlo.add %b81, %cst : tensor<4xf32>
        %b83 = stablehlo.add %b82, %cst : tensor<4xf32>
        %b84 = stablehlo.add %b83, %cst : tensor<4xf32>
        %b85 = stablehlo.add %b84, %cst : tensor<4xf32>
        %b86 = stablehlo.add %b85, %cst : tensor<4xf32>
        %b87 = stablehlo.add %b86, %cst : tensor<4xf32>
        %b88 = stablehlo.add %b87, %cst : tensor<4xf32>
        %b89 = stablehlo.add %b88, %cst : tensor<4xf32>
        %b90 = stablehlo.add %b89, %cst : tensor<4xf32>
        %b91 = stablehlo.add %b90, %cst : tensor<4xf32>
        %b92 = stablehlo.add %b91, %cst : tensor<4xf32>
        %b93 = stablehlo.add %b92, %cst : tensor<4xf32>
        %b94 = stablehlo.add %b93, %cst : tensor<4xf32>
        %b95 = stablehlo.add %b94, %cst : tensor<4xf32>
        %b96 = stablehlo.add %b95, %cst : tensor<4xf32>
        %b97 = stablehlo.add %b96, %cst : tensor<4xf32>
        %b98 = stablehlo.add %b97, %cst : tensor<4xf32>
        %b99 = stablehlo.add %b98, %cst : tensor<4xf32>
        %b100 = stablehlo.add %b99, %cst : tensor<4xf32>
        %b101 = stablehlo.add %b100, %cst : tensor<4xf32>
        %b102 = stablehlo.add %b101, %cst : tensor<4xf32>
        %b103 = stablehlo.add %b102, %cst : tensor<4xf32>
        %b104 = stablehlo.add %b103, %cst : tensor<4xf32>
        %b105 = stablehlo.add %b104, %cst : tensor<4xf32>
        %b106 = stablehlo.add %b105, %cst : tensor<4xf32>
        %b107 = stablehlo.add %b106, %cst : tensor<4xf32>
        %b108 = stablehlo.add %b107, %cst : tensor<4xf32>
        %b109 = stablehlo.add %b108, %cst : tensor<4xf32>
        %b110 = stablehlo.add %b109, %cst : tensor<4xf32>
        %b111 = stablehlo.add %b110, %cst : tensor<4xf32>
        %b112 = stablehlo.add %b111, %cst : tensor<4xf32>
        %b113 = stablehlo.add %b112, %cst : tensor<4xf32>
        %b114 = stablehlo.add %b113, %cst : tensor<4xf32>
        %b115 = stablehlo.add %b114, %cst : tensor<4xf32>
        %b116 = stablehlo.add %b115, %cst : tensor<4xf32>
        %b117 = stablehlo.add %b116, %cst : tensor<4xf32>
        %b118 = stablehlo.add %b117, %cst : tensor<4xf32>
        %b119 = stablehlo.add %b118, %cst : tensor<4xf32>
        %b120 = stablehlo.add %b119, %cst : tensor<4xf32>
        %b121 = stablehlo.add %b120, %cst : tensor<4xf32>
        %b122 = stablehlo.add %b121, %cst : tensor<4xf32>
        %b123 = stablehlo.add %b122, %cst : tensor<4xf32>
        %b124 = stablehlo.add %b123, %cst : tensor<4xf32>
        %b125 = stablehlo.add %b124, %cst : tensor<4xf32>
        %b126 = stablehlo.add %b125, %cst : tensor<4xf32>
        %b127 = stablehlo.add %b126, %cst : tensor<4xf32>
        %b128 = stablehlo.add %b127, %cst : tensor<4xf32>
        %b129 = stablehlo.add %b128, %cst : tensor<4xf32>
        %3 = stablehlo.add %iterArg_i, %c_1 : tensor<i64>
        stablehlo.return %3, %b129 : tensor<i64>, tensor<4xf32>
      }
      %2 = stablehlo.add %iterArg, %c_1 : tensor<i64>
      stablehlo.return %2, %1#1 : tensor<i64>, tensor<4xf32>
    }
    return %0#1 : tensor<4xf32>
  }
}

// CHECK-LABEL: func.func @main
// CHECK:         %[[OUTER:.+]]:2 = stablehlo.while
// CHECK:         } do {
// CHECK-NEXT:      %[[INNER:.+]]:2 = stablehlo.while
// CHECK:           stablehlo.return %{{.+}}, %[[INNER]]#1
// CHECK:         return %[[OUTER]]#1
