// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt | FileCheck %s

module @all_reduce_101_repro attributes {mhlo.num_partitions = 4 : i64, mhlo.num_replicas = 1 : i64} {
  sdy.mesh @mesh = <["x"=2, "y"=2]>

  func.func @main(
    %arg0: tensor<20x1536x3056xf64> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}, {"x"}]>},
    %arg1: tensor<20x1536x3056xf64> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}, {"x"}]>},
    %arg2: tensor<20x1536x3056xf64> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}, {"x"}]>},
    %arg3: tensor<20x1536x3056xf64> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}, {"x"}]>}
  ) -> (
    tensor<1534x3070xf64> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x"}]>},
    tensor<1534x3070xf64> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x"}]>},
    tensor<1534x3070xf64> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x"}]>},
    tensor<1534x3070xf64> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x"}]>}
  ) {

    %e_full = stablehlo.slice %arg0 [12:13, 1:1535, 0:3056]
      : (tensor<20x1536x3056xf64>) -> tensor<1x1534x3056xf64>

    %e_r5 = stablehlo.slice %arg0 [12:13, 1:1535, 3054:3055]
      : (tensor<20x1536x3056xf64>) -> tensor<1x1534x1xf64>
    %e_r4 = stablehlo.slice %arg0 [12:13, 1:1535, 3053:3054]
      : (tensor<20x1536x3056xf64>) -> tensor<1x1534x1xf64>
    %e_r3 = stablehlo.slice %arg0 [12:13, 1:1535, 3052:3053]
      : (tensor<20x1536x3056xf64>) -> tensor<1x1534x1xf64>
    %e_r2 = stablehlo.slice %arg0 [12:13, 1:1535, 3051:3052]
      : (tensor<20x1536x3056xf64>) -> tensor<1x1534x1xf64>
    %e_r1 = stablehlo.slice %arg0 [12:13, 1:1535, 3050:3051]
      : (tensor<20x1536x3056xf64>) -> tensor<1x1534x1xf64>
    %e_r0 = stablehlo.slice %arg0 [12:13, 1:1535, 3049:3050]
      : (tensor<20x1536x3056xf64>) -> tensor<1x1534x1xf64>

    %e_l5 = stablehlo.slice %arg0 [12:13, 1:1535, 6:7]
      : (tensor<20x1536x3056xf64>) -> tensor<1x1534x1xf64>
    %e_l4 = stablehlo.slice %arg0 [12:13, 1:1535, 5:6]
      : (tensor<20x1536x3056xf64>) -> tensor<1x1534x1xf64>
    %e_l3 = stablehlo.slice %arg0 [12:13, 1:1535, 4:5]
      : (tensor<20x1536x3056xf64>) -> tensor<1x1534x1xf64>
    %e_l2 = stablehlo.slice %arg0 [12:13, 1:1535, 3:4]
      : (tensor<20x1536x3056xf64>) -> tensor<1x1534x1xf64>
    %e_l1 = stablehlo.slice %arg0 [12:13, 1:1535, 2:3]
      : (tensor<20x1536x3056xf64>) -> tensor<1x1534x1xf64>
    %e_l0 = stablehlo.slice %arg0 [12:13, 1:1535, 1:2]
      : (tensor<20x1536x3056xf64>) -> tensor<1x1534x1xf64>

    // Concat order: [3049, 3050, 3051, 3052, 3053, 3054, wrap(interior), 1, 2, 3, 4, 5, 6]
    %e_c0 = stablehlo.reshape %e_r0 : (tensor<1x1534x1xf64>) -> tensor<1534x1xf64>
    %e_c1 = stablehlo.reshape %e_r1 : (tensor<1x1534x1xf64>) -> tensor<1534x1xf64>
    %e_c2 = stablehlo.reshape %e_r2 : (tensor<1x1534x1xf64>) -> tensor<1534x1xf64>
    %e_c3 = stablehlo.reshape %e_r3 : (tensor<1x1534x1xf64>) -> tensor<1534x1xf64>
    %e_c4 = stablehlo.reshape %e_r4 : (tensor<1x1534x1xf64>) -> tensor<1534x1xf64>
    %e_c5 = stablehlo.reshape %e_r5 : (tensor<1x1534x1xf64>) -> tensor<1534x1xf64>
    %e_c6 = stablehlo.reshape %e_l0 : (tensor<1x1534x1xf64>) -> tensor<1534x1xf64>
    %e_c7 = stablehlo.reshape %e_l1 : (tensor<1x1534x1xf64>) -> tensor<1534x1xf64>
    %e_c8 = stablehlo.reshape %e_l2 : (tensor<1x1534x1xf64>) -> tensor<1534x1xf64>
    %e_c9 = stablehlo.reshape %e_l3 : (tensor<1x1534x1xf64>) -> tensor<1534x1xf64>
    %e_c10 = stablehlo.reshape %e_l4 : (tensor<1x1534x1xf64>) -> tensor<1534x1xf64>
    %e_c11 = stablehlo.reshape %e_l5 : (tensor<1x1534x1xf64>) -> tensor<1534x1xf64>

    %e_interior = stablehlo.reshape %e_full : (tensor<1x1534x3056xf64>) -> tensor<1534x3056xf64>
    %e_wrapped = "enzymexla.wrap"(%e_interior) <{dimension = 1 : i64, lhs = 1 : i64, rhs = 1 : i64}>
      : (tensor<1534x3056xf64>) -> tensor<1534x3058xf64>

    %out_e = stablehlo.concatenate
      %e_c0, %e_c1, %e_c2, %e_c3, %e_c4, %e_c5,
      %e_wrapped,
      %e_c6, %e_c7, %e_c8, %e_c9, %e_c10, %e_c11, dim = 1
      : (tensor<1534x1xf64>, tensor<1534x1xf64>, tensor<1534x1xf64>,
         tensor<1534x1xf64>, tensor<1534x1xf64>, tensor<1534x1xf64>,
         tensor<1534x3058xf64>,
         tensor<1534x1xf64>, tensor<1534x1xf64>, tensor<1534x1xf64>,
         tensor<1534x1xf64>, tensor<1534x1xf64>, tensor<1534x1xf64>)
      -> tensor<1534x3070xf64>

    %g_full = stablehlo.slice %arg1 [11:12, 1:1535, 0:3056]
      : (tensor<20x1536x3056xf64>) -> tensor<1x1534x3056xf64>

    %g_r0 = stablehlo.slice %arg1 [11:12, 1:1535, 3049:3050]
      : (tensor<20x1536x3056xf64>) -> tensor<1x1534x1xf64>
    %g_r1 = stablehlo.slice %arg1 [11:12, 1:1535, 3050:3051]
      : (tensor<20x1536x3056xf64>) -> tensor<1x1534x1xf64>
    %g_r2 = stablehlo.slice %arg1 [11:12, 1:1535, 3051:3052]
      : (tensor<20x1536x3056xf64>) -> tensor<1x1534x1xf64>
    %g_r3 = stablehlo.slice %arg1 [11:12, 1:1535, 3052:3053]
      : (tensor<20x1536x3056xf64>) -> tensor<1x1534x1xf64>
    %g_r4 = stablehlo.slice %arg1 [11:12, 1:1535, 3053:3054]
      : (tensor<20x1536x3056xf64>) -> tensor<1x1534x1xf64>
    %g_r5 = stablehlo.slice %arg1 [11:12, 1:1535, 3054:3055]
      : (tensor<20x1536x3056xf64>) -> tensor<1x1534x1xf64>

    %g_l0 = stablehlo.slice %arg1 [11:12, 1:1535, 1:2]
      : (tensor<20x1536x3056xf64>) -> tensor<1x1534x1xf64>
    %g_l1 = stablehlo.slice %arg1 [11:12, 1:1535, 2:3]
      : (tensor<20x1536x3056xf64>) -> tensor<1x1534x1xf64>
    %g_l2 = stablehlo.slice %arg1 [11:12, 1:1535, 3:4]
      : (tensor<20x1536x3056xf64>) -> tensor<1x1534x1xf64>
    %g_l3 = stablehlo.slice %arg1 [11:12, 1:1535, 4:5]
      : (tensor<20x1536x3056xf64>) -> tensor<1x1534x1xf64>
    %g_l4 = stablehlo.slice %arg1 [11:12, 1:1535, 5:6]
      : (tensor<20x1536x3056xf64>) -> tensor<1x1534x1xf64>
    %g_l5 = stablehlo.slice %arg1 [11:12, 1:1535, 6:7]
      : (tensor<20x1536x3056xf64>) -> tensor<1x1534x1xf64>

    %g_c0 = stablehlo.reshape %g_r0 : (tensor<1x1534x1xf64>) -> tensor<1534x1xf64>
    %g_c1 = stablehlo.reshape %g_r1 : (tensor<1x1534x1xf64>) -> tensor<1534x1xf64>
    %g_c2 = stablehlo.reshape %g_r2 : (tensor<1x1534x1xf64>) -> tensor<1534x1xf64>
    %g_c3 = stablehlo.reshape %g_r3 : (tensor<1x1534x1xf64>) -> tensor<1534x1xf64>
    %g_c4 = stablehlo.reshape %g_r4 : (tensor<1x1534x1xf64>) -> tensor<1534x1xf64>
    %g_c5 = stablehlo.reshape %g_r5 : (tensor<1x1534x1xf64>) -> tensor<1534x1xf64>
    %g_c6 = stablehlo.reshape %g_l0 : (tensor<1x1534x1xf64>) -> tensor<1534x1xf64>
    %g_c7 = stablehlo.reshape %g_l1 : (tensor<1x1534x1xf64>) -> tensor<1534x1xf64>
    %g_c8 = stablehlo.reshape %g_l2 : (tensor<1x1534x1xf64>) -> tensor<1534x1xf64>
    %g_c9 = stablehlo.reshape %g_l3 : (tensor<1x1534x1xf64>) -> tensor<1534x1xf64>
    %g_c10 = stablehlo.reshape %g_l4 : (tensor<1x1534x1xf64>) -> tensor<1534x1xf64>
    %g_c11 = stablehlo.reshape %g_l5 : (tensor<1x1534x1xf64>) -> tensor<1534x1xf64>

    %g_interior = stablehlo.reshape %g_full : (tensor<1x1534x3056xf64>) -> tensor<1534x3056xf64>
    %g_wrapped = "enzymexla.wrap"(%g_interior) <{dimension = 1 : i64, lhs = 1 : i64, rhs = 1 : i64}>
      : (tensor<1534x3056xf64>) -> tensor<1534x3058xf64>

    %out_g = stablehlo.concatenate
      %g_c0, %g_c1, %g_c2, %g_c3, %g_c4, %g_c5,
      %g_wrapped,
      %g_c6, %g_c7, %g_c8, %g_c9, %g_c10, %g_c11, dim = 1
      : (tensor<1534x1xf64>, tensor<1534x1xf64>, tensor<1534x1xf64>,
         tensor<1534x1xf64>, tensor<1534x1xf64>, tensor<1534x1xf64>,
         tensor<1534x3058xf64>,
         tensor<1534x1xf64>, tensor<1534x1xf64>, tensor<1534x1xf64>,
         tensor<1534x1xf64>, tensor<1534x1xf64>, tensor<1534x1xf64>)
      -> tensor<1534x3070xf64>

    // =========================================================================
    // Group F: from %arg2 at level [12:13]
    //   Source: %5749 in original, lines 6330-6341 (slices), 6546-6560 (reshape/wrap/concat)
    // =========================================================================

    %f_full = stablehlo.slice %arg2 [12:13, 1:1535, 0:3056]
      : (tensor<20x1536x3056xf64>) -> tensor<1x1534x3056xf64>

    %f_r5 = stablehlo.slice %arg2 [12:13, 1:1535, 3054:3055]
      : (tensor<20x1536x3056xf64>) -> tensor<1x1534x1xf64>
    %f_r4 = stablehlo.slice %arg2 [12:13, 1:1535, 3053:3054]
      : (tensor<20x1536x3056xf64>) -> tensor<1x1534x1xf64>
    %f_r3 = stablehlo.slice %arg2 [12:13, 1:1535, 3052:3053]
      : (tensor<20x1536x3056xf64>) -> tensor<1x1534x1xf64>
    %f_r2 = stablehlo.slice %arg2 [12:13, 1:1535, 3051:3052]
      : (tensor<20x1536x3056xf64>) -> tensor<1x1534x1xf64>
    %f_r1 = stablehlo.slice %arg2 [12:13, 1:1535, 3050:3051]
      : (tensor<20x1536x3056xf64>) -> tensor<1x1534x1xf64>
    %f_r0 = stablehlo.slice %arg2 [12:13, 1:1535, 3049:3050]
      : (tensor<20x1536x3056xf64>) -> tensor<1x1534x1xf64>

    %f_l5 = stablehlo.slice %arg2 [12:13, 1:1535, 6:7]
      : (tensor<20x1536x3056xf64>) -> tensor<1x1534x1xf64>
    %f_l4 = stablehlo.slice %arg2 [12:13, 1:1535, 5:6]
      : (tensor<20x1536x3056xf64>) -> tensor<1x1534x1xf64>
    %f_l3 = stablehlo.slice %arg2 [12:13, 1:1535, 4:5]
      : (tensor<20x1536x3056xf64>) -> tensor<1x1534x1xf64>
    %f_l2 = stablehlo.slice %arg2 [12:13, 1:1535, 3:4]
      : (tensor<20x1536x3056xf64>) -> tensor<1x1534x1xf64>
    %f_l1 = stablehlo.slice %arg2 [12:13, 1:1535, 2:3]
      : (tensor<20x1536x3056xf64>) -> tensor<1x1534x1xf64>
    %f_l0 = stablehlo.slice %arg2 [12:13, 1:1535, 1:2]
      : (tensor<20x1536x3056xf64>) -> tensor<1x1534x1xf64>

    %f_c0 = stablehlo.reshape %f_r0 : (tensor<1x1534x1xf64>) -> tensor<1534x1xf64>
    %f_c1 = stablehlo.reshape %f_r1 : (tensor<1x1534x1xf64>) -> tensor<1534x1xf64>
    %f_c2 = stablehlo.reshape %f_r2 : (tensor<1x1534x1xf64>) -> tensor<1534x1xf64>
    %f_c3 = stablehlo.reshape %f_r3 : (tensor<1x1534x1xf64>) -> tensor<1534x1xf64>
    %f_c4 = stablehlo.reshape %f_r4 : (tensor<1x1534x1xf64>) -> tensor<1534x1xf64>
    %f_c5 = stablehlo.reshape %f_r5 : (tensor<1x1534x1xf64>) -> tensor<1534x1xf64>
    %f_c6 = stablehlo.reshape %f_l0 : (tensor<1x1534x1xf64>) -> tensor<1534x1xf64>
    %f_c7 = stablehlo.reshape %f_l1 : (tensor<1x1534x1xf64>) -> tensor<1534x1xf64>
    %f_c8 = stablehlo.reshape %f_l2 : (tensor<1x1534x1xf64>) -> tensor<1534x1xf64>
    %f_c9 = stablehlo.reshape %f_l3 : (tensor<1x1534x1xf64>) -> tensor<1534x1xf64>
    %f_c10 = stablehlo.reshape %f_l4 : (tensor<1x1534x1xf64>) -> tensor<1534x1xf64>
    %f_c11 = stablehlo.reshape %f_l5 : (tensor<1x1534x1xf64>) -> tensor<1534x1xf64>

    %f_interior = stablehlo.reshape %f_full : (tensor<1x1534x3056xf64>) -> tensor<1534x3056xf64>
    %f_wrapped = "enzymexla.wrap"(%f_interior) <{dimension = 1 : i64, lhs = 1 : i64, rhs = 1 : i64}>
      : (tensor<1534x3056xf64>) -> tensor<1534x3058xf64>

    %out_f = stablehlo.concatenate
      %f_c0, %f_c1, %f_c2, %f_c3, %f_c4, %f_c5,
      %f_wrapped,
      %f_c6, %f_c7, %f_c8, %f_c9, %f_c10, %f_c11, dim = 1
      : (tensor<1534x1xf64>, tensor<1534x1xf64>, tensor<1534x1xf64>,
         tensor<1534x1xf64>, tensor<1534x1xf64>, tensor<1534x1xf64>,
         tensor<1534x3058xf64>,
         tensor<1534x1xf64>, tensor<1534x1xf64>, tensor<1534x1xf64>,
         tensor<1534x1xf64>, tensor<1534x1xf64>, tensor<1534x1xf64>)
      -> tensor<1534x3070xf64>

    // =========================================================================
    // Group H: from %arg3 at level [11:12]
    //   Source: %5766 in original, lines 6397-6408 (slices), 6409-6423 (reshape/wrap/concat)
    // =========================================================================

    %h_full = stablehlo.slice %arg3 [11:12, 1:1535, 0:3056]
      : (tensor<20x1536x3056xf64>) -> tensor<1x1534x3056xf64>

    %h_r0 = stablehlo.slice %arg3 [11:12, 1:1535, 3049:3050]
      : (tensor<20x1536x3056xf64>) -> tensor<1x1534x1xf64>
    %h_r1 = stablehlo.slice %arg3 [11:12, 1:1535, 3050:3051]
      : (tensor<20x1536x3056xf64>) -> tensor<1x1534x1xf64>
    %h_r2 = stablehlo.slice %arg3 [11:12, 1:1535, 3051:3052]
      : (tensor<20x1536x3056xf64>) -> tensor<1x1534x1xf64>
    %h_r3 = stablehlo.slice %arg3 [11:12, 1:1535, 3052:3053]
      : (tensor<20x1536x3056xf64>) -> tensor<1x1534x1xf64>
    %h_r4 = stablehlo.slice %arg3 [11:12, 1:1535, 3053:3054]
      : (tensor<20x1536x3056xf64>) -> tensor<1x1534x1xf64>
    %h_r5 = stablehlo.slice %arg3 [11:12, 1:1535, 3054:3055]
      : (tensor<20x1536x3056xf64>) -> tensor<1x1534x1xf64>

    %h_l0 = stablehlo.slice %arg3 [11:12, 1:1535, 1:2]
      : (tensor<20x1536x3056xf64>) -> tensor<1x1534x1xf64>
    %h_l1 = stablehlo.slice %arg3 [11:12, 1:1535, 2:3]
      : (tensor<20x1536x3056xf64>) -> tensor<1x1534x1xf64>
    %h_l2 = stablehlo.slice %arg3 [11:12, 1:1535, 3:4]
      : (tensor<20x1536x3056xf64>) -> tensor<1x1534x1xf64>
    %h_l3 = stablehlo.slice %arg3 [11:12, 1:1535, 4:5]
      : (tensor<20x1536x3056xf64>) -> tensor<1x1534x1xf64>
    %h_l4 = stablehlo.slice %arg3 [11:12, 1:1535, 5:6]
      : (tensor<20x1536x3056xf64>) -> tensor<1x1534x1xf64>
    %h_l5 = stablehlo.slice %arg3 [11:12, 1:1535, 6:7]
      : (tensor<20x1536x3056xf64>) -> tensor<1x1534x1xf64>

    %h_c0 = stablehlo.reshape %h_r0 : (tensor<1x1534x1xf64>) -> tensor<1534x1xf64>
    %h_c1 = stablehlo.reshape %h_r1 : (tensor<1x1534x1xf64>) -> tensor<1534x1xf64>
    %h_c2 = stablehlo.reshape %h_r2 : (tensor<1x1534x1xf64>) -> tensor<1534x1xf64>
    %h_c3 = stablehlo.reshape %h_r3 : (tensor<1x1534x1xf64>) -> tensor<1534x1xf64>
    %h_c4 = stablehlo.reshape %h_r4 : (tensor<1x1534x1xf64>) -> tensor<1534x1xf64>
    %h_c5 = stablehlo.reshape %h_r5 : (tensor<1x1534x1xf64>) -> tensor<1534x1xf64>
    %h_c6 = stablehlo.reshape %h_l0 : (tensor<1x1534x1xf64>) -> tensor<1534x1xf64>
    %h_c7 = stablehlo.reshape %h_l1 : (tensor<1x1534x1xf64>) -> tensor<1534x1xf64>
    %h_c8 = stablehlo.reshape %h_l2 : (tensor<1x1534x1xf64>) -> tensor<1534x1xf64>
    %h_c9 = stablehlo.reshape %h_l3 : (tensor<1x1534x1xf64>) -> tensor<1534x1xf64>
    %h_c10 = stablehlo.reshape %h_l4 : (tensor<1x1534x1xf64>) -> tensor<1534x1xf64>
    %h_c11 = stablehlo.reshape %h_l5 : (tensor<1x1534x1xf64>) -> tensor<1534x1xf64>

    %h_interior = stablehlo.reshape %h_full : (tensor<1x1534x3056xf64>) -> tensor<1534x3056xf64>
    %h_wrapped = "enzymexla.wrap"(%h_interior) <{dimension = 1 : i64, lhs = 1 : i64, rhs = 1 : i64}>
      : (tensor<1534x3056xf64>) -> tensor<1534x3058xf64>

    %out_h = stablehlo.concatenate
      %h_c0, %h_c1, %h_c2, %h_c3, %h_c4, %h_c5,
      %h_wrapped,
      %h_c6, %h_c7, %h_c8, %h_c9, %h_c10, %h_c11, dim = 1
      : (tensor<1534x1xf64>, tensor<1534x1xf64>, tensor<1534x1xf64>,
         tensor<1534x1xf64>, tensor<1534x1xf64>, tensor<1534x1xf64>,
         tensor<1534x3058xf64>,
         tensor<1534x1xf64>, tensor<1534x1xf64>, tensor<1534x1xf64>,
         tensor<1534x1xf64>, tensor<1534x1xf64>, tensor<1534x1xf64>)
      -> tensor<1534x3070xf64>

    return %out_e, %out_g, %out_f, %out_h
      : tensor<1534x3070xf64>, tensor<1534x3070xf64>,
        tensor<1534x3070xf64>, tensor<1534x3070xf64>
  }
}

// CHECK:  func.func @main(%arg0: tensor<20x1536x3056xf64> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}, {"x"}]>}, %arg1: tensor<20x1536x3056xf64> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}, {"x"}]>}, %arg2: tensor<20x1536x3056xf64> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}, {"x"}]>}, %arg3: tensor<20x1536x3056xf64> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}, {"x"}]>}) -> (tensor<1534x3070xf64> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x"}]>}, tensor<1534x3070xf64> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x"}]>}, tensor<1534x3070xf64> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x"}]>}, tensor<1534x3070xf64> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x"}]>}) {
// CHECK-NEXT:    %0 = stablehlo.slice %arg0 [12:13, 1:1535, 0:3056] : (tensor<20x1536x3056xf64>) -> tensor<1x1534x3056xf64>
// CHECK-NEXT:    %1 = stablehlo.reshape %0 : (tensor<1x1534x3056xf64>) -> tensor<1534x3056xf64>
// CHECK-NEXT:    %2 = "enzymexla.wrap"(%1) <{dimension = 1 : i64, lhs = 7 : i64, rhs = 7 : i64}> : (tensor<1534x3056xf64>) -> tensor<1534x3070xf64>
// CHECK-NEXT:    %3 = stablehlo.slice %arg1 [11:12, 1:1535, 0:3056] : (tensor<20x1536x3056xf64>) -> tensor<1x1534x3056xf64>
// CHECK-NEXT:    %4 = stablehlo.reshape %3 : (tensor<1x1534x3056xf64>) -> tensor<1534x3056xf64>
// CHECK-NEXT:    %5 = "enzymexla.wrap"(%4) <{dimension = 1 : i64, lhs = 7 : i64, rhs = 7 : i64}> : (tensor<1534x3056xf64>) -> tensor<1534x3070xf64>
// CHECK-NEXT:    %6 = stablehlo.slice %arg2 [12:13, 1:1535, 0:3056] : (tensor<20x1536x3056xf64>) -> tensor<1x1534x3056xf64>
// CHECK-NEXT:    %7 = stablehlo.reshape %6 : (tensor<1x1534x3056xf64>) -> tensor<1534x3056xf64>
// CHECK-NEXT:    %8 = "enzymexla.wrap"(%7) <{dimension = 1 : i64, lhs = 7 : i64, rhs = 7 : i64}> : (tensor<1534x3056xf64>) -> tensor<1534x3070xf64>
// CHECK-NEXT:    %9 = stablehlo.slice %arg3 [11:12, 1:1535, 0:3056] : (tensor<20x1536x3056xf64>) -> tensor<1x1534x3056xf64>
// CHECK-NEXT:    %10 = stablehlo.reshape %9 : (tensor<1x1534x3056xf64>) -> tensor<1534x3056xf64>
// CHECK-NEXT:    %11 = "enzymexla.wrap"(%10) <{dimension = 1 : i64, lhs = 7 : i64, rhs = 7 : i64}> : (tensor<1534x3056xf64>) -> tensor<1534x3070xf64>
// CHECK-NEXT:    return %2, %5, %8, %11 : tensor<1534x3070xf64>, tensor<1534x3070xf64>, tensor<1534x3070xf64>, tensor<1534x3070xf64>
// CHECK-NEXT:  }
