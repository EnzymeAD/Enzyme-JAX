// RUN: enzymexlamlir-opt --split-input-file --sdy-propagation-pipeline --shardy-to-distributed-pipeline=dump-value-axes=true %s 2>&1 >/dev/null | FileCheck %s --check-prefix=DUMP
// RUN: enzymexlamlir-opt --split-input-file --mlir-print-op-on-diagnostic=false --sdy-propagation-pipeline --shardy-to-distributed-pipeline --cse --canonicalize --insert-physical-mesh=$'configuration-string="distributed.PhysicalMesh @mesh16 device_target \"cpu\" axes [!distributed.physical_comm_axis<4, 4>, !distributed.physical_comm_axis<4, 1>]"' --distributed-search-strategies="search-strategy=beam beam-size=4 search-samples=8 search-seed=0" --distributed-search-lowering-pipeline %s 2>/dev/null | FileCheck %s --check-prefix=LOWER

// Gather and scatter run unsharded: every factor of their rule needs
// replication. The reshard points around them keep their neighbours
// shardable, so only the values between those points are unshardable.

// An embedding lookup with i32 indices feeding a dot. The table, the indices,
// the weight and the dot keep shardable axes.
// DUMP: ===== Logical Axes per Value =====
// DUMP-NEXT: %arg0 : {{\[}}%{{[0-9]+}}{{\]}} {{\[}}%{{[0-9]+}}{{\]}}{{$}}
// DUMP-NEXT: %arg1 : {{\[}}%{{[0-9]+}}{{\]}} [] []{{$}}
// DUMP-NEXT: %arg2 : {{\[}}%{{[0-9]+}}{{\]}} {{\[}}%{{[0-9]+}}{{\]}}{{$}}
// DUMP-NEXT: %{{[0-9]+}} : {{\[}}a{{[0-9]+}}(unshardable){{\]}} {{\[}}a{{[0-9]+}}(unshardable){{\]}}{{$}}
// DUMP-NEXT: %{{[0-9]+}} : {{\[}}a{{[0-9]+}}(unshardable){{\]}} [] []{{$}}
// DUMP-NEXT: %{{[0-9]+}} : {{\[}}a{{[0-9]+}}(unshardable){{\]}} [] {{\[}}a{{[0-9]+}}(unshardable){{\]}}{{$}}
// DUMP-NEXT: %{{[0-9]+}} : {{\[}}[[BATCH:a[0-9]+]]{{\]}} [] {{\[}}%{{[0-9]+}}{{\]}}{{$}}
// DUMP-NEXT: %{{[0-9]+}} : {{\[}}[[BATCH]]{{\]}} [] {{\[}}%{{[0-9]+}}{{\]}}{{$}}
// DUMP-NEXT: ===== End Logical Axes per Value =====

// The search lowers the gather on whole tensors, so Shardy never masks or
// clamps its indices.
// LOWER-LABEL: module @gather_i32_indices
// LOWER-NOT: stablehlo.clamp
// LOWER: "stablehlo.gather"({{.*}}: (tensor<512x64xf32>, tensor<8x1x1xi32>) -> tensor<8x1x64xf32>
// LOWER-NOT: stablehlo.clamp
// LOWER: stablehlo.dot_general
module @gather_i32_indices {
  func.func public @main(%emb: tensor<512x64xf32>, %ids: tensor<8x1x1xi32>, %w: tensor<64x64xf32>) -> tensor<8x1x64xf32> {
    %g = "stablehlo.gather"(%emb, %ids) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 64>}> : (tensor<512x64xf32>, tensor<8x1x1xi32>) -> tensor<8x1x64xf32>
    %y = stablehlo.dot_general %g, %w, contracting_dims = [2] x [0] : (tensor<8x1x64xf32>, tensor<64x64xf32>) -> tensor<8x1x64xf32>
    return %y : tensor<8x1x64xf32>
  }
}

// -----

// A scatter-add into a table feeding a dot. The arguments and the dot keep
// shardable axes.
// DUMP: ===== Logical Axes per Value =====
// DUMP-NEXT: %arg0 : {{\[}}%{{[0-9]+}}{{\]}} {{\[}}%{{[0-9]+}}{{\]}}{{$}}
// DUMP-NEXT: %arg1 : {{\[}}%{{[0-9]+}}{{\]}} []{{$}}
// DUMP-NEXT: %arg2 : {{\[}}%{{[0-9]+}}{{\]}} {{\[}}%{{[0-9]+}}{{\]}}{{$}}
// DUMP-NEXT: %arg3 : {{\[}}%{{[0-9]+}}{{\]}} {{\[}}%{{[0-9]+}}{{\]}}{{$}}
// DUMP-NEXT: %{{[0-9]+}} : {{\[}}a{{[0-9]+}}(unshardable){{\]}} {{\[}}a{{[0-9]+}}(unshardable){{\]}}{{$}}
// DUMP-NEXT: %{{[0-9]+}} : {{\[}}a{{[0-9]+}}(unshardable){{\]}} []{{$}}
// DUMP-NEXT: %{{[0-9]+}} : {{\[}}a{{[0-9]+}}(unshardable){{\]}} {{\[}}a{{[0-9]+}}(unshardable){{\]}}{{$}}
// DUMP-NEXT: %{{[0-9]+}} : {{\[}}a{{[0-9]+}}(unshardable){{\]}} {{\[}}a{{[0-9]+}}(unshardable){{\]}}{{$}}
// DUMP-NEXT: %{{[0-9]+}} : {{\[}}[[ROWS:a[0-9]+]]{{\]}} {{\[}}%{{[0-9]+}}{{\]}}{{$}}
// DUMP-NEXT: %{{[0-9]+}} : {{\[}}[[ROWS]]{{\]}} {{\[}}%{{[0-9]+}}{{\]}}{{$}}
// DUMP-NEXT: ===== End Logical Axes per Value =====

// LOWER-LABEL: module @scatter_i32_indices
// LOWER: "stablehlo.scatter"
// LOWER: }) {{.*}}: (tensor<512x64xf32>, tensor<8x1xi32>, tensor<8x64xf32>) -> tensor<512x64xf32>
// LOWER: stablehlo.dot_general
module @scatter_i32_indices {
  func.func public @main(%table: tensor<512x64xf32>, %ids: tensor<8x1xi32>, %upd: tensor<8x64xf32>, %w: tensor<64x64xf32>) -> tensor<512x64xf32> {
    %s = "stablehlo.scatter"(%table, %ids, %upd) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, indices_are_sorted = false, unique_indices = false}> ({
    ^bb0(%a: tensor<f32>, %b: tensor<f32>):
      %r = stablehlo.add %a, %b : tensor<f32>
      stablehlo.return %r : tensor<f32>
    }) : (tensor<512x64xf32>, tensor<8x1xi32>, tensor<8x64xf32>) -> tensor<512x64xf32>
    %y = stablehlo.dot_general %s, %w, contracting_dims = [1] x [0] : (tensor<512x64xf32>, tensor<64x64xf32>) -> tensor<512x64xf32>
    return %y : tensor<512x64xf32>
  }
}

// -----

// A gather whose result Shardy reshards for a sharding constraint converts.
// DUMP: ===== Logical Axes per Value =====
// DUMP: ===== End Logical Axes per Value =====
// LOWER-LABEL: module @gather_feeds_explicit_reshard
// LOWER: "stablehlo.gather"
module @gather_feeds_explicit_reshard {
  sdy.mesh @mesh = <["x"=4]>
  func.func public @main(%emb: tensor<512x64xf32>, %ids: tensor<8x1x1xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}, {}]>}, %w: tensor<64x64xf32>) -> tensor<8x1x64xf32> {
    %g = "stablehlo.gather"(%emb, %ids) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 64>}> : (tensor<512x64xf32>, tensor<8x1x1xi32>) -> tensor<8x1x64xf32>
    %c = sdy.sharding_constraint %g <@mesh, [{}, {}, {"x"}]> : tensor<8x1x64xf32>
    %y = stablehlo.dot_general %c, %w, contracting_dims = [2] x [0] : (tensor<8x1x64xf32>, tensor<64x64xf32>) -> tensor<8x1x64xf32>
    return %y : tensor<8x1x64xf32>
  }
}
