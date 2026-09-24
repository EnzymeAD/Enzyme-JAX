// RUN: rm -rf %t.dir && mkdir -p %t.dir
// RUN: enzymexlamlir-opt --mlir-print-op-on-diagnostic=false --insert-physical-mesh=$'configuration-string="distributed.PhysicalMesh @mesh4 device_target \"cpu\" axes [!distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>] {device_flops = 1.0e15, device_mem_bandwidth = 1.0e18}"' --shardy-to-distributed-pipeline=preserve-shardy-sharding=true --set-distributed-logging="root=%t.dir module-name=testmod" --distributed-search-strategies="beam-size=20" %s -o /dev/null 2> /dev/null
// RUN: cat %t.dir/testmod-*/Main/logical_axes.json | FileCheck %s --check-prefix=AXES
// RUN: cat %t.dir/testmod-*/Main/op_axis_tags.json | FileCheck %s --check-prefix=TAGS

// In a Shardy baseline module every partitioning axis is a product over
// physical factors, so no logical axis exists to name. Logging must not assume
// one does: axes are listed as empty, and tagged dims resolve to no ids.
// AXES: {"axes":[]}
// TAGS: "argument_shardings":{{.*}}"dims":{{.*}}"output_shardings":{{.*}}"dims"

module {
  sdy.mesh @mesh = <["x"=2, "y"=2]>
  func.func @main(
      %arg0: tensor<1x2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}]>},
      %arg1: tensor<2x1xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>},
      %arg2: tensor<1x2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>},
      %arg3: tensor<2x1xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>})
      -> (tensor<1x1xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>},
          tensor<1x1xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}) {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] :
      (tensor<1x2xf32>, tensor<2x1xf32>) -> tensor<1x1xf32>
    %1 = stablehlo.dot_general %arg2, %arg3, contracting_dims = [1] x [0] :
      (tensor<1x2xf32>, tensor<2x1xf32>) -> tensor<1x1xf32>
    return %0, %1 : tensor<1x1xf32>, tensor<1x1xf32>
  }
}
