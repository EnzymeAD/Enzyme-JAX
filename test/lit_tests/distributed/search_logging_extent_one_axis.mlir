// RUN: rm -rf %t.dir && mkdir -p %t.dir
// RUN: enzymexlamlir-opt --mlir-print-op-on-diagnostic=false --sdy-propagation-pipeline --shardy-to-distributed-pipeline --cse --canonicalize --set-distributed-logging="root=%t.dir module-name=testmod" --insert-physical-mesh=$'configuration-string="distributed.PhysicalMesh @mesh2 device_target \"cpu\" axes [!distributed.physical_comm_axis<2, 1>] {device_flops = 1.0e14, device_mem_bandwidth = 1.0e12}"' --distributed-search-strategies="beam-size=8" %s -o /dev/null
// RUN: cat %t.dir/testmod-*/Main/op_axis_tags.json | FileCheck %s

// Appending a trailing size-1 dimension gives the kernel a partitioning axis
// that is an empty axis.product (an extent-1 axis has no logical axis behind
// it). Logging must tolerate such an axis: it has no logging id, so it is
// omitted from the resolved dims instead of being named.
//
// The output is sharded on "x" along dim 0 and has the empty axis on dim 1, so
// the tagged output dims are one axis id followed by an empty list.
// CHECK: "output_shardings":{{\[}}{"dims":{{\[}}{{\[}}{{[0-9]+}}{{\]}},{{\[}}{{\]}}{{\]}},"unreduced_axes":{{\[}}{{\]}}}{{\]}}

module {
  sdy.mesh @mesh = <["x"=2]>
  func.func @main(
      %arg0: tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>})
      -> (tensor<4x1xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [0] : (tensor<4xf32>) -> tensor<4x1xf32>
    return %0 : tensor<4x1xf32>
  }
}
