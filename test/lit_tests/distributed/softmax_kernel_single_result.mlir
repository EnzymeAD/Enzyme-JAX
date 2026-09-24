// RUN: rm -rf %t.dir && mkdir -p %t.dir
// RUN: enzymexlamlir-opt --mlir-print-op-on-diagnostic=false --sdy-propagation-pipeline --shardy-to-distributed-pipeline --cse --canonicalize --insert-physical-mesh=$'configuration-string="distributed.PhysicalMesh @mesh1 device_target \"cpu\" axes [!distributed.physical_comm_axis<1, 1>] {device_flops = 1.0e12, device_mem_bandwidth = 1.0e11}"' --distributed-search-strategies="beam-size=4 dump-kernel-modules-to=%t.dir" %s -o /dev/null
// RUN: cat %t.dir/*.mlir | FileCheck %s

// A stable softmax on one device becomes a single kernel. Clustering yields
// every value crossing a cluster boundary, so merging leaves intermediates
// yielded but unused; they must be pruned before kernels are dispatched, or
// the kernel body is not a closed subgraph and cannot match a fused softmax.
// CHECK: func.func @kernel
// CHECK: stablehlo.divide
// CHECK-NEXT: return %{{[^,]*}} : tensor<1x4096xf32>
module {
  func.func @main(%x: tensor<1x4096xf32>) -> tensor<1x4096xf32> {
    %ninf = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %zero = stablehlo.constant dense<0.0> : tensor<f32>
    %m = stablehlo.reduce(%x init: %ninf) applies stablehlo.maximum across dimensions = [1] : (tensor<1x4096xf32>, tensor<f32>) -> tensor<1xf32>
    %mb = stablehlo.broadcast_in_dim %m, dims = [0] : (tensor<1xf32>) -> tensor<1x4096xf32>
    %s = stablehlo.subtract %x, %mb : tensor<1x4096xf32>
    %e = stablehlo.exponential %s : tensor<1x4096xf32>
    %sum = stablehlo.reduce(%e init: %zero) applies stablehlo.add across dimensions = [1] : (tensor<1x4096xf32>, tensor<f32>) -> tensor<1xf32>
    %sb = stablehlo.broadcast_in_dim %sum, dims = [0] : (tensor<1xf32>) -> tensor<1x4096xf32>
    %r = stablehlo.divide %e, %sb : tensor<1x4096xf32>
    return %r : tensor<1x4096xf32>
  }
}
