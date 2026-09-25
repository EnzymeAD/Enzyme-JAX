// RUN: sed s/NNN/8/g %s | enzymexlamlir-opt --mlir-print-op-on-diagnostic=false --sdy-propagation-pipeline --shardy-to-distributed-pipeline --cse --canonicalize --insert-physical-mesh=$'configuration-string="distributed.PhysicalMesh @mesh8 device_target \"cpu\" axes [!distributed.physical_comm_axis<8, 1>]"' --distributed-search-strategies="search-strategy=beam beam-size=1000 log-progress=true" -o /dev/null 2>&1 | FileCheck %s --check-prefix=DEFAULT
// RUN: sed s/NNN/8/g %s | enzymexlamlir-opt --mlir-print-op-on-diagnostic=false --sdy-propagation-pipeline --shardy-to-distributed-pipeline --cse --canonicalize --insert-physical-mesh=$'configuration-string="distributed.PhysicalMesh @mesh8 device_target \"cpu\" axes [!distributed.physical_comm_axis<8, 1>]"' --distributed-search-strategies="search-strategy=beam beam-size=1000 log-progress=true max-decision-extent=2" -o /dev/null 2>&1 | FileCheck %s --check-prefix=PRIMES
// RUN: sed s/NNN/8/g %s | not enzymexlamlir-opt --mlir-print-op-on-diagnostic=false --sdy-propagation-pipeline --shardy-to-distributed-pipeline --cse --canonicalize --insert-physical-mesh=$'configuration-string="distributed.PhysicalMesh @mesh8 device_target \"cpu\" axes [!distributed.physical_comm_axis<8, 1>]"' --distributed-search-strategies="max-decision-extent=1" -o /dev/null 2>&1 | FileCheck %s --check-prefix=REJECT
// RUN: sed s/NNN/12/g %s | enzymexlamlir-opt --mlir-print-op-on-diagnostic=false --sdy-propagation-pipeline --shardy-to-distributed-pipeline --cse --canonicalize --insert-physical-mesh=$'configuration-string="distributed.PhysicalMesh @mesh8 device_target \"cpu\" axes [!distributed.physical_comm_axis<12, 1>]"' --distributed-search-strategies="search-strategy=beam beam-size=1000 log-progress=true" -o /dev/null 2>&1 | FileCheck %s --check-prefix=E12 --implicit-check-not='child 1: took chunk'
// RUN: sed s/NNN/9/g %s | enzymexlamlir-opt --mlir-print-op-on-diagnostic=false --sdy-propagation-pipeline --shardy-to-distributed-pipeline --cse --canonicalize --insert-physical-mesh=$'configuration-string="distributed.PhysicalMesh @mesh8 device_target \"cpu\" axes [!distributed.physical_comm_axis<9, 1>]"' --distributed-search-strategies="search-strategy=beam beam-size=1000 log-progress=true" -o /dev/null 2>&1 | FileCheck %s --check-prefix=E9 --implicit-check-not='child 1: took chunk'
// RUN: sed s/NNN/11/g %s | enzymexlamlir-opt --mlir-print-op-on-diagnostic=false --sdy-propagation-pipeline --shardy-to-distributed-pipeline --cse --canonicalize --insert-physical-mesh=$'configuration-string="distributed.PhysicalMesh @mesh8 device_target \"cpu\" axes [!distributed.physical_comm_axis<11, 1>]"' --distributed-search-strategies="search-strategy=beam beam-size=1000 log-progress=true" -o /dev/null 2>&1 | FileCheck %s --check-prefix=E11 --implicit-check-not='child 1: took chunk'
// RUN: sed s/NNN/27/g %s | enzymexlamlir-opt --mlir-print-op-on-diagnostic=false --sdy-propagation-pipeline --shardy-to-distributed-pipeline --cse --canonicalize --insert-physical-mesh=$'configuration-string="distributed.PhysicalMesh @mesh8 device_target \"cpu\" axes [!distributed.physical_comm_axis<27, 1>]"' --distributed-search-strategies="search-strategy=beam beam-size=1000 log-progress=true" -o /dev/null 2>&1 | FileCheck %s --check-prefix=E27 --implicit-check-not='child 1: took chunk'
// RUN: sed s/NNN/16/g %s | enzymexlamlir-opt --mlir-print-op-on-diagnostic=false --sdy-propagation-pipeline --shardy-to-distributed-pipeline --cse --canonicalize --insert-physical-mesh=$'configuration-string="distributed.PhysicalMesh @mesh8 device_target \"cpu\" axes [!distributed.physical_comm_axis<16, 1>]"' --distributed-search-strategies="search-strategy=beam beam-size=1000 log-progress=true" -o /dev/null 2>&1 | FileCheck %s --check-prefix=E16 --implicit-check-not='child 1: took chunk'
// RUN: sed s/NNN/30/g %s | enzymexlamlir-opt --mlir-print-op-on-diagnostic=false --sdy-propagation-pipeline --shardy-to-distributed-pipeline --cse --canonicalize --insert-physical-mesh=$'configuration-string="distributed.PhysicalMesh @mesh8 device_target \"cpu\" axes [!distributed.physical_comm_axis<30, 1>]"' --distributed-search-strategies="search-strategy=beam beam-size=1000 log-progress=true" -o /dev/null 2>&1 | FileCheck %s --check-prefix=E30 --implicit-check-not='child 1: took chunk'

// nextChunkSize takes one chunk per free factor: the smallest prime of
// gcd(remaining extent, factor extent), then further primes while the product
// stays <= max-decision-extent. An extent-8 factor is consumed whole by one
// child by default, and 2 at a time with max-decision-extent=2.
// DEFAULT: took chunk (extent 8) from a physical factor of extent 8
// PRIMES: took chunk (extent 2) from a physical factor of extent 8
// PRIMES: took chunk (extent 2) from a physical factor of extent 4
// REJECT: max-decision-extent must be >= 2, got 1

// Chunk results (the first decision on an axis whose extent equals its
// single physical factor, first child only):
// E12: took chunk (extent 4) from a physical factor of extent 12
// E9: took chunk (extent 3) from a physical factor of extent 9
// E11: took chunk (extent 11) from a physical factor of extent 11
// E27: took chunk (extent 3) from a physical factor of extent 27
// E16: took chunk (extent 8) from a physical factor of extent 16
// E30: took chunk (extent 6) from a physical factor of extent 30

module {
  sdy.mesh @mesh = <["x"=NNN]>
  func.func @main(
      %arg0: tensor<NNNxf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>})
      -> (tensor<NNNxf32>) {
    %0 = stablehlo.add %arg0, %arg0 : tensor<NNNxf32>
    return %0 : tensor<NNNxf32>
  }
}
