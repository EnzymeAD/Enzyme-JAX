// RUN: enzymexlamlir-opt --split-input-file --shardy-to-distributed-pipeline=preserve-shardy-sharding=true %s | FileCheck %s

// The sdy meshes need not match the physical mesh's axis structure, only
// its device count and factor boundaries.

// sdy [dp=4, tp=8] over physical <8,4> <4,1>: dp is the high digit of the
// major physical axis (extent 4, stride 2) and tp is the low digit of it
// (extent 2, stride 1) times the whole minor axis.
// CHECK-LABEL: module
module {
  sdy.mesh @mesh = <["dp"=4, "tp"=8]>
  distributed.PhysicalMesh @m device_target "cpu" axes [!distributed.physical_comm_axis<8, 4>, !distributed.physical_comm_axis<4, 1>]
  func.func @main(%x: tensor<8xf32>) -> tensor<8xf32> {
    return %x : tensor<8xf32>
  }
}

// -----

// Two meshes with different splits of the same 8 devices.
// CHECK-LABEL: module
module {
  sdy.mesh @m1 = <["a"=2, "b"=4]>
  sdy.mesh @m2 = <["x"=4, "y"=2]>
  distributed.PhysicalMesh @m device_target "cpu" axes [!distributed.physical_comm_axis<2, 4>, !distributed.physical_comm_axis<4, 1>]
  func.func @main(%x: tensor<8xf32>) -> tensor<8xf32> {
    return %x : tensor<8xf32>
  }
}
