// RUN: not enzymexlamlir-opt --split-input-file --shardy-to-distributed-pipeline=preserve-shardy-sharding=true %s 2>&1 | FileCheck %s

// Every case must be rejected by the flag's mesh validation before any
// conversion runs.

// CHECK: requires exactly one distributed.PhysicalMesh
module {
  sdy.mesh @mesh = <["a"=2, "b"=2]>
  func.func @main(%x: tensor<8xf32>) -> tensor<8xf32> {
    return %x : tensor<8xf32>
  }
}

// -----

// CHECK: mesh has 8 devices but the physical mesh has 4
module {
  sdy.mesh @mesh = <["a"=2, "b"=4]>
  distributed.PhysicalMesh @m device_target "cpu" axes [!distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>]
  func.func @main(%x: tensor<8xf32>) -> tensor<8xf32> {
    return %x : tensor<8xf32>
  }
}

// -----

// Axis "b" (extent 6) needs the whole extent-4 minor axis plus 3/2 of the
// next one.
// CHECK: axis "b": extent 6 straddles the boundary of physical axis
module {
  sdy.mesh @mesh = <["a"=2, "b"=6]>
  distributed.PhysicalMesh @m device_target "cpu" axes [!distributed.physical_comm_axis<3, 4>, !distributed.physical_comm_axis<4, 1>]
  func.func @main(%x: tensor<8xf32>) -> tensor<8xf32> {
    return %x : tensor<8xf32>
  }
}

// -----

// Axis "b" (extent 2) does not divide the extent-3 minor physical axis.
// CHECK: axis "b": extent 2 straddles the boundary of physical axis
module {
  sdy.mesh @mesh = <["a"=3, "b"=2]>
  distributed.PhysicalMesh @m device_target "cpu" axes [!distributed.physical_comm_axis<2, 3>, !distributed.physical_comm_axis<3, 1>]
  func.func @main(%x: tensor<8xf32>) -> tensor<8xf32> {
    return %x : tensor<8xf32>
  }
}

// -----

// CHECK: default (iota) device order
module {
  sdy.mesh @mesh = <["a"=2, "b"=2], device_ids=[0, 2, 1, 3]>
  distributed.PhysicalMesh @m device_target "cpu" axes [!distributed.physical_comm_axis<2, 2>, !distributed.physical_comm_axis<2, 1>]
  func.func @main(%x: tensor<8xf32>) -> tensor<8xf32> {
    return %x : tensor<8xf32>
  }
}
