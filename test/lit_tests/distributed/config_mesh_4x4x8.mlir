// RUN: true
distributed.PhysicalMesh @mesh448 device_target "cpu" axes [!distributed.physical_comm_axis<4, 32>,!distributed.physical_comm_axis<4, 8>, !distributed.physical_comm_axis<8, 1>]
