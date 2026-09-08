// RUN: enzymexlamlir-opt --insert-physical-mesh=$'configuration-string="distributed.PhysicalMesh @mesh448 device_target \"cpu\" axes [!distributed.physical_comm_axis<4, 32>,!distributed.physical_comm_axis<4, 8>, !distributed.physical_comm_axis<8, 1>]"' %s
module {}
// CHECK-NEXT: module {
// CHECK-NEXT:   distributed.PhysicalMesh @mesh448 device_target "cpu" axes [!distributed.physical_comm_axis<4, 32>, !distributed.physical_comm_axis<4, 8>, !distributed.physical_comm_axis<8, 1>]
// CHECK-NEXT:   %0:3 = distributed.GetPhysicalMeshAxes @mesh448 : !distributed.physical_comm_axis<4, 32>, !distributed.physical_comm_axis<4, 8>, !distributed.physical_comm_axis<8, 1>
// CHECK-NEXT: }