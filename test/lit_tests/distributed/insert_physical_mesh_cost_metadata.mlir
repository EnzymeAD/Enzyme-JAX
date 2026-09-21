// RUN: enzymexlamlir-opt --insert-physical-mesh=$'configuration-string="distributed.PhysicalMesh @meshm device_target \"gpu\" axes [!distributed.physical_comm_axis<2, 4>, !distributed.physical_comm_axis<4, 1>] {axis_bandwidths = [9.0e2, 5.0e1], axis_latencies = [1.0e-3, 5.0e-3], launch_latency = 2.0e-2, device_flops = 1.0e5, device_mem_bandwidth = 3.0e3}"' %s | FileCheck %s

// The cost metadata of the configured physical mesh survives insertion.

// CHECK: distributed.PhysicalMesh @meshm device_target "gpu" axes [!distributed.physical_comm_axis<2, 4>, !distributed.physical_comm_axis<4, 1>] {axis_bandwidths = [9.000000e+02, 5.000000e+01], axis_latencies = [1.000000e-03, 5.000000e-03], device_flops = 1.000000e+05 : f64, device_mem_bandwidth = 3.000000e+03 : f64, launch_latency = 2.000000e-02 : f64}

module {
}
