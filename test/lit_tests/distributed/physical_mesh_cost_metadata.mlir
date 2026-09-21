// RUN: enzymexlamlir-opt --split-input-file --verify-diagnostics %s | FileCheck %s
// A roundtrip is only complete if the printed form parses back.
// RUN: enzymexlamlir-opt --split-input-file --verify-diagnostics %s | enzymexlamlir-opt --split-input-file | FileCheck %s

// The cost metadata is optional and roundtrips when present.
module {
  // CHECK: distributed.PhysicalMesh @mesh0 device_target "gpu" axes [!distributed.physical_comm_axis<2, 4>, !distributed.physical_comm_axis<4, 1>] {axis_bandwidths = [9.000000e+11, 5.000000e+10], axis_latencies = [1.000000e-03, 5.000000e-03], device_flops = 1.000000e+15 : f64, device_mem_bandwidth = 3.000000e+12 : f64, launch_latency = 2.000000e-05 : f64}
  distributed.PhysicalMesh @mesh0 device_target "gpu" axes [!distributed.physical_comm_axis<2, 4>, !distributed.physical_comm_axis<4, 1>] {axis_bandwidths = [9.0e11, 5.0e10], axis_latencies = [1.0e-3, 5.0e-3], device_flops = 1.0e15, device_mem_bandwidth = 3.0e12, launch_latency = 2.0e-5}
}

// -----

module {
  // CHECK: distributed.PhysicalMesh @bare device_target "cpu" axes [!distributed.physical_comm_axis<2, 1>]
  distributed.PhysicalMesh @bare device_target "cpu" axes [!distributed.physical_comm_axis<2, 1>]
}

// -----

module {
  // expected-error @+1 {{requires axis_bandwidths to have one entry per axis (2), got 1}}
  distributed.PhysicalMesh @m device_target "gpu" axes [!distributed.physical_comm_axis<2, 4>, !distributed.physical_comm_axis<4, 1>] {axis_bandwidths = [9.0e11]}
}

// -----

module {
  // expected-error @+1 {{requires axis_bandwidths[0] to be positive and finite}}
  distributed.PhysicalMesh @m device_target "gpu" axes [!distributed.physical_comm_axis<2, 1>] {axis_bandwidths = [0.0e0]}
}

// -----

module {
  // expected-error @+1 {{requires axis_latencies[0] to be non-negative and finite}}
  distributed.PhysicalMesh @m device_target "gpu" axes [!distributed.physical_comm_axis<2, 1>] {axis_latencies = [-1.0e-6]}
}

// -----

module {
  // expected-error @+1 {{requires device_flops to be positive and finite}}
  distributed.PhysicalMesh @m device_target "gpu" axes [!distributed.physical_comm_axis<2, 1>] {device_flops = 0.0e0}
}

// -----

module {
  // expected-error @+1 {{requires launch_latency to be non-negative and finite}}
  distributed.PhysicalMesh @m device_target "gpu" axes [!distributed.physical_comm_axis<2, 1>] {launch_latency = -1.0e0}
}

// -----

module {
  // expected-error @+1 {{requires device_mem_bandwidth to be positive and finite}}
  distributed.PhysicalMesh @m device_target "gpu" axes [!distributed.physical_comm_axis<2, 1>] {device_mem_bandwidth = 0.0e0}
}
