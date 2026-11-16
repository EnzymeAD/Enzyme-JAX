// RUN: enzymexlamlir-opt %s -split-input-file -verify-diagnostics

func.func @yield_type_mismatch(%peer : tensor<i32>, %comm : !comm.nccl.comm) -> tensor<4xf32> {
    // expected-error @+1 {{expected yielded types}}
    %result = comm.nccl.group -> tensor<4xf32> {
      %recv = comm.nccl.recv %peer, %comm : (tensor<i32>, !comm.nccl.comm) -> tensor<2xf32>
      comm.nccl.yield %recv : tensor<2xf32>
    }
    return %result : tensor<4xf32>
}

// -----

func.func @yield_outside_group(%buf : tensor<4xf32>) {
    // expected-error @+1 {{expects parent op 'comm.nccl.group'}}
    comm.nccl.yield %buf : tensor<4xf32>
}
