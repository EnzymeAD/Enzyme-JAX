// RUN: enzymexlamlir-opt %s | FileCheck %s --check-prefix=CPU

module {
  func.func @main() -> tensor<i32> {
    %0 = enzymexla.comm_rank : tensor<i32>
    return %0 : tensor<i32>
  }
}

// CPU:  func.func @main() -> tensor<i32> {
// CPU-NEXT:    %0 = enzymexla.comm_rank : tensor<i32>
// CPU-NEXT:    return %0 : tensor<i32>
