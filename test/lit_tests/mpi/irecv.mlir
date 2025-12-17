// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(lower-enzymexla-mpi{backend=cpu})" %s | FileCheck %s --check-prefix=CPU

module {
  func.func @main(%arg0: tensor<5xf64> {enzymexla.memory_effects = ["read", "write", "allocate", "free"], tf.aliasing_output = 0 : i32}) -> tensor<5xf64> attributes {enzymexla.memory_effects = ["read", "write", "allocate", "free"]} {
    %0 = stablehlo.transpose %arg0, dims = [0] : (tensor<5xf64>) -> tensor<5xf64>
    %c = stablehlo.constant dense<1> : tensor<i32>
    %c_0 = stablehlo.constant dense<42> : tensor<i32>
    %c_1 = stablehlo.constant dense<5> : tensor<i32>
    %c_2 = stablehlo.constant dense<-1> : tensor<i64>
    %outbuf, %outrequest = enzymexla.irecv(%0, %c_1, %c, %c_0, %c_2) : (tensor<5xf64>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i64>) -> (tensor<5xf64>, tensor<i64>)
    enzymexla.wait(%outrequest) : tensor<i64>
    %1 = stablehlo.transpose %outbuf, dims = [0] : (tensor<5xf64>) -> tensor<5xf64>
    return %1 : tensor<5xf64>
  }
}
