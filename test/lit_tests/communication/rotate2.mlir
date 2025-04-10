// RUN: enzymexlamlir-opt --optimize-communication %s | FileCheck %s

module {
  sdy.mesh @mesh = <["x"=2, "y"=4]>
  func.func @rotate(%arg0: tensor<12x1024xi64> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x"}]>}) -> (tensor<12x1024xi64> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x"}]>}) {
    %0 = "enzymexla.rotate"(%arg0) <{amount = 100 : si32, dimension = 1 : si32}> : (tensor<12x1024xi64>) -> tensor12x1024xi64>
    return %0 : tensor<12x1024xi64>
  }                     
} 

// CHECK: "enzymexla.rotate"
