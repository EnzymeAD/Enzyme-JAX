// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt --allow-unregistered-dialect | FileCheck %s

module @"reactant_loop!" attributes {mhlo.num_partitions = 1 : i64, mhlo.num_replicas = 1 : i64} {
  func.func @main(%arg31: tensor<144x1024x1008xf64>, %arg41: tensor<144x1024x1008xf64>, %start : tensor<i64>, %arg2 : tensor<i64>) -> tensor<144x1024x1008xf64> {
    %c_270 = stablehlo.constant dense<1> : tensor<i64>

    %0:2 = stablehlo.while(%iterArg = %start, %iterArg_37 = %arg31) : tensor<i64>, tensor<144x1024x1008xf64>
     cond {
       "test.use"(%iterArg_37) : (tensor<144x1024x1008xf64>) -> ()
      %1 = stablehlo.compare  LT, %iterArg, %arg2 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %1 : tensor<i1>
    } do {
       "test.use"(%iterArg_37) : (tensor<144x1024x1008xf64>) -> ()

      %9224 = stablehlo.add %iterArg, %c_270 : tensor<i64>
      stablehlo.return %9224, %arg41 : tensor<i64>, tensor<144x1024x1008xf64>
    }
    return %0#1 : tensor<144x1024x1008xf64>
  }
}

// CHECK:  func.func @main(%arg0: tensor<144x1024x1008xf64>, %arg1: tensor<144x1024x1008xf64>, %arg2: tensor<i64>, %arg3: tensor<i64>) -> tensor<144x1024x1008xf64> {
// CHECK-NEXT:    %c = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:    "test.use"(%arg0) : (tensor<144x1024x1008xf64>) -> ()
// CHECK-NEXT:    %0 = stablehlo.compare  LT, %arg2, %arg3 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:    %1 = stablehlo.select %0, %arg0, %arg1 : tensor<i1>, tensor<144x1024x1008xf64>
// CHECK-NEXT:    %2 = stablehlo.while(%iterArg = %arg2) : tensor<i64>
// CHECK-NEXT:     cond {
// CHECK-NEXT:      %3 = stablehlo.compare  EQ, %iterArg, %arg2 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:      %4 = stablehlo.select %3, %arg0, %arg1 : tensor<i1>, tensor<144x1024x1008xf64>
// CHECK-NEXT:      "test.use"(%4) : (tensor<144x1024x1008xf64>) -> ()
// CHECK-NEXT:      %5 = stablehlo.compare  LT, %iterArg, %arg3 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:      stablehlo.return %5 : tensor<i1>
// CHECK-NEXT:    } do {
// CHECK-NEXT:      %3 = stablehlo.compare  EQ, %iterArg, %arg2 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:      %4 = stablehlo.select %3, %arg0, %arg1 : tensor<i1>, tensor<144x1024x1008xf64>
// CHECK-NEXT:      "test.use"(%4) : (tensor<144x1024x1008xf64>) -> ()
// CHECK-NEXT:      %5 = stablehlo.add %iterArg, %c : tensor<i64>
// CHECK-NEXT:      stablehlo.return %5 : tensor<i64>
// CHECK-NEXT:    }
// CHECK-NEXT:    return %1 : tensor<144x1024x1008xf64>
// CHECK-NEXT:  }