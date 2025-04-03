// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=while_induction_reduction" --transform-interpreter --enzyme-hlo-remove-transform -allow-unregistered-dialect %s

module @"reactant_loop!" attributes {mhlo.num_partitions = 1 : i64, mhlo.num_replicas = 1 : i64} {
  func.func @main(%arg31: tensor<144x1024x1008xf64>, %arg2 : tensor<i64>, %up1 : tensor<128x1009x1008xf64>, %up2 : tensor<1x1008x1008xf64>, %up3 : tensor<1x1008x1008xf64>) -> (tensor<144x1024x1008xf64>) {
    %c_26 = stablehlo.constant dense<136> : tensor<i64>
    %c_28 = stablehlo.constant dense<7> : tensor<i64>
    %c_30 = stablehlo.constant dense<8> : tensor<i64>
    %c_33 = stablehlo.constant dense<0> : tensor<i64>
    %c_271 = stablehlo.constant dense<0> : tensor<i64>
    %c_270 = stablehlo.constant dense<1> : tensor<i64>

    %0:2 = stablehlo.while(%iterArg = %c_271, %iterArg_37 = %arg31) : tensor<i64>, tensor<144x1024x1008xf64>
     cond {
      %1 = stablehlo.compare  LT, %iterArg, %arg2 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %1 : tensor<i1>
    } do {
      %80 = stablehlo.dynamic_update_slice %iterArg_37, %up1, %c_30, %c_30, %c_33 : (tensor<144x1024x1008xf64>, tensor<128x1009x1008xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<144x1024x1008xf64>

      %117 = stablehlo.slice %80 [8:136, 2:1024, 1001:1008] : (tensor<144x1024x1008xf64>) -> tensor<128x1022x7xf64>
      %118 = stablehlo.slice %80 [8:136, 2:1024, 0:1008] : (tensor<144x1024x1008xf64>) -> tensor<128x1022x1008xf64>
      %119 = stablehlo.slice %80 [8:136, 2:1024, 0:7] : (tensor<144x1024x1008xf64>) -> tensor<128x1022x7xf64>

      %123 = stablehlo.slice %80 [8:136, 1:1023, 1001:1008] : (tensor<144x1024x1008xf64>) -> tensor<128x1022x7xf64>
      %124 = stablehlo.slice %80 [8:136, 1:1023, 0:1008] : (tensor<144x1024x1008xf64>) -> tensor<128x1022x1008xf64>
      %125 = stablehlo.slice %80 [8:136, 1:1023, 0:7] : (tensor<144x1024x1008xf64>) -> tensor<128x1022x7xf64>


      "test.use"(%117, %118, %119, %123, %124, %125) : (tensor<128x1022x7xf64>, tensor<128x1022x1008xf64>, tensor<128x1022x7xf64>, tensor<128x1022x7xf64>, tensor<128x1022x1008xf64>, tensor<128x1022x7xf64>) -> ()

      %87 = stablehlo.dynamic_update_slice %80, %up2, %c_28, %c_30, %c_33 : (tensor<144x1024x1008xf64>, tensor<1x1008x1008xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<144x1024x1008xf64>


      %90 = stablehlo.dynamic_update_slice %87, %up3, %c_26, %c_30, %c_33 : (tensor<144x1024x1008xf64>, tensor<1x1008x1008xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<144x1024x1008xf64>

      %9224 = stablehlo.add %iterArg, %c_270 : tensor<i64>
      stablehlo.return %9224, %90 : tensor<i64>, tensor<144x1024x1008xf64>
    }
    return %0#1 : tensor<144x1024x1008xf64>
  }
}

// CHECK:  func.func @main(%arg0: tensor<24x38x62xf32>, %arg1: tensor<i64>) -> tensor<24x38x62xf32> {
// CHECK-NEXT:    %c = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:    %c_1 = stablehlo.constant dense<7> : tensor<i64>
// CHECK-NEXT:    %0 = stablehlo.slice %arg0 [7:17, 7:31, 7:55] : (tensor<24x38x62xf32>) -> tensor<10x24x48xf32>
// CHECK-NEXT:    %1:2 = stablehlo.while(%iterArg = %c, %iterArg_2 = %0) : tensor<i64>, tensor<10x24x48xf32>
// CHECK-NEXT:     cond {
// CHECK-NEXT:      %3 = stablehlo.compare  LT, %iterArg, %arg1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:      stablehlo.return %3 : tensor<i1>
// CHECK-NEXT:    } do {
// CHECK-NEXT:      %3 = stablehlo.slice %iterArg_2 [0:10, 0:24, 0:48] : (tensor<10x24x48xf32>) -> tensor<10x24x48xf32>
// CHECK-NEXT:      "test.use"(%3) : (tensor<10x24x48xf32>) -> ()
// CHECK-NEXT:      %4 = "test.update"() : () -> tensor<10x24x48xf32>
// CHECK-NEXT:      %5 = stablehlo.add %iterArg, %c_0 : tensor<i64>
// CHECK-NEXT:      stablehlo.return %5, %4 : tensor<i64>, tensor<10x24x48xf32>
// CHECK-NEXT:    }
// CHECK-NEXT:    %2 = stablehlo.dynamic_update_slice %arg0, %1#1, %c_1, %c_1, %c_1 : (tensor<24x38x62xf32>, tensor<10x24x48xf32>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<24x38x62xf32>
// CHECK-NEXT:    return %2 : tensor<24x38x62xf32>
// CHECK-NEXT:  }