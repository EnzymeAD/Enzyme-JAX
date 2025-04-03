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

// CHECK:  func.func @main(%arg0: tensor<144x1024x1008xf64>, %arg1: tensor<i64>, %arg2: tensor<128x1009x1008xf64>, %arg3: tensor<1x1008x1008xf64>, %arg4: tensor<1x1008x1008xf64>) -> tensor<144x1024x1008xf64> {
// CHECK-NEXT:    %c = stablehlo.constant dense<8> : tensor<i64>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<129> : tensor<i64>
// CHECK-NEXT:    %c_1 = stablehlo.constant dense<7> : tensor<i64>
// CHECK-NEXT:    %c_2 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %c_3 = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:    %0 = stablehlo.slice %arg0 [7:137, 1:1024, 0:1008] : (tensor<144x1024x1008xf64>) -> tensor<130x1023x1008xf64>
// CHECK-NEXT:    %1 = stablehlo.slice %arg0 [0:130, 7:1016, 0:1008] : (tensor<144x1024x1008xf64>) -> tensor<130x1009x1008xf64>
// CHECK-NEXT:    %2:2 = stablehlo.while(%iterArg = %c_2, %iterArg_4 = %1) : tensor<i64>, tensor<130x1009x1008xf64>
// CHECK-NEXT:     cond {
// CHECK-NEXT:      %4 = stablehlo.compare  LT, %iterArg, %arg1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:      stablehlo.return %4 : tensor<i1>
// CHECK-NEXT:    } do {
// CHECK-NEXT:      %4 = stablehlo.dynamic_update_slice %0, %iterArg_4, %c_2, %c_1, %c_2 : (tensor<130x1023x1008xf64>, tensor<130x1009x1008xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<130x1023x1008xf64>
// CHECK-NEXT:      %5 = stablehlo.dynamic_update_slice %4, %arg2, %c_3, %c_1, %c_2 : (tensor<130x1023x1008xf64>, tensor<128x1009x1008xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<130x1023x1008xf64>
// CHECK-NEXT:      %6 = stablehlo.slice %5 [1:129, 1:1023, 1001:1008] : (tensor<130x1023x1008xf64>) -> tensor<128x1022x7xf64>
// CHECK-NEXT:      %7 = stablehlo.slice %5 [1:129, 1:1023, 0:1008] : (tensor<130x1023x1008xf64>) -> tensor<128x1022x1008xf64>
// CHECK-NEXT:      %8 = stablehlo.slice %5 [1:129, 1:1023, 0:7] : (tensor<130x1023x1008xf64>) -> tensor<128x1022x7xf64>
// CHECK-NEXT:      %9 = stablehlo.slice %5 [1:129, 0:1022, 1001:1008] : (tensor<130x1023x1008xf64>) -> tensor<128x1022x7xf64>
// CHECK-NEXT:      %10 = stablehlo.slice %5 [1:129, 0:1022, 0:1008] : (tensor<130x1023x1008xf64>) -> tensor<128x1022x1008xf64>
// CHECK-NEXT:      %11 = stablehlo.slice %5 [1:129, 0:1022, 0:7] : (tensor<130x1023x1008xf64>) -> tensor<128x1022x7xf64>
// CHECK-NEXT:      "test.use"(%6, %7, %8, %9, %10, %11) : (tensor<128x1022x7xf64>, tensor<128x1022x1008xf64>, tensor<128x1022x7xf64>, tensor<128x1022x7xf64>, tensor<128x1022x1008xf64>, tensor<128x1022x7xf64>) -> ()
// CHECK-NEXT:      %12 = stablehlo.dynamic_update_slice %5, %arg3, %c_2, %c_1, %c_2 : (tensor<130x1023x1008xf64>, tensor<1x1008x1008xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<130x1023x1008xf64>
// CHECK-NEXT:      %13 = stablehlo.dynamic_update_slice %12, %arg4, %c_0, %c_1, %c_2 : (tensor<130x1023x1008xf64>, tensor<1x1008x1008xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<130x1023x1008xf64>
// CHECK-NEXT:      %14 = stablehlo.add %iterArg, %c_3 : tensor<i64>
// CHECK-NEXT:      %15 = stablehlo.slice %13 [0:130, 7:1016, 0:1008] : (tensor<130x1023x1008xf64>) -> tensor<130x1009x1008xf64>
// CHECK-NEXT:      stablehlo.return %14, %15 : tensor<i64>, tensor<130x1009x1008xf64>
// CHECK-NEXT:    }
// CHECK-NEXT:    %3 = stablehlo.dynamic_update_slice %arg0, %2#1, %c_1, %c, %c_2 : (tensor<144x1024x1008xf64>, tensor<130x1009x1008xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<144x1024x1008xf64>
// CHECK-NEXT:    return %3 : tensor<144x1024x1008xf64>
// CHECK-NEXT:  }


// FULL:  func.func @main(%arg0: tensor<144x1024x1008xf64>, %arg1: tensor<i64>, %arg2: tensor<128x1009x1008xf64>, %arg3: tensor<1x1008x1008xf64>, %arg4: tensor<1x1008x1008xf64>) -> tensor<144x1024x1008xf64> {
// FULL-NEXT:    %c = stablehlo.constant dense<8> : tensor<i64>
// FULL-NEXT:    %c_0 = stablehlo.constant dense<7> : tensor<i64>
// FULL-NEXT:    %c_1 = stablehlo.constant dense<0> : tensor<i64>
// FULL-NEXT:    %c_2 = stablehlo.constant dense<1> : tensor<i64>
// FULL-NEXT:    %0 = stablehlo.slice %arg0 [0:130, 7:1016, 0:1008] : (tensor<144x1024x1008xf64>) -> tensor<130x1009x1008xf64>
// FULL-NEXT:    %1:2 = stablehlo.while(%iterArg = %c_1, %iterArg_3 = %0) : tensor<i64>, tensor<130x1009x1008xf64>
// FULL-NEXT:     cond {
// FULL-NEXT:      %3 = stablehlo.compare  LT, %iterArg, %arg1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// FULL-NEXT:      stablehlo.return %3 : tensor<i1>
// FULL-NEXT:    } do {
// FULL-NEXT:      %3 = stablehlo.slice %arg0 [8:136, 2:8, 1001:1008] : (tensor<144x1024x1008xf64>) -> tensor<128x6x7xf64>
// FULL-NEXT:      %4 = stablehlo.slice %arg2 [0:128, 0:1009, 1001:1008] : (tensor<128x1009x1008xf64>) -> tensor<128x1009x7xf64>
// FULL-NEXT:      %5 = stablehlo.slice %arg0 [8:136, 1017:1024, 1001:1008] : (tensor<144x1024x1008xf64>) -> tensor<128x7x7xf64>
// FULL-NEXT:      %6 = stablehlo.concatenate %3, %4, %5, dim = 1 : (tensor<128x6x7xf64>, tensor<128x1009x7xf64>, tensor<128x7x7xf64>) -> tensor<128x1022x7xf64>
// FULL-NEXT:      %7 = stablehlo.slice %arg0 [8:136, 2:8, 0:1008] : (tensor<144x1024x1008xf64>) -> tensor<128x6x1008xf64>
// FULL-NEXT:      %8 = stablehlo.slice %arg0 [8:136, 1017:1024, 0:1008] : (tensor<144x1024x1008xf64>) -> tensor<128x7x1008xf64>
// FULL-NEXT:      %9 = stablehlo.concatenate %7, %arg2, %8, dim = 1 : (tensor<128x6x1008xf64>, tensor<128x1009x1008xf64>, tensor<128x7x1008xf64>) -> tensor<128x1022x1008xf64>
// FULL-NEXT:      %10 = stablehlo.slice %arg0 [8:136, 2:8, 0:7] : (tensor<144x1024x1008xf64>) -> tensor<128x6x7xf64>
// FULL-NEXT:      %11 = stablehlo.slice %arg2 [0:128, 0:1009, 0:7] : (tensor<128x1009x1008xf64>) -> tensor<128x1009x7xf64>
// FULL-NEXT:      %12 = stablehlo.slice %arg0 [8:136, 1017:1024, 0:7] : (tensor<144x1024x1008xf64>) -> tensor<128x7x7xf64>
// FULL-NEXT:      %13 = stablehlo.concatenate %10, %11, %12, dim = 1 : (tensor<128x6x7xf64>, tensor<128x1009x7xf64>, tensor<128x7x7xf64>) -> tensor<128x1022x7xf64>
// FULL-NEXT:      %14 = stablehlo.slice %arg0 [8:136, 1:8, 1001:1008] : (tensor<144x1024x1008xf64>) -> tensor<128x7x7xf64>
// FULL-NEXT:      %15 = stablehlo.slice %arg0 [8:136, 1017:1023, 1001:1008] : (tensor<144x1024x1008xf64>) -> tensor<128x6x7xf64>
// FULL-NEXT:      %16 = stablehlo.concatenate %14, %4, %15, dim = 1 : (tensor<128x7x7xf64>, tensor<128x1009x7xf64>, tensor<128x6x7xf64>) -> tensor<128x1022x7xf64>
// FULL-NEXT:      %17 = stablehlo.slice %arg0 [8:136, 1:8, 0:1008] : (tensor<144x1024x1008xf64>) -> tensor<128x7x1008xf64>
// FULL-NEXT:      %18 = stablehlo.slice %arg0 [8:136, 1017:1023, 0:1008] : (tensor<144x1024x1008xf64>) -> tensor<128x6x1008xf64>
// FULL-NEXT:      %19 = stablehlo.concatenate %17, %arg2, %18, dim = 1 : (tensor<128x7x1008xf64>, tensor<128x1009x1008xf64>, tensor<128x6x1008xf64>) -> tensor<128x1022x1008xf64>
// FULL-NEXT:      %20 = stablehlo.slice %arg0 [8:136, 1:8, 0:7] : (tensor<144x1024x1008xf64>) -> tensor<128x7x7xf64>
// FULL-NEXT:      %21 = stablehlo.slice %arg0 [8:136, 1017:1023, 0:7] : (tensor<144x1024x1008xf64>) -> tensor<128x6x7xf64>
// FULL-NEXT:      %22 = stablehlo.concatenate %20, %11, %21, dim = 1 : (tensor<128x7x7xf64>, tensor<128x1009x7xf64>, tensor<128x6x7xf64>) -> tensor<128x1022x7xf64>
// FULL-NEXT:      "test.use"(%6, %9, %13, %16, %19, %22) : (tensor<128x1022x7xf64>, tensor<128x1022x1008xf64>, tensor<128x1022x7xf64>, tensor<128x1022x7xf64>, tensor<128x1022x1008xf64>, tensor<128x1022x7xf64>) -> ()
// FULL-NEXT:      %23 = stablehlo.slice %iterArg_3 [0:1, 1008:1009, 0:1008] : (tensor<130x1009x1008xf64>) -> tensor<1x1x1008xf64>
// FULL-NEXT:      %24 = stablehlo.concatenate %arg3, %23, dim = 1 : (tensor<1x1008x1008xf64>, tensor<1x1x1008xf64>) -> tensor<1x1009x1008xf64>
// FULL-NEXT:      %25 = stablehlo.slice %iterArg_3 [129:130, 1008:1009, 0:1008] : (tensor<130x1009x1008xf64>) -> tensor<1x1x1008xf64>
// FULL-NEXT:      %26 = stablehlo.concatenate %arg4, %25, dim = 1 : (tensor<1x1008x1008xf64>, tensor<1x1x1008xf64>) -> tensor<1x1009x1008xf64>
// FULL-NEXT:      %27 = stablehlo.concatenate %24, %arg2, %26, dim = 0 : (tensor<1x1009x1008xf64>, tensor<128x1009x1008xf64>, tensor<1x1009x1008xf64>) -> tensor<130x1009x1008xf64>
// FULL-NEXT:      %28 = stablehlo.add %iterArg, %c_2 : tensor<i64>
// FULL-NEXT:      stablehlo.return %28, %27 : tensor<i64>, tensor<130x1009x1008xf64>
// FULL-NEXT:    }
// FULL-NEXT:    %2 = stablehlo.dynamic_update_slice %arg0, %1#1, %c_0, %c, %c_1 : (tensor<144x1024x1008xf64>, tensor<130x1009x1008xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<144x1024x1008xf64>
// FULL-NEXT:    return %2 : tensor<144x1024x1008xf64>
// FULL-NEXT:  }