// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=group_comms_concat" --transform-interpreter --enzyme-hlo-remove-transform -allow-unregistered-dialect %s | FileCheck %s

module @"reactant_loop!" attributes {mhlo.num_partitions = 1 : i64, mhlo.num_replicas = 1 : i64} {
  func.func @main(%arg15: tensor<1x24x96xf64>, %6755 : tensor<1x8x80xf64>) -> (tensor<1x24x96xf64>) {

    %6839 = stablehlo.slice %arg15 [0:1, 0:7, 0:8] : (tensor<1x24x96xf64>) -> tensor<1x7x8xf64>

    %6840 = stablehlo.slice %6755 [0:1, 0:1, 72:80] : (tensor<1x8x80xf64>) -> tensor<1x1x8xf64>
    %6841 = stablehlo.slice %6755 [0:1, 0:8, 72:80] : (tensor<1x8x80xf64>) -> tensor<1x8x8xf64>
    %6850 = stablehlo.slice %6755 [0:1, 7:8, 72:80] : (tensor<1x8x80xf64>) -> tensor<1x1x8xf64>

    %6851 = stablehlo.slice %arg15 [0:1, 17:24, 0:8] : (tensor<1x24x96xf64>) -> tensor<1x7x8xf64>

    %6852 = stablehlo.concatenate %6839, %6840, %6841, %6850, %6851, dim = 1 : (tensor<1x7x8xf64>, tensor<1x1x8xf64>, tensor<1x8x8xf64>, tensor<1x1x8xf64>, tensor<1x7x8xf64>) -> tensor<1x24x8xf64>

    %6838 = stablehlo.slice %arg15 [0:1, 0:7, 8:88] : (tensor<1x24x96xf64>) -> tensor<1x7x80xf64>
    
    %6836 = stablehlo.slice %6755 [0:1, 0:1, 0:80] : (tensor<1x8x80xf64>) -> tensor<1x1x80xf64>
    %6837 = stablehlo.slice %6755 [0:1, 7:8, 0:80] : (tensor<1x8x80xf64>) -> tensor<1x1x80xf64>
    
    %6845 = stablehlo.slice %arg15 [0:1, 17:24, 8:88] : (tensor<1x24x96xf64>) -> tensor<1x7x80xf64>

    %6846 = stablehlo.concatenate %6838, %6836, %6755, %6837, %6845, dim = 1 : (tensor<1x7x80xf64>, tensor<1x1x80xf64>, tensor<1x8x80xf64>, tensor<1x1x80xf64>, tensor<1x7x80xf64>) -> tensor<1x24x80xf64>

    %6842 = stablehlo.slice %arg15 [0:1, 0:7, 88:96] : (tensor<1x24x96xf64>) -> tensor<1x7x8xf64>

    %6843 = stablehlo.slice %6755 [0:1, 0:1, 0:8] : (tensor<1x8x80xf64>) -> tensor<1x1x8xf64>
    %6844 = stablehlo.slice %6755 [0:1, 0:8, 0:8] : (tensor<1x8x80xf64>) -> tensor<1x8x8xf64>
    %6847 = stablehlo.slice %6755 [0:1, 7:8, 0:8] : (tensor<1x8x80xf64>) -> tensor<1x1x8xf64>

    %6848 = stablehlo.slice %arg15 [0:1, 17:24, 88:96] : (tensor<1x24x96xf64>) -> tensor<1x7x8xf64>

    %6849 = stablehlo.concatenate %6842, %6843, %6844, %6847, %6848, dim = 1 : (tensor<1x7x8xf64>, tensor<1x1x8xf64>, tensor<1x8x8xf64>, tensor<1x1x8xf64>, tensor<1x7x8xf64>) -> tensor<1x24x8xf64>

    %6853 = stablehlo.concatenate %6852, %6846, %6849, dim = 2 : (tensor<1x24x8xf64>, tensor<1x24x80xf64>, tensor<1x24x8xf64>) -> tensor<1x24x96xf64>

    return %6853 : tensor<1x24x96xf64>
  }
}

// CHECK:  func.func @main(%arg0: tensor<1x24x96xf64>, %arg1: tensor<1x8x80xf64>) -> tensor<1x24x96xf64> {
// CHECK-NEXT:    %0 = "enzymexla.comm_region"() ({
// CHECK-NEXT:      %1 = stablehlo.slice %arg0 [0:1, 0:7, 0:8] : (tensor<1x24x96xf64>) -> tensor<1x7x8xf64>
// CHECK-NEXT:      %2 = stablehlo.slice %arg1 [0:1, 0:1, 72:80] : (tensor<1x8x80xf64>) -> tensor<1x1x8xf64>
// CHECK-NEXT:      %3 = stablehlo.slice %arg1 [0:1, 0:8, 72:80] : (tensor<1x8x80xf64>) -> tensor<1x8x8xf64>
// CHECK-NEXT:      %4 = stablehlo.slice %arg1 [0:1, 7:8, 72:80] : (tensor<1x8x80xf64>) -> tensor<1x1x8xf64>
// CHECK-NEXT:      %5 = stablehlo.slice %arg0 [0:1, 17:24, 0:8] : (tensor<1x24x96xf64>) -> tensor<1x7x8xf64>
// CHECK-NEXT:      %6 = stablehlo.concatenate %1, %2, %3, %4, %5, dim = 1 : (tensor<1x7x8xf64>, tensor<1x1x8xf64>, tensor<1x8x8xf64>, tensor<1x1x8xf64>, tensor<1x7x8xf64>) -> tensor<1x24x8xf64>
// CHECK-NEXT:      %7 = stablehlo.slice %arg0 [0:1, 0:7, 8:88] : (tensor<1x24x96xf64>) -> tensor<1x7x80xf64>
// CHECK-NEXT:      %8 = stablehlo.slice %arg1 [0:1, 0:1, 0:80] : (tensor<1x8x80xf64>) -> tensor<1x1x80xf64>
// CHECK-NEXT:      %9 = stablehlo.slice %arg1 [0:1, 7:8, 0:80] : (tensor<1x8x80xf64>) -> tensor<1x1x80xf64>
// CHECK-NEXT:      %10 = stablehlo.slice %arg0 [0:1, 17:24, 8:88] : (tensor<1x24x96xf64>) -> tensor<1x7x80xf64>
// CHECK-NEXT:      %11 = stablehlo.concatenate %7, %8, %arg1, %9, %10, dim = 1 : (tensor<1x7x80xf64>, tensor<1x1x80xf64>, tensor<1x8x80xf64>, tensor<1x1x80xf64>, tensor<1x7x80xf64>) -> tensor<1x24x80xf64>
// CHECK-NEXT:      %12 = stablehlo.slice %arg0 [0:1, 0:7, 88:96] : (tensor<1x24x96xf64>) -> tensor<1x7x8xf64>
// CHECK-NEXT:      %13 = stablehlo.slice %arg1 [0:1, 0:1, 0:8] : (tensor<1x8x80xf64>) -> tensor<1x1x8xf64>
// CHECK-NEXT:      %14 = stablehlo.slice %arg1 [0:1, 0:8, 0:8] : (tensor<1x8x80xf64>) -> tensor<1x8x8xf64>
// CHECK-NEXT:      %15 = stablehlo.slice %arg1 [0:1, 7:8, 0:8] : (tensor<1x8x80xf64>) -> tensor<1x1x8xf64>
// CHECK-NEXT:      %16 = stablehlo.slice %arg0 [0:1, 17:24, 88:96] : (tensor<1x24x96xf64>) -> tensor<1x7x8xf64>
// CHECK-NEXT:      %17 = stablehlo.concatenate %12, %13, %14, %15, %16, dim = 1 : (tensor<1x7x8xf64>, tensor<1x1x8xf64>, tensor<1x8x8xf64>, tensor<1x1x8xf64>, tensor<1x7x8xf64>) -> tensor<1x24x8xf64>
// CHECK-NEXT:      %18 = stablehlo.concatenate %6, %11, %17, dim = 2 : (tensor<1x24x8xf64>, tensor<1x24x80xf64>, tensor<1x24x8xf64>) -> tensor<1x24x96xf64>
// CHECK-NEXT:      stablehlo.return %18 : tensor<1x24x96xf64>
// CHECK-NEXT:    }) : () -> tensor<1x24x96xf64>
// CHECK-NEXT:    return %0 : tensor<1x24x96xf64>
// CHECK-NEXT:  }
