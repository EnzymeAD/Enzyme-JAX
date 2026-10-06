// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=slice_internal" --transform-interpreter --enzyme-hlo-remove-transform -allow-unregistered-dialect --enzyme-hlo-opt %s | FileCheck %s

// extend(T, lhs, rhs) along dim 2 is [T[0:lhs] | T | T[80-rhs:80]]; a slice that lies entirely inside one
// of the three regions is a slice of T. The rhs region repeats the LAST rhs elements, so a slice of it
// maps to T[80-rhs ...], not T[80 ...].
module @"reactant_loop!" {

  func.func @start(%arg0: tensor<1x8x80xf64>) -> tensor<1x8x8xf64> {
    %extend = "enzymexla.extend"(%arg0) <{dimension = 2 : i64, lhs = 8 : i64, rhs = 8 : i64}> : (tensor<1x8x80xf64>) -> tensor<1x8x96xf64>
    %res = stablehlo.slice %extend [0:1, 0:8, 0:8] : (tensor<1x8x96xf64>) -> tensor<1x8x8xf64>
    stablehlo.return %res : tensor<1x8x8xf64>
  }
  func.func @middle(%arg0: tensor<1x8x80xf64>) -> tensor<1x8x80xf64> {
    %extend = "enzymexla.extend"(%arg0) <{dimension = 2 : i64, lhs = 8 : i64, rhs = 8 : i64}> : (tensor<1x8x80xf64>) -> tensor<1x8x96xf64>
    %res = stablehlo.slice %extend [0:1, 0:8, 8:88] : (tensor<1x8x96xf64>) -> tensor<1x8x80xf64>
    stablehlo.return %res : tensor<1x8x80xf64>
  }
  func.func @end(%arg0: tensor<1x8x80xf64>) -> tensor<1x8x8xf64> {
    %extend = "enzymexla.extend"(%arg0) <{dimension = 2 : i64, lhs = 8 : i64, rhs = 8 : i64}> : (tensor<1x8x80xf64>) -> tensor<1x8x96xf64>
    %res = stablehlo.slice %extend [0:1, 0:8, 88:96] : (tensor<1x8x96xf64>) -> tensor<1x8x8xf64>
    stablehlo.return %res : tensor<1x8x8xf64>
  }
  func.func @last_plane(%arg0: tensor<7x8xf32>) -> tensor<7x1xf32> {
    %extend = "enzymexla.extend"(%arg0) <{dimension = 1 : i64, lhs = 1 : i64, rhs = 1 : i64}> : (tensor<7x8xf32>) -> tensor<7x10xf32>
    %res = stablehlo.slice %extend [0:7, 9:10] : (tensor<7x10xf32>) -> tensor<7x1xf32>
    stablehlo.return %res : tensor<7x1xf32>
  }
  func.func @none(%arg0: tensor<1x8x80xf64>) -> tensor<1x8x14xf64> {
    %extend = "enzymexla.extend"(%arg0) <{dimension = 2 : i64, lhs = 8 : i64, rhs = 8 : i64}> : (tensor<1x8x80xf64>) -> tensor<1x8x96xf64>
    %res = stablehlo.slice %extend [0:1, 0:8, 6:20] : (tensor<1x8x96xf64>) -> tensor<1x8x14xf64>
    stablehlo.return %res : tensor<1x8x14xf64>
  }
}

// CHECK:    func.func @start(%arg0: tensor<1x8x80xf64>) -> tensor<1x8x8xf64> {
// CHECK-NEXT:      %0 = stablehlo.slice %arg0 [0:1, 0:8, 0:8] : (tensor<1x8x80xf64>) -> tensor<1x8x8xf64>
// CHECK-NEXT:      stablehlo.return %0 : tensor<1x8x8xf64>
// CHECK-NEXT:    }

// CHECK:    func.func @middle(%arg0: tensor<1x8x80xf64>) -> tensor<1x8x80xf64> {
// CHECK-NEXT:      stablehlo.return %arg0 : tensor<1x8x80xf64>
// CHECK-NEXT:    }

// CHECK:    func.func @end(%arg0: tensor<1x8x80xf64>) -> tensor<1x8x8xf64> {
// CHECK-NEXT:      %0 = stablehlo.slice %arg0 [0:1, 0:8, 72:80] : (tensor<1x8x80xf64>) -> tensor<1x8x8xf64>
// CHECK-NEXT:      stablehlo.return %0 : tensor<1x8x8xf64>
// CHECK-NEXT:    }

// CHECK:    func.func @last_plane(%arg0: tensor<7x8xf32>) -> tensor<7x1xf32> {
// CHECK-NEXT:      %0 = stablehlo.slice %arg0 [0:7, 7:8] : (tensor<7x8xf32>) -> tensor<7x1xf32>
// CHECK-NEXT:      stablehlo.return %0 : tensor<7x1xf32>
// CHECK-NEXT:    }

// CHECK:    func.func @none(%arg0: tensor<1x8x80xf64>) -> tensor<1x8x14xf64> {
// CHECK-NEXT:      %0 = "enzymexla.extend"(%arg0) <{dimension = 2 : i64, lhs = 8 : i64, rhs = 8 : i64}> : (tensor<1x8x80xf64>) -> tensor<1x8x96xf64>
// CHECK-NEXT:      %1 = stablehlo.slice %0 [0:1, 0:8, 6:20] : (tensor<1x8x96xf64>) -> tensor<1x8x14xf64>
// CHECK-NEXT:      stablehlo.return %1 : tensor<1x8x14xf64>
// CHECK-NEXT:    }
