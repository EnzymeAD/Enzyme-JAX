// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt="max_constant_expansion=0" | FileCheck %s

module {
  func.func @main() -> (tensor<4x6128x12272xi64>, tensor<4x6128x12272xi64>, tensor<4x6128x12272xi64>, tensor<4x6128x12272xi64>, tensor<4x6128x12272xi64>, tensor<4x6128x12272xi64>, tensor<4x6128x12272xi64>, tensor<4x6128x12272xi64>, tensor<4x6128x12272xi64>, tensor<4x6128x12272xi64>, tensor<4x6128x12272xi64>, tensor<4x6128x12272xi64>, tensor<4x6128x12272xi64>, tensor<4x6128x12272xi64>, tensor<4x6128x12272xi64>, tensor<4x6128x12272xi64>, tensor<4x6128x12272xi64>, tensor<4x6128x12272xi64>) {
    %1384 = stablehlo.iota dim = 1 : tensor<4x6128x12272xi64>
    %1385 = stablehlo.iota dim = 1 : tensor<4x6128x12272xi64>
    %1386 = stablehlo.iota dim = 1 : tensor<4x6128x12272xi64>
    %1387 = stablehlo.iota dim = 1 : tensor<4x6128x12272xi64>
    %1388 = stablehlo.iota dim = 1 : tensor<4x6128x12272xi64>
    %1389 = stablehlo.iota dim = 1 : tensor<4x6128x12272xi64>
    %1390 = stablehlo.iota dim = 1 : tensor<4x6128x12272xi64>
    %1391 = stablehlo.iota dim = 1 : tensor<4x6128x12272xi64>
    %1392 = stablehlo.iota dim = 1 : tensor<4x6128x12272xi64>
    %1393 = stablehlo.iota dim = 1 : tensor<4x6128x12272xi64>
    %1394 = stablehlo.iota dim = 1 : tensor<4x6128x12272xi64>
    %1395 = stablehlo.iota dim = 1 : tensor<4x6128x12272xi64>
    %1396 = stablehlo.iota dim = 1 : tensor<4x6128x12272xi64>
    %1397 = stablehlo.iota dim = 1 : tensor<4x6128x12272xi64>
    %1398 = stablehlo.iota dim = 1 : tensor<4x6128x12272xi64>
    %1399 = stablehlo.iota dim = 1 : tensor<4x6128x12272xi64>
    %1400 = stablehlo.iota dim = 1 : tensor<4x6128x12272xi64>
    %1401 = stablehlo.iota dim = 1 : tensor<4x6128x12272xi64>
    return %1384, %1385, %1386, %1387, %1388, %1389, %1390, %1391, %1392, %1393, %1394, %1395, %1396, %1397, %1398, %1399, %1400, %1401: tensor<4x6128x12272xi64>, tensor<4x6128x12272xi64>, tensor<4x6128x12272xi64>, tensor<4x6128x12272xi64>, tensor<4x6128x12272xi64>, tensor<4x6128x12272xi64>, tensor<4x6128x12272xi64>, tensor<4x6128x12272xi64>, tensor<4x6128x12272xi64>, tensor<4x6128x12272xi64>, tensor<4x6128x12272xi64>, tensor<4x6128x12272xi64>, tensor<4x6128x12272xi64>, tensor<4x6128x12272xi64>, tensor<4x6128x12272xi64>, tensor<4x6128x12272xi64>, tensor<4x6128x12272xi64>, tensor<4x6128x12272xi64>
  }
}

// CHECK:    func.func @main() -> (tensor<4x6128x12272xi64>, tensor<4x6128x12272xi64>, tensor<4x6128x12272xi64>, tensor<4x6128x12272xi64>, tensor<4x6128x12272xi64>, tensor<4x6128x12272xi64>, tensor<4x6128x12272xi64>, tensor<4x6128x12272xi64>, tensor<4x6128x12272xi64>, tensor<4x6128x12272xi64>, tensor<4x6128x12272xi64>, tensor<4x6128x12272xi64>, tensor<4x6128x12272xi64>, tensor<4x6128x12272xi64>, tensor<4x6128x12272xi64>, tensor<4x6128x12272xi64>, tensor<4x6128x12272xi64>, tensor<4x6128x12272xi64>) {
// CHECK-NEXT:    %[[IOTA:.+]] = stablehlo.iota dim = 1 : tensor<4x6128x12272xi64>
// CHECK-NEXT:    return %[[IOTA]], %[[IOTA]], %[[IOTA]], %[[IOTA]], %[[IOTA]], %[[IOTA]], %[[IOTA]], %[[IOTA]], %[[IOTA]], %[[IOTA]], %[[IOTA]], %[[IOTA]], %[[IOTA]], %[[IOTA]], %[[IOTA]], %[[IOTA]], %[[IOTA]], %[[IOTA]] : tensor<4x6128x12272xi64>, tensor<4x6128x12272xi64>, tensor<4x6128x12272xi64>, tensor<4x6128x12272xi64>, tensor<4x6128x12272xi64>, tensor<4x6128x12272xi64>, tensor<4x6128x12272xi64>, tensor<4x6128x12272xi64>, tensor<4x6128x12272xi64>, tensor<4x6128x12272xi64>, tensor<4x6128x12272xi64>, tensor<4x6128x12272xi64>, tensor<4x6128x12272xi64>, tensor<4x6128x12272xi64>, tensor<4x6128x12272xi64>, tensor<4x6128x12272xi64>, tensor<4x6128x12272xi64>, tensor<4x6128x12272xi64>
// CHECK-NEXT:  }
