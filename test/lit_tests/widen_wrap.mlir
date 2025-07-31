// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

module @"reactant_loop!" {
  func.func @main(%iterArg_326 : tensor<1x1536x3072xf64>, %1125 : tensor<1x1520x3056xf64>) -> tensor<1x1536x3072xf64> {

            %1185 = stablehlo.slice %iterArg_326 [0:1, 0:7, 3056:3058] : (tensor<1x1536x3072xf64>) -> tensor<1x7x2xf64>
    
                %1140 = "enzymexla.extend"(%1125) <{dimension = 1 : i64, lhs = 1 : i64, rhs = 1 : i64}> : (tensor<1x1520x3056xf64>) -> tensor<1x1522x3056xf64>
          %1141 = stablehlo.slice %1140 [0:1, 0:1522, 3048:3050] : (tensor<1x1522x3056xf64>) -> tensor<1x1522x2xf64>

          %1189 = stablehlo.slice %iterArg_326 [0:1, 1529:1536, 3056:3058] : (tensor<1x1536x3072xf64>) -> tensor<1x7x2xf64>
      
    %1190 = stablehlo.concatenate %1185, %1141, %1189, dim = 1 : (tensor<1x7x2xf64>, tensor<1x1522x2xf64>, tensor<1x7x2xf64>) -> tensor<1x1536x2xf64> 


                %1184 = stablehlo.slice %iterArg_326 [0:1, 0:7, 8:3064] : (tensor<1x1536x3072xf64>) -> tensor<1x7x3056xf64>
                %1186 = stablehlo.slice %iterArg_326 [0:1, 1529:1536, 8:3064] : (tensor<1x1536x3072xf64>) -> tensor<1x7x3056xf64> 

          %1187 = stablehlo.concatenate %1184, %1140, %1186, dim = 1 : (tensor<1x7x3056xf64>, tensor<1x1522x3056xf64>, tensor<1x7x3056xf64>) -> tensor<1x1536x3056xf64> 

     %1188 = "enzymexla.wrap"(%1187) <{dimension = 2 : i64, lhs = 6 : i64, rhs = 8 : i64}> : (tensor<1x1536x3056xf64>) -> tensor<1x1536x3070xf64>

%1191 = stablehlo.concatenate %1190, %1188, dim = 2 : (tensor<1x1536x2xf64>, tensor<1x1536x3070xf64>) -> tensor<1x1536x3072xf64>

    stablehlo.return %1191 : tensor<1x1536x3072xf64>
  }
}

// CHECK:    func.func @main() -> tensor<1x8x96xf64> {
// CHECK-NEXT:      %cst = stablehlo.constant dense<3.000000e+00> : tensor<1x8x96xf64>
// CHECK-NEXT:      stablehlo.return %cst : tensor<1x8x96xf64>
// CHECK-NEXT:    }