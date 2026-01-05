// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=recognize_updatewithoutcorners" --transform-interpreter --enzyme-hlo-remove-transform %s --split-input-file | FileCheck %s

  func.func @main(%iterArg_388: tensor<6x6130x12272xf64>, %385: tensor<4x6128x12272xf64>) -> (tensor<6x6130x12272xf64>) {
           %1275 = stablehlo.slice %iterArg_388 [0:1, 0:1, 0:12272] : (tensor<6x6130x12272xf64>) -> tensor<1x1x12272xf64>

          %1264 = stablehlo.slice %385 [0:4, 0:1, 0:12272] : (tensor<4x6128x12272xf64>) -> tensor<4x1x12272xf64>

          %1257 = stablehlo.slice %iterArg_388 [5:6, 0:1, 0:12272] : (tensor<6x6130x12272xf64>) -> tensor<1x1x12272xf64>

      %1276 = stablehlo.concatenate %1275, %1264, %1257, dim = 0 : (tensor<1x1x12272xf64>, tensor<4x1x12272xf64>, tensor<1x1x12272xf64>) -> tensor<6x1x12272xf64>


      %1254 = "enzymexla.extend"(%385) <{dimension = 0 : i64, lhs = 1 : i64, rhs = 1 : i64}> : (tensor<4x6128x12272xf64>) -> tensor<6x6128x12272xf64>

          %1287 = stablehlo.slice %iterArg_388 [0:1, 6129:6130, 0:12272] : (tensor<6x6130x12272xf64>) -> tensor<1x1x12272xf64>
          %1267 = stablehlo.slice %385 [0:4, 6127:6128, 0:12272] : (tensor<4x6128x12272xf64>) -> tensor<4x1x12272xf64>

          %1258 = stablehlo.slice %iterArg_388 [5:6, 6129:6130, 0:12272] : (tensor<6x6130x12272xf64>) -> tensor<1x1x12272xf64>
      %1288 = stablehlo.concatenate %1287, %1267, %1258, dim = 0 : (tensor<1x1x12272xf64>, tensor<4x1x12272xf64>, tensor<1x1x12272xf64>) -> tensor<6x1x12272xf64>


    %RES = stablehlo.concatenate %1276, %1254, %1288, dim = 1 : (tensor<6x1x12272xf64>, tensor<6x6128x12272xf64>, tensor<6x1x12272xf64>) -> tensor<6x6130x12272xf64>
      func.return %RES : tensor<6x6130x12272xf64>
  }

