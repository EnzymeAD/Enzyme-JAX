// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(raise-affine-to-stablehlo,arith-raise,enzyme-hlo-opt{max_constant_expansion=1})" | FileCheck %s

module {
  func.func private @"##call__Z25gpu__mask_immersed_field_16CompilerMetadataI10StaticSizeI13_180__85__20_E12DynamicCheckvv7NDRangeILi3ES0_I11_12__6__20_ES0_I11_16__16__1_EvvEE11OffsetArrayI7Float64Li3E13CuTracedArrayIS9_Li3ELi1E13_194__99__34_EE5TupleI6Center4FaceSE_E20ImmersedBoundaryGridIS9_8Periodic14RightConnected7Bounded28OrthogonalSphericalShellGridIS9_SI_SJ_SK_28StaticVerticalDiscretizationIS8_IS9_Li1ESA_IS9_Li1ELi1E5_35__EES8_IS9_Li1ESA_IS9_Li1ELi1E5_34__EESO_SQ_ES8_IS9_Li2ESA_IS9_Li2ELi1E9_194__99_EE8TripolarI5Int64SV_SV_EvE16GridFittedBottomI5FieldISE_SE_vvvvS8_IS9_Li3ESA_IS9_Li3ELi1E12_194__99__1_EES9_vvvE23CenterImmersedConditionEvvvES9_#1335$par243"(%arg0: memref<34x99x194xf64, 1>, %arg1: memref<34xf64, 1>, %arg2: memref<1x99x194xf64, 1>) {
    %c-1_i64 = arith.constant -1 : i64
    %c1 = arith.constant 1 : index
    %c1_i64 = arith.constant 1 : i64
    %cst = arith.constant 0.000000e+00 : f64
    affine.parallel (%arg3, %arg4, %arg5) = (0, 0, 0) to (20, 85, 180) {
      %0 = arith.addi %arg4, %c1 : index
      %1 = arith.index_castui %0 : index to i64
      %2 = affine.load %arg1[%arg3 + 7] : memref<34xf64, 1>
      %3 = affine.load %arg2[0, %arg4 + 7, %arg5 + 7] : memref<1x99x194xf64, 1>
      %4 = arith.cmpf ole, %2, %3 {fastmathFlags = #llvm.fastmath<none>} : f64
      %5 = arith.addi %1, %c-1_i64 : i64
      %6 = affine.load %arg2[0, %arg4 + 6, %arg5 + 7] : memref<1x99x194xf64, 1>
      %7 = arith.cmpf ole, %2, %6 {fastmathFlags = #llvm.fastmath<none>} : f64
      %8 = arith.cmpi ult, %5, %c1_i64 : i64
      %9 = arith.ori %8, %7 : i1
      %10 = arith.ori %4, %9 : i1
      %11 = arith.cmpi uge, %5, %c1_i64 : i64
      %12 = arith.andi %11, %10 : i1
      %13 = affine.load %arg0[%arg3 + 7, %arg4 + 7, %arg5 + 7] : memref<34x99x194xf64, 1>
      %14 = arith.select %12, %cst, %13 : f64
      affine.store %14, %arg0[%arg3 + 7, %arg4 + 7, %arg5 + 7] : memref<34x99x194xf64, 1>
    }
    return
  }
}

// CHECK:  func.func private @"##call__Z25gpu__mask_immersed_field_16CompilerMetadataI10StaticSizeI13_180__85__20_E12DynamicCheckvv7NDRangeILi3ES0_I11_12__6__20_ES0_I11_16__16__1_EvvEE11OffsetArrayI7Float64Li3E13CuTracedArrayIS9_Li3ELi1E13_194__99__34_EE5TupleI6Center4FaceSE_E20ImmersedBoundaryGridIS9_8Periodic14RightConnected7Bounded28OrthogonalSphericalShellGridIS9_SI_SJ_SK_28StaticVerticalDiscretizationIS8_IS9_Li1ESA_IS9_Li1ELi1E5_35__EES8_IS9_Li1ESA_IS9_Li1ELi1E5_34__EESO_SQ_ES8_IS9_Li2ESA_IS9_Li2ELi1E9_194__99_EE8TripolarI5Int64SV_SV_EvE16GridFittedBottomI5FieldISE_SE_vvvvS8_IS9_Li3ESA_IS9_Li3ELi1E12_194__99__1_EES9_vvvE23CenterImmersedConditionEvvvES9_#1335$par243_raised"(%arg0: tensor<34x99x194xf64>, %arg1: tensor<34xf64>, %arg2: tensor<1x99x194xf64>) -> (tensor<34x99x194xf64>, tensor<34xf64>, tensor<1x99x194xf64>) {
// CHECK-NEXT:    %c = stablehlo.constant dense<-1> : tensor<i64>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:    %c_1 = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:    %0 = stablehlo.iota dim = 0 : tensor<20xi64>
// CHECK-NEXT:    %c_2 = stablehlo.constant dense<0> : tensor<20xi64>
// CHECK-NEXT:    %1 = stablehlo.add %0, %c_2 : tensor<20xi64>
// CHECK-NEXT:    %c_3 = stablehlo.constant dense<1> : tensor<20xi64>
// CHECK-NEXT:    %2 = stablehlo.multiply %1, %c_3 : tensor<20xi64>
// CHECK-NEXT:    %3 = stablehlo.iota dim = 0 : tensor<85xi64>
// CHECK-NEXT:    %c_4 = stablehlo.constant dense<0> : tensor<85xi64>
// CHECK-NEXT:    %4 = stablehlo.add %3, %c_4 : tensor<85xi64>
// CHECK-NEXT:    %c_5 = stablehlo.constant dense<1> : tensor<85xi64>
// CHECK-NEXT:    %5 = stablehlo.multiply %4, %c_5 : tensor<85xi64>
// CHECK-NEXT:    %6 = stablehlo.iota dim = 0 : tensor<180xi64>
// CHECK-NEXT:    %c_6 = stablehlo.constant dense<0> : tensor<180xi64>
// CHECK-NEXT:    %7 = stablehlo.add %6, %c_6 : tensor<180xi64>
// CHECK-NEXT:    %c_7 = stablehlo.constant dense<1> : tensor<180xi64>
// CHECK-NEXT:    %8 = stablehlo.multiply %7, %c_7 : tensor<180xi64>
// CHECK-NEXT:    %9 = stablehlo.broadcast_in_dim %5, dims = [0] : (tensor<85xi64>) -> tensor<85xi64>
// CHECK-NEXT:    %10 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i64>) -> tensor<85xi64>
// CHECK-NEXT:    %11 = arith.addi %9, %10 : tensor<85xi64>
// CHECK-NEXT:    %12 = stablehlo.slice %arg1 [7:27] : (tensor<34xf64>) -> tensor<20xf64>
// CHECK-NEXT:    %13 = stablehlo.reshape %12 : (tensor<20xf64>) -> tensor<20xf64>
// CHECK-NEXT:    %14 = stablehlo.slice %arg2 [0:1, 7:92, 7:187] : (tensor<1x99x194xf64>) -> tensor<1x85x180xf64>
// CHECK-NEXT:    %15 = stablehlo.reshape %14 : (tensor<1x85x180xf64>) -> tensor<85x180xf64>
// CHECK-NEXT:    %16 = stablehlo.broadcast_in_dim %13, dims = [0] : (tensor<20xf64>) -> tensor<20x85x180xf64>
// CHECK-NEXT:    %17 = stablehlo.broadcast_in_dim %15, dims = [1, 2] : (tensor<85x180xf64>) -> tensor<20x85x180xf64>
// CHECK-NEXT:    %18 = arith.cmpf ole, %16, %17 {fastmathFlags = #llvm.fastmath<none>} : tensor<20x85x180xf64>
// CHECK-NEXT:    %19 = stablehlo.broadcast_in_dim %11, dims = [0] : (tensor<85xi64>) -> tensor<85xi64>
// CHECK-NEXT:    %20 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i64>) -> tensor<85xi64>
// CHECK-NEXT:    %21 = arith.addi %19, %20 : tensor<85xi64>
// CHECK-NEXT:    %22 = stablehlo.slice %arg2 [0:1, 6:91, 7:187] : (tensor<1x99x194xf64>) -> tensor<1x85x180xf64>
// CHECK-NEXT:    %23 = stablehlo.reshape %22 : (tensor<1x85x180xf64>) -> tensor<85x180xf64>
// CHECK-NEXT:    %24 = stablehlo.broadcast_in_dim %13, dims = [0] : (tensor<20xf64>) -> tensor<20x85x180xf64>
// CHECK-NEXT:    %25 = stablehlo.broadcast_in_dim %23, dims = [1, 2] : (tensor<85x180xf64>) -> tensor<20x85x180xf64>
// CHECK-NEXT:    %26 = arith.cmpf ole, %24, %25 {fastmathFlags = #llvm.fastmath<none>} : tensor<20x85x180xf64>
// CHECK-NEXT:    %27 = stablehlo.broadcast_in_dim %21, dims = [0] : (tensor<85xi64>) -> tensor<85xi64>
// CHECK-NEXT:    %28 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i64>) -> tensor<85xi64>
// CHECK-NEXT:    %29 = arith.cmpi ult, %27, %28 : tensor<85xi64>
// CHECK-NEXT:    %30 = stablehlo.broadcast_in_dim %29, dims = [0] : (tensor<85xi1>) -> tensor<85x20x180xi1>
// CHECK-NEXT:    %31 = stablehlo.broadcast_in_dim %26, dims = [1, 0, 2] : (tensor<20x85x180xi1>) -> tensor<85x20x180xi1>
// CHECK-NEXT:    %32 = arith.ori %30, %31 : tensor<85x20x180xi1>
// CHECK-NEXT:    %33 = stablehlo.broadcast_in_dim %18, dims = [0, 1, 2] : (tensor<20x85x180xi1>) -> tensor<20x85x180xi1>
// CHECK-NEXT:    %34 = stablehlo.broadcast_in_dim %32, dims = [1, 0, 2] : (tensor<85x20x180xi1>) -> tensor<20x85x180xi1>
// CHECK-NEXT:    %35 = arith.ori %33, %34 : tensor<20x85x180xi1>
// CHECK-NEXT:    %36 = stablehlo.broadcast_in_dim %21, dims = [0] : (tensor<85xi64>) -> tensor<85xi64>
// CHECK-NEXT:    %37 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i64>) -> tensor<85xi64>
// CHECK-NEXT:    %38 = arith.cmpi uge, %36, %37 : tensor<85xi64>
// CHECK-NEXT:    %39 = stablehlo.broadcast_in_dim %38, dims = [0] : (tensor<85xi1>) -> tensor<85x20x180xi1>
// CHECK-NEXT:    %40 = stablehlo.broadcast_in_dim %35, dims = [1, 0, 2] : (tensor<20x85x180xi1>) -> tensor<85x20x180xi1>
// CHECK-NEXT:    %41 = arith.andi %39, %40 : tensor<85x20x180xi1>
// CHECK-NEXT:    %42 = stablehlo.slice %arg0 [7:27, 7:92, 7:187] : (tensor<34x99x194xf64>) -> tensor<20x85x180xf64>
// CHECK-NEXT:    %43 = stablehlo.reshape %42 : (tensor<20x85x180xf64>) -> tensor<20x85x180xf64>
// CHECK-NEXT:    %44 = stablehlo.broadcast_in_dim %41, dims = [0, 1, 2] : (tensor<85x20x180xi1>) -> tensor<85x20x180xi1>
// CHECK-NEXT:    %45 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<85x20x180xf64>
// CHECK-NEXT:    %46 = stablehlo.broadcast_in_dim %43, dims = [1, 0, 2] : (tensor<20x85x180xf64>) -> tensor<85x20x180xf64>
// CHECK-NEXT:    %47 = arith.select %44, %45, %46 : tensor<85x20x180xi1>, tensor<85x20x180xf64>
// CHECK-NEXT:    %c_8 = stablehlo.constant dense<7> : tensor<i64>
// CHECK-NEXT:    %c_9 = stablehlo.constant dense<7> : tensor<i64>
// CHECK-NEXT:    %c_10 = stablehlo.constant dense<7> : tensor<i64>
// CHECK-NEXT:    %48 = stablehlo.broadcast_in_dim %47, dims = [1, 0, 2] : (tensor<85x20x180xf64>) -> tensor<20x85x180xf64>
// CHECK-NEXT:    %49 = stablehlo.dynamic_update_slice %arg0, %48, %c_8, %c_9, %c_10 : (tensor<34x99x194xf64>, tensor<20x85x180xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<34x99x194xf64>
// CHECK-NEXT:    return %49, %arg1, %arg2 : tensor<34x99x194xf64>, tensor<34xf64>, tensor<1x99x194xf64>
// CHECK-NEXT:  }