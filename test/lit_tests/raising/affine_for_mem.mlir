// RUN: enzymexlamlir-opt %s --affine-cfg --canonicalize --raise-affine-to-stablehlo --arith-raise --enzyme-hlo-opt=max_constant_expansion=0 | FileCheck %s

module {
  func.func private @"##call__Z38gpu__fill_periodic_west_and_east_halo_16CompilerMetadataI16OffsetStaticSizeI12_1_50__1_20_E12DynamicCheckvv7NDRangeILi2E10StaticSizeI6_4__2_ES4_I8_16__16_E5TupleI5Int64S8_ES0_I6_0__0_EEE11OffsetArrayI7Float64Li3E13CuTracedArrayISE_Li3ELi1E12_50__50__20_EE3ValILi40EESI_ILi5EE#864$par100"(%arg0: memref<20x50x50xf64, 1>, %arg3: memref<8x256xf64>) {
    %c50 = arith.constant 50 : index
    %c9 = arith.constant 9 : index
    %c44 = arith.constant 44 : index
    %c8 = arith.constant 8 : index
    %c43 = arith.constant 43 : index
    %c7 = arith.constant 7 : index
    %c42 = arith.constant 42 : index
    %c6 = arith.constant 6 : index
    %c41 = arith.constant 41 : index
    %c5 = arith.constant 5 : index
    %c40 = arith.constant 40 : index
    %c49_i64 = arith.constant 49 : i64
    %c48_i64 = arith.constant 48 : i64
    %c47_i64 = arith.constant 47 : i64
    %c46_i64 = arith.constant 46 : i64
    %c16 = arith.constant 16 : index
    %c0_i64 = arith.constant 0 : i64
    %c3_i64 = arith.constant 3 : i64
    %c2_i64 = arith.constant 2 : i64
    %c1_i64 = arith.constant 1 : i64
    %c4_i64 = arith.constant 4 : i64
    %c16_i64 = arith.constant 16 : i64
    %0 = arith.muli %c1_i64, %c16_i64 : i64
    %c50_i64 = arith.constant 50 : i64
    %c20_i64 = arith.constant 20 : i64
    %c2500_i64 = arith.constant 2500 : i64
    %c45_i64 = arith.constant 45 : i64
    affine.parallel (%arg1, %arg2) = (0, 0) to (8, 256) {
      %1 = arith.index_castui %arg1 : index to i64
      %2 = affine.if affine_set<(d0) : (d0 >= 0, -d0 + 3 >= 0)>(%arg1) -> (f64) {
        %6 = arith.muli %c0_i64, %c4_i64 : i64
        %7 = arith.subi %1, %6 : i64
        %8 = arith.muli %7, %c16_i64 : i64
        %9 = arith.index_castui %arg2 : index to i64
        %10 = arith.divui %arg2, %c16 : index
        %11 = arith.muli %10, %c16 : index
        %12 = arith.index_castui %11 : index to i64
        %13 = arith.subi %9, %12 : i64
        %14 = arith.addi %8, %13 : i64
        %15 = arith.muli %14, %c50_i64 : i64
        %16 = arith.index_cast %15 : i64 to index
        %17 = arith.muli %c0_i64, %c16_i64 : i64
        %18 = arith.index_castui %10 : index to i64
        %19 = arith.addi %17, %18 : i64
        %20 = arith.muli %19, %c2500_i64 : i64
        %21 = arith.index_cast %20 : i64 to index
        %22 = arith.addi %16, %21 : index
        %23 = arith.addi %22, %c9 : index
        %24 = arith.divui %23, %c50 : index
        %25 = arith.divui %24, %c50 : index
        %26 = arith.remui %24, %c50 : index
        %27 = arith.remui %23, %c50 : index
        %28 = memref.load %arg0[%25, %26, %27] : memref<20x50x50xf64, 1>
        affine.yield %28 : f64
      } else {
        %6 = arith.muli %c1_i64, %c4_i64 : i64
        %7 = arith.subi %1, %6 : i64
        %8 = arith.muli %7, %c16_i64 : i64
        %9 = arith.index_castui %arg2 : index to i64
        %10 = arith.divui %arg2, %c16 : index
        %11 = arith.muli %10, %c16 : index
        %12 = arith.index_castui %11 : index to i64
        %13 = arith.subi %9, %12 : i64
        %14 = arith.addi %8, %13 : i64
        %15 = arith.muli %14, %c50_i64 : i64
        %16 = arith.index_cast %15 : i64 to index
        %17 = arith.muli %c1_i64, %c16_i64 : i64
        %18 = arith.index_castui %10 : index to i64
        %19 = arith.addi %17, %18 : i64
        %20 = arith.muli %19, %c2500_i64 : i64
        %21 = arith.index_cast %20 : i64 to index
        %22 = arith.addi %16, %21 : index
        %23 = arith.addi %22, %c9 : index
        %24 = arith.divui %23, %c50 : index
        %25 = arith.divui %24, %c50 : index
        %26 = arith.remui %24, %c50 : index
        %27 = arith.remui %23, %c50 : index
        %28 = memref.load %arg0[%25, %26, %27] : memref<20x50x50xf64, 1>
        affine.yield %28 : f64
      }
      affine.store %2, %arg3[%arg1, %arg2] : memref<8x256xf64>
    }
    return
  }
}

// CHECK:  func.func private @"##call__Z38gpu__fill_periodic_west_and_east_halo_16CompilerMetadataI16OffsetStaticSizeI12_1_50__1_20_E12DynamicCheckvv7NDRangeILi2E10StaticSizeI6_4__2_ES4_I8_16__16_E5TupleI5Int64S8_ES0_I6_0__0_EEE11OffsetArrayI7Float64Li3E13CuTracedArrayISE_Li3ELi1E12_50__50__20_EE3ValILi40EESI_ILi5EE#864$par100_raised"(%arg0: tensor<20x50x50xf64>, %arg1: tensor<8x256xf64>) -> (tensor<20x50x50xf64>, tensor<8x256xf64>) {
// CHECK-NEXT:    %c = stablehlo.constant dense<16> : tensor<16xi64>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<-50> : tensor<8x16xi64>
// CHECK-NEXT:    %c_1 = stablehlo.constant dense<15> : tensor<16x8x16x1xi64>
// CHECK-NEXT:    %c_2 = stablehlo.constant dense<-14> : tensor<8x16xi64>
// CHECK-NEXT:    %c_3 = stablehlo.constant dense<9> : tensor<i64>
// CHECK-NEXT:    %c_4 = stablehlo.constant dense<50> : tensor<8x16xi64>
// CHECK-NEXT:    %c_5 = stablehlo.constant dense<16> : tensor<8xi64>
// CHECK-NEXT:    %c_6 = stablehlo.constant {enzymexla.non_negative = [#enzymexla<guaranteed GUARANTEED>]} dense<3> : tensor<8xi64>
// CHECK-NEXT:    %c_7 = stablehlo.constant dense<1> : tensor<8x16xi64>
// CHECK-NEXT:    %c_8 = stablehlo.constant dense<0> : tensor<8x16xi64>
// CHECK-NEXT:    %c_9 = stablehlo.constant {enzymexla.non_negative = [#enzymexla<guaranteed GUARANTEED>]} dense<0> : tensor<8xi64>
// CHECK-NEXT:    %0 = stablehlo.iota dim = 0 {enzymexla.non_negative = [#enzymexla<guaranteed GUARANTEED>]} : tensor<8xi64>
// CHECK-NEXT:    %1 = stablehlo.iota dim = 0 : tensor<16xi64>
// CHECK-NEXT:    %2 = stablehlo.subtract %c_6, %0 {enzymexla.non_negative = [#enzymexla<guaranteed NOTGUARANTEED>]} : tensor<8xi64>
// CHECK-NEXT:    %3 = stablehlo.compare GE, %2, %c_9 : (tensor<8xi64>, tensor<8xi64>) -> tensor<8xi1>
// CHECK-NEXT:    %4 = stablehlo.multiply %0, %c_5 : tensor<8xi64>
// CHECK-NEXT:    %5 = stablehlo.broadcast_in_dim %4, dims = [0] : (tensor<8xi64>) -> tensor<8x16xi64>
// CHECK-NEXT:    %6 = stablehlo.iota dim = 1 : tensor<8x16xi64>
// CHECK-NEXT:    %7 = stablehlo.add %5, %6 {enzymexla.non_negative = [#enzymexla<guaranteed GUARANTEED>]} : tensor<8x16xi64>
// CHECK-NEXT:    %8 = stablehlo.compare LE, %7, %c_8 : (tensor<8x16xi64>, tensor<8x16xi64>) -> tensor<8x16xi1>
// CHECK-NEXT:    %9 = stablehlo.negate %7 : tensor<8x16xi64>
// CHECK-NEXT:    %10 = stablehlo.subtract %9, %c_7 : tensor<8x16xi64>
// CHECK-NEXT:    %11 = stablehlo.select %8, %10, %7 : tensor<8x16xi1>, tensor<8x16xi64>
// CHECK-NEXT:    %12 = stablehlo.divide %11, %c_4 : tensor<8x16xi64>
// CHECK-NEXT:    %13 = stablehlo.divide %11, %c_0 : tensor<8x16xi64>
// CHECK-NEXT:    %14 = stablehlo.subtract %13, %c_7 : tensor<8x16xi64>
// CHECK-NEXT:    %15 = stablehlo.select %8, %14, %12 : tensor<8x16xi1>, tensor<8x16xi64>
// CHECK-NEXT:    %16 = stablehlo.iota dim = 0 : tensor<16x8x16x1xi64>
// CHECK-NEXT:    %17 = stablehlo.broadcast_in_dim %15, dims = [1, 2] : (tensor<8x16xi64>) -> tensor<16x8x16x1xi64>
// CHECK-NEXT:    %18 = stablehlo.add %16, %17 : tensor<16x8x16x1xi64>
// CHECK-NEXT:    %19 = stablehlo.remainder %7, %c_4 : tensor<8x16xi64>
// CHECK-NEXT:    %20 = stablehlo.compare LT, %7, %c_8 : (tensor<8x16xi64>, tensor<8x16xi64>) -> tensor<8x16xi1>
// CHECK-NEXT:    %21 = stablehlo.add %19, %c_4 : tensor<8x16xi64>
// CHECK-NEXT:    %22 = stablehlo.select %20, %21, %19 : tensor<8x16xi1>, tensor<8x16xi64>
// CHECK-NEXT:    %23 = stablehlo.broadcast_in_dim %22, dims = [1, 2] : (tensor<8x16xi64>) -> tensor<16x8x16x1xi64>
// CHECK-NEXT:    %24 = stablehlo.concatenate %18, %23, dim = 3 : (tensor<16x8x16x1xi64>, tensor<16x8x16x1xi64>) -> tensor<16x8x16x2xi64>
// CHECK-NEXT:    %25 = stablehlo.pad %24, %c_3, low = [0, 0, 0, 0], high = [0, 0, 0, 1], interior = [0, 0, 0, 0] : (tensor<16x8x16x2xi64>, tensor<i64>) -> tensor<16x8x16x3xi64>
// CHECK-NEXT:    %26 = stablehlo.reshape %25 : (tensor<16x8x16x3xi64>) -> tensor<2048x3xi64>
// CHECK-NEXT:    %27 = "stablehlo.gather"(%arg0, %26) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1, 2], start_index_map = [0, 1, 2], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 1>}> : (tensor<20x50x50xf64>, tensor<2048x3xi64>) -> tensor<2048xf64>
// CHECK-NEXT:    %28 = stablehlo.reshape %27 : (tensor<2048xf64>) -> tensor<16x8x16xf64>
// CHECK-NEXT:    %29 = stablehlo.add %7, %c_2 {enzymexla.non_negative = [#enzymexla<guaranteed NOTGUARANTEED>]} : tensor<8x16xi64>
// CHECK-NEXT:    %30 = stablehlo.compare LE, %29, %c_8 : (tensor<8x16xi64>, tensor<8x16xi64>) -> tensor<8x16xi1>
// CHECK-NEXT:    %31 = stablehlo.negate %29 : tensor<8x16xi64>
// CHECK-NEXT:    %32 = stablehlo.subtract %31, %c_7 : tensor<8x16xi64>
// CHECK-NEXT:    %33 = stablehlo.select %30, %32, %29 : tensor<8x16xi1>, tensor<8x16xi64>
// CHECK-NEXT:    %34 = stablehlo.divide %33, %c_4 : tensor<8x16xi64>
// CHECK-NEXT:    %35 = stablehlo.divide %33, %c_0 : tensor<8x16xi64>
// CHECK-NEXT:    %36 = stablehlo.subtract %35, %c_7 : tensor<8x16xi64>
// CHECK-NEXT:    %37 = stablehlo.select %30, %36, %34 : tensor<8x16xi1>, tensor<8x16xi64>
// CHECK-NEXT:    %38 = stablehlo.broadcast_in_dim %37, dims = [1, 2] : (tensor<8x16xi64>) -> tensor<16x8x16x1xi64>
// CHECK-NEXT:    %39 = stablehlo.add %16, %38 : tensor<16x8x16x1xi64>
// CHECK-NEXT:    %40 = stablehlo.add %39, %c_1 : tensor<16x8x16x1xi64>
// CHECK-NEXT:    %41 = stablehlo.multiply %37, %c_0 : tensor<8x16xi64>
// CHECK-NEXT:    %42 = stablehlo.add %7, %41 : tensor<8x16xi64>
// CHECK-NEXT:    %43 = stablehlo.add %42, %c_2 : tensor<8x16xi64>
// CHECK-NEXT:    %44 = stablehlo.broadcast_in_dim %43, dims = [1, 2] : (tensor<8x16xi64>) -> tensor<16x8x16x1xi64>
// CHECK-NEXT:    %45 = stablehlo.concatenate %40, %44, dim = 3 : (tensor<16x8x16x1xi64>, tensor<16x8x16x1xi64>) -> tensor<16x8x16x2xi64>
// CHECK-NEXT:    %46 = stablehlo.pad %45, %c_3, low = [0, 0, 0, 0], high = [0, 0, 0, 1], interior = [0, 0, 0, 0] : (tensor<16x8x16x2xi64>, tensor<i64>) -> tensor<16x8x16x3xi64>
// CHECK-NEXT:    %47 = stablehlo.reshape %46 : (tensor<16x8x16x3xi64>) -> tensor<2048x3xi64>
// CHECK-NEXT:    %48 = "stablehlo.gather"(%arg0, %47) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1, 2], start_index_map = [0, 1, 2], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 1>}> : (tensor<20x50x50xf64>, tensor<2048x3xi64>) -> tensor<2048xf64>
// CHECK-NEXT:    %49 = stablehlo.reshape %48 : (tensor<2048xf64>) -> tensor<16x8x16xf64>
// CHECK-NEXT:    %50 = stablehlo.broadcast_in_dim %3, dims = [0] : (tensor<8xi1>) -> tensor<8x16x16xi1>
// CHECK-NEXT:    %51 = stablehlo.transpose %28, dims = [1, 0, 2] : (tensor<16x8x16xf64>) -> tensor<8x16x16xf64>
// CHECK-NEXT:    %52 = stablehlo.transpose %49, dims = [1, 0, 2] : (tensor<16x8x16xf64>) -> tensor<8x16x16xf64>
// CHECK-NEXT:    %53 = stablehlo.select %50, %51, %52 : tensor<8x16x16xi1>, tensor<8x16x16xf64>
// CHECK-NEXT:    %54 = stablehlo.multiply %1, %c : tensor<16xi64>
// CHECK-NEXT:    %55 = stablehlo.broadcast_in_dim %54, dims = [0] : (tensor<16xi64>) -> tensor<16x16xi64>
// CHECK-NEXT:    %56 = stablehlo.iota dim = 1 : tensor<16x16xi64>
// CHECK-NEXT:    %57 = stablehlo.add %55, %56 : tensor<16x16xi64>
// CHECK-NEXT:    %58 = stablehlo.reshape %57 : (tensor<16x16xi64>) -> tensor<256x1xi64>
// CHECK-NEXT:    %59 = stablehlo.iota dim = 0 : tensor<8x256x1xi64>
// CHECK-NEXT:    %60 = stablehlo.reshape %59 : (tensor<8x256x1xi64>) -> tensor<2048x1xi64>
// CHECK-NEXT:    %61 = stablehlo.broadcast_in_dim %58, dims = [1, 0] : (tensor<256x1xi64>) -> tensor<8x256xi64>
// CHECK-NEXT:    %62 = stablehlo.reshape %61 : (tensor<8x256xi64>) -> tensor<2048x1xi64>
// CHECK-NEXT:    %63 = stablehlo.concatenate %60, %62, dim = 1 : (tensor<2048x1xi64>, tensor<2048x1xi64>) -> tensor<2048x2xi64>
// CHECK-NEXT:    %64 = stablehlo.reshape %53 : (tensor<8x16x16xf64>) -> tensor<2048xf64>
// CHECK-NEXT:    %65 = "stablehlo.scatter"(%arg1, %63, %64) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = true}> ({
// CHECK-NEXT:    ^bb0(%arg2: tensor<f64>, %arg3: tensor<f64>):
// CHECK-NEXT:      stablehlo.return %arg3 : tensor<f64>
// CHECK-NEXT:    }) : (tensor<8x256xf64>, tensor<2048x2xi64>, tensor<2048xf64>) -> tensor<8x256xf64>
// CHECK-NEXT:    return %arg0, %65 : tensor<20x50x50xf64>, tensor<8x256xf64>
// CHECK-NEXT:  }
