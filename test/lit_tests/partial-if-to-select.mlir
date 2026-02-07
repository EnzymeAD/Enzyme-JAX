// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(func.func(canonicalize-loops))" %s | FileCheck %s

// CHECK-NOT: = scf.yield{{.*}}false

module {
func.func private @"##call__Z16gpu_halo_kernel_16CompilerMetadataI10StaticSizeI6_8__8_E12DynamicCheckvv7NDRangeILi2ES0_I6_1__1_ES1_vvEE13CuTracedArrayI7Float64Li2ELi1E6_8__8_E5Int64SA_#245$par0"(%arg0: memref<8x8xf64, 1>) {
  %c3_i64 = arith.constant 3 : i64
  %c8 = arith.constant 8 : index
  %false = arith.constant false
  %c-1_i64 = arith.constant -1 : i64
  %0 = ub.poison : i64
  %c1_i64 = arith.constant 1 : i64
  %c64_i64 = arith.constant 64 : i64
  %c8_i64 = arith.constant 8 : i64
  %c2_i64 = arith.constant 2 : i64
  affine.parallel (%arg1) = (0) to (64) {
    %1 = arith.index_castui %arg1 : index to i64
    %2 = arith.divui %arg1, %c8 : index
    %3 = arith.muli %2, %c8 : index
    %4 = arith.index_castui %3 : index to i64
    %5 = arith.subi %1, %4 : i64
    %6 = arith.muli %2, %c8 : index
    %7 = arith.index_castui %6 : index to i64
    %8 = arith.addi %7, %5 : i64
    %9 = arith.muli %8, %c8_i64 : i64
    %10 = scf.while (%arg2 = %c1_i64) : (i64) -> i64 {
      %11 = arith.addi %arg2, %9 : i64
      %12 = arith.addi %11, %c3_i64 : i64
      %13 = arith.index_cast %arg2 : i64 to index
      %14 = arith.cmpi uge, %12, %c64_i64 : i64
      %15:2 = scf.if %14 -> (i64, i1) {
        scf.yield %0, %false : i64, i1
      } else {
        %16 = affine.apply affine_map<(d0) -> (d0 * 8 + 3)>(%arg1)
        %17 = arith.addi %16, %13 : index
        %18 = arith.remui %17, %c8 : index
        %19 = arith.divui %17, %c8 : index
        %20 = memref.load %arg0[%19, %18] : memref<8x8xf64, 1>
        %21 = arith.addi %arg2, %9 : i64
        %22 = arith.addi %21, %c-1_i64 : i64
        %23 = arith.cmpi uge, %22, %c64_i64 : i64
        %24:2 = scf.if %23 -> (i64, i1) {
          scf.yield %0, %false : i64, i1
        } else {
          %25 = arith.index_cast %22 : i64 to index
          %26 = arith.remui %25, %c8 : index
          %27 = arith.divui %25, %c8 : index
          memref.store %20, %arg0[%27, %26] : memref<8x8xf64, 1>
          %28 = arith.addi %arg2, %c1_i64 : i64
          %29 = arith.cmpi ne, %arg2, %c2_i64 : i64
          scf.yield %28, %29 : i64, i1
        }
        scf.yield %24#0, %24#1 : i64, i1
      }
      scf.condition(%15#1) %15#0 : i64
    } do {
    ^bb0(%arg2: i64):
      scf.yield %arg2 : i64
    }
  }
  return
}
}
