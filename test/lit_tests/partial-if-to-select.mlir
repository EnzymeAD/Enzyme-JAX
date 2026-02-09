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

// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0) -> (d0 * 8 + 3)>
// CHECK-LABEL:   func.func private @"##call__Z16gpu_halo_kernel_16CompilerMetadataI10StaticSizeI6_8__8_E12DynamicCheckvv7NDRangeILi2ES0_I6_1__1_ES1_vvEE13CuTracedArrayI7Float64Li2ELi1E6_8__8_E5Int64SA_#245$par0"(
// CHECK-SAME:      %[[ARG0:.*]]: memref<8x8xf64, 1>) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 3 : i64
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 8 : index
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant false
// CHECK:           %[[CONSTANT_3:.*]] = arith.constant -1 : i64
// CHECK:           %[[POISON_0:.*]] = ub.poison : i64
// CHECK:           %[[CONSTANT_4:.*]] = arith.constant 1 : i64
// CHECK:           %[[CONSTANT_5:.*]] = arith.constant 64 : i64
// CHECK:           %[[CONSTANT_6:.*]] = arith.constant 8 : i64
// CHECK:           %[[CONSTANT_7:.*]] = arith.constant 2 : i64
// CHECK:           affine.parallel (%[[VAL_0:.*]]) = (0) to (64) {
// CHECK:             %[[INDEX_CASTUI_0:.*]] = arith.index_castui %[[VAL_0]] : index to i64
// CHECK:             %[[DIVUI_0:.*]] = arith.divui %[[VAL_0]], %[[CONSTANT_1]] : index
// CHECK:             %[[MULI_0:.*]] = arith.muli %[[DIVUI_0]], %[[CONSTANT_1]] : index
// CHECK:             %[[INDEX_CASTUI_1:.*]] = arith.index_castui %[[MULI_0]] : index to i64
// CHECK:             %[[SUBI_0:.*]] = arith.subi %[[INDEX_CASTUI_0]], %[[INDEX_CASTUI_1]] : i64
// CHECK:             %[[MULI_1:.*]] = arith.muli %[[DIVUI_0]], %[[CONSTANT_1]] : index
// CHECK:             %[[INDEX_CASTUI_2:.*]] = arith.index_castui %[[MULI_1]] : index to i64
// CHECK:             %[[ADDI_0:.*]] = arith.addi %[[INDEX_CASTUI_2]], %[[SUBI_0]] : i64
// CHECK:             %[[MULI_2:.*]] = arith.muli %[[ADDI_0]], %[[CONSTANT_6]] : i64
// CHECK:             %[[WHILE_0:.*]] = scf.while (%[[VAL_1:.*]] = %[[CONSTANT_4]]) : (i64) -> i64 {
// CHECK:               %[[ADDI_1:.*]] = arith.addi %[[VAL_1]], %[[MULI_2]] : i64
// CHECK:               %[[ADDI_2:.*]] = arith.addi %[[ADDI_1]], %[[CONSTANT_0]] : i64
// CHECK:               %[[INDEX_CAST_0:.*]] = arith.index_cast %[[VAL_1]] : i64 to index
// CHECK:               %[[CMPI_0:.*]] = arith.cmpi uge, %[[ADDI_2]], %[[CONSTANT_5]] : i64
// CHECK:               %[[ADDI_3:.*]] = arith.addi %[[VAL_1]], %[[CONSTANT_4]] : i64
// CHECK:               %[[ADDI_4:.*]] = arith.addi %[[VAL_1]], %[[MULI_2]] : i64
// CHECK:               %[[ADDI_5:.*]] = arith.addi %[[ADDI_4]], %[[CONSTANT_3]] : i64
// CHECK:               %[[CMPI_1:.*]] = arith.cmpi uge, %[[ADDI_5]], %[[CONSTANT_5]] : i64
// CHECK:               %[[CMPI_2:.*]] = arith.cmpi ne, %[[VAL_1]], %[[CONSTANT_7]] : i64
// CHECK:               %[[SELECT_0:.*]] = arith.select %[[CMPI_1]], %[[CONSTANT_2]], %[[CMPI_2]] : i1
// CHECK:               %[[SELECT_1:.*]] = arith.select %[[CMPI_0]], %[[CONSTANT_2]], %[[SELECT_0]] : i1
// CHECK:               %[[IF_0:.*]]:2 = scf.if %[[CMPI_0]] -> (i64, i1) {
// CHECK:                 scf.yield %[[POISON_0]], %[[CONSTANT_2]] : i64, i1
// CHECK:               } else {
// CHECK:                 %[[APPLY_0:.*]] = affine.apply #[[$ATTR_0]](%[[VAL_0]])
// CHECK:                 %[[ADDI_6:.*]] = arith.addi %[[APPLY_0]], %[[INDEX_CAST_0]] : index
// CHECK:                 %[[REMUI_0:.*]] = arith.remui %[[ADDI_6]], %[[CONSTANT_1]] : index
// CHECK:                 %[[DIVUI_1:.*]] = arith.divui %[[ADDI_6]], %[[CONSTANT_1]] : index
// CHECK:                 %[[LOAD_0:.*]] = memref.load %[[ARG0]]{{\[}}%[[DIVUI_1]], %[[REMUI_0]]] : memref<8x8xf64, 1>
// CHECK:                 %[[ADDI_7:.*]] = arith.addi %[[VAL_1]], %[[MULI_2]] : i64
// CHECK:                 %[[ADDI_8:.*]] = arith.addi %[[ADDI_7]], %[[CONSTANT_3]] : i64
// CHECK:                 %[[CMPI_3:.*]] = arith.cmpi uge, %[[ADDI_8]], %[[CONSTANT_5]] : i64
// CHECK:                 %[[ADDI_9:.*]] = arith.addi %[[VAL_1]], %[[CONSTANT_4]] : i64
// CHECK:                 %[[CMPI_4:.*]] = arith.cmpi ne, %[[VAL_1]], %[[CONSTANT_7]] : i64
// CHECK:                 %[[SELECT_2:.*]] = arith.select %[[CMPI_3]], %[[CONSTANT_2]], %[[CMPI_4]] : i1
// CHECK:                 %[[IF_1:.*]]:2 = scf.if %[[CMPI_3]] -> (i64, i1) {
// CHECK:                   scf.yield %[[POISON_0]], %[[CONSTANT_2]] : i64, i1
// CHECK:                 } else {
// CHECK:                   %[[INDEX_CAST_1:.*]] = arith.index_cast %[[ADDI_8]] : i64 to index
// CHECK:                   %[[REMUI_1:.*]] = arith.remui %[[INDEX_CAST_1]], %[[CONSTANT_1]] : index
// CHECK:                   %[[DIVUI_2:.*]] = arith.divui %[[INDEX_CAST_1]], %[[CONSTANT_1]] : index
// CHECK:                   memref.store %[[LOAD_0]], %[[ARG0]]{{\[}}%[[DIVUI_2]], %[[REMUI_1]]] : memref<8x8xf64, 1>
// CHECK:                   %[[ADDI_10:.*]] = arith.addi %[[VAL_1]], %[[CONSTANT_4]] : i64
// CHECK:                   %[[CMPI_5:.*]] = arith.cmpi ne, %[[VAL_1]], %[[CONSTANT_7]] : i64
// CHECK:                   scf.yield %[[ADDI_10]], %[[CMPI_5]] : i64, i1
// CHECK:                 }
// CHECK:                 scf.yield %[[ADDI_9]], %[[SELECT_2]] : i64, i1
// CHECK:               }
// CHECK:               scf.condition(%[[SELECT_1]]) %[[ADDI_3]] : i64
// CHECK:             } do {
// CHECK:             ^bb0(%[[VAL_2:.*]]: i64):
// CHECK:               scf.yield %[[VAL_2]] : i64
// CHECK:             }
// CHECK:           }
// CHECK:           return
// CHECK:         }

