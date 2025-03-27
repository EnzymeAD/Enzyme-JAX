// RUN: enzymexlamlir-opt --affine-cfg --split-input-file --allow-unregistered-dialect %s | FileCheck %s

module {
  func.func @_Z7runTestiPPc(%arg0: index, %arg2: memref<?xi32>) {
    %c1_i64 = arith.constant 1 : i64
    affine.parallel (%arg12, %arg13, %arg14) = (0, 0, 0) to (100, 1017, 1010) {
      %0 = arith.index_castui %arg12 : index to i64
      %1 = arith.index_castui %arg13 : index to i64
      %32 = arith.cmpi ult, %1, %c1_i64 : i64
      %34 = arith.cmpi uge, %1, %c1_i64 : i64
      "test.use"(%32, %34) : (i1, i1) -> ()
      %s32 = arith.cmpi slt, %1, %c1_i64 : i64
      %s34 = arith.cmpi sge, %1, %c1_i64 : i64
      "test.use"(%s32, %s34) : (i1, i1) -> ()
    }
    return
  }
}

//CHECK:  func.func @_Z7runTestiPPc(%arg0: index, %arg1: memref<?xi32>) {
//CHECK-NEXT:    %c0 = arith.constant 0 : index
//CHECK-NEXT:    affine.parallel (%arg2, %arg3, %arg4) = (0, 0, 0) to (100, 1017, 1010) {
//CHECK-NEXT:      %0 = arith.cmpi eq, %arg3, %c0 : index
//CHECK-NEXT:      %1 = arith.cmpi ne, %arg3, %c0 : index
//CHECK-NEXT:      "test.use"(%0, %1) : (i1, i1) -> ()
//CHECK-NEXT:      %2 = arith.cmpi eq, %arg3, %c0 : index
//CHECK-NEXT:      %3 = arith.cmpi ne, %arg3, %c0 : index
//CHECK-NEXT:      "test.use"(%2, %3) : (i1, i1) -> ()
//CHECK-NEXT:    }
//CHECK-NEXT:    return
//CHECK-NEXT:  }