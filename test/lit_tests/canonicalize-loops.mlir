// RUN: enzymexlamlir-opt --canonicalize-loops %s | FileCheck %s

#map31 = affine_map<(d0, d1) -> (d0 + d1 * 16 - (d0 floordiv 16) * 16 + 2625)>  
#map32 = affine_map<(d0, d1) -> (d0 + d1 * 16 - (d0 floordiv 16) * 16 + 2251)>
#map33 = affine_map<(d0, d1) -> (d0 + d1 * 16 - (d0 floordiv 16) * 16 + 47131)>
#map34 = affine_map<(d0, d1) -> (d0 + d1 * 16 - (d0 floordiv 16) * 16 + 47505)>

// CHECK-LABEL: foo
func.func private @foo(%arg0: !llvm.ptr<1>) {
    %c4_i32 = arith.constant 4 : i32
    %c4_i64 = arith.constant 4 : i64
    %c1_i64 = arith.constant 1 : i64
    %c-361_i64 = arith.constant -361 : i64
    %c-360_i64 = arith.constant -360 : i64
    // CHECK: affine.parallel (%{{.+}}, %{{.+}}) = (0, 0) to (23, 256)
    affine.parallel (%arg1, %arg2, %arg3, %arg4, %arg5, %arg6) = (0, 0, 0, 0, 0, 0) to (23, 1, 1, 256, 1, 1) {
      %0 = arith.index_cast %arg4 : index to i32
      %1 = arith.extui %0 : i32 to i64
      %2 = arith.index_cast %arg1 : index to i32
      %3 = arith.extui %2 : i32 to i64
      %4 = arith.shrui %0, %c4_i32 : i32
      %5 = arith.extui %4 : i32 to i64
      %6 = arith.subi %3, %5 : i64
      %7 = arith.shli %6, %c4_i64 : i64
      %8 = arith.addi %1, %c1_i64 : i64
      %9 = arith.addi %8, %7 : i64
      %10 = arith.addi %5, %c1_i64 : i64
      %11 = arith.addi %9, %c-361_i64 : i64
      %12 = arith.cmpi ult, %11, %c-360_i64 : i64
      %13 = arith.cmpi ne, %10, %c1_i64 : i64
      %14 = arith.ori %13, %12 : i1
      scf.if %14 {
      } else {
        %15 = affine.apply #map31(%arg4, %arg1)
        %16 = arith.index_cast %15 : index to i32
        %17 = llvm.getelementptr %arg0[%16] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f64
        %18 = llvm.load %17 : !llvm.ptr<1> -> f64
        %19 = affine.apply #map32(%arg4, %arg1)
        %20 = arith.index_cast %19 : index to i32
        %21 = llvm.getelementptr %arg0[%20] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f64
        llvm.store %18, %21 : f64, !llvm.ptr<1>
        %22 = affine.apply #map33(%arg4, %arg1)
        %23 = arith.index_cast %22 : index to i32
        %24 = llvm.getelementptr %arg0[%23] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f64
        %25 = llvm.load %24 : !llvm.ptr<1> -> f64
        %26 = affine.apply #map34(%arg4, %arg1)
        %27 = arith.index_cast %26 : index to i32
        %28 = llvm.getelementptr %arg0[%27] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f64
        llvm.store %25, %28 : f64, !llvm.ptr<1>
      }
    }
    return
  }
