// RUN: enzymexlamlir-opt %s --affine-cfg --canonicalize | FileCheck %s

#map1 = affine_map<(d0) -> (-d0 + 18435)>
#map2 = affine_map<(d0) -> (-d0 + 18241)>
#map3 = affine_map<(d0) -> (-d0 + 18047)>
#map4 = affine_map<(d0) -> (-d0 + 17853)>
#map5 = affine_map<(d0) -> (-d0 + 17659)>
#map6 = affine_map<(d0) -> (-d0 + 17465)>
#map7 = affine_map<(d0) -> (-d0 + 17271)>
#map8 = affine_map<(d0) -> (-d0 + 18629)>
#map9 = affine_map<(d0) -> (d0 * -194 + 18436)>
#map10 = affine_map<(d0) -> (d0 * -194 + 18811)>
#set = affine_set<(d0) : (d0 - 1 >= 0)>
#set1 = affine_set<(d0) : (-d0 + 89 >= 0)>

func.func private @par6(%arg0: memref<1x104x194xf64, 1>) {
  %c104 = arith.constant 104 : index
  %c194 = arith.constant 194 : index
  %c2_i64 = arith.constant 2 : i64
  %c182_i64 = arith.constant 182 : i64
  %c1_i64 = arith.constant 1 : i64
  %c-1_i64 = arith.constant -1 : i64
  affine.parallel (%arg1) = (0) to (180) {
    %0 = affine.load %arg0[0, 7, %arg1 + 7] : memref<1x104x194xf64, 1>
    affine.store %0, %arg0[0, 6, %arg1 + 7] : memref<1x104x194xf64, 1>
    // CHECK: %[[if:.+]]:10 = affine.if
    // CHECK: %[[v0:.+]] = affine.load
    // CHECK: %[[v1:.+]] = affine.load
    // CHECK: %[[v2:.+]] = affine.load
    // CHECK: %[[v3:.+]] = affine.load
    // CHECK: %[[v4:.+]] = affine.load
    // CHECK: %[[v5:.+]] = affine.load
    // CHECK: %[[v6:.+]] = affine.load
    // CHECK: %[[v7:.+]] = affine.load
    %1:2 = affine.if #set(%arg1) -> (i64, i64) {
      affine.yield %c-1_i64, %c182_i64 : i64, i64
    // CHECK: else
    // CHECK: %[[e0:.+]] = affine.load
    // CHECK: %[[e1:.+]] = affine.load
    // CHECK: %[[e2:.+]] = affine.load
    // CHECK: %[[e3:.+]] = affine.load
    // CHECK: %[[e4:.+]] = affine.load
    // CHECK: %[[e5:.+]] = affine.load
    // CHECK: %[[e6:.+]] = affine.load
    // CHECK: %[[e7:.+]] = affine.load
    } else {
      affine.yield %c1_i64, %c2_i64 : i64, i64
    }
    // CHECK-NOT: memref.load
    // CHECK: arith.mulf %{{.*}}, %[[if]]#9
    // CHECK: arith.mulf %{{.*}}, %[[if]]#8
    // CHECK: arith.mulf %{{.*}}, %[[if]]#7
    // CHECK: arith.mulf %{{.*}}, %[[if]]#6
    // CHECK: arith.mulf %{{.*}}, %[[if]]#5
    // CHECK: arith.mulf %{{.*}}, %[[if]]#4
    // CHECK: arith.mulf %{{.*}}, %[[if]]#3
    // CHECK: arith.mulf %{{.*}}, %[[if]]#2
    %2 = arith.sitofp %1#0 : i64 to f64
    %3 = arith.index_cast %1#1 : i64 to index
    %4 = affine.apply #map1(%arg1)
    %5 = arith.addi %4, %3 : index
    %6 = arith.remui %5, %c194 : index
    %7 = arith.divui %5, %c194 : index
    %8 = arith.remui %7, %c104 : index
    %9 = arith.divui %7, %c104 : index
    %10 = memref.load %arg0[%9, %8, %6] : memref<1x104x194xf64, 1>
    %11 = arith.mulf %2, %10 {fastmathFlags = #llvm.fastmath<none>} : f64
    affine.store %11, %arg0[0, 97, %arg1 + 7] : memref<1x104x194xf64, 1>
    %12 = affine.apply #map2(%arg1)
    %13 = arith.addi %12, %3 : index
    %14 = arith.remui %13, %c194 : index
    %15 = arith.divui %13, %c194 : index
    %16 = arith.remui %15, %c104 : index
    %17 = arith.divui %15, %c104 : index
    %18 = memref.load %arg0[%17, %16, %14] : memref<1x104x194xf64, 1>
    %19 = arith.mulf %2, %18 {fastmathFlags = #llvm.fastmath<none>} : f64
    affine.store %19, %arg0[0, 98, %arg1 + 7] : memref<1x104x194xf64, 1>
    %20 = affine.apply #map3(%arg1)
    %21 = arith.addi %20, %3 : index
    %22 = arith.remui %21, %c194 : index
    %23 = arith.divui %21, %c194 : index
    %24 = arith.remui %23, %c104 : index
    %25 = arith.divui %23, %c104 : index
    %26 = memref.load %arg0[%25, %24, %22] : memref<1x104x194xf64, 1>
    %27 = arith.mulf %2, %26 {fastmathFlags = #llvm.fastmath<none>} : f64
    affine.store %27, %arg0[0, 99, %arg1 + 7] : memref<1x104x194xf64, 1>
    %28 = affine.apply #map4(%arg1)
    %29 = arith.addi %28, %3 : index
    %30 = arith.remui %29, %c194 : index
    %31 = arith.divui %29, %c194 : index
    %32 = arith.remui %31, %c104 : index
    %33 = arith.divui %31, %c104 : index
    %34 = memref.load %arg0[%33, %32, %30] : memref<1x104x194xf64, 1>
    %35 = arith.mulf %2, %34 {fastmathFlags = #llvm.fastmath<none>} : f64
    affine.store %35, %arg0[0, 100, %arg1 + 7] : memref<1x104x194xf64, 1>
    %36 = affine.apply #map5(%arg1)
    %37 = arith.addi %36, %3 : index
    %38 = arith.remui %37, %c194 : index
    %39 = arith.divui %37, %c194 : index
    %40 = arith.remui %39, %c104 : index
    %41 = arith.divui %39, %c104 : index
    %42 = memref.load %arg0[%41, %40, %38] : memref<1x104x194xf64, 1>
    %43 = arith.mulf %2, %42 {fastmathFlags = #llvm.fastmath<none>} : f64
    affine.store %43, %arg0[0, 101, %arg1 + 7] : memref<1x104x194xf64, 1>
    %44 = affine.apply #map6(%arg1)
    %45 = arith.addi %44, %3 : index
    %46 = arith.remui %45, %c194 : index
    %47 = arith.divui %45, %c194 : index
    %48 = arith.remui %47, %c104 : index
    %49 = arith.divui %47, %c104 : index
    %50 = memref.load %arg0[%49, %48, %46] : memref<1x104x194xf64, 1>
    %51 = arith.mulf %2, %50 {fastmathFlags = #llvm.fastmath<none>} : f64
    affine.store %51, %arg0[0, 102, %arg1 + 7] : memref<1x104x194xf64, 1>
    %52 = affine.apply #map7(%arg1)
    %53 = arith.addi %52, %3 : index
    %54 = arith.remui %53, %c194 : index
    %55 = arith.divui %53, %c194 : index
    %56 = arith.remui %55, %c104 : index
    %57 = arith.divui %55, %c104 : index
    %58 = memref.load %arg0[%57, %56, %54] : memref<1x104x194xf64, 1>
    %59 = arith.mulf %2, %58 {fastmathFlags = #llvm.fastmath<none>} : f64
    affine.store %59, %arg0[0, 103, %arg1 + 7] : memref<1x104x194xf64, 1>
    %60 = arith.index_cast %1#1 : i64 to index
    %61 = affine.apply #map8(%arg1)
    %62 = arith.addi %61, %60 : index
    %63 = arith.remui %62, %c194 : index
    %64 = arith.divui %62, %c194 : index
    %65 = arith.remui %64, %c104 : index
    %66 = arith.divui %64, %c104 : index
    %67 = memref.load %arg0[%66, %65, %63] : memref<1x104x194xf64, 1>
    %68 = arith.mulf %2, %67 {fastmathFlags = #llvm.fastmath<none>} : f64
    %69 = affine.load %arg0[0, 96, %arg1 + 7] : memref<1x104x194xf64, 1>
    %70 = affine.if #set1(%arg1) -> f64 {
      affine.yield %69 : f64
    } else {
      affine.yield %68 : f64
    }
    affine.store %70, %arg0[0, 96, %arg1 + 7] : memref<1x104x194xf64, 1>
  }
  return
}
