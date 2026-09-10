// RUN: enzymexlamlir-opt %s --llvm-to-affine-access | FileCheck %s

// A stack tensor<double, 3> is zeroed as one aggregate store, written one
// element at a time and read back as scalars: the allocation is an aggregate
// of one scalar throughout, so it converts to a flat memref of that scalar,
// and the aggregate view's store becomes one store per leaf. The lifetime
// markers go with the allocation. A view of an aggregate with mixed leaves
// keeps the allocation as it is.

func.func @unit_vector(%d: index) -> (f64, f64, f64) {
  %c1 = arith.constant 1 : i32
  %cst = arith.constant 1.000000e+00 : f64
  %zero = arith.constant 0.000000e+00 : f64
  %poison = llvm.mlir.poison : !llvm.array<3 x f64>
  %undef = llvm.mlir.undef : !llvm.struct<"struct.mfem::future::tensor", (array<3 x f64>)>
  %e = llvm.alloca %c1 x !llvm.struct<"struct.mfem::future::tensor", (array<3 x f64>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  llvm.intr.lifetime.start %e : !llvm.ptr
  %agg = "enzymexla.pointer2memref"(%e) : (!llvm.ptr) -> memref<?x!llvm.struct<"struct.mfem::future::tensor", (array<3 x f64>)>>
  %0 = llvm.insertvalue %zero, %poison[0] : !llvm.array<3 x f64>
  %1 = llvm.insertvalue %zero, %0[1] : !llvm.array<3 x f64>
  %2 = llvm.insertvalue %zero, %1[2] : !llvm.array<3 x f64>
  %3 = llvm.insertvalue %2, %undef[0] : !llvm.struct<"struct.mfem::future::tensor", (array<3 x f64>)>
  affine.store %3, %agg[0] : memref<?x!llvm.struct<"struct.mfem::future::tensor", (array<3 x f64>)>>
  %v = "enzymexla.pointer2memref"(%e) : (!llvm.ptr) -> memref<?xf64>
  affine.store %cst, %v[%d] : memref<?xf64>
  %x = affine.load %v[0] : memref<?xf64>
  %y = affine.load %v[1] : memref<?xf64>
  %z = affine.load %v[2] : memref<?xf64>
  llvm.intr.lifetime.end %e : !llvm.ptr
  return %x, %y, %z : f64, f64, f64
}

func.func @copy_out(%d: index, %in: f64) -> !llvm.array<2 x array<2 x f64>> {
  %c1 = arith.constant 1 : i32
  %m = llvm.alloca %c1 x !llvm.array<2 x array<2 x f64>> : (i32) -> !llvm.ptr
  %v = "enzymexla.pointer2memref"(%m) : (!llvm.ptr) -> memref<?xf64>
  affine.store %in, %v[%d] : memref<?xf64>
  %agg = "enzymexla.pointer2memref"(%m) : (!llvm.ptr) -> memref<?x!llvm.array<2 x array<2 x f64>>>
  %r = affine.load %agg[0] : memref<?x!llvm.array<2 x array<2 x f64>>>
  return %r : !llvm.array<2 x array<2 x f64>>
}

func.func @mixed(%d: index, %in: f64) -> !llvm.struct<(f64, i32)> {
  %c1 = arith.constant 1 : i32
  %m = llvm.alloca %c1 x !llvm.struct<(f64, i32)> : (i32) -> !llvm.ptr
  %v = "enzymexla.pointer2memref"(%m) : (!llvm.ptr) -> memref<?xf64>
  affine.store %in, %v[0] : memref<?xf64>
  %agg = "enzymexla.pointer2memref"(%m) : (!llvm.ptr) -> memref<?x!llvm.struct<(f64, i32)>>
  %r = affine.load %agg[0] : memref<?x!llvm.struct<(f64, i32)>>
  return %r : !llvm.struct<(f64, i32)>
}

// CHECK:    func.func @unit_vector(%[[a1:.+]]: index) -> (f64, f64, f64) {
// CHECK-NEXT:    %[[a2:.+]] = arith.constant 1.000000e+00 : f64
// CHECK-NEXT:    %[[a3:.+]] = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:    %[[a4:.+]] = memref.alloca() {alignment = 8 : i64} : memref<3xf64>
// CHECK-NEXT:    affine.store %[[a3]], %[[a4]][0] : memref<3xf64>
// CHECK-NEXT:    affine.store %[[a3]], %[[a4]][1] : memref<3xf64>
// CHECK-NEXT:    affine.store %[[a3]], %[[a4]][2] : memref<3xf64>
// CHECK-NEXT:    affine.store %[[a2]], %[[a4]][%[[a1]]] : memref<3xf64>
// CHECK-NEXT:    %[[a5:.+]] = affine.load %[[a4]][0] : memref<3xf64>
// CHECK-NEXT:    %[[a6:.+]] = affine.load %[[a4]][1] : memref<3xf64>
// CHECK-NEXT:    %[[a7:.+]] = affine.load %[[a4]][2] : memref<3xf64>
// CHECK-NEXT:    return %[[a5]], %[[a6]], %[[a7]] : f64, f64, f64
// CHECK-NEXT:  }
// CHECK-NEXT:  func.func @copy_out(%[[a1]]: index, %[[a8:.+]]: f64) -> !llvm.array<2 x array<2 x f64>> {
// CHECK-NEXT:    %[[a5]] = llvm.mlir.undef : !llvm.array<2 x array<2 x f64>>
// CHECK-NEXT:    %[[a4]] = memref.alloca() : memref<4xf64>
// CHECK-NEXT:    affine.store %[[a8]], %[[a4]][%[[a1]]] : memref<4xf64>
// CHECK-NEXT:    %[[a6]] = affine.load %[[a4]][0] : memref<4xf64>
// CHECK-NEXT:    %[[a7]] = llvm.insertvalue %[[a6]], %[[a5]][0, 0] : !llvm.array<2 x array<2 x f64>> 
// CHECK-NEXT:    %[[a9:.+]] = affine.load %[[a4]][1] : memref<4xf64>
// CHECK-NEXT:    %[[a10:.+]] = llvm.insertvalue %[[a9]], %[[a7]][0, 1] : !llvm.array<2 x array<2 x f64>> 
// CHECK-NEXT:    %[[a11:.+]] = affine.load %[[a4]][2] : memref<4xf64>
// CHECK-NEXT:    %[[a12:.+]] = llvm.insertvalue %[[a11]], %[[a10]][1, 0] : !llvm.array<2 x array<2 x f64>> 
// CHECK-NEXT:    %[[a13:.+]] = affine.load %[[a4]][3] : memref<4xf64>
// CHECK-NEXT:    %[[a14:.+]] = llvm.insertvalue %[[a13]], %[[a12]][1, 1] : !llvm.array<2 x array<2 x f64>> 
// CHECK-NEXT:    return %[[a14]] : !llvm.array<2 x array<2 x f64>>
// CHECK-NEXT:  }
// CHECK-NEXT:  func.func @mixed(%[[a1]]: index, %[[a8]]: f64) -> !llvm.struct<(f64, i32)> {
// CHECK-NEXT:    %[[a15:.+]] = arith.constant 1 : i32
// CHECK-NEXT:    %[[a5]] = llvm.alloca %[[a15]] x !llvm.struct<(f64, i32)> : (i32) -> !llvm.ptr
// CHECK-NEXT:    %[[a6]] = "enzymexla.pointer2memref"(%[[a5]]) : (!llvm.ptr) -> memref<?xf64>
// CHECK-NEXT:    affine.store %[[a8]], %[[a6]][0] : memref<?xf64>
// CHECK-NEXT:    %[[a7]] = "enzymexla.pointer2memref"(%[[a5]]) : (!llvm.ptr) -> memref<?x!llvm.struct<(f64, i32)>>
// CHECK-NEXT:    %[[a9]] = affine.load %[[a7]][0] : memref<?x!llvm.struct<(f64, i32)>>
// CHECK-NEXT:    return %[[a9]] : !llvm.struct<(f64, i32)>
// CHECK-NEXT:  }
