// RUN: enzymexlamlir-opt %s --llvm-to-affine-access | FileCheck %s

// A lambda's captures: an i32, a pointer and two doubles. The host reads them
// as one aggregate and each lane copies them into a private alloca, of which it
// reads only the doubles. Each aggregate access becomes one access per leaf
// through a view of that leaf's type, and the lane's stores of the leaves it
// never reads go away, the pointer among them.

module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f64 = dense<64> : vector<2xi64>>} {
  func.func @captures(%cap: !llvm.ptr, %out: memref<4xf64>) {
    %c1 = arith.constant 1 : i64
    %v = "enzymexla.pointer2memref"(%cap) : (!llvm.ptr) -> memref<?x!llvm.struct<(i32, ptr, array<2 x f64>)>>
    %agg = affine.load %v[0] : memref<?x!llvm.struct<(i32, ptr, array<2 x f64>)>>
    affine.parallel (%i) = (0) to (4) {
      %copy = llvm.alloca %c1 x !llvm.struct<(i32, ptr, array<2 x f64>)> : (i64) -> !llvm.ptr
      %w = "enzymexla.pointer2memref"(%copy) : (!llvm.ptr) -> memref<?x!llvm.struct<(i32, ptr, array<2 x f64>)>>
      affine.store %agg, %w[0] : memref<?x!llvm.struct<(i32, ptr, array<2 x f64>)>>
      %d = "enzymexla.pointer2memref"(%copy) : (!llvm.ptr) -> memref<?xf64>
      affine.for %k = 0 to 2 {
        %x = affine.load %d[%k + 2] : memref<?xf64>
        affine.store %x, %out[%i] : memref<4xf64>
      }
    }
    return
  }

  // A read at an index of unknown range could reach any leaf: every store stays.
  func.func @unknown_read(%cap: !llvm.ptr, %out: memref<4xf64>, %n: index) {
    %c1 = arith.constant 1 : i64
    %v = "enzymexla.pointer2memref"(%cap) : (!llvm.ptr) -> memref<?x!llvm.struct<(i32, ptr, array<2 x f64>)>>
    %agg = affine.load %v[0] : memref<?x!llvm.struct<(i32, ptr, array<2 x f64>)>>
    affine.parallel (%i) = (0) to (4) {
      %copy = llvm.alloca %c1 x !llvm.struct<(i32, ptr, array<2 x f64>)> : (i64) -> !llvm.ptr
      %w = "enzymexla.pointer2memref"(%copy) : (!llvm.ptr) -> memref<?x!llvm.struct<(i32, ptr, array<2 x f64>)>>
      affine.store %agg, %w[0] : memref<?x!llvm.struct<(i32, ptr, array<2 x f64>)>>
      %d = "enzymexla.pointer2memref"(%copy) : (!llvm.ptr) -> memref<?xf64>
      %x = affine.load %d[symbol(%n)] : memref<?xf64>
      affine.store %x, %out[%i] : memref<4xf64>
    }
    return
  }
}

// CHECK-LABEL: func.func @captures
// CHECK-SAME: (%[[CAP:.+]]: !llvm.ptr, %[[OUT:.+]]: memref<4xf64>)
// CHECK-NEXT:    %[[D:.+]] = "enzymexla.pointer2memref"(%[[CAP]]) : (!llvm.ptr) -> memref<?xf64>
// CHECK-NEXT:    %[[X0:.+]] = affine.load %[[D]][2] : memref<?xf64>
// CHECK-NEXT:    %[[X1:.+]] = affine.load %[[D]][3] : memref<?xf64>
// CHECK-NEXT:    affine.parallel (%[[I:.+]]) = (0) to (4) {
// CHECK-NEXT:      %[[COPY:.+]] = memref.alloca() : memref<4xf64>
// CHECK-NEXT:      affine.store %[[X0]], %[[COPY]][2] : memref<4xf64>
// CHECK-NEXT:      affine.store %[[X1]], %[[COPY]][3] : memref<4xf64>
// CHECK-NEXT:      affine.for %[[K:.+]] = 0 to 2 {
// CHECK-NEXT:        %[[X:.+]] = affine.load %[[COPY]][%[[K]] + 2] : memref<4xf64>
// CHECK-NEXT:        affine.store %[[X]], %[[OUT]][%[[I]]] : memref<4xf64>
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:    return

// CHECK-LABEL: func.func @unknown_read
// CHECK-SAME: (%[[CAP:.+]]: !llvm.ptr, %[[OUT:.+]]: memref<4xf64>, %[[N:.+]]: index)
// CHECK-NEXT:    %[[C1:.+]] = arith.constant 1 : i64
// CHECK-NEXT:    %[[VI:.+]] = "enzymexla.pointer2memref"(%[[CAP]]) : (!llvm.ptr) -> memref<?xi32>
// CHECK-NEXT:    %[[VP:.+]] = "enzymexla.pointer2memref"(%[[CAP]]) : (!llvm.ptr) -> memref<?x!llvm.ptr>
// CHECK-NEXT:    %[[VD:.+]] = "enzymexla.pointer2memref"(%[[CAP]]) : (!llvm.ptr) -> memref<?xf64>
// CHECK-NEXT:    %[[A:.+]] = affine.load %[[VI]][0] : memref<?xi32>
// CHECK-NEXT:    %[[B:.+]] = affine.load %[[VP]][1] : memref<?x!llvm.ptr>
// CHECK-NEXT:    %[[X0:.+]] = affine.load %[[VD]][2] : memref<?xf64>
// CHECK-NEXT:    %[[X1:.+]] = affine.load %[[VD]][3] : memref<?xf64>
// CHECK-NEXT:    affine.parallel (%[[I:.+]]) = (0) to (4) {
// CHECK-NEXT:      %[[COPY:.+]] = llvm.alloca %[[C1]] x !llvm.struct<(i32, ptr, array<2 x f64>)> : (i64) -> !llvm.ptr
// CHECK-NEXT:      %[[WI:.+]] = "enzymexla.pointer2memref"(%[[COPY]]) : (!llvm.ptr) -> memref<?xi32>
// CHECK-NEXT:      %[[WP:.+]] = "enzymexla.pointer2memref"(%[[COPY]]) : (!llvm.ptr) -> memref<?x!llvm.ptr>
// CHECK-NEXT:      %[[WD:.+]] = "enzymexla.pointer2memref"(%[[COPY]]) : (!llvm.ptr) -> memref<?xf64>
// CHECK-NEXT:      affine.store %[[A]], %[[WI]][0] : memref<?xi32>
// CHECK-NEXT:      affine.store %[[B]], %[[WP]][1] : memref<?x!llvm.ptr>
// CHECK-NEXT:      affine.store %[[X0]], %[[WD]][2] : memref<?xf64>
// CHECK-NEXT:      affine.store %[[X1]], %[[WD]][3] : memref<?xf64>
// CHECK-NEXT:      %[[R:.+]] = "enzymexla.pointer2memref"(%[[COPY]]) : (!llvm.ptr) -> memref<?xf64>
// CHECK-NEXT:      %[[X:.+]] = affine.load %[[R]][symbol(%[[N]])] : memref<?xf64>
// CHECK-NEXT:      affine.store %[[X]], %[[OUT]][%[[I]]] : memref<4xf64>
// CHECK-NEXT:    }
// CHECK-NEXT:    return
