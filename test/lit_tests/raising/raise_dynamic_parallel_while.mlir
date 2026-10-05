// RUN: enzymexlamlir-opt %s --raise-affine-to-stablehlo --split-input-file | FileCheck %s

// A parallel axis whose extent is only known at runtime raises as a
// stablehlo.while over the raised body, tagged enzymexla.parallel so
// downstream passes know the iterations are independent.
func.func @dynkern(%out: memref<100xf64, 1>, %nbuf: memref<i64, 1>) {
  %n = affine.load %nbuf[] : memref<i64, 1>
  %ni = arith.index_cast %n : i64 to index
  affine.parallel (%i) = (0) to (symbol(%ni)) {
    %iv = arith.index_castui %i : index to i64
    %v = arith.sitofp %iv : i64 to f64
    affine.store %v, %out[%i] : memref<100xf64, 1>
  }
  return
}

// CHECK-LABEL: func.func private @dynkern_raised(
// CHECK-SAME: %[[OUT:.+]]: tensor<100xf64>, %[[N:.+]]: tensor<i64>
// CHECK: %[[WHILE:.+]]:3 = stablehlo.while(%[[IV:.+]] = %{{.+}}, %[[BUF:.+]] = %[[OUT]], %{{.+}} = %[[N]]) : tensor<i64>, tensor<100xf64>, tensor<i64> attributes {enzymexla.parallel}
// CHECK: cond {
// CHECK: stablehlo.compare LT, %[[IV]], %{{.+}} : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK: } do {
// CHECK: %[[V:.+]] = stablehlo.broadcast_in_dim %{{.+}}, dims = [] : (tensor<f64>) -> tensor<1xf64>
// CHECK: stablehlo.dynamic_update_slice %[[BUF]], %[[V]], %[[IV]] : (tensor<100xf64>, tensor<1xf64>, tensor<i64>) -> tensor<100xf64>
// CHECK: return %[[WHILE]]#1, %[[WHILE]]#2 : tensor<100xf64>, tensor<i64>

// -----

// Mixed extents: the static axis stays batched inside the while body, only
// the dynamic axis is peeled into the while.
func.func @mixed(%out: memref<16x100xf64, 1>, %nbuf: memref<i64, 1>) {
  %n = affine.load %nbuf[] : memref<i64, 1>
  %ni = arith.index_cast %n : i64 to index
  affine.parallel (%e, %j) = (0, 0) to (symbol(%ni), 16) {
    %iv = arith.index_castui %j : index to i64
    %v = arith.sitofp %iv : i64 to f64
    affine.store %v, %out[%j, %e] : memref<16x100xf64, 1>
  }
  return
}

// CHECK-LABEL: func.func private @mixed_raised(
// CHECK: %[[W:.+]]:3 = stablehlo.while(%[[IV:.+]] = %{{.+}}, %[[BUF:.+]] = %{{.+}}, %{{.+}} = %{{.+}}) : tensor<i64>, tensor<16x100xf64>, tensor<i64> attributes {enzymexla.parallel}
// CHECK: } do {
// CHECK: %[[UPD:.+]] = stablehlo.broadcast_in_dim %{{.+}}, dims = [0] : (tensor<16xf64>) -> tensor<16x1xf64>
// CHECK: stablehlo.dynamic_update_slice %[[BUF]], %[[UPD]], %{{.+}}, %[[IV]] : (tensor<16x100xf64>, tensor<16x1xf64>, tensor<i64>, tensor<i64>) -> tensor<16x100xf64>

// -----

// A dynamic parallel loop outside any raised region is left untouched: the
// peel only applies where raising will consume the result.
func.func @host(%out: memref<100xf64, 1>, %n: index) -> index {
  affine.parallel (%i) = (0) to (%n) {
    %c = arith.constant 1.0 : f64
    affine.store %c, %out[%i] : memref<100xf64, 1>
  }
  return %n : index
}

// CHECK-LABEL: func.func @host(
// CHECK: affine.parallel
// CHECK-NOT: enzymexla.parallel

// -----

// A parallel axis whose extent differs between the lanes of a static one
// raises as a masked while, the lanes past their extent masked off: still
// tagged, since each lane's iterations are those of the parallel loop.
func.func @lanes(%out: memref<4x100xf64, 1>, %nbuf: memref<i64, 1>) {
  %n = affine.load %nbuf[] : memref<i64, 1>
  %ni = arith.index_cast %n : i64 to index
  affine.parallel (%l) = (0) to (4) {
    affine.parallel (%i) = (0) to (symbol(%ni) + %l) {
      %iv = arith.index_castui %i : index to i64
      %v = arith.sitofp %iv : i64 to f64
      affine.store %v, %out[%l, %i] : memref<4x100xf64, 1>
    }
  }
  return
}

// CHECK:  func.func private @lanes_raised(%arg0: tensor<4x100xf64>, %arg1: tensor<i64>) -> (tensor<4x100xf64>, tensor<i64>) {
// CHECK-NEXT:   %0 = stablehlo.iota dim = 0 : tensor<4xi64>
// CHECK-NEXT:   %c = stablehlo.constant dense<0> : tensor<4xi64>
// CHECK-NEXT:   %1 = stablehlo.add %0, %c : tensor<4xi64>
// CHECK-NEXT:   %c_0 = stablehlo.constant dense<1> : tensor<4xi64>
// CHECK-NEXT:   %2 = stablehlo.multiply %1, %c_0 : tensor<4xi64>
// CHECK-NEXT:   %3 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<i64>) -> tensor<4xi64>
// CHECK-NEXT:   %4 = stablehlo.add %2, %3 : tensor<4xi64>
// CHECK-NEXT:   %c_1 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:   %5 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<i64>) -> tensor<4xi64>
// CHECK-NEXT:   %6 = stablehlo.add %2, %5 : tensor<4xi64>
// CHECK-NEXT:   %c_2 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:   %7 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<i64>) -> tensor<4xi64>
// CHECK-NEXT:   %8 = stablehlo.add %2, %7 : tensor<4xi64>
// CHECK-NEXT:   %c_3 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:   %9 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<i64>) -> tensor<4xi64>
// CHECK-NEXT:   %10 = stablehlo.add %2, %9 : tensor<4xi64>
// CHECK-NEXT:   %c_4 = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:   %11 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<i64>) -> tensor<4xi64>
// CHECK-NEXT:   %12 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<i64>) -> tensor<4xi64>
// CHECK-NEXT:   %13 = stablehlo.subtract %10, %12 : tensor<4xi64>
// CHECK-NEXT:   %c_5 = stablehlo.constant dense<0> : tensor<4xi64>
// CHECK-NEXT:   %14 = stablehlo.maximum %13, %c_5 : tensor<4xi64>
// CHECK-NEXT:   %c_6 = stablehlo.constant dense<1> : tensor<4xi64>
// CHECK-NEXT:   %15 = stablehlo.subtract %11, %c_6 : tensor<4xi64>
// CHECK-NEXT:   %16 = stablehlo.add %14, %15 : tensor<4xi64>
// CHECK-NEXT:   %17 = stablehlo.divide %16, %11 : tensor<4xi64>
// CHECK-NEXT:   %c_7 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:   %18 = stablehlo.reduce(%17 init: %c_7) applies stablehlo.maximum across dimensions = [0] : (tensor<4xi64>, tensor<i64>) -> tensor<i64>
// CHECK-NEXT:   %c_8 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:   %19:3 = stablehlo.while(%iterArg = %c_8, %iterArg_9 = %arg0, %iterArg_10 = %arg1) : tensor<i64>, tensor<4x100xf64>, tensor<i64> attributes {enzymexla.parallel}
// CHECK-NEXT:   cond {
// CHECK-NEXT:     %20 = stablehlo.compare LT, %iterArg, %18 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:     stablehlo.return %20 : tensor<i1>
// CHECK-NEXT:   } do {
// CHECK-NEXT:     %20 = stablehlo.broadcast_in_dim %iterArg, dims = [] : (tensor<i64>) -> tensor<4xi64>
// CHECK-NEXT:     %21 = stablehlo.multiply %20, %11 : tensor<4xi64>
// CHECK-NEXT:     %22 = stablehlo.add %12, %21 : tensor<4xi64>
// CHECK-NEXT:     %23 = stablehlo.compare LT, %22, %10 : (tensor<4xi64>, tensor<4xi64>) -> tensor<4xi1>
// CHECK-NEXT:     %24 = arith.sitofp %22 : tensor<4xi64> to tensor<4xf64>
// CHECK-NEXT:     %25 = stablehlo.reshape %2 : (tensor<4xi64>) -> tensor<4x1xi64>
// CHECK-NEXT:     %26 = stablehlo.broadcast_in_dim %22, dims = [0] : (tensor<4xi64>) -> tensor<4x1xi64>
// CHECK-NEXT:     %27 = stablehlo.concatenate %25, %26, dim = 1 : (tensor<4x1xi64>, tensor<4x1xi64>) -> tensor<4x2xi64>
// CHECK-NEXT:     %c_11 = stablehlo.constant dense<-1> : tensor<4x2xi64>
// CHECK-NEXT:     %28 = stablehlo.broadcast_in_dim %23, dims = [0] : (tensor<4xi1>) -> tensor<4x2xi1>
// CHECK-NEXT:     %29 = stablehlo.select %28, %27, %c_11 : tensor<4x2xi1>, tensor<4x2xi64>
// CHECK-NEXT:     %30 = "stablehlo.scatter"(%iterArg_9, %29, %24) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = false}> ({
// CHECK-NEXT:     ^bb0(%arg2: tensor<f64>, %arg3: tensor<f64>):
// CHECK-NEXT:       stablehlo.return %arg3 : tensor<f64>
// CHECK-NEXT:     }) : (tensor<4x100xf64>, tensor<4x2xi64>, tensor<4xf64>) -> tensor<4x100xf64>
// CHECK-NEXT:     %c_12 = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:     %31 = stablehlo.add %iterArg, %c_12 : tensor<i64>
// CHECK-NEXT:     stablehlo.return %31, %30, %iterArg_10 : tensor<i64>, tensor<4x100xf64>, tensor<i64>
// CHECK-NEXT:   }
// CHECK-NEXT:   return %19#1, %19#2 : tensor<4x100xf64>, tensor<i64>
// CHECK-NEXT: }
