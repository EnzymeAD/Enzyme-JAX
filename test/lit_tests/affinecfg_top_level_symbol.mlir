// RUN: enzymexlamlir-opt --affine-cfg %s | FileCheck %s

module {
  gpu.module @kernels {
    gpu.func @select_index(%buffer: memref<?xi64>, %index: index,
                           %enter: i1, %choose_doubled: i1,
                           %value: i64) kernel {
      scf.if %enter {
        %doubled = arith.addi %index, %index : index
        %selected = arith.select %choose_doubled, %doubled, %index : index
        memref.store %value, %buffer[%selected] : memref<?xi64>
      }

      gpu.return
    }
  }
}

// CHECK-LABEL:   gpu.func @select_index(
// CHECK-SAME:      %[[buffer:.*]]: memref<?xi64>, %[[index:.*]]: index,
// CHECK-SAME:      %[[enter:.*]]: i1, %[[choose:.*]]: i1, %[[value:.*]]: i64) kernel {
// CHECK:           %[[doubled:.*]] = arith.addi %[[index]], %[[index]] : index
// CHECK:           %[[selected:.*]] = arith.select %[[choose]], %[[doubled]], %[[index]] : index
// CHECK:           scf.if %[[enter]] {
// CHECK:             affine.store %[[value]], %[[buffer]]{{\[}}symbol(%[[selected]])] : memref<?xi64>
// CHECK:           }
// CHECK:           gpu.return
