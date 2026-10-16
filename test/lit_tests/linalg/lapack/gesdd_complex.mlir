// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(lower-enzymexla-lapack{backend=cpu blas_int_width=64},enzyme-hlo-opt,drop-unsupported-attributes)" %s | FileCheck %s --check-prefix=CPU

// `?gesdd` takes the real workspace `rwork` between `lwork` and `iwork`

module {
  func.func @main(%arg0: tensor<8x6xcomplex<f64>>) -> (tensor<8x6xcomplex<f64>>, tensor<6xf64>, tensor<6x6xcomplex<f64>>, tensor<i64>) {
    %0:4 = enzymexla.lapack.gesdd %arg0 : (tensor<8x6xcomplex<f64>>) -> (tensor<8x6xcomplex<f64>>, tensor<6xf64>, tensor<6x6xcomplex<f64>>, tensor<i64>)
    return %0#0, %0#1, %0#2, %0#3 : tensor<8x6xcomplex<f64>>, tensor<6xf64>, tensor<6x6xcomplex<f64>>, tensor<i64>
  }
}

module {
  func.func @main(%arg0: tensor<8x6xcomplex<f64>>) -> tensor<6xf64> {
    %0:4 = enzymexla.lapack.gesdd %arg0 <{compute_uv = false}> : (tensor<8x6xcomplex<f64>>) -> (tensor<8x6xcomplex<f64>>, tensor<6xf64>, tensor<6x6xcomplex<f64>>, tensor<i64>)
    return %0#1 : tensor<6xf64>
  }
}

// The workspaces are sized by LAPACK and can be large, so they live on the heap

// CPU: llvm.func private @enzymexla_wrapper_lapack_zgesdd_(
// CPU-DAG:   %[[EIGHT:.*]] = llvm.mlir.constant(8 : i64) : i64
// CPU-DAG:   %[[TWO:.*]] = llvm.mlir.constant(2 : i64) : i64
// CPU:   %[[LWORK:.*]] = llvm.alloca %{{.*}} x i64 : (i64) -> !llvm.ptr
// CPU:   %[[QUERY:.*]] = llvm.alloca %[[TWO]] x f64 : (i64) -> !llvm.ptr
// CPU:   %[[IWORK:.*]] = llvm.call @malloc(%{{.*}}) : (i64) -> !llvm.ptr
// CPU:   %[[RBYTES:.*]] = llvm.mul %{{.*}}, %[[EIGHT]] : i64
// CPU-NEXT:   %[[RWORK:.*]] = llvm.call @malloc(%[[RBYTES]]) : (i64) -> !llvm.ptr
// CPU-NEXT:   llvm.call @enzymexla_lapack_zgesdd_(%{{.*}}, %arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %[[QUERY]], %[[LWORK]], %[[RWORK]], %[[IWORK]], %arg9, %{{.*}})
// CPU-NEXT:   %[[SIZEF:.*]] = llvm.load %[[QUERY]] : !llvm.ptr -> f64
// CPU-NEXT:   %[[SIZE:.*]] = llvm.fptosi %[[SIZEF]] : f64 to i64
// CPU-NEXT:   %[[REALS:.*]] = llvm.mul %[[SIZE]], %[[TWO]] : i64
// CPU-NEXT:   %[[WBYTES:.*]] = llvm.mul %[[REALS]], %[[EIGHT]] : i64
// CPU-NEXT:   %[[WORK:.*]] = llvm.call @malloc(%[[WBYTES]]) : (i64) -> !llvm.ptr
// CPU-NEXT:   llvm.store %[[SIZE]], %[[LWORK]] : i64, !llvm.ptr
// CPU-NEXT:   llvm.call @enzymexla_lapack_zgesdd_(%{{.*}}, %arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %[[WORK]], %[[LWORK]], %[[RWORK]], %[[IWORK]], %arg9, %{{.*}})
// CPU-NEXT:   llvm.call @free(%[[IWORK]]) : (!llvm.ptr) -> ()
// CPU-NEXT:   llvm.call @free(%[[RWORK]]) : (!llvm.ptr) -> ()
// CPU-NEXT:   llvm.call @free(%[[WORK]]) : (!llvm.ptr) -> ()
// CPU-NEXT:   llvm.return

// CPU: llvm.func private @enzymexla_wrapper_lapack_zgesdd__nouv(
// CPU-DAG:   %[[SEVEN:.*]] = llvm.mlir.constant(7 : i64) : i64
// CPU-DAG:   %[[EIGHT:.*]] = llvm.mlir.constant(8 : i64) : i64
// CPU:   %[[LWORK:.*]] = llvm.alloca %{{.*}} x i64 : (i64) -> !llvm.ptr
// CPU:   %[[IWORK:.*]] = llvm.call @malloc(%{{.*}}) : (i64) -> !llvm.ptr
// CPU:   %[[RSIZE:.*]] = arith.muli %{{.*}}, %[[SEVEN]] : i64
// CPU-NEXT:   %[[RBYTES:.*]] = llvm.mul %[[RSIZE]], %[[EIGHT]] : i64
// CPU-NEXT:   %[[RWORK:.*]] = llvm.call @malloc(%[[RBYTES]]) : (i64) -> !llvm.ptr
// CPU-NEXT:   llvm.call @enzymexla_lapack_zgesdd_(%{{.*}}, %arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %{{.*}}, %[[LWORK]], %[[RWORK]], %[[IWORK]], %arg9, %{{.*}})
