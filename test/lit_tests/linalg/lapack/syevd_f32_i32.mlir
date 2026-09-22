// RUN: enzymexlamlir-opt --lower-enzymexla-lapack="backend=cpu blas_int_width=32" %s | FileCheck %s --check-prefix=CPU

func.func @main(%arg0: tensor<3x3xf32>) -> (tensor<3x3xf32>, tensor<3xf32>, tensor<i32>) {
  %0:3 = enzymexla.lapack.syevd %arg0 {uplo = #enzymexla.uplo<U>} : (tensor<3x3xf32>) -> (tensor<3x3xf32>, tensor<3xf32>, tensor<i32>)
  return %0#0, %0#1, %0#2 : tensor<3x3xf32>, tensor<3xf32>, tensor<i32>
}

// i32 LAPACK sizes are sign-extended to i64 before reaching malloc.
// CPU-LABEL:  llvm.func @enzymexla_wrapper_lapack_ssyevd_
// CPU:          %[[OPTF:.+]] = llvm.load %{{.+}} : !llvm.ptr -> f32
// CPU-NEXT:     %[[OPTW:.+]] = llvm.fptosi %[[OPTF]] : f32 to i32
// CPU-NEXT:     %[[OPTIW:.+]] = llvm.load %{{.+}} : !llvm.ptr -> i32
// CPU-NEXT:     %[[OPTW64:.+]] = llvm.sext %[[OPTW]] : i32 to i64
// CPU-NEXT:     %[[WBYTES:.+]] = llvm.mul %[[OPTW64]], %[[FOUR:.+]] : i64
// CPU-NEXT:     %[[WORK:.+]] = llvm.call @malloc(%[[WBYTES]]) : (i64) -> !llvm.ptr
// CPU-NEXT:     %[[OPTIW64:.+]] = llvm.sext %[[OPTIW]] : i32 to i64
// CPU-NEXT:     %[[IWBYTES:.+]] = llvm.mul %[[OPTIW64]], %[[FOUR]] : i64
// CPU-NEXT:     %[[IWORK:.+]] = llvm.call @malloc(%[[IWBYTES]]) : (i64) -> !llvm.ptr
// CPU:          llvm.call @enzymexla_lapack_ssyevd_(%{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %[[WORK]], %{{.+}}, %[[IWORK]], %{{.+}}, %{{.+}})
// CPU-NEXT:     llvm.call @free(%[[WORK]]) : (!llvm.ptr) -> ()
// CPU-NEXT:     llvm.call @free(%[[IWORK]]) : (!llvm.ptr) -> ()
// CPU-NEXT:     llvm.return
