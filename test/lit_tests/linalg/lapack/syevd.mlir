// RUN: enzymexlamlir-opt --lower-enzymexla-lapack={backend=cpu,blas_int_width=64} --enzyme-hlo-opt %s | FileCheck %s --check-prefix=CPU

func.func @main(%arg0: tensor<4x4xf64>) -> (tensor<4x4xf64>, tensor<4xf64>, tensor<i64>) {
  %0:3 = enzymexla.lapack.syevd %arg0 <{uplo = #enzymexla.uplo<L>}> : (tensor<4x4xf64>) -> (tensor<4x4xf64>, tensor<4xf64>, tensor<i64>)
  return %0#0, %0#1, %0#2 : tensor<4x4xf64>, tensor<4xf64>, tensor<i64>
}

// CPU:        llvm.func @free(!llvm.ptr)
// CPU-NEXT:   llvm.func @malloc(i64) -> !llvm.ptr

// Heap, not alloca: the workspace overflows the 8 MB worker stack from N = 723.
// CPU-LABEL:  llvm.func @enzymexla_wrapper_lapack_dsyevd_
// CPU-SAME:     (%[[A:.+]]: !llvm.ptr, %[[W:.+]]: !llvm.ptr, %[[INFO:.+]]: !llvm.ptr)
// CPU-NEXT:     %[[EIGHT:.+]] = llvm.mlir.constant(8 : i64) : i64
// CPU-NEXT:     %[[N:.+]] = llvm.mlir.constant(4 : i64) : i64
// CPU-NEXT:     %[[UPLO:.+]] = llvm.mlir.constant(76 : i8) : i8
// CPU-NEXT:     %[[JOBZ:.+]] = llvm.mlir.constant(86 : i8) : i8
// CPU-NEXT:     %[[ONE:.+]] = llvm.mlir.constant(1 : i64) : i64
// CPU-NEXT:     %[[M1:.+]] = llvm.mlir.constant(-1 : i64) : i64
// CPU-NEXT:     %[[JOBZP:.+]] = llvm.alloca %[[ONE]] x i8 : (i64) -> !llvm.ptr
// CPU-NEXT:     llvm.store %[[JOBZ]], %[[JOBZP]] : i8, !llvm.ptr
// CPU-NEXT:     %[[UPLOP:.+]] = llvm.alloca %[[ONE]] x i8 : (i64) -> !llvm.ptr
// CPU-NEXT:     llvm.store %[[UPLO]], %[[UPLOP]] : i8, !llvm.ptr
// CPU-NEXT:     %[[NP:.+]] = llvm.alloca %[[ONE]] x i64 : (i64) -> !llvm.ptr
// CPU-NEXT:     llvm.store %[[N]], %[[NP]] : i64, !llvm.ptr
// CPU-NEXT:     %[[LDAP:.+]] = llvm.alloca %[[ONE]] x i64 : (i64) -> !llvm.ptr
// CPU-NEXT:     llvm.store %[[N]], %[[LDAP]] : i64, !llvm.ptr
// CPU-NEXT:     %[[LWORK:.+]] = llvm.alloca %[[ONE]] x i64 : (i64) -> !llvm.ptr
// CPU-NEXT:     llvm.store %[[M1]], %[[LWORK]] : i64, !llvm.ptr
// CPU-NEXT:     %[[LIWORK:.+]] = llvm.alloca %[[ONE]] x i64 : (i64) -> !llvm.ptr
// CPU-NEXT:     llvm.store %[[M1]], %[[LIWORK]] : i64, !llvm.ptr
// CPU-NEXT:     %[[WQ:.+]] = llvm.alloca %[[ONE]] x f64 : (i64) -> !llvm.ptr
// CPU-NEXT:     %[[IWQ:.+]] = llvm.alloca %[[ONE]] x i64 : (i64) -> !llvm.ptr
// CPU-NEXT:     llvm.call @enzymexla_lapack_dsyevd_(%[[JOBZP]], %[[UPLOP]], %[[NP]], %[[A]], %[[LDAP]], %[[W]], %[[WQ]], %[[LWORK]], %[[IWQ]], %[[LIWORK]], %[[INFO]])
// CPU-NEXT:     %[[OPTF:.+]] = llvm.load %[[WQ]] : !llvm.ptr -> f64
// CPU-NEXT:     %[[OPTW:.+]] = llvm.fptosi %[[OPTF]] : f64 to i64
// CPU-NEXT:     %[[OPTIW:.+]] = llvm.load %[[IWQ]] : !llvm.ptr -> i64
// CPU-NEXT:     %[[WBYTES:.+]] = llvm.mul %[[OPTW]], %[[EIGHT]] : i64
// CPU-NEXT:     %[[WORK:.+]] = llvm.call @malloc(%[[WBYTES]]) : (i64) -> !llvm.ptr
// CPU-NEXT:     %[[IWBYTES:.+]] = llvm.mul %[[OPTIW]], %[[EIGHT]] : i64
// CPU-NEXT:     %[[IWORK:.+]] = llvm.call @malloc(%[[IWBYTES]]) : (i64) -> !llvm.ptr
// CPU-NEXT:     llvm.store %[[OPTW]], %[[LWORK]] : i64, !llvm.ptr
// CPU-NEXT:     llvm.store %[[OPTIW]], %[[LIWORK]] : i64, !llvm.ptr
// CPU-NEXT:     llvm.call @enzymexla_lapack_dsyevd_(%[[JOBZP]], %[[UPLOP]], %[[NP]], %[[A]], %[[LDAP]], %[[W]], %[[WORK]], %[[LWORK]], %[[IWORK]], %[[LIWORK]], %[[INFO]])
// CPU-NEXT:     llvm.call @free(%[[WORK]]) : (!llvm.ptr) -> ()
// CPU-NEXT:     llvm.call @free(%[[IWORK]]) : (!llvm.ptr) -> ()
// CPU-NEXT:     llvm.return

// CPU-LABEL:  llvm.func @enzymexla_lapack_dsyevd_

// CPU:  func.func @main(%[[ARG:.*]]: tensor<4x4xf64>) -> (tensor<4x4xf64>, tensor<4xf64>, tensor<i64>) {
// CPU-NEXT:    %[[W:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<4xf64>
// CPU-NEXT:    %[[INFO:.*]] = stablehlo.constant dense<-1> : tensor<i64>
// CPU-NEXT:    %[[RES:.*]]:3 = enzymexla.jit_call @enzymexla_wrapper_lapack_dsyevd_ (%[[ARG]], %[[W]], %[[INFO]])
// CPU-SAME:      output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [1], operand_index = 1, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [2], operand_index = 2, operand_tuple_indices = []>]
// CPU-SAME:      xla_side_effect_free
// CPU-SAME:      (tensor<4x4xf64>, tensor<4xf64>, tensor<i64>) -> (tensor<4x4xf64>, tensor<4xf64>, tensor<i64>)
// CPU-NEXT:    return %[[RES]]#0, %[[RES]]#1, %[[RES]]#2 : tensor<4x4xf64>, tensor<4xf64>, tensor<i64>
// CPU-NEXT:  }
