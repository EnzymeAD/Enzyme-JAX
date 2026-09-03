// RUN: enzymexlamlir-opt %s | FileCheck %s


#op_null = #comm.mpi.op<MPI_OP_NULL>
#sum = #comm.mpi.op<"MPI_SUM">
#min = #comm.mpi.op<MPI_MIN>
#max = #comm.mpi.op<MPI_MAX>
#prod = #comm.mpi.op<MPI_PROD>
#band = #comm.mpi.op<MPI_BAND>
#bor = #comm.mpi.op<MPI_BOR>
#bxor = #comm.mpi.op<MPI_BXOR>
#land = #comm.mpi.op<MPI_LAND>
#lor = #comm.mpi.op<MPI_LOR>
#lxor = #comm.mpi.op<MPI_LXOR>
#minloc = #comm.mpi.op<MPI_MINLOC>
#maxloc = #comm.mpi.op<MPI_MAXLOC>
#replace = #comm.mpi.op<MPI_REPLACE>
#no_op = #comm.mpi.op<MPI_NO_OP>

// XFAIL: *
#error = #comm.mpi.op<MPI_OP_ERROR>
