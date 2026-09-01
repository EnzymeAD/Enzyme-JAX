// RUN: enzymexlamlir-opt %s | FileCheck %s


// CHECK: #op_null = #comm.mpi.comm<"MPI_OP_NULL">
#op_null = #comm.mpi.comm<"MPI_OP_NULL">

// CHECK: #sum = #comm.mpi.comm<"MPI_SUM">
#sum = #comm.mpi.comm<"MPI_SUM">

// CHECK: #min = #comm.mpi.comm<"MPI_MIN">
#min = #comm.mpi.comm<"MPI_MIN">

// CHECK: #max = #comm.mpi.comm<"MPI_MAX">
#max = #comm.mpi.comm<"MPI_MAX">

// CHECK: #prod = #comm.mpi.comm<"MPI_PROD">
#prod = #comm.mpi.comm<"MPI_PROD">

// CHECK: #band = #comm.mpi.comm<"MPI_BAND">
#band = #comm.mpi.comm<"MPI_BAND">

// CHECK: #bor = #comm.mpi.comm<"MPI_BOR">
#bor = #comm.mpi.comm<"MPI_BOR">

// CHECK: #bxor = #comm.mpi.comm<"MPI_BXOR">
#bxor = #comm.mpi.comm<"MPI_BXOR">

// CHECK: #land = #comm.mpi.comm<"MPI_LAND">
#land = #comm.mpi.comm<"MPI_LAND">

// CHECK: #lor = #comm.mpi.comm<"MPI_LOR">
#lor = #comm.mpi.comm<"MPI_LOR">

// CHECK: #lxor = #comm.mpi.comm<"MPI_LXOR">
#lxor = #comm.mpi.comm<"MPI_LXOR">

// CHECK: #minloc = #comm.mpi.comm<"MPI_MINLOC">
#minloc = #comm.mpi.comm<"MPI_MINLOC">

// CHECK: #maxloc = #comm.mpi.comm<"MPI_MAXLOC">
#maxloc = #comm.mpi.comm<"MPI_MAXLOC">

// CHECK: #replace = #comm.mpi.comm<"MPI_REPLACE">
#replace = #comm.mpi.comm<"MPI_REPLACE">

// CHECK: #no_op = #comm.mpi.comm<"MPI_NO_OP">
#no_op = #comm.mpi.comm<"MPI_NO_OP">
