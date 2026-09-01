// RUN: enzymexlamlir-opt %s | FileCheck %s

#comm.mpi_comm<MPI_COMM_WORLD>
#comm.mpi_comm<MPI_COMM_SELF>
#comm.mpi_comm<MPI_COMM_NULL>
