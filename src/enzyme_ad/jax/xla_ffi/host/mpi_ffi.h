#pragma once

extern "C" void enzymexla_set_mpi_comm_rank(void *);
extern "C" void enzymexla_set_mpi_comm_size(void *);
extern "C" void enzymexla_set_mpi_comm_split(void *);
extern "C" void enzymexla_set_mpi_barrier(void *);
extern "C" void enzymexla_set_mpi_send(void *);
extern "C" void enzymexla_set_mpi_isend(void *);
extern "C" void enzymexla_set_mpi_recv(void *);
extern "C" void enzymexla_set_mpi_irecv(void *);
extern "C" void enzymexla_set_mpi_wait(void *);
extern "C" void enzymexla_set_mpi_waitall(void *);
extern "C" void enzymexla_set_mpi_allreduce(void *);
extern "C" void enzymexla_set_mpi_bcast(void *);

extern "C" size_t enzymexla_set_mpi_status_size();
