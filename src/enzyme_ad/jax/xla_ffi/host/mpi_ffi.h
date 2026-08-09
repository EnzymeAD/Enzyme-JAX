#pragma once

#if defined(__cplusplus)
extern "C" {
#endif

void enzymexla_set_mpi_comm_rank(void *);
void enzymexla_set_mpi_comm_size(void *);
void enzymexla_set_mpi_comm_split(void *);
void enzymexla_set_mpi_barrier(void *);
void enzymexla_set_mpi_send(void *);
void enzymexla_set_mpi_isend(void *);
void enzymexla_set_mpi_recv(void *);
void enzymexla_set_mpi_irecv(void *);
void enzymexla_set_mpi_wait(void *);
void enzymexla_set_mpi_waitall(void *);
void enzymexla_set_mpi_allreduce(void *);
void enzymexla_set_mpi_bcast(void *);

#if defined(__cplusplus)
}
#endif
