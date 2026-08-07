#include <type_traits>

#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"
#include "xla/ffi/ffi_api.h"

#include "mpi.h"

#include "mpi_ffi.h"

ffi::Error MPI_Comm_rank_impl(MPI_Comm comm, ffi::Result<int> rank) {
  int err = MPI_Comm_rank(comm, rank.operator->());
  if (err != MPI_SUCCESS) {
    return ffi::Error::InvalidArgument(
        absl::StrFormat("MPI_Comm_rank failed with error code %d", err));
  }
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER(MPI_Comm_rank_ffi, MPI_Comm_rank_impl,
                       xla::ffi::Ffi::Bind()
                           .Arg<MPI_Comm>() // comm
                           .Ret<int>()      // rank
);

ffi::Error MPI_Comm_size_impl(MPI_Comm comm, ffi::Result<int> size) {
  int err = MPI_Comm_size(comm, size.operator->());
  if (err != MPI_SUCCESS) {
    return ffi::Error::InvalidArgument(
        absl::StrFormat("MPI_Comm_size failed with error code %d", err));
  }
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER(MPI_Comm_size_ffi, MPI_Comm_size_impl,
                       xla::ffi::Ffi::Bind()
                           .Arg<MPI_Comm>() // comm
                           .Ret<int>()      // size
);

ffi::Error MPI_Comm_split_impl(MPI_Comm comm, int color, int key,
                               ffi::Result<MPI_Comm> newcomm) {
  int err = MPI_Comm_split(comm, color, key, newcomm.operator->());
  if (err != MPI_SUCCESS) {
    return ffi::Error::InvalidArgument(
        absl::StrFormat("MPI_Comm_split failed with error code %d", err));
  }
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER(MPI_Comm_split_ffi, MPI_Comm_split_impl,
                       xla::ffi::Ffi::Bind()
                           .Arg<MPI_Comm>() // comm
                           .Arg<int>()      // color
                           .Arg<int>()      // key
                           .Ret<MPI_Comm>() // newcomm
);

ffi::Error MPI_Barrier_impl(MPI_Comm comm) {
  int err = MPI_Barrier(comm);
  if (err != MPI_SUCCESS) {
    return ffi::Error::InvalidArgument(
        absl::StrFormat("MPI_Barrier failed with error code %d", err));
  }
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER(MPI_Barrier_ffi, MPI_Barrier_impl,
                       xla::ffi::Ffi::Bind().Arg<MPI_Comm>());

ffi::Error MPI_Send_impl(AnyBuffer buf, MPI_Datatype datatype, int dest,
                         int tag, MPI_Comm comm) {
  int count = buf.element_count();
  int err = MPI_Send(buf.untyped_data(), count, datatype, dest, tag, comm);
  if (err != MPI_SUCCESS) {
    return ffi::Error::InvalidArgument(
        absl::StrFormat("MPI_Send failed with error code %d", err));
  }
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER(MPI_Send_ffi, MPI_Send_impl,
                       xla::ffi::Ffi::Bind()
                           .Arg<AnyBuffer>()    // buf
                           .Arg<MPI_Datatype>() // datatype
                           .Arg<int>()          // dest
                           .Arg<int>()          // tag
                           .Arg<MPI_Comm>()     // comm
);

ffi::Error MPI_Isend_impl(AnyBuffer buf, MPI_Datatype datatype, int dest,
                          int tag, MPI_Comm comm,
                          ffi::Result<MPI_Request> request) {
  int count = buf.element_count();
  int err = MPI_Isend(buf.untyped_data(), count, datatype, dest, tag, comm,
                      request.operator->());
  if (err != MPI_SUCCESS) {
    return ffi::Error::InvalidArgument(
        absl::StrFormat("MPI_Isend failed with error code %d", err));
  }
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER(MPI_Isend_ffi, MPI_Isend_impl,
                       xla::ffi::Ffi::Bind()
                           .Arg<AnyBuffer>()    // buf
                           .Arg<MPI_Datatype>() // datatype
                           .Arg<int>()          // dest
                           .Arg<int>()          // tag
                           .Arg<MPI_Comm>()     // comm
                           .Ret<MPI_Request>()  // request
);

ffi::Error MPI_Recv_impl(MPI_Datatype datatype, int source, int tag,
                         MPI_Comm comm, ffi::Result<AnyBuffer> buf,
                         ffi::Result<MPI_Status> status) {
  int count = buf->element_count();
  int err = MPI_Recv(buf->untyped_data(), count, datatype, source, tag, comm,
                     status.operator->());
  if (err != MPI_SUCCESS) {
    return ffi::Error::InvalidArgument(
        absl::StrFormat("MPI_Recv failed with error code %d", err));
  }
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER(MPI_Recv_ffi, MPI_Recv_impl,
                       xla::ffi::Ffi::Bind()
                           .Arg<MPI_Datatype>() // datatype
                           .Arg<int>()          // source
                           .Arg<int>()          // tag
                           .Arg<MPI_Comm>()     // comm
                           .Ret<AnyBuffer>()    // buf
                           .Ret<MPI_Status>()   // status
);

ffi::Error MPI_Irecv_impl(AnyBuffer buf, MPI_Datatype datatype, int source,
                          int tag, MPI_Comm comm,
                          ffi::Result<MPI_Request> request) {
  int count = buf.element_count();
  int err = MPI_Irecv(buf.untyped_data(), count, datatype, source, tag, comm,
                      request.operator->());
  if (err != MPI_SUCCESS) {
    return ffi::Error::InvalidArgument(
        absl::StrFormat("MPI_Irecv failed with error code %d", err));
  }
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER(MPI_Irecv_ffi, MPI_Irecv_impl,
                       xla::ffi::Ffi::Bind()
                           .Arg<MPI_Datatype>() // datatype
                           .Arg<int>()          // source
                           .Arg<int>()          // tag
                           .Arg<MPI_Comm>()     // comm
                           .Ret<AnyBuffer>()    // buf
                           .Ret<MPI_Request>()  // request
);

ffi::Error MPI_Wait_impl(MPI_Request request, ffi::Result<MPI_Status> status) {
  int err = MPI_Wait(&request, status.operator->());
  if (err != MPI_SUCCESS) {
    return ffi::Error::InvalidArgument(
        absl::StrFormat("MPI_Wait failed with error code %d", err));
  }
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER(MPI_Wait_ffi, MPI_Wait_impl,
                       xla::ffi::Ffi::Bind()
                           .Arg<MPI_Request>() // request
                           .Ret<MPI_Status>()  // status
);

ffi::Error MPI_Waitall_impl(RemainingArgs requests, RemainingRets statuses) {
  if (requests.size() != statuses.size()) {
    return ffi::Error::InvalidArgument(
        absl::StrFormat("MPI_Waitall: requests and statuses must have the same "
                        "size, but got %d and %d",
                        requests.size(), statuses.size()));
  }
  int count = requests.size();

  std::vector<MPI_Request> request_vector;
  std::vector<MPI_Status> status_vector;
  request_vector.reserve(count);
  status_vector.reserve(count);

  for (int i = 0; i < count; ++i) {
    auto req_or_error = requests.get<MPI_Request>(i);
    if (req_or_error.has_error())
      return req_or_error.error();
    request_vector.push_back(req_or_error.value());
  }

  int err = MPI_Waitall(count, request_vector.data(), status_vector.data());
  if (err != MPI_SUCCESS) {
    return ffi::Error::InvalidArgument(
        absl::StrFormat("MPI_Waitall failed with error code %d", err));
  }

  for (int i = 0; i < count; ++i) {
    auto status_or_error = statuses.get<MPI_Status>(i);
    if (status_or_error.has_error())
      return status_or_error.error();
    *status_or_error.value() = status_vector[i];
  }

  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER(MPI_Waitall_ffi, MPI_Waitall_impl,
                       xla::ffi::Ffi::Bind()
                           .RemainingArgs() // requests
                           .RemainingRets() // statuses
);

ffi::Error MPI_Allreduce_impl(AnyBuffer sendbuf, MPI_Datatype datatype,
                              MPI_Op op, MPI_Comm comm,
                              ffi::Result<AnyBuffer> recvbuf) {
  if (sendbuf.element_count() <= recvbuf->element_count()) {
    return ffi::Error::InvalidArgument(absl::StrFormat(
        "MPI_Allreduce: recvbuf size (%d) must be at least sendbuf size (%d)",
        recvbuf->element_count(), sendbuf.element_count()));
  }
  int count = sendbuf.element_count();
  int err = MPI_Allreduce(sendbuf.untyped_data(), recvbuf.untyped_data(), count,
                          datatype, op, comm);
  if (err != MPI_SUCCESS) {
    return ffi::Error::InvalidArgument(
        absl::StrFormat("MPI_Allreduce failed with error code %d", err));
  }
  return ffi::Error::Success();
}

ffi::Error MPI_Bcast_impl(AnyBuffer buf, MPI_Datatype datatype, int root,
                          MPI_Comm comm) {
  int count = buf.element_count();
  int err = MPI_Bcast(buf.untyped_data(), count, datatype, root, comm);
  if (err != MPI_SUCCESS) {
    return ffi::Error::InvalidArgument(
        absl::StrFormat("MPI_Bcast failed with error code %d", err));
  }
  return ffi::Error::Success();
}

void registerEnzymeJaXXLAMPIFFI() {
  XLA_FFI_REGISTER_HANDLER(xla::ffi::GetXlaFfiApi(),
                           "enzymexla_ffi_mpi_comm_rank", "Host",
                           MPI_Comm_rank_ffi);
  XLA_FFI_REGISTER_HANDLER(xla::ffi::GetXlaFfiApi(),
                           "enzymexla_ffi_mpi_comm_size", "Host",
                           MPI_Comm_size_ffi);
  XLA_FFI_REGISTER_HANDLER(xla::ffi::GetXlaFfiApi(),
                           "enzymexla_ffi_mpi_comm_split", "Host",
                           MPI_Comm_split_ffi);
  XLA_FFI_REGISTER_HANDLER(xla::ffi::GetXlaFfiApi(),
                           "enzymexla_ffi_mpi_barrier", "Host",
                           MPI_Barrier_ffi);
  XLA_FFI_REGISTER_HANDLER(xla::ffi::GetXlaFfiApi(), "enzymexla_ffi_mpi_send",
                           "Host", MPI_Send_ffi);
  XLA_FFI_REGISTER_HANDLER(xla::ffi::GetXlaFfiApi(), "enzymexla_ffi_mpi_isend",
                           "Host", MPI_Isend_ffi);
  XLA_FFI_REGISTER_HANDLER(xla::ffi::GetXlaFfiApi(), "enzymexla_ffi_mpi_recv",
                           "Host", MPI_Recv_ffi);
  XLA_FFI_REGISTER_HANDLER(xla::ffi::GetXlaFfiApi(), "enzymexla_ffi_mpi_irecv",
                           "Host", MPI_Irecv_ffi);
  XLA_FFI_REGISTER_HANDLER(xla::ffi::GetXlaFfiApi(), "enzymexla_ffi_mpi_wait",
                           "Host", MPI_Wait_ffi);
  XLA_FFI_REGISTER_HANDLER(xla::ffi::GetXlaFfiApi(),
                           "enzymexla_ffi_mpi_waitall", "Host",
                           MPI_Waitall_ffi);
  XLA_FFI_REGISTER_HANDLER(xla::ffi::GetXlaFfiApi(), "enzymexla_ffi_mpi_bcast",
                           "Host", MPI_Bcast_ffi);
}
