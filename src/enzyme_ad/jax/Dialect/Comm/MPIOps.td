#ifndef ENZYMEXLA_COMM_DIALECT_MPI_OPS_TD
#define ENZYMEXLA_COMM_DIALECT_MPI_OPS_TD

include "Dialect.td"

include "mlir/Interfaces/SideEffectInterfaces.td"

include "Types.td"
include "Attrs.td"

def MPI_PredefinedCommOp : Comm_Op<"mpi.predefined_comm", [Pure]> {
  let summary = "Get the named predefined MPI communicator";
  let arguments = (ins MPI_CommsAttr : $value);
  let results = (outs MPI_Comm : $comm);
  let assemblyFormat = "attr-dict `:` type(results)";
}

def MPI_PredefinedOpOp : Comm_Op<"mpi.predefined_op", [Pure]> {
  let summary = "Get the named predefined MPI reduction operator";
  let arguments = (ins MPI_OpsAttr : $value);
  let results = (outs MPI_Op : $op);
  let assemblyFormat = "attr-dict `:` type(results)";
}

// NOTE not sure if necessary since we take the datatype from the MLIR tensor eltype
// def MPI_PredefinedDatatypeOp : Comm_Op<"mpi.predefined_datatype", [Pure]> {
//   let summary = "Get the named predefined MPI datatype";
//   let arguments = (ins MPI_DatatypesAttr : $value);
//   let results = (outs MPI_Datatype : $datatype);
//   let assemblyFormat = "attr-dict `:` type(results)";
// }

def MPI_CommRankOp : Comm_Op<"mpi.comm_rank", [Pure]> {
  let summary = "Equivalent to `MPI_Comm_rank`";
  let arguments = (ins MPI_Comm : $comm);
  let results = (outs I32 : $rank);
  let assemblyFormat = "$comm attr-dict `:` type(results)";
}

def MPI_CommSizeOp : Comm_Op<"mpi.comm_size", [Pure]> {
  let summary = "Equivalent to `MPI_Comm_size`";
  let arguments = (ins MPI_Comm : $comm);
  let results = (outs I32 : $size);
  let assemblyFormat = "$comm attr-dict `:` type(results)";
}

def MPI_CommSplitOp : Comm_Op<"mpi.comm_split", []> {
  let summary = "Equivalent to `MPI_Comm_split`";

  let arguments = (ins
    MPI_Comm : $comm,
    I32 : $color,
    I32 : $key
  );

  let results = (outs MPI_Comm : $newcomm);
  let assemblyFormat = "operands attr-dict `:` functional-type($comm, results)";
}

def MPI_BarrierOp : Comm_Op<"mpi.barrier", []> {
  let summary = "Equivalent to `MPI_Barrier`";
  let arguments = (ins MPI_Comm : $comm);
  let assemblyFormat = "$comm attr-dict";
}

def MPI_SendOp : Comm_Op<"mpi.send", []> {
  let summary = "Equivalent to `MPI_Send`";

  // TODO add optional `count` operand for dynamic-shaped tensors?
  let arguments = (ins
    AnyTensor : $buf,
    I32 : $tag,
    I32 : $dest,
    MPI_Comm : $comm
  );

  let assemblyFormat = "operands attr-dict `:` type($buf)";
}

def MPI_IsendOp : Comm_Op<"mpi.isend", []> {
  let summary = "Equivalent to `MPI_Isend`";

  // TODO add optional `count` operand for dynamic-shaped tensors?
  let arguments = (ins
    AnyTensor : $buf,
    I32 : $tag,
    I32 : $dest,
    MPI_Comm : $comm
  );

  let results = (outs MPI_Request : $request);
  let assemblyFormat = "operands attr-dict `:` functional-type($buf, results)";
}

def MPI_RecvOp : Comm_Op<"mpi.recv", []> {
  let summary = "Equivalent to `MPI_Recv`";

  // TODO add optional `count` operand for dynamic-shaped tensors?
  let arguments = (ins
    I32 : $tag,
    I32 : $source,
    MPI_Comm : $comm
  );

  let results = (outs AnyTensor : $buf);
  let assemblyFormat = "operands attr-dict `:` type(results)";
}

def MPI_IrecvOp : Comm_Op<"mpi.irecv", []> {
  let summary = "Equivalent to `MPI_Irecv`";

  // TODO add optional `count` operand for dynamic-shaped tensors?
  let arguments = (ins
    I32 : $tag,
    I32 : $source,
    MPI_Comm : $comm
  );
  let results = (outs
    AnyTensor : $buf,
    MPI_Request : $request
  );
  let assemblyFormat = "operands attr-dict `:` type(results)";
}

def MPI_WaitOp : Comm_Op<"mpi.wait", []> {
  let summary = "Equivalent to `MPI_Wait`";
  let arguments = (ins MPI_Request : $request);
  let assemblyFormat = "operands attr-dict `:` type(operands)";
}

def MPI_WaitallOp : Comm_Op<"mpi.waitall", []> {
  let summary = "Equivalent to `MPI_Waitall`";
  let arguments = (ins Variadic<MPI_Request> : $requests);
  let assemblyFormat = "operands attr-dict `:` type(operands)";
}

def MPI_AllreduceOp : Comm_Op<"mpi.allreduce", []> {
  let summary = "Equivalent to `MPI_Allreduce`";

  let arguments = (ins
    AnyTensor : $sendbuf,
    MPI_OpsAttr : $reduceOpAttr,
    MPI_Comm : $comm
  );

  let results = (outs AnyTensor : $recvbuf);
  let assemblyFormat = "$sendbuf `,` $reduceOpAttr `,` $comm attr-dict `:` functional-type($sendbuf, results)";
}

def MPI_BcastOp : Comm_Op<"mpi.bcast", []> {
  let summary = "Equivalent to `MPI_Bcast`";

  let arguments = (ins
    AnyTensor : $inbuf,
    I32 : $root,
    MPI_Comm : $comm
  );

  let results = (outs AnyTensor : $outbuf);
  let assemblyFormat = "operands attr-dict `:` functional-type($inbuf, results)";
}

#endif // ENZYMEXLA_COMM_DIALECT_MPI_OPS_TD
