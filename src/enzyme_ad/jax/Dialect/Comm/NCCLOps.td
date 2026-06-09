#ifndef ENZYMEXLA_COMM_DIALECT_NCCL_OPS_TD
#define ENZYMEXLA_COMM_DIALECT_NCCL_OPS_TD

include "Dialect.td"

include "mlir/Interfaces/SideEffectInterfaces.td"

include "Types.td"
include "NCCLAttrs.td"

def NCCL_CommSplitOp : Comm_Op<"nccl.comm_split", []> {
  let summary = "Equivalent to `NCCL_CommSplit`";

  let arguments = (ins
    NCCL_Comm : $comm,
    I32 : $color,
    I32 : $key
  );

  let results = (outs NCCL_Comm : $new_comm);
  let assemblyFormat = "operands attr-dict `:` functional-type($comm, results)";
}

def NCCL_CommFinalizeOp : Comm_Op<"nccl.comm_finalize", []> {
  let summary = "Equivalent to `NCCL_CommFinalize`";
  let arguments = (ins NCCL_Comm : $comm);
  let assemblyFormat = "$comm attr-dict";
}

def NCCL_CommDestroyOp : Comm_Op<"nccl.comm_destroy", []> {
  let summary = "Equivalent to `NCCL_CommDestroy`";
  let arguments = (ins NCCL_Comm : $comm);
  let assemblyFormat = "$comm attr-dict";
}

def NCCL_CommAbortOp : Comm_Op<"nccl.comm_abort", []> {
  let summary = "Equivalent to `NCCL_CommAbort`";
  let arguments = (ins NCCL_Comm : $comm);
  let assemblyFormat = "$comm attr-dict";
}

def NCCL_CommCountOp : Comm_Op<"nccl.comm_count", [Pure]> {
  let summary = "Equivalent to `NCCL_CommCount`";
  let arguments = (ins NCCL_Comm : $comm);
  let results = (outs I32 : $count);
  let assemblyFormat = "$comm attr-dict `:` type(results)";
}

def NCCL_CommCuDeviceOp : Comm_Op<"nccl.comm_cudevice", [Pure]> {
  let summary = "Equivalent to `NCCL_CommCuDevice`";
  let arguments = (ins NCCL_Comm : $comm);
  let results = (outs I32 : $cu_device);
  let assemblyFormat = "$comm attr-dict `:` type(results)";
}

def NCCL_CommUserRankOp : Comm_Op<"nccl.comm_user_rank", [Pure]> {
  let summary = "Equivalent to `NCCL_CommUserRank`";
  let arguments = (ins NCCL_Comm : $comm);
  let results = (outs I32 : $user_rank);
  let assemblyFormat = "$comm attr-dict `:` type(results)";
}

def NCCL_AllreduceOp : Comm_Op<"nccl.allreduce", []> {
  let summary = "Equivalent to `NCCL_Allreduce`";

  let arguments = (ins
    AnyTensor : $sendbuf,
    NCCL_RedOpAttr : $reduceOpAttr,
    NCCL_Comm : $comm,
    NCCL_Stream : $stream
    // TODO $inplace unit attr for reusing sendbuf as recvbuf
  );

  let results = (outs AnyTensor : $recvbuf);
  let assemblyFormat = "$sendbuf `,` $reduceOpAttr `,` $comm attr-dict `:` functional-type($sendbuf, $recvbuf)";
}

def NCCL_BroadcastOp : Comm_Op<"nccl.bcast", []> {
  let summary = "Equivalent to `NCCL_Bcast`";

  let arguments = (ins
    AnyTensor : $sendbuff,
    I32 : $root,
    NCCL_Comm : $comm,
    NCCL_Stream : $stream
    // TODO $inplace unit attr for reusing sendbuf as recvbuf
  );

  let results = (outs AnyTensor : $recvbuff);
  let assemblyFormat = "operands attr-dict `:` functional-type($sendbuff, results)";
}

def NCCL_SendOp : Comm_Op<"nccl.send", []> {
  let summary = "Equivalent to `NCCL_Send`";

  let arguments = (ins
    AnyTensor : $sendbuff,
    I32 : $peer,
    NCCL_Comm : $comm,
    NCCL_Stream : $stream
  );

  let assemblyFormat = "operands attr-dict `:` type(operands)";
}

def NCCL_RecvOp : Comm_Op<"nccl.recv", []> {
  let summary = "Equivalent to `NCCL_Recv`";

  let arguments = (ins
    I32 : $peer,
    NCCL_Comm : $comm,
    NCCL_Stream : $stream
  );
  let results = (outs AnyTensor : $recvbuff);

  let assemblyFormat = "operands attr-dict `:` type(results)";
}

#endif // ENZYMEXLA_COMM_DIALECT_NCCL_OPS_TD
