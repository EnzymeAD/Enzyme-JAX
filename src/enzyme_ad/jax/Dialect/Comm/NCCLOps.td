#ifndef ENZYMEXLA_COMM_DIALECT_NCCL_OPS_TD
#define ENZYMEXLA_COMM_DIALECT_NCCL_OPS_TD

include "Dialect.td"

include "mlir/Interfaces/SideEffectInterfaces.td"

include "Types.td"
include "Attrs.td"

def NCCL_CommSplitOp : Comm_Op<"nccl.comm_split", []> {
  let summary = "Equivalent to `ncclCommSplit`";

  let arguments = (ins
    NCCL_Comm : $comm,
    I32 : $color,
    I32 : $key
  );

  let results = (outs NCCL_Comm : $new_comm);
  let assemblyFormat = "operands attr-dict `:` functional-type($comm, results)";
}

def NCCL_CommFinalizeOp : Comm_Op<"nccl.comm_finalize", []> {
  let summary = "Equivalent to `ncclCommFinalize`";
  let arguments = (ins NCCL_Comm : $comm);
  let assemblyFormat = "$comm attr-dict";
}

def NCCL_CommDestroyOp : Comm_Op<"nccl.comm_destroy", []> {
  let summary = "Equivalent to `ncclCommDestroy`";
  let arguments = (ins NCCL_Comm : $comm);
  let assemblyFormat = "$comm attr-dict";
}

def NCCL_CommAbortOp : Comm_Op<"nccl.comm_abort", []> {
  let summary = "Equivalent to `ncclCommAbort`";
  let arguments = (ins NCCL_Comm : $comm);
  let assemblyFormat = "$comm attr-dict";
}

def NCCL_CommCountOp : Comm_Op<"nccl.comm_count", [Pure]> {
  let summary = "Equivalent to `ncclCommCount`";
  let arguments = (ins NCCL_Comm : $comm);
  let results = (outs I32 : $count);
  let assemblyFormat = "$comm attr-dict `:` type(results)";
}

def NCCL_CommCuDeviceOp : Comm_Op<"nccl.comm_cudevice", [Pure]> {
  let summary = "Equivalent to `ncclCommCuDevice`";
  let arguments = (ins NCCL_Comm : $comm);
  let results = (outs I32 : $cu_device);
  let assemblyFormat = "$comm attr-dict `:` type(results)";
}

def NCCL_CommUserRankOp : Comm_Op<"nccl.comm_user_rank", [Pure]> {
  let summary = "Equivalent to `ncclCommUserRank`";
  let arguments = (ins NCCL_Comm : $comm);
  let results = (outs I32 : $user_rank);
  let assemblyFormat = "$comm attr-dict `:` type(results)";
}

def NCCL_AllReduceOp : Comm_Op<"nccl.all_reduce", []> {
  let summary = "Equivalent to `ncclAllReduce`";

  let arguments = (ins
    AnyTensor : $sendbuf,
    NCCL_RedOpAttr : $reduceOpAttr,
    NCCL_Comm : $comm,
    NCCL_Stream : $stream
    // TODO $inplace unit attr for reusing sendbuf as recvbuf
  );

  let results = (outs AnyTensor : $recvbuf);
  let assemblyFormat = "$sendbuf `,` $reduceOpAttr `,` $comm `,` $stream attr-dict `:` functional-type($sendbuf, $recvbuf)";
}

def NCCL_BroadcastOp : Comm_Op<"nccl.bcast", []> {
  let summary = "Equivalent to `ncclBroadcast`";

  let arguments = (ins
    AnyTensor : $sendbuff,
    I32 : $root,
    NCCL_Comm : $comm,
    NCCL_Stream : $stream
    // TODO $inplace unit attr for reusing sendbuf as recvbuf
  );

  let results = (outs AnyTensor : $recvbuff);
  let assemblyFormat = "operands attr-dict `:` functional-type($sendbuff, $recvbuff)";
}

def NCCL_SendOp : Comm_Op<"nccl.send", []> {
  let summary = "Equivalent to `ncclSend`";

  let arguments = (ins
    AnyTensor : $sendbuff,
    I32 : $peer,
    NCCL_Comm : $comm,
    NCCL_Stream : $stream
  );

  let assemblyFormat = "operands attr-dict `:` type($sendbuff)";
}

def NCCL_RecvOp : Comm_Op<"nccl.recv", []> {
  let summary = "Equivalent to `ncclRecv`";

  let arguments = (ins
    I32 : $peer,
    NCCL_Comm : $comm,
    NCCL_Stream : $stream
  );
  let results = (outs AnyTensor : $recvbuff);

  let assemblyFormat = "operands attr-dict `:` type($recvbuff)";
}

#endif // ENZYMEXLA_COMM_DIALECT_NCCL_OPS_TD
