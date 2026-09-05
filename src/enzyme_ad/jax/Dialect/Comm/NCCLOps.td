#ifndef ENZYMEXLA_COMM_DIALECT_NCCL_OPS_TD
#define ENZYMEXLA_COMM_DIALECT_NCCL_OPS_TD

include "Dialect.td"

include "mlir/Interfaces/SideEffectInterfaces.td"

include "Utils.td"
include "Types.td"
include "Attrs.td"

def NcclCommSplitOp : Comm_Op<"nccl.comm_split", []> {
  let summary = "Equivalent to `ncclCommSplit`";

  let arguments = (ins
    NcclComm : $comm,
    TensorI32 : $color,
    TensorI32 : $key
  );

  let results = (outs NcclComm : $new_comm);
  let assemblyFormat = "operands attr-dict `:` functional-type($comm, results)";
}

def NcclCommFinalizeOp : Comm_Op<"nccl.comm_finalize", []> {
  let summary = "Equivalent to `ncclCommFinalize`";
  let arguments = (ins NcclComm : $comm);
  let assemblyFormat = "$comm attr-dict";
}

def NcclCommDestroyOp : Comm_Op<"nccl.comm_destroy", []> {
  let summary = "Equivalent to `ncclCommDestroy`";
  let arguments = (ins NcclComm : $comm);
  let assemblyFormat = "$comm attr-dict";
}

def NcclCommAbortOp : Comm_Op<"nccl.comm_abort", []> {
  let summary = "Equivalent to `ncclCommAbort`";
  let arguments = (ins NcclComm : $comm);
  let assemblyFormat = "$comm attr-dict";
}

def NcclCommCountOp : Comm_Op<"nccl.comm_count", [Pure]> {
  let summary = "Equivalent to `ncclCommCount`";
  let arguments = (ins NcclComm : $comm);
  let results = (outs TensorI32 : $count);
  let assemblyFormat = "$comm attr-dict `:` type(results)";
}

def NcclCommCuDeviceOp : Comm_Op<"nccl.comm_cudevice", [Pure]> {
  let summary = "Equivalent to `ncclCommCuDevice`";
  let arguments = (ins NcclComm : $comm);
  let results = (outs TensorI32 : $cu_device);
  let assemblyFormat = "$comm attr-dict `:` type(results)";
}

def NcclCommUserRankOp : Comm_Op<"nccl.comm_user_rank", [Pure]> {
  let summary = "Equivalent to `ncclCommUserRank`";
  let arguments = (ins NcclComm : $comm);
  let results = (outs TensorI32 : $user_rank);
  let assemblyFormat = "$comm attr-dict `:` type(results)";
}

def NcclAllReduceOp : Comm_Op<"nccl.all_reduce", []> {
  let summary = "Equivalent to `ncclAllReduce`";

  let arguments = (ins
    AnyTensor : $sendbuf,
    NcclRedOpAttr : $reduceOp,
    NcclComm : $comm
    // TODO $inplace unit attr for reusing sendbuf as recvbuf
  );

  let results = (outs AnyTensor : $recvbuf);
  let assemblyFormat = "$sendbuf `,` $reduceOp `,` $comm attr-dict `:` functional-type($sendbuf, $recvbuf)";
}

def NcclBroadcastOp : Comm_Op<"nccl.broadcast", []> {
  let summary = "Equivalent to `ncclBroadcast`";

  let arguments = (ins
    AnyTensor : $sendbuff,
    TensorI32 : $root,
    NcclComm : $comm
    // TODO $inplace unit attr for reusing sendbuf as recvbuf
  );

  let results = (outs AnyTensor : $recvbuff);
  let assemblyFormat = "operands attr-dict `:` functional-type($sendbuff, $recvbuff)";
}

def NcclSendOp : Comm_Op<"nccl.send", []> {
  let summary = "Equivalent to `ncclSend`";

  let arguments = (ins
    AnyTensor : $sendbuff,
    TensorI32 : $peer,
    NcclComm : $comm
  );

  let assemblyFormat = "operands attr-dict `:` type($sendbuff)";
}

def NcclRecvOp : Comm_Op<"nccl.recv", []> {
  let summary = "Equivalent to `ncclRecv`";

  let arguments = (ins
    TensorI32 : $peer,
    NcclComm : $comm
  );
  let results = (outs AnyTensor : $recvbuff);

  let assemblyFormat = "operands attr-dict `:` type($recvbuff)";
}

#endif // ENZYMEXLA_COMM_DIALECT_NCCL_OPS_TD
