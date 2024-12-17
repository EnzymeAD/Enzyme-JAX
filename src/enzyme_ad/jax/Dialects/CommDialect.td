include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/DialectBase.td"
include "mlir/IR/Traits.td"

def CommunicationDialect : Dialect {
  let name = "comm";
  let summary = "A prototype dialect for various communication ops";
  let description = [{}];
  let cppNamespace = "::mlir::comm";
}

class CommOp<string name, list<Trait> traits = []> : Op<CommunicationDialect, name, traits>;

def CommFoo : CommOp<"foo"> {
  let summary = "do-nothing test op";
  let arguments = (ins );
  let results = (outs );
  let assemblyFormat = [{
    attr-dict
  }];
}

def CommJoin : CommOp<"join", traits = [Terminator]> {
  let summary = "Denotes the end of a split block, similar to ret for a function";
  let arguments = (ins );
  let results = (outs );
  let assemblyFormat = [{
    attr-dict
  }];
}

def CommSplitBranch : CommOp<"split_branch"> {
  let summary = "One of the branches taken by different devices in a split op.";
  let description = [{
    Inside of a split op, this branch will execute on the provided static list of devices.
    The code inside of this branch will have access to any communication items declared in
    its parent split block, or anything in an outside scope.
  }];

  // Takes in list of participating devices (TODO- currently list of int literals), code region as attributes
  let arguments = (ins 
    I32Attr:$device_list
  );
  let regions = (region 
    AnyRegion:$branch_code // TODO constraint on region?
  );
  
  let results = (outs );   // nothing?
  let assemblyFormat = [{
    attr-dict  `<` $device_list `>` $branch_code
  }];
}

// def CommSplit : CommOp<"Split"> {
//   let summary = "The highest level split node in the communication dialect."
// }