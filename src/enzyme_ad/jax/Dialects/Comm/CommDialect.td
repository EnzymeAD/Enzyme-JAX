include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/DialectBase.td"
include "mlir/IR/Traits.td"

def CommunicationDialect : Dialect {
  let name = "comm";
  let summary = "A prototype dialect for various communication ops";
  let description = [{}];
  let cppNamespace = "::mlir::comm";
}

// Dialect inheritence shortcuts
class CommOp<string name, list<Trait> traits = []> : Op<CommunicationDialect, name, traits>;
class CommAttr<string name, string mnemomic, list<Trait> traits = []> : AttrDef<CommunicationDialect, name, traits>{
  let mnemonic = mnemomic;
}
// class CommType<string name, string type_mnemonic, list<Trait> traits = []> : TypeDef<CommunicationDialect, name, traits> {
//   let mnemonic = type_mnemonic;
// }

/*
* Dialect Types
*/
// def DeviceIdType : CommType<"DeviceId", "device_id"> {
//   let summary="Wrapper around int to specify a device from our device set";
//   let parameters=(ins "unsigned":$id);
//   let assemblyFormat=[{`d` $id }];
// }

/*
* Dialect Attributes
*/

// def CommDeviceIdAttr : CommAttr<"DeviceIdAttr", "device_id_attr">{
//   let parameters=(ins DeviceIdType:$id);
//   let assemblyFormat = [{ $id }];
// }

def CommSplitBranchDescriptor : CommAttr<"SplitBranchDescriptor", "branch_descriptor"> {
  let parameters= (ins ArrayRefParameter<"unsigned">:$device_ids);
  let assemblyFormat = [{ `{` custom<SplitBranchDescriptor>($device_ids) `}` }];
  // let assemblyFormat = [{ $device_ids }];
}

/*
* Dialect Ops
*/

def CommFoo : CommOp<"foo"> {
  let summary = "do-nothing test op";
  let arguments = (ins );
  let results = (outs );
  let assemblyFormat = [{
    attr-dict
  }];
}

// Return, for end of split blocks. We may just be able to use return- lets see if there's any special
// semantics we want join to have
def CommJoin : CommOp<"join", traits = [Terminator]> {
  let summary = "Denotes the end of a split block, similar to ret for a function";
  let arguments = (ins );
  let results = (outs );
  let assemblyFormat = [{
    attr-dict
  }];
}

def CommSplit : CommOp<"split"> {
  let summary = "The highest level split node in the communication dialect.";
  let description = [{
    Takes in a definition of communication items and a list of split branches for devices to take.
  }];

  let arguments = (ins
    ArrayAttr:$branch_metadata // array of attributes of type CommSplitBranchDescriptor. Parallel array with $branches.
  );
  let regions = (region 
    VariadicRegion<AnyRegion>:$branches // regions for each branch. Parallel array with $branch_metadata
  );  
  let results = (outs );

  let assemblyFormat = [{
    attr-dict `{` custom<SplitBranch>($branch_metadata, $branches) `}`
  }];
}