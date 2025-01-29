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
  let useDefaultTypePrinterParser = 1;
}

// Dialect inheritence shortcuts
class CommOp<string name, list<Trait> traits = []> : Op<CommunicationDialect, name, traits>;
class CommAttr<string name, string mnemomic, list<Trait> traits = []> : AttrDef<CommunicationDialect, name, traits>{
  let mnemonic = mnemomic;
}
class CommType<string name, string type_mnemonic, list<Trait> traits = []> : TypeDef<CommunicationDialect, name, traits> {
  let mnemonic = type_mnemonic;
}

/*
* Dialect Types
*/
def DeviceIdType : CommType<"DeviceId", "device_id"> {
  let summary="Wrapper around int to specify a device from our device set";
  let parameters=(ins "unsigned":$id);
  let assemblyFormat=[{`d` $id }];
}
def MessageTokenType : CommType<"MessageToken", "token"> {
  let summary = "Represents a consumable message token";
  let mnemonic = "msg_token";
}

/*
* Dialect traits and interfaces
*/
def CommSplitMemberOpTrait :  NativeOpTrait<"SplitMemberOp", 
  /*traits=*/[],
  /*extraOpDeclaration = */[{
    mlir::comm::CommSplit getParentSplit();
  }],
  /*extraOpDefinition = */[{
    mlir::comm::CommSplit $cppClass::getParentSplit(){
      // Verifier checks that this is indeed of the correct type
      return dyn_cast<mlir::comm::CommSplit>(getOperation()->getParentOp());
    }
  }]
>{
  let cppNamespace = "::mlir::comm";
}

def CommMessage : OpInterface<"CommMessage"> {
  let cppNamespace = "::mlir::comm";
  let methods = [
    InterfaceMethod<[{
      Returns what type this message takes as inputs
    }], "mlir::Type", "getInputType">,
    InterfaceMethod<[{
      Returns what type will result from recieving this message
    }], "mlir::Type", "getOutputType">,
    InterfaceMethod<[{
      Returns the token handle to this message
    }], "mlir::TypedValue<mlir::comm::MessageTokenType>", "getToken">
  ];
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

def CommSplit : CommOp<"split", traits = [SingleBlock, NoTerminator]> {
  let summary = "The highest level split node in the communication dialect.";
  let description = [{
    Takes in a definition of communication items and a list of split branches for devices to take.
    Encoded as a single-block no-terminator region that consists only of branches and communcation token declarations.
    Example syntax:
      comm.split {
        %1 = comm.simple_msg msg_type
        comm.branch [1, 4] {
          // ... comm branch region
        }
        comm.branch [2] {
          // ... comm branch region
        }
      }
  }];

  let arguments = (ins ); // no inputs yet, encoded in the region
  let regions = (region SizedRegion<1>:$declarations);
  let results = (outs );

  let assemblyFormat = [{
     $declarations attr-dict
  }];

  let hasVerifier = 1;

  // Add some convenience getters to hide the mess around having a declarations region
  let extraClassDeclaration = [{
    auto getMessages() {
      return getDeclarations().getOps<::mlir::comm::CommMessage>();
    }
    auto getBranches() {
      return getDeclarations().getOps<::mlir::comm::CommBranch>();
    }
  }];
}

def CommBranch : CommOp<"branch", traits = [CommSplitMemberOpTrait]> {
  let summary = "Represents one branch that can be taken by a split node";
  let arguments = (ins DenseI32ArrayAttr:$device_ids);
  let regions = (region AnyRegion:$region);
  let assemblyFormat = [{
    attr-dict $device_ids $region
  }];
}

def CommSend: CommOp<"send"> {
  let summary = "An op to fulfill (part of) a messages input.";
  let arguments = (ins MessageTokenType:$token, AnyType:$data);
  let results = (outs );
  let assemblyFormat = [{
    attr-dict $token $data `:` type($data)
  }];
  let extraClassDeclaration = [{
    CommSimpleMessage getMessage();
  }];
  let hasVerifier = 1;
}

def CommRecv: CommOp<"recv"> {
  let summary = "An op that blocks and returns the messages output";
  let arguments = (ins MessageTokenType:$token);
  let results = (outs AnyType:$data);
  let assemblyFormat = [{
    attr-dict $token `:` type($data)
  }];
}

/*
* Different types of message ops
*/
// Base class for messages
class CommMessageBase<string name, list<Trait> extra_traits = []>: CommOp<name, traits = extra_traits # [DeclareOpInterfaceMethods<CommMessage>, CommSplitMemberOpTrait]>;

// Message types. In the future we will likely want to have a common base class
def CommSimpleMessage: CommMessageBase<"simple_msg"> {
  let summary = "A simple single-usage, one-way message token";
  let arguments = (ins 
    TypeAttr:$data_type
  );
  let results = (outs
    MessageTokenType:$token
  );
  let assemblyFormat = [{
    attr-dict $data_type
  }];
}

def CommMultiplexMessage: CommMessageBase<"multiplex_msg"> {
  let summary = "A phi node-like message that allows the compiler to choose from any of the input messages";
  let arguments = (ins TypeAttr:$data_type, Variadic<MessageTokenType>:$in_tokens);
  let results = (outs
    MessageTokenType:$token
  );
  let assemblyFormat = [{
    attr-dict $data_type $in_tokens
  }];
  let hasVerifier = 1;
}