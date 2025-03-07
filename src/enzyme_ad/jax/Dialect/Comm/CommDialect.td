include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/DialectBase.td"
include "mlir/IR/Traits.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"  

def CommDialect : Dialect {
  let name = "comm";
  let summary = "A prototype dialect for various communication ops";
  let description = [{}];
  let cppNamespace = "::mlir::comm";
  let useDefaultTypePrinterParser = 1;
}

// Dialect inheritence shortcuts
class CommOp<string name, list<Trait> traits = []> : Op<CommDialect, name, traits>;

class CommType<string name, string type_mnemonic, list<Trait> traits = []> : TypeDef<CommDialect, name, traits> {
  let mnemonic = type_mnemonic;
}

/*
* Dialect Types
*/
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

    A split block may be marked as a loop, in which case it may have loopy branches:
    branches may consume the same communication token once per loop.
  }];

  let arguments = (ins UnitAttr:$is_loopy);
  let regions = (region SizedRegion<1>:$declarations);
  let results = (outs );
  let assemblyFormat = [{
    (`loop` $is_loopy^)? attr-dict-with-keyword $declarations
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

def CommBranch : CommOp<"branch", traits = [CommSplitMemberOpTrait, SingleBlock, HasParent<"CommSplit">]> {
  let summary = "Represents one branch that can be taken by a split node";
  let description = [{
      Represents a device subset-specific execution path in a split node. A simple (non-loopy) branch
      consists of a single entry region containing a single basic block of code. A loopy branch contains
      the same entry branch, a loop region (analogous to a do-while loop or the "cond" region in a scf.while),
      a reentry region (analogous to the "do" region in a scf.while), and an exit region executed once after the
      loop completes. 
        * loop, reentry, and exit regions are optional and may be empty, but if any are present all must be present.
        * Values are passed between regions using the sfc.yield operation
        * The loop condition is indicated by a scf.condition operation in the loop region
        * Communication tokens are refreshed before starting the reentry region. To be semantically correct, all branches that
          communicate to each other must have loop conditions that keep them in a consistent state. This is not checked (and is
          not decidable in general), but will rather be assumed of the input; the compiler will make sure to maintain correctness.
  }];

  let arguments = (ins DenseI32ArrayAttr:$device_ids);
  let regions = (region SizedRegion<1>:$entry, MaxSizedRegion<1>:$loop, MaxSizedRegion<1>:$reentry, MaxSizedRegion<1>:$exit);
  let assemblyFormat = [{
    attr-dict $device_ids `enter` $entry custom<BranchLoopRegions>($loop, $reentry, $exit)
  }];

  // Extras: a function to quickly determine if the branch is loopy (has any instructions in loop or reentry other than continue)
  let extraClassDeclaration = [{
    /**
      * Returns true if the branch loops. If false all code must be placed in the entry block.
      */
    bool isLoop();
  }];

  // Verifier to check exit block is used only if the branch is loopy
  let hasVerifier = 1;
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
class CommMessageBase<string name, list<Trait> extra_traits = []>: CommOp<name, traits = extra_traits # [DeclareOpInterfaceMethods<CommMessage>, CommSplitMemberOpTrait, HasParent<"CommSplit">]>;

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

// Terminators, modeled almost identically after scf.yield and scf.condition
// (we can't use scf.yield directly because it requires parent ops to be scf ops)
def CommYield: CommOp<"yield", traits = [HasParent<"CommBranch">, Terminator, ReturnLike]> {
  let summary = "Yields a value to the parent branch";
  let arguments = (ins Variadic<AnyType>:$results);
  let results = (outs );
  let assemblyFormat = [{
    attr-dict ($results^ `:` type($results))?
  }];
}

def CommCondition: CommOp<"condition", traits = [HasParent<"CommBranch">, Terminator, ReturnLike]> {
  let summary = "A condition that determines if the loop should continue";
  let arguments = (ins I1:$condition, Variadic<AnyType>:$results);
  let results = (outs );
  let assemblyFormat = [{
    `(` $condition `)` attr-dict ($results^ `:` type($results))?
  }];
}