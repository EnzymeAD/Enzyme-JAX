include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/DialectBase.td"
include "mlir/IR/Traits.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

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
def TokenType : CommType<"Token", "token"> {
  let summary = "Represents a consumable message token. No runtime counterpart";
  let mnemonic = "msg_token";
}

def DeviceType : CommType<"Device", "device"> {
  let summary = "Represents a device declaration, with no runtime counterpart";
  let mnemonic = "device";
}

def ChannelType : CommType<"Channel", "channel"> {
  let summary = "Represents a communication channel, with no runtime counterpart";
  let mnemonic = "channel";
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
    auto getBranches() {
      return getDeclarations().getOps<::mlir::comm::CommBranch>();
    }
  }];
}

def CommBranch : CommOp<"branch", traits = [CommSplitMemberOpTrait, SingleBlock, HasParent<"CommSplit">, NoTerminator]> {
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
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = [{
    attr-dict $device_ids $body
  }];

  let hasVerifier = 1;
}

// Token op
def CommToken: CommOp<"token", traits = [CommSplitMemberOpTrait]> {
  let summary = "A simple single-usage, one-way message token";
  let arguments = (ins 
    TypeAttr:$data_type // TODO make a dummy type for condition-matching tokens, which don't carry datas
  );
  let results = (outs
    TokenType:$token
  );
  let assemblyFormat = [{
    attr-dict $data_type
  }];
  let hasVerifier = 1;
}

// I would really like to have the TokenConsumer interface include a
// utility method that returns the token op, but am running into issues
// with referencing CommToken to troubleshoot later.
def TokenConsumer : OpInterface<"TokenConsumer"> {
  let cppNamespace = "::mlir::comm";
  let methods = [
    // InterfaceMethod<[{
    //   Returns the token that this operation consumes
    // }], "CommToken", "getTokenOp">
    ];
}
// def DefaultTokenConsumer : NativeOpTrait<"DefaultTokenConsumer",
//   /*traits =  */ [TokenConsumer],
//   /*extraOpDeclaration = */[{
//     mlir::comm::CommToken getTokenOp(){
//       return getToken();
//     }
//   }]> {
//   let cppNamespace = "::mlir::comm";
// }


// Primitives
def CommSend: CommOp<"send", traits = [TokenConsumer]> {
  let summary = "An op to fulfill (part of) a messages input.";
  // TODO When we want to make send asynchronous remove the read from the token
  let arguments = (ins Arg<TokenType, "token", [MemWriteAt<1, FullEffect>, MemReadAt<2, FullEffect>]>:$token,
                    Arg<AnyType, "message data", [MemReadAt<0, FullEffect>]>:$data);
  let results = (outs );
  let assemblyFormat = [{
    attr-dict $token $data `:` type($data)
  }];
  let hasVerifier = 1;
}

def CommRecv: CommOp<"recv", traits = [TokenConsumer]> {
  let summary = "An op that blocks and returns the messages output";
  // Need to encode write back to the token because the sender may be waiting.
  let arguments = (ins Arg<TokenType, "token", [MemReadAt<0, FullEffect>, MemWriteAt<2>]>:$token);
  let results = (outs AnyType:$data);
  let assemblyFormat = [{
    attr-dict $token `:` type($data)
  }];
}

// Conditional Communication Op- op that wraps a SCF control flow operation
// and adds a communication token to it
def CommControl : CommOp<"control", traits = [SingleBlock, NoTerminator, TokenConsumer]> {
  let summary = "A wrapper operation to allow communication in control flow.";
  let description = [{
    This operation wraps a control flow operation (e.g. scf.if, scf.while) and adds a communication token to it.
    This associates control flow accross branches, establishing a bisimulation between the branches. Passes and 
    programmers MUST assure that the condition of each control flow on each branch is equivalent, otherwise
    we lose communication consistency. This property is assumed by the compiler on its inputs, but can't
    be checked in general.
  }];
  let arguments = (ins 
    Arg<TokenType, "token", [MemReadAt<0, FullEffect>, MemWriteAt<1>]>:$token
  );
  let results = (outs AnyType:$data); // pass through the values of the control flow operation
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = [{
    attr-dict $token $body `:` type($data)
  }];
  let hasVerifier = 1;
}