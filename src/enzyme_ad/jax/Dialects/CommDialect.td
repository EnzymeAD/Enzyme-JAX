include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/DialectBase.td"

def CommunicationDialect : Dialect {
  let name = "comm";
  let summary = "A prototype dialect for various communication ops";
  let description = [{}];
  let cppNamespace = "::mlir::comm";
}

class CommOp<string name, list<Trait> traits = []> : Op<CommunicationDialect, name, traits>;

def CommFoo : CommOp<"Foo"> {
  let summary = "do-nothing test op";
  let arguments = (ins );
  let results = (outs );
  let assemblyFormat = [{
    attr-dict
  }];
}
