include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.td"
include "mlir/Dialect/Transform/IR/TransformDialect.td"

class RaisingPatternOp<string mnemonic, list<Trait> traits = []>
    : Op<Transform_Dialect,
         "apply_patterns.raising." # mnemonic,
          // For some reason, inherited methods are not getting declared...
         !listconcat(
            [DeclareOpInterfaceMethods<PatternDescriptorOpInterface>],
            traits)> {
  let arguments = (ins OptionalAttr<I64Attr>:$benefit);
  list<string> patterns = [];
  let assemblyFormat = "attr-dict";
}

def ApplyRemoveIVsPatterns : RaisingPatternOp<
    "remove_ivs"> {
  let patterns = ["RemoveIVs"];
}

def ApplyNormalizeLoopPatterns : RaisingPatternOp<
    "normalize_loop"> {
  let patterns = ["NormalizeLoop"];
}
