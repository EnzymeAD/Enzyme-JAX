include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.td"
include "mlir/Dialect/Transform/IR/TransformDialect.td"

def SearchablePatternDescriptorOpInterface :
    OpInterface<"SearchablePatternDescriptorOpInterface",
                [PatternDescriptorOpInterface]> {
  let methods = [
    StaticInterfaceMethod<
      [{Generates possible attribute combinations for opaque op construction.}],
      "::llvm::SmallVector<::mlir::DictionaryAttr>",
      "getPossibleAttrCombinations",
      (ins "::mlir::Builder &":$builder),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        return {builder.getDictionaryAttr({})};
      }]>,
  ];
}

class EnzymeHLOPatternOp<string mnemonic, list<Trait> traits = []> 
    : Op<Transform_Dialect,
         "apply_patterns.enzyme_hlo." # mnemonic,
          // For some reason, inherited methods are not getting declared...
         !listconcat(
            [DeclareOpInterfaceMethods<SearchablePatternDescriptorOpInterface>,
             DeclareOpInterfaceMethods<PatternDescriptorOpInterface>],
            traits)> {
  list<string> patterns = [];
  let assemblyFormat = "attr-dict";
}

class EnzymeHLOParameterizedPatternOp<string mnemonic, list<Trait> traits = []> 
    : Op<Transform_Dialect,
         "apply_patterns.enzyme_hlo." # mnemonic,
          // For some reason, inherited methods are not getting declared...
         !listconcat(
            [SearchablePatternDescriptorOpInterface,
             DeclareOpInterfaceMethods<PatternDescriptorOpInterface>],
            traits)> {
}

def ApplyTransposeConcatReshapePatterns : EnzymeHLOPatternOp<
    "transpose_concat_reshape"> {
  let patterns = [
    "TransposeDotReorder",
    "DotTranspose",
    "ConvertConvertFloat",
    "ConcatToPad",
    "ConcatAppendingReshape",
    "ReshapeIota",
    "ReshapePad",
  ];
}

def ApplyFullReduceReshapeOrTransposePatterns : EnzymeHLOPatternOp<
    "full_reduce_reshape_or_transpose"> {
  let patterns = [
    "FullReduceReshapeOrTranspose",
  ];
}

def ApplySliceTransposeOrBroadcastPatterns : EnzymeHLOPatternOp<
    "slice_transpose_or_broadcast"> {
  let patterns = [
    "SliceTranspose",
    "SliceBroadcast",
  ];
}

def ApplyReducePadPatterns : EnzymeHLOPatternOp<
    "reduce_pad"> {
  let patterns = [
    "ReducePad",
  ];
}

def ApplyZeroPadPatterns : EnzymeHLOPatternOp<
    "zero_pad"> {
  let patterns = [
    "MulZeroPad",
    "DivZeroPad",
  ];
}

def ApplyBinopConstPadPatterns : EnzymeHLOPatternOp<
    "binop_const_pad"> {
  let patterns = [
    "BinopConstPad<stablehlo::AddOp>",
    "BinopConstPad<stablehlo::SubtractOp>",
    "BinopConstPad<stablehlo::MulOp>",
    "BinopConstPad<stablehlo::DivOp>",
  ];
}

def ApplyBinopBinopPadPatterns : EnzymeHLOPatternOp<
    "binop_binop_pad"> {
  let patterns = [
    "BinopBinopPadPad<stablehlo::AddOp>",
    "BinopBinopPadPad<stablehlo::MulOp>",
  ];
}

def ApplyUnaryPushPadPatterns : EnzymeHLOPatternOp<
    "unary_push_pad"> {
  let patterns = [
    "UnaryPadPush<stablehlo::ConvertOp>",
    "UnaryPadPush<stablehlo::TanhOp>",
    "UnaryPadPush<stablehlo::ExpOp>",
  ];
}

def ApplyTransposePadPatterns : EnzymeHLOPatternOp<
    "transpose_pad"> {
  let patterns = [
    "TransposePad",
  ];
}

def ApplyTransposeConvertPatterns : EnzymeHLOPatternOp<
    "transpose_convert"> {
  let patterns = [
    "TransposeConvert",
  ];  
}

def ApplyTransposeTransposePatterns : EnzymeHLOPatternOp<
    "transpose_transpose"> {
  let patterns = [
    "TransposeTranspose",
  ];
}

def ApplyBroadcastReducePatterns : EnzymeHLOPatternOp<
    "broadcast_reduce"> {
  let patterns = [
    "BroadcastReduce",
  ];
}

def ApplyPadDotGeneralPatterns : EnzymeHLOParameterizedPatternOp<
    "pad_dot_general"> {
  let arguments = (ins BoolAttr:$postPad);
  let assemblyFormat = "`postPad` `=` $postPad attr-dict";
  // TODO: the following can be automated by tablegen or some sort of
  // lighter-weight introspection of searchable attributes.
  let extraClassDeclaration = [{
    ::llvm::SmallVector<::mlir::DictionaryAttr>
    static getPossibleAttrCombinations(::mlir::Builder &builder) {
      return {builder.getDictionaryAttr(
                  builder.getNamedAttr("postPad", builder.getBoolAttr(true))),
              builder.getDictionaryAttr(
                  builder.getNamedAttr("postPad", builder.getBoolAttr(false)))};
    }
  }];
}
