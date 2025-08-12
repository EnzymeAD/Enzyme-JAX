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
  let arguments = (ins OptionalAttr<I64Attr>:$benefit);
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

// benefit 65k
def ApplyAddSimplifyPatterns : EnzymeHLOPatternOp<
    "add_simplify"> {
  let patterns = ["AddSimplify"];
}
def ApplyReplaceNegAddWithSubtract: EnzymeHLOPatternOp<
    "replace_neg_add_with_subtract"> {
  let patterns = ["ReplaceNegAddWithSubtract"];
}
def ApplySubSimplifyPatterns : EnzymeHLOPatternOp<
    "sub_simplify"> {
  let patterns = ["SubSimplify"];
}
def ApplyAndSimplifyPatterns : EnzymeHLOPatternOp<
    "and_simplify"> {
  let patterns = ["AndSimplify"];
}
def ApplyMaxSimplifyPatterns : EnzymeHLOPatternOp<
    "max_simplify"> {
  let patterns = ["MaxSimplify"];
}
def ApplyMinSimplifyPatterns : EnzymeHLOPatternOp<
    "min_simplify"> {
  let patterns = ["MinSimplify"];
}
def ApplyOrSimplifyPatterns : EnzymeHLOPatternOp<
    "or_simplify"> {
  let patterns = ["OrSimplify"];
}
def ApplyMulSimplifyPatterns : EnzymeHLOPatternOp<
    "mul_simplify"> {
  let patterns = ["MulSimplify"];
}
def ApplyDivSimplifyPatterns : EnzymeHLOPatternOp<
    "div_simplify"> {
  let patterns = ["DivSimplify"];
}
def ApplyPowSimplifyPatterns : EnzymeHLOPatternOp<
    "pow_simplify"> {
  let patterns = ["PowSimplify"];
}
def ApplyNoopSlicePatterns : EnzymeHLOPatternOp<
    "noop_slice"> {
  let patterns = ["NoopSlice"];
}
def ApplyNoopReversePatterns : EnzymeHLOPatternOp<
    "noop_reverse"> {
  let patterns = ["NoopReverse"];
}
def ApplySliceSlicePatterns : EnzymeHLOPatternOp<
    "slice_slice"> {
  let patterns = ["SliceSlice"];
}
def ApplyPadSimplifyPatterns : EnzymeHLOParameterizedPatternOp<
    "pad_simplify"> {
  let arguments = (ins OptionalAttr<I64Attr>:$benefit, I64Attr:$parameter);
  let assemblyFormat = "attr-dict";
  // TODO: this should be made better searchable.
  let extraClassDeclaration = [{
    ::llvm::SmallVector<::mlir::DictionaryAttr>
    static getPossibleAttrCombinations(::mlir::Builder &builder) {
      return {builder.getDictionaryAttr(
                  builder.getNamedAttr("parameter",
                                       builder.getI64IntegerAttr(1024)))};
    }
  }];
}
def ApplyNegativePadToSlicePatterns : EnzymeHLOPatternOp<
    "negative_pad_to_slice"> {
  let patterns = ["NegativePadToSlice"];
}
def ApplySliceSimplifyPatterns : EnzymeHLOPatternOp<
    "slice_simplify"> {
  let patterns = ["SliceSimplify"];
}
def ApplyConvertSimplifyPatterns : EnzymeHLOPatternOp<
    "convert_simplify"> {
  let patterns = ["ConvertSimplify"];
}
def ApplyDynamicSliceToStaticPatterns : EnzymeHLOPatternOp<
    "dynamic_slice_to_static"> {
  let patterns = ["DynamicSliceToStatic"];
}
def ApplyDynamicUpdateSliceElimPatterns : EnzymeHLOPatternOp<
    "dynamic_update_slice_elim"> {
  let patterns = ["DynamicUpdateSliceElim"];
}
def ApplyReduceToReshapePatterns : EnzymeHLOPatternOp<
    "reduce_to_reshape"> {
  let patterns = ["ReduceToReshape"];
}
def ApplyBroadcastToReshapePatterns : EnzymeHLOPatternOp<
    "broadcast_to_reshape"> {
  let patterns = ["BroadcastToReshape"];
}
def ApplyNotConstProp : EnzymeHLOPatternOp<
    "not_const_prop"> {
  let patterns = ["UnaryConstProp<stablehlo::NotOp,stablehlo::notOp>"];
}
def ApplyIsFiniteConstProp : EnzymeHLOPatternOp<
    "is_finite_const_prop"> {
  let patterns = ["UnaryConstProp<stablehlo::IsFiniteOp,stablehlo::isFiniteOp>"];
}
def ApplyLogConstProp : EnzymeHLOPatternOp<
    "log_const_prop">{
  let patterns = ["UnaryConstProp<stablehlo::LogOp,stablehlo::logOp>"];
}
def ApplyLogPlusConstProp : EnzymeHLOPatternOp<
    "log_plus_one_const_prop">{
  let patterns = ["UnaryConstProp<stablehlo::Log1pOp,stablehlo::log1pOp>"];
}
def ApplyNegConstProp : EnzymeHLOPatternOp<
    "neg_const_prop"> {
  let patterns = ["UnaryConstProp<stablehlo::NegOp,stablehlo::negOp>"];
}
def ApplySqrtConstProp : EnzymeHLOPatternOp<
    "sqrt_const_prop"> {
  let patterns = ["UnaryConstProp<stablehlo::SqrtOp,stablehlo::sqrtOp>"];
}
def ApplyRsqrtConstProp : EnzymeHLOPatternOp<
    "rsqrt_const_prop"> {
  let patterns = ["UnaryConstProp<stablehlo::RsqrtOp,stablehlo::rsqrtOp>"];
}
def ApplyCosConstProp : EnzymeHLOPatternOp<
    "cos_const_prop"> {
  let patterns = ["UnaryConstProp<stablehlo::CosineOp,stablehlo::cosineOp>"];
}
def ApplySinConstProp : EnzymeHLOPatternOp<
    "sin_const_prop"> {
  let patterns = ["UnaryConstProp<stablehlo::SineOp,stablehlo::sineOp>"];
}
def ApplyExpConstProp : EnzymeHLOPatternOp<
    "exp_const_prop"> {
  let patterns = ["UnaryConstProp<stablehlo::ExpOp,stablehlo::exponentialOp>"];
}
def ApplyExpm1ConstProp : EnzymeHLOPatternOp<
    "expm1_const_prop"> {
  let patterns = ["UnaryConstProp<stablehlo::Expm1Op,stablehlo::expm1Op>"];
}
def ApplyTanhConstProp : EnzymeHLOPatternOp<
    "tanh_const_prop"> {
  let patterns = ["UnaryConstProp<stablehlo::TanhOp,stablehlo::tanhOp>"];
}
def AbsConstPropPatterns : EnzymeHLOPatternOp<
    "abs_const_prop"> {
  let patterns = ["UnaryConstProp<stablehlo::AbsOp,stablehlo::absOp>"];
}
def ApplyLogisticConstProp : EnzymeHLOPatternOp<
    "logistic_const_prop"> {
  let patterns = ["UnaryConstProp<stablehlo::LogisticOp,stablehlo::logisticOp>"];
}
def ApplyConjConstProp : EnzymeHLOPatternOp<
    "conj_const_prop"> {
  let patterns = ["UnaryConstProp<chlo::ConjOp,conjOp>"];
}
def ApplyChloInfConstProp : EnzymeHLOPatternOp<
    "chlo_inf_const_prop">{
  let patterns = ["ChloInfConstProp"];
}
def ApplyGammaConstProp : EnzymeHLOPatternOp<
    "gamma_const_prop">{
  let patterns = ["GammaConstProp"];
}
def ApplyCeilConstProp : EnzymeHLOPatternOp<
    "ceil_const_prop">{
  let patterns = ["UnaryConstProp<stablehlo::CeilOp,stablehlo::ceilOp>"];
}
def ApplyCbrtConstProp : EnzymeHLOPatternOp<
    "cbrt_const_prop">{
  let patterns = ["UnaryConstProp<stablehlo::CbrtOp,stablehlo::cbrtOp>"];
}
def ApplyRealConstProp : EnzymeHLOPatternOp<
    "real_const_prop">{
  let patterns = ["UnaryConstProp<stablehlo::RealOp,stablehlo::realOp>"];
}
def ApplyImagConstProp : EnzymeHLOPatternOp<
    "imag_const_prop">{
  let patterns = ["UnaryConstProp<stablehlo::ImagOp,stablehlo::imagOp>"];
}
def ApplyRoundConstProp : EnzymeHLOPatternOp<
    "round_const_prop">{
  let patterns = ["UnaryConstProp<stablehlo::RoundOp,stablehlo::roundOp>"];
}
def ApplyRoundNearestEvenConstProp : EnzymeHLOPatternOp<
    "round_nearest_even_const_prop">{
  let patterns = [
    "UnaryConstProp<stablehlo::RoundNearestEvenOp,stablehlo::roundNearestEvenOp>"
  ];
}
def ApplySignConstProp : EnzymeHLOPatternOp<
    "sign_const_prop">{
  let patterns = ["UnaryConstProp<stablehlo::SignOp,stablehlo::signOp>"];
}
def ApplyFloorConstProp : EnzymeHLOPatternOp<
    "floor_const_prop">{
  let patterns = ["UnaryConstProp<stablehlo::FloorOp,stablehlo::floorOp>"];
}
def ApplyTanConstProp : EnzymeHLOPatternOp<
    "tan_const_prop">{
  let patterns = ["UnaryConstProp<stablehlo::TanOp,stablehlo::tanOp>"];
}
def SignAbsSimplifyPatterns : EnzymeHLOPatternOp<
    "sign_abs_simplify"> {
  let patterns = ["SignAbsSimplify"];
}
def AbsPositiveSimplifyPatterns : EnzymeHLOPatternOp<
    "abs_positive_simplify"> {
  let patterns = ["AbsPositiveSimplify"];
}

def ApplySquareAbsSimplifyPatterns : EnzymeHLOPatternOp<
    "square_abs_simplify"> {
  let patterns = ["SquareAbsSimplify"];
}

def ApplyDivideDivideSimplifyPatterns : EnzymeHLOPatternOp<
    "divide_divide_simplify"> {
  let patterns = ["DivideDivideSimplify"];
}

// regular benefit
def ApplyConvertConcatPatterns : EnzymeHLOPatternOp<
    "convert_concat"> {
  let patterns = ["ConvertConcat"];
}
def ApplyDynamicUpdateToConcatPatterns : EnzymeHLOPatternOp<
    "dynamic_update_to_concat"> {
  let patterns = ["DynamicUpdateToConcat"];
}
def ApplySliceOfDynamicUpdatePatterns : EnzymeHLOPatternOp<
    "slice_of_dynamic_update"> {
  let patterns = ["SliceOfDynamicUpdate"];
}
def ApplySliceElementwisePatterns : EnzymeHLOPatternOp<
    "slice_elementwise"> {
  let patterns = ["SliceElementwise"];
}
def ApplySlicePadPatterns : EnzymeHLOPatternOp<
    "slice_pad"> {
  let patterns = ["SlicePad"];
}
def ApplyPadReduceWindowPatterns : EnzymeHLOPatternOp<
    "pad_reduce_window"> {
  let patterns = ["PadReduceWindow"];
}
def ApplyConvolutionPadPatterns : EnzymeHLOPatternOp<
    "convolution_pad"> {
  let patterns = ["ConvolutionPad"];
}
def ReshapeEmptyBroadcastPatterns : EnzymeHLOPatternOp<
    "reshape_empty_broadcast"> {
  let patterns = ["ReshapeEmptyBroadcast"];
}
def ApplySliceReshapePadPatterns : EnzymeHLOPatternOp<
    "slice_reshape_pad"> {
  let patterns = ["SliceReshapePad"];
}
def ApplyDotReshapeDotPatterns : EnzymeHLOPatternOp<
    "dot_reshape_dot"> {
  let patterns = ["DotReshapeDot"];
}
def ApplyConcatConstPropPatterns : EnzymeHLOParameterizedPatternOp<
    "concat_const_prop"> {
  let arguments = (ins OptionalAttr<I64Attr>:$benefit, I64Attr:$parameter);
  let assemblyFormat = "attr-dict";
  // TODO: this should be made better searchable.
  let extraClassDeclaration = [{
    ::llvm::SmallVector<::mlir::DictionaryAttr>
    static getPossibleAttrCombinations(::mlir::Builder &builder) {
      return {builder.getDictionaryAttr(
                  builder.getNamedAttr("parameter",
                                       builder.getI64IntegerAttr(1024)))};
    }
  }];
}
def ApplyDynamicUpdateSliceConstPropPatterns : EnzymeHLOParameterizedPatternOp<
    "dynamic_update_slice_const_prop"> {
  let arguments = (ins OptionalAttr<I64Attr>:$benefit, I64Attr:$parameter);
  let assemblyFormat = "attr-dict";
  // TODO: this should be made better searchable.
  let extraClassDeclaration = [{
    ::llvm::SmallVector<::mlir::DictionaryAttr>
    static getPossibleAttrCombinations(::mlir::Builder &builder) {
      return {builder.getDictionaryAttr(
                  builder.getNamedAttr("parameter",
                                       builder.getI64IntegerAttr(1024)))};
    }
  }];
}
def ApplyConcatFusePatterns : EnzymeHLOPatternOp<
    "concat_fuse"> {
  let patterns = ["ConcatFuse"];
}
def ApplyPadPadPatterns : EnzymeHLOPatternOp<
    "pad_pad"> {
  let patterns = ["PadPad"];
}
def ApplyScatterToDynamicUpdateSlicePatterns : EnzymeHLOPatternOp<
    "scatter_to_dynamic_update_slice"> {
  let patterns = ["ScatterToDynamicUpdateSlice"];
}
def ApplyReduceConcatPatterns : EnzymeHLOPatternOp<
    "reduce_concat"> {
  let patterns = ["ReduceConcat"];
}
def ApplyConcatSlicePatterns : EnzymeHLOPatternOp<
    "concat_slice"> {
  let patterns = ["ConcatSlice"];
}
def ApplySliceConcatPatterns : EnzymeHLOPatternOp<
    "slice_concat"> {
  let patterns = ["SliceConcat"];
}
def ApplyTransposeDotReorderPatterns : EnzymeHLOPatternOp<
    "transpose_dot_reorder"> {
  let patterns = ["TransposeDotReorder"];
}
def ApplyTransposeEinsumPatterns : EnzymeHLOPatternOp<
    "transpose_einsum"> {
  let patterns = ["TransposeEinsum"];
}
def ApplyEinsumTransposePatterns : EnzymeHLOPatternOp<
    "einsum_transpose"> {
  let patterns = ["EinsumTranspose"];
}
def ApplyTransposeConvolutionPatterns : EnzymeHLOPatternOp<
    "transpose_convolution"> {
  let patterns = ["TransposeConvolution"];
}
def ApplyConvolutionTransposePatterns : EnzymeHLOPatternOp<
    "convolution_transpose"> {
  let patterns = ["ConvolutionTranspose"];
}
def ApplyDotTransposePatterns : EnzymeHLOPatternOp<
    "dot_transpose"> {
  let patterns = ["DotTranspose"];
}
def ApplyConvertConvertFloatPatterns : EnzymeHLOPatternOp<
    "convert_convert_float"> {
  let patterns = ["ConvertConvertFloat"];
}
def ApplyConcatToPadPatterns : EnzymeHLOPatternOp<
    "concat_to_pad"> {
  let patterns = ["ConcatToPad"];
}
def ApplyConcatAppendingReshapePatterns : EnzymeHLOPatternOp<
    "concat_appending_reshape"> {
  let patterns = ["ConcatAppendingReshape"];
}
def ApplyReshapeIotaPatterns : EnzymeHLOPatternOp<
    "reshape_iota"> {
  let patterns = ["ReshapeIota"];
}
def ApplyFullReduceReshapeOrTransposePatterns : EnzymeHLOPatternOp<
    "full_reduce_reshape_or_transpose"> {
  let patterns = ["FullReduceReshapeOrTranspose"];
}
def ApplySliceTransposePatterns : EnzymeHLOPatternOp<
    "slice_transpose"> {
  let patterns = ["SliceTranspose"];
}
def ApplySliceBroadcastPatterns : EnzymeHLOPatternOp<
    "slice_broadcast"> {
  let patterns = ["SliceBroadcast"];
}
def ApplyReducePadPatterns : EnzymeHLOPatternOp<
    "reduce_pad"> {
  let patterns = ["ReducePad"];
}
def ApplyBroadcastPadPatterns : EnzymeHLOPatternOp<
    "broadcast_pad"> {
  let patterns = ["BroadcastPad"];
}
def ApplyMulZeroPadPatterns : EnzymeHLOPatternOp<
    "mul_zero_pad"> {
  let patterns = ["MulZeroPad"];
}
def ApplyDivZeroPadPatterns : EnzymeHLOPatternOp<
    "div_zero_pad"> {
  let patterns = ["DivZeroPad"];
}
def ApplyTransposePadPatterns : EnzymeHLOPatternOp<
    "transpose_pad"> {
  let patterns = ["TransposePad"];
}
def ApplyReshapePadPatterns : EnzymeHLOPatternOp<
    "reshape_pad"> {
  let patterns = ["ReshapePad"];
}
def ApplyReshapeSelectPatterns : EnzymeHLOPatternOp<
    "reshape_select"> {
  let patterns = ["ReshapeSelect"];
}
def ApplyTransposeConvertPatterns : EnzymeHLOPatternOp<
    "transpose_convert"> {
  let patterns = ["TransposeConvert"];
}
def ApplyTransposeTransposePatterns : EnzymeHLOPatternOp<
    "transpose_transpose"> {
  let patterns = ["TransposeTranspose"];
}
def ApplyBroadcastReducePatterns : EnzymeHLOPatternOp<
    "broadcast_reduce"> {
  let patterns = ["BroadcastReduce"];
}
def ApplySliceDotGeneralPatterns : EnzymeHLOPatternOp<
    "slice_dot_general"> {
  let patterns = ["SliceDotGeneral"];
}
def ApplySliceReshapePatterns : EnzymeHLOPatternOp<
    "slice_reshape"> {
  let patterns = ["SliceReshape"];
}
def ApplyAllFiniteIsFinitePatterns : EnzymeHLOPatternOp<
    "all_finite_is_finite"> {
  let patterns = ["AllFiniteIsFinite"];
}
def ApplyAllFiniteIsInfPatterns : EnzymeHLOPatternOp<
    "all_finite_is_inf"> {
  let patterns = ["AllFiniteIsInf"];
}
def ApplyAllFiniteIsPosInfPatterns : EnzymeHLOPatternOp<
    "all_finite_is_pos_inf"> {
  let patterns = ["AllFiniteIsPosInf"];
}
def ApplyAllFiniteIsNegInfPatterns : EnzymeHLOPatternOp<
    "all_finite_is_neg_inf"> {
  let patterns = ["AllFiniteIsNegInf"];
}

def ApplyConcatReshapeSlicePatterns : EnzymeHLOPatternOp<
    "concat_reshape_slice"> {
  let patterns = ["ConcatReshapeSlice"];
}

def ApplyConcatReshapeReducePatterns : EnzymeHLOPatternOp<
    "concat_reshape_reduce"> {
  let patterns = ["ConcatReshapeReduce"];
}

def ApplyConcatElementwisePatterns : EnzymeHLOPatternOp<
    "concat_elementwise"> {
  let patterns = ["ConcatElementwise"];
}

def ApplyConcatReshapeElementwisePatterns : EnzymeHLOPatternOp<
    "concat_reshape_elementwise"> {
  let patterns = ["ConcatReshapeElementwise"];
}

def ApplyNoNanCompareSimplify : EnzymeHLOParameterizedPatternOp<
    "no_nan_compare_simplify"> {
  let arguments = (ins OptionalAttr<I64Attr>:$benefit, BoolAttr:$parameter);
  let assemblyFormat = "attr-dict";
  // TODO: this should be made better searchable.
  let extraClassDeclaration = [{
    ::llvm::SmallVector<::mlir::DictionaryAttr>
    static getPossibleAttrCombinations(::mlir::Builder &builder) {
      return {builder.getDictionaryAttr(
                  builder.getNamedAttr("parameter",
                                       builder.getBoolAttr(true)))};
    }
  }];
}

def ApplyNoNanSelfSubSimplify : EnzymeHLOParameterizedPatternOp<
    "no_nan_self_sub_simplify"> {
  let arguments = (ins OptionalAttr<I64Attr>:$benefit, BoolAttr:$parameter);
  let assemblyFormat = "attr-dict";
  // TODO: this should be made better searchable.
  let extraClassDeclaration = [{
    ::llvm::SmallVector<::mlir::DictionaryAttr>
    static getPossibleAttrCombinations(::mlir::Builder &builder) {
      return {builder.getDictionaryAttr(
                  builder.getNamedAttr("parameter",
                                       builder.getBoolAttr(true)))};
    }
  }];
}

// benefit 65k + max_constant_expansion flag
def ApplyNoNanAddSubSimplify : EnzymeHLOParameterizedPatternOp<
    "no_nan_add_sub_simplify"> {
  let arguments = (ins OptionalAttr<I64Attr>:$benefit, BoolAttr:$parameter);
  let assemblyFormat = "attr-dict";
  // TODO: this should be made better searchable.
  let extraClassDeclaration = [{
    ::llvm::SmallVector<::mlir::DictionaryAttr>
    static getPossibleAttrCombinations(::mlir::Builder &builder) {
      return {builder.getDictionaryAttr(
                  builder.getNamedAttr("parameter",
                                       builder.getBoolAttr(true)))};
    }
  }];
}

def ApplyNoNanMulSimplify : EnzymeHLOParameterizedPatternOp<
    "no_nan_mul_simplify"> {
  let arguments = (ins OptionalAttr<I64Attr>:$benefit, BoolAttr:$parameter);
  let assemblyFormat = "attr-dict";
  // TODO: this should be made better searchable.
  let extraClassDeclaration = [{
    ::llvm::SmallVector<::mlir::DictionaryAttr>
    static getPossibleAttrCombinations(::mlir::Builder &builder) {
      return {builder.getDictionaryAttr(
                  builder.getNamedAttr("parameter",
                                       builder.getBoolAttr(true)))};
    }
  }];
}

def ApplyNoNanDivSimplify : EnzymeHLOParameterizedPatternOp<
    "no_nan_div_simplify"> {
  let arguments = (ins OptionalAttr<I64Attr>:$benefit, BoolAttr:$parameter);
  let assemblyFormat = "attr-dict";
  // TODO: this should be made better searchable.
  let extraClassDeclaration = [{
    ::llvm::SmallVector<::mlir::DictionaryAttr>
    static getPossibleAttrCombinations(::mlir::Builder &builder) {
      return {builder.getDictionaryAttr(
                  builder.getNamedAttr("parameter",
                                       builder.getBoolAttr(true)))};
    }
  }];
}

def ApplyNoNanZeroBasePowSimplify : EnzymeHLOParameterizedPatternOp<
    "no_nan_zero_base_pow_simplify"> {
  let arguments = (ins OptionalAttr<I64Attr>:$benefit, BoolAttr:$parameter);
  let assemblyFormat = "attr-dict";
  // TODO: this should be made better searchable.
  let extraClassDeclaration = [{
    ::llvm::SmallVector<::mlir::DictionaryAttr>
    static getPossibleAttrCombinations(::mlir::Builder &builder) {
      return {builder.getDictionaryAttr(
                  builder.getNamedAttr("parameter",
                                       builder.getBoolAttr(true)))};
    }
  }];
}

def ApplyTransposeElementwisePatterns : EnzymeHLOParameterizedPatternOp<
    "transpose_elementwise"> {
  let arguments = (ins OptionalAttr<I64Attr>:$benefit, BoolAttr:$parameter);
  let assemblyFormat = "attr-dict";
  // TODO: this should be made better searchable.
  let extraClassDeclaration = [{
    ::llvm::SmallVector<::mlir::DictionaryAttr>
    static getPossibleAttrCombinations(::mlir::Builder &builder) {
      return {builder.getDictionaryAttr(
                  builder.getNamedAttr("parameter",
                                       builder.getBoolAttr(true)))};
    }
  }];
}

def ApplyConcatPushBinopAddPatterns : EnzymeHLOPatternOp<
    "concat_push_binop_add"> {
  let patterns = ["ConcatPushBinop<stablehlo::AddOp>"];
}
def ApplyConcatPushBinopMulPatterns : EnzymeHLOPatternOp<
    "concat_push_binop_mul"> {
  let patterns = ["ConcatPushBinop<stablehlo::MulOp>"];
}
def ApplyBinBroadcastSplatAddPatterns : EnzymeHLOPatternOp<
    "bin_broadcast_splat_add"> {
  let patterns = ["BinBroadcastSplat<stablehlo::AddOp>"];
}
def ApplyBinBroadcastSplatSubtractPatterns : EnzymeHLOPatternOp<
    "bin_broadcast_splat_subtract"> {
  let patterns = ["BinBroadcastSplat<stablehlo::SubtractOp>"];
}
def ApplyBinBroadcastSplatDivPatterns : EnzymeHLOPatternOp<
    "bin_broadcast_splat_div"> {
  let patterns = ["BinBroadcastSplat<stablehlo::DivOp>"];
}
def ApplyBinBroadcastSplatMulPatterns : EnzymeHLOPatternOp<
    "bin_broadcast_splat_mul"> {
  let patterns = ["BinBroadcastSplat<stablehlo::MulOp>"];
}

def ApplyBinaryOpTransposeSimplifyAddPatterns : EnzymeHLOPatternOp<
    "binary_op_transpose_simplify_add"> {
  let patterns = ["BinaryOpTransposeSimplify<stablehlo::AddOp>"];
}
def ApplyBinaryOpTransposeSimplifySubPatterns : EnzymeHLOPatternOp<
    "binary_op_transpose_simplify_sub"> {
  let patterns = ["BinaryOpTransposeSimplify<stablehlo::SubtractOp>"];
}
def ApplyBinaryOpTransposeSimplifyMulPatterns : EnzymeHLOPatternOp<
    "binary_op_transpose_simplify_mul"> {
  let patterns = ["BinaryOpTransposeSimplify<stablehlo::MulOp>"];
}
def ApplyBinaryOpTransposeSimplifyDivPatterns : EnzymeHLOPatternOp<
    "binary_op_transpose_simplify_div"> {
  let patterns = ["BinaryOpTransposeSimplify<stablehlo::DivOp>"];
}
def ApplyBinaryOpTransposeSimplifyMinPatterns : EnzymeHLOPatternOp<
    "binary_op_transpose_simplify_min"> {
  let patterns = ["BinaryOpTransposeSimplify<stablehlo::MinOp>"];
}
def ApplyBinaryOpTransposeSimplifyMaxPatterns : EnzymeHLOPatternOp<
    "binary_op_transpose_simplify_max"> {
  let patterns = ["BinaryOpTransposeSimplify<stablehlo::MaxOp>"];
}
def ApplyBinaryOpTransposeSimplifyPowPatterns : EnzymeHLOPatternOp<
    "binary_op_transpose_simplify_pow"> {
  let patterns = ["BinaryOpTransposeSimplify<stablehlo::PowOp>"];
}
def ApplyBinaryOpTransposeSimplifyAndPatterns : EnzymeHLOPatternOp<
    "binary_op_transpose_simplify_and"> {
  let patterns = ["BinaryOpTransposeSimplify<stablehlo::AndOp>"];
}
def ApplyBinaryOpTransposeSimplifyOrPatterns : EnzymeHLOPatternOp<
    "binary_op_transpose_simplify_or"> {
  let patterns = ["BinaryOpTransposeSimplify<stablehlo::OrOp>"];
}
def ApplyBinaryOpTransposeSimplifyXorPatterns : EnzymeHLOPatternOp<
    "binary_op_transpose_simplify_xor"> {
  let patterns = ["BinaryOpTransposeSimplify<stablehlo::XorOp>"];
}
def ApplyBinaryOpTransposeSimplifyRemPatterns : EnzymeHLOPatternOp<
    "binary_op_transpose_simplify_rem"> {
  let patterns = ["BinaryOpTransposeSimplify<stablehlo::RemOp>"];
}

def  AssociativeCommonMulOpReorderingPatterns : EnzymeHLOPatternOp<
    "associative_common_mul_op_reordering"> {
  let patterns = ["AssociativeCommonMulOpReordering<stablehlo::AddOp>",
                  "AssociativeCommonMulOpReordering<stablehlo::SubtractOp>"];
}

def ApplyAssociativeBinaryOpReorderingPatterns : EnzymeHLOPatternOp<
    "associative_binary_op_reordering"> {
  let patterns = ["AssociativeBinaryOpReordering<stablehlo::AddOp>",
                  "AssociativeBinaryOpReordering<stablehlo::MulOp>",
                  "AssociativeBinaryOpReordering<stablehlo::MinOp>",
                  "AssociativeBinaryOpReordering<stablehlo::MaxOp>",
                  "AssociativeBinaryOpReordering<stablehlo::AndOp>",
                  "AssociativeBinaryOpReordering<stablehlo::OrOp>",
                  "AssociativeBinaryOpReordering<stablehlo::XorOp>"];
}

def ApplyCommonAssociativeCommutativeOpReorderPatterns : EnzymeHLOPatternOp<
    "common_associative_commutative_op_reorder"> {
  let patterns = ["CommonAssociativeCommutativeOpReorder<stablehlo::AddOp>",
                  "CommonAssociativeCommutativeOpReorder<stablehlo::MulOp>",
                  "CommonAssociativeCommutativeOpReorder<stablehlo::MinOp>",
                  "CommonAssociativeCommutativeOpReorder<stablehlo::MaxOp>",
                  "CommonAssociativeCommutativeOpReorder<stablehlo::AndOp>",
                  "CommonAssociativeCommutativeOpReorder<stablehlo::OrOp>",
                  "CommonAssociativeCommutativeOpReorder<stablehlo::XorOp>"];
}

def ApplyTransposeUnaryTransposeAbsPatterns : EnzymeHLOPatternOp<
    "transpose_unary_transpose_abs"> {
  let patterns = ["TransposeUnaryTransposeSimplify<stablehlo::AbsOp>"];
}
def ApplyTransposeUnaryTransposeNegPatterns : EnzymeHLOPatternOp<
    "transpose_unary_transpose_neg"> {
  let patterns = ["TransposeUnaryTransposeSimplify<stablehlo::NegOp>"];
}
def ApplyTransposeUnaryTransposeSqrtPatterns : EnzymeHLOPatternOp<
    "transpose_unary_transpose_sqrt"> {
  let patterns = ["TransposeUnaryTransposeSimplify<stablehlo::SqrtOp>"];
}
def ApplyTransposeUnaryTransposeRsqrtPatterns : EnzymeHLOPatternOp<
    "transpose_unary_transpose_rsqrt"> {
  let patterns = ["TransposeUnaryTransposeSimplify<stablehlo::RsqrtOp>"];
}
def ApplyTransposeUnaryTransposeCeilPatterns : EnzymeHLOPatternOp<
    "transpose_unary_transpose_ceil"> {
  let patterns = ["TransposeUnaryTransposeSimplify<stablehlo::CeilOp>"];
}
def ApplyTransposeUnaryTransposeConvertPatterns : EnzymeHLOPatternOp<
    "transpose_unary_transpose_convert"> {
  let patterns = ["TransposeUnaryTransposeSimplify<stablehlo::ConvertOp>"];
}
def ApplyTransposeUnaryTransposeCosinePatterns : EnzymeHLOPatternOp<
    "transpose_unary_transpose_cosine"> {
  let patterns = ["TransposeUnaryTransposeSimplify<stablehlo::CosineOp>"];
}
def ApplyTransposeUnaryTransposeExpPatterns : EnzymeHLOPatternOp<
    "transpose_unary_transpose_exp"> {
  let patterns = ["TransposeUnaryTransposeSimplify<stablehlo::ExpOp>"];
}
def ApplyTransposeUnaryTransposeExpm1Patterns : EnzymeHLOPatternOp<
    "transpose_unary_transpose_expm1"> {
  let patterns = ["TransposeUnaryTransposeSimplify<stablehlo::Expm1Op>"];
}
def ApplyTransposeUnaryTransposeLogPatterns : EnzymeHLOPatternOp<
    "transpose_unary_transpose_log"> {
  let patterns = ["TransposeUnaryTransposeSimplify<stablehlo::LogOp>"];
}
def ApplyTransposeUnaryTransposeLog1pPatterns : EnzymeHLOPatternOp<
    "transpose_unary_transpose_log1p"> {
  let patterns = ["TransposeUnaryTransposeSimplify<stablehlo::Log1pOp>"];
}
def ApplyTransposeUnaryTransposeSignPatterns : EnzymeHLOPatternOp<
    "transpose_unary_transpose_sign"> {
  let patterns = ["TransposeUnaryTransposeSimplify<stablehlo::SignOp>"];
}
def ApplyTransposeUnaryTransposeSinePatterns : EnzymeHLOPatternOp<
    "transpose_unary_transpose_sine"> {
  let patterns = ["TransposeUnaryTransposeSimplify<stablehlo::SineOp>"];
}
def ApplyTransposeUnaryTransposeTanhPatterns : EnzymeHLOPatternOp<
    "transpose_unary_transpose_tanh"> {
  let patterns = ["TransposeUnaryTransposeSimplify<stablehlo::TanhOp>"];
}

def TransposeBroadcastInDimToBroadcastInDimPatterns : EnzymeHLOPatternOp<
    "transpose_broadcast_in_dim_to_broadcast_in_dim"> {
  let patterns = ["TransposeBroadcastInDimToBroadcastInDim", "BroadcastInDimTransposeToBroadcastInDim"];
}

def AddPadPadToConcatPatterns : EnzymeHLOPatternOp<
    "add_pad_pad_to_concat"> {
  let patterns = ["AddPadPadToConcat"];
}
def ApplyBinopPadToConcatAddPatterns : EnzymeHLOPatternOp<
    "binop_pad_to_concat_add"> {
  let patterns = ["BinopPadToConcat<stablehlo::AddOp>"];
}
def ApplyBinopPadToConcatMulPatterns : EnzymeHLOPatternOp<
    "binop_pad_to_concat_mul"> {
  let patterns = ["BinopPadToConcat<stablehlo::MulOp>"];
}
def ApplyBinopConstPadAddPatterns : EnzymeHLOPatternOp<
    "binop_const_pad_add"> {
  let patterns = ["BinopConstPad<stablehlo::AddOp>"];
}
def ApplyBinopConstPadSubtractPatterns : EnzymeHLOPatternOp<
    "binop_const_pad_subtract"> {
  let patterns = ["BinopConstPad<stablehlo::SubtractOp>"];
}
def ApplyBinopConstPadMulPatterns : EnzymeHLOPatternOp<
    "binop_const_pad_mul"> {
  let patterns = ["BinopConstPad<stablehlo::MulOp>"];
}
def ApplyBinopConstPadDivPatterns : EnzymeHLOPatternOp<
    "binop_const_pad_div"> {
  let patterns = ["BinopConstPad<stablehlo::DivOp>"];
}
def ApplyBinopBinopPadPadAddPatterns : EnzymeHLOPatternOp<
    "binop_binop_pad_pad_add"> {
  let patterns = ["BinopBinopPadPad<stablehlo::AddOp>"];
}
def ApplyBinopBinopPadPadMulPatterns : EnzymeHLOPatternOp<
    "binop_binop_pad_pad_mul"> {
  let patterns = ["BinopBinopPadPad<stablehlo::MulOp>"];
}
def ApplyBinopPadPadAddPatterns : EnzymeHLOPatternOp<
    "binop_pad_pad_add"> {
  let patterns = ["BinopPadPad<stablehlo::AddOp>"];
}
def ApplyBinopPadPadSubtractPatterns : EnzymeHLOPatternOp<
    "binop_pad_pad_subtract"> {
  let patterns = ["BinopPadPad<stablehlo::SubtractOp>"];
}
def ApplyBinopPadPadMulPatterns : EnzymeHLOPatternOp<
    "binop_pad_pad_mul"> {
  let patterns = ["BinopPadPad<stablehlo::MulOp>"];
}
def ApplyBinopPadPadDivPatterns : EnzymeHLOPatternOp<
    "binop_pad_pad_div"> {
  let patterns = ["BinopPadPad<stablehlo::DivOp>"];
}
def ApplyBinopPadPadMinPatterns : EnzymeHLOPatternOp<
    "binop_pad_pad_min"> {
  let patterns = ["BinopPadPad<stablehlo::MinOp>"];
}
def ApplyBinopPadPadMaxPatterns : EnzymeHLOPatternOp<
    "binop_pad_pad_max"> {
  let patterns = ["BinopPadPad<stablehlo::MaxOp>"];
}
def ApplyBinopConstSimplify : EnzymeHLOPatternOp<
    "binop_const_simplify"> {
  let patterns = [
    "MulDivConst", "MulMulConst", "DivMulConst", "DivDivConst",
    "AddSubConst", "AddAddConst", "SubAddConst", "SubSubConst"
  ];
}
def ApplyUnaryPadPushConvertPatterns : EnzymeHLOPatternOp<
    "unary_pad_push_convert"> {
  let patterns = ["UnaryPadPush<stablehlo::ConvertOp>"];
}
def ApplyUnaryPadPushTanhPatterns : EnzymeHLOPatternOp<
    "unary_pad_push_tanh"> {
  let patterns = ["UnaryPadPush<stablehlo::TanhOp>"];
}
def ApplyUnaryPadPushExpPatterns : EnzymeHLOPatternOp<
    "unary_pad_push_exp"> {
  let patterns = ["UnaryPadPush<stablehlo::ExpOp>"];
}
def ApplyCSEIotaPatterns : EnzymeHLOPatternOp<
    "cse_iota"> {
  let patterns = ["CSEIota"];
}
def ApplyCSEBroadcastInDimPatterns : EnzymeHLOPatternOp<
    "cse_broadcast_in_dim"> {
  let patterns = ["CSE<stablehlo::BroadcastInDimOp>"];
}
def ApplyCSESlicePatterns : EnzymeHLOPatternOp<
    "cse_slice"> {
  let patterns = ["CSE<stablehlo::SliceOp>"];
}
def ApplyCSETransposePatterns : EnzymeHLOPatternOp<
    "cse_transpose"> {
  let patterns = ["CSE<stablehlo::TransposeOp>"];
}
def ApplyCSEConvertPatterns : EnzymeHLOPatternOp<
    "cse_convert"> {
  let patterns = ["CSE<stablehlo::ConvertOp>"];
}
def ApplyCSEPadPatterns : EnzymeHLOPatternOp<
    "cse_pad"> {
  let patterns = ["CSE<stablehlo::PadOp>"];
}
def ApplyCSEDotGeneralPatterns : EnzymeHLOPatternOp<
    "cse_dot_general"> {
  let patterns = ["CSE<stablehlo::DotGeneralOp>"];
}
def ApplyCSEReshapePatterns : EnzymeHLOPatternOp<
    "cse_reshape"> {
  let patterns = ["CSE<stablehlo::ReshapeOp>"];
}
def ApplyCSEMulPatterns : EnzymeHLOPatternOp<
    "cse_mul"> {
  let patterns = ["CSE<stablehlo::MulOp>"];
}
def ApplyCSEDivPatterns : EnzymeHLOPatternOp<
    "cse_div"> {
  let patterns = ["CSE<stablehlo::DivOp>"];
}
def ApplyCSEAddPatterns : EnzymeHLOPatternOp<
    "cse_add"> {
  let patterns = ["CSE<stablehlo::AddOp>"];
}
def ApplyCSESubtractPatterns : EnzymeHLOPatternOp<
    "cse_subtract"> {
  let patterns = ["CSE<stablehlo::SubtractOp>"];
}
def ApplyCSEMinPatterns : EnzymeHLOPatternOp<
    "cse_min"> {
  let patterns = ["CSE<stablehlo::MinOp>"];
}
def ApplyCSEMaxPatterns : EnzymeHLOPatternOp<
    "cse_max"> {
  let patterns = ["CSE<stablehlo::MaxOp>"];
}
def ApplyCSENegPatterns : EnzymeHLOPatternOp<
    "cse_neg"> {
  let patterns = ["CSE<stablehlo::NegOp>"];
}
def ApplyCSEAbsPatterns : EnzymeHLOPatternOp<
    "cse_abs"> {
  let patterns = ["CSE<stablehlo::AbsOp>"];
}
def ApplyCSEWrapPatterns : EnzymeHLOPatternOp<
    "cse_wrap"> {
  let patterns = ["CSE<enzymexla::WrapOp>"];
}
def ApplyCSERotatePatterns : EnzymeHLOPatternOp<
    "cse_rotate"> {
  let patterns = ["CSE<enzymexla::RotateOp>"];
}
def ApplyCSEExtendPatterns : EnzymeHLOPatternOp<
    "cse_extend"> {
  let patterns = ["CSE<enzymexla::ExtendOp>"];
}

def CompareAbs : EnzymeHLOPatternOp<
    "compare_abs"> {
  let patterns = ["CompareAbs"];
}

def CompareMul : EnzymeHLOPatternOp<
    "compare_mul"> {
  let patterns = ["CompareMul"];
}

def CompareConvert : EnzymeHLOPatternOp<
    "compare_convert"> {
  let patterns = ["CompareConvert"];
}

def AddSelects : EnzymeHLOPatternOp<
    "add_selects"> {
  let patterns = ["AddSelects"];
}

def CompareOpCanonPatterns : EnzymeHLOPatternOp<
    "compare_op_canon"> {
  let patterns = ["CompareOpCanon"];
}

def CompareExtPatterns : EnzymeHLOPatternOp<
    "compare_ext"> {
  let patterns = ["CompareExt"];
}

def BroadcastInDimOpCanonPatterns : EnzymeHLOPatternOp<
    "broadcast_in_dim_op_canon"> {
  let patterns = ["BroadcastInDimOpCanon"];
}

def ConvertOpCanonPatterns : EnzymeHLOPatternOp<
    "convert_op_canon"> {
  let patterns = ["ConvertOpCanon"];
}

def DynamicBroadcastInDimOpNotActuallyDunamicPatterns : EnzymeHLOPatternOp<
    "dynamic_broadcast_in_dim_op_not_actually_dynamic"> {
  let patterns = ["DynamicBroadcastInDimOpNotActuallyDynamic"];
}

def ChainedDynamicBroadcastInDimCanonicalizationPatterns : EnzymeHLOPatternOp<
    "chained_dynamic_broadcast_in_dim_canonicalization"> {
  let patterns = ["ChainedDynamicBroadcastInDimCanonicalization"];
}

def DynamicBroadcastInDimAlDimsNonExpandingPatterns : EnzymeHLOPatternOp<
    "dynamic_broadcast_in_dim_all_dims_non_expanding"> {
  let patterns = ["DynamicBroadcastInDimAllDimsNonExpanding"];
}

def NoopReduceOpCanonPatterns : EnzymeHLOPatternOp<
    "noop_reduce_op_canon"> {
  let patterns = ["NoopReduceOpCanon"];
}

def EmptyReduceOpCanonPatterns : EnzymeHLOPatternOp<
    "empty_reduce_op_canon"> {
  let patterns = ["EmptyReduceOpCanon"];
}

def DynamicReshapeOpCanonPatterns : EnzymeHLOPatternOp<
    "dynamic_reshape_op_canon"> {
  let patterns = ["DynamicReshapeOpCanon"];
}

def GetTupleElementOpCanonPatterns : EnzymeHLOPatternOp<
    "get_tuple_element_op_canon"> {
  let patterns = ["GetTupleElementOpCanon"];
}

def RealOpCanonPatterns : EnzymeHLOPatternOp<
    "real_op_canon"> {
  let patterns = ["RealOpCanon"];
}

def ImagOpCanonPatterns : EnzymeHLOPatternOp<
    "imag_op_canon"> {
  let patterns = ["ImagOpCanon"];
}

def ConjComplexNegatePatterns : EnzymeHLOPatternOp<
    "conj_complex_negate"> {
  let patterns = ["ConjComplexNegate"];
}

def GetDimensionSizeOpCanonPatterns : EnzymeHLOPatternOp<
    "get_dimension_size_op_canon"> {
  let patterns = ["GetDimensionSizeOpCanon"];
}

def GatherOpCanonPatterns : EnzymeHLOPatternOp<
    "gather_op_canon"> {
  let patterns = ["GatherOpCanon"];
}

def ReshapeOpCanonPatterns : EnzymeHLOPatternOp<
    "reshape_op_canon"> {
  let patterns = ["ReshapeOpCanon"];
}

def MergeConsecutiveReshapesPatterns : EnzymeHLOPatternOp<
    "merge_consecutive_reshapes"> {
  let patterns = ["MergeConsecutiveReshapes"];
}

def TransposeIsReshapePatterns : EnzymeHLOPatternOp<
    "transpose_is_reshape"> {
  let patterns = ["TransposeIsReshape"];
}

def IfRemoveUnused : EnzymeHLOPatternOp<
    "if_remove_unused"> {
  let patterns = ["IfRemoveUnused"];
}

def IfInline : EnzymeHLOPatternOp<
    "if_inline"> {
  let patterns = ["IfInline"];
}

def IfToSelect : EnzymeHLOPatternOp<
    "if_to_select"> {
  let patterns = ["IfToSelect"];
}

def IfPredPropagation : EnzymeHLOPatternOp<
    "if_pred_propagation"> {
  let patterns = ["IfPredPropagation"];
}

def ApplyTransposeReversePatterns : EnzymeHLOPatternOp<
    "transpose_reverse"> {
  let patterns = ["TransposeReverse"];
}

def ApplyReverseTransposePatterns : EnzymeHLOPatternOp<
    "reverse_transpose"> {
  let patterns = ["ReverseTranspose"];
}

// benefit 65k + max_constant_expansion flag
def ApplyWhileSimplifyPatterns : EnzymeHLOParameterizedPatternOp<
    "while_simplify"> {
  let arguments = (ins OptionalAttr<I64Attr>:$benefit, BoolAttr:$parameter);
  let assemblyFormat = "attr-dict";
  // TODO: this should be made better searchable.
  let extraClassDeclaration = [{
    ::llvm::SmallVector<::mlir::DictionaryAttr>
    static getPossibleAttrCombinations(::mlir::Builder &builder) {
      return {builder.getDictionaryAttr(
                  builder.getNamedAttr("parameter",
                                       builder.getBoolAttr(false)))};
    }
  }];
}

def ApplyWhileLICMPatterns : EnzymeHLOParameterizedPatternOp<
    "while_licm"> {
  let arguments = (ins OptionalAttr<I64Attr>:$benefit, BoolAttr:$parameter);
  let assemblyFormat = "attr-dict";
  // TODO: this should be made better searchable.
  let extraClassDeclaration = [{
    ::llvm::SmallVector<::mlir::DictionaryAttr>
    static getPossibleAttrCombinations(::mlir::Builder &builder) {
      return {builder.getDictionaryAttr(
                  builder.getNamedAttr("parameter",
                                       builder.getBoolAttr(false)))};
    }
  }];
}

def ApplySliceLICMPatterns : EnzymeHLOParameterizedPatternOp<
    "slice_licm"> {
  let arguments = (ins OptionalAttr<I64Attr>:$benefit, BoolAttr:$parameter);
  let assemblyFormat = "attr-dict";
  // TODO: this should be made better searchable.
  let extraClassDeclaration = [{
    ::llvm::SmallVector<::mlir::DictionaryAttr>
    static getPossibleAttrCombinations(::mlir::Builder &builder) {
      return {builder.getDictionaryAttr(
                  builder.getNamedAttr("parameter",
                                       builder.getBoolAttr(true)))};
    }
  }];
}

def ApplyDUSLICMPatterns : EnzymeHLOParameterizedPatternOp<
    "dus_licm"> {
  let arguments = (ins OptionalAttr<I64Attr>:$benefit, BoolAttr:$parameter);
  let assemblyFormat = "attr-dict";
  // TODO: this should be made better searchable.
  let extraClassDeclaration = [{
    ::llvm::SmallVector<::mlir::DictionaryAttr>
    static getPossibleAttrCombinations(::mlir::Builder &builder) {
      return {builder.getDictionaryAttr(
                  builder.getNamedAttr("parameter",
                                       builder.getBoolAttr(true)))};
    }
  }];
}

def ApplyPadLICMPatterns : EnzymeHLOParameterizedPatternOp<
    "pad_licm"> {
  let arguments = (ins OptionalAttr<I64Attr>:$benefit, BoolAttr:$parameter);
  let assemblyFormat = "attr-dict";
  // TODO: this should be made better searchable.
  let extraClassDeclaration = [{
    ::llvm::SmallVector<::mlir::DictionaryAttr>
    static getPossibleAttrCombinations(::mlir::Builder &builder) {
      return {builder.getDictionaryAttr(
                  builder.getNamedAttr("parameter",
                                       builder.getBoolAttr(true)))};
    }
  }];
}

def ApplyElementwiseLICMPatterns : EnzymeHLOParameterizedPatternOp<
    "elementwise_licm"> {
  let arguments = (ins OptionalAttr<I64Attr>:$benefit, BoolAttr:$parameter);
  let assemblyFormat = "attr-dict";
  // TODO: this should be made better searchable.
  let extraClassDeclaration = [{
    ::llvm::SmallVector<::mlir::DictionaryAttr>
    static getPossibleAttrCombinations(::mlir::Builder &builder) {
      return {builder.getDictionaryAttr(
                  builder.getNamedAttr("parameter",
                                       builder.getBoolAttr(true)))};
    }
  }];
}

def ApplyConcatenateLICMPatterns : EnzymeHLOParameterizedPatternOp<
    "concatenate_licm"> {
  let arguments = (ins OptionalAttr<I64Attr>:$benefit, BoolAttr:$parameter);
  let assemblyFormat = "attr-dict";
  // TODO: this should be made better searchable.
  let extraClassDeclaration = [{
    ::llvm::SmallVector<::mlir::DictionaryAttr>
    static getPossibleAttrCombinations(::mlir::Builder &builder) {
      return {builder.getDictionaryAttr(
                  builder.getNamedAttr("parameter",
                                       builder.getBoolAttr(true)))};
    }
  }];
}

def ApplyBroadcastInDimLICMPatterns : EnzymeHLOParameterizedPatternOp<
    "broadcastindim_licm"> {
  let arguments = (ins OptionalAttr<I64Attr>:$benefit, BoolAttr:$parameter);
  let assemblyFormat = "attr-dict";
  // TODO: this should be made better searchable.
  let extraClassDeclaration = [{
    ::llvm::SmallVector<::mlir::DictionaryAttr>
    static getPossibleAttrCombinations(::mlir::Builder &builder) {
      return {builder.getDictionaryAttr(
                  builder.getNamedAttr("parameter",
                                       builder.getBoolAttr(true)))};
    }
  }];
}

def ApplyReshapeLICMPatterns : EnzymeHLOParameterizedPatternOp<
    "reshape_licm"> {
  let arguments = (ins OptionalAttr<I64Attr>:$benefit, BoolAttr:$parameter);
  let assemblyFormat = "attr-dict";
  // TODO: this should be made better searchable.
  let extraClassDeclaration = [{
    ::llvm::SmallVector<::mlir::DictionaryAttr>
    static getPossibleAttrCombinations(::mlir::Builder &builder) {
      return {builder.getDictionaryAttr(
                  builder.getNamedAttr("parameter",
                                       builder.getBoolAttr(true)))};
    }
  }];
}

def ApplyTransposeLICMPatterns : EnzymeHLOParameterizedPatternOp<
    "transpose_licm"> {
  let arguments = (ins OptionalAttr<I64Attr>:$benefit, BoolAttr:$parameter);
  let assemblyFormat = "attr-dict";
  // TODO: this should be made better searchable.
  let extraClassDeclaration = [{
    ::llvm::SmallVector<::mlir::DictionaryAttr>
    static getPossibleAttrCombinations(::mlir::Builder &builder) {
      return {builder.getDictionaryAttr(
                  builder.getNamedAttr("parameter",
                                       builder.getBoolAttr(true)))};
    }
  }];
}

def SelectCompIotaConstSimplify : EnzymeHLOPatternOp<
    "select_comp_iota_const_simplify"> {
  let patterns = ["SelectCompIotaConstSimplify"];
}

def SelectCompIotaConstToDUS : EnzymeHLOPatternOp<
    "select_comp_iota_to_dus"> {
  let patterns = ["SelectCompIotaConstToDUS"];
}

def SelectPadToDUS : EnzymeHLOPatternOp<
    "select_pad_to_dus"> {
  let patterns = ["SelectPadToDUS"];
}

def AndPadPad : EnzymeHLOPatternOp<
    "and_pad_pad"> {
  let patterns = ["AndPadPad"];
}

def SelectOpUsedWithinIf : EnzymeHLOPatternOp<
    "select_op_used_within_if"> {
  let patterns = ["SelectOpUsedWithinIf"];
}

def ZeroExtentTensorCanonPatterns : EnzymeHLOPatternOp<
    "zero_extent_tensor_canon"> {
  let patterns = ["ZeroExtentTensorCanon"];
}

def ReorderElementwiseAndShapeOpPatterns : EnzymeHLOPatternOp<
    "reorder_elementwise_and_shape_op"> {
  let patterns = ["ReorderElementwiseAndShapeOp"];
}

def TransposeSimplifyPatterns : EnzymeHLOPatternOp<
    "transpose_simplify"> {
  let patterns = ["TransposeSimplify"];
}

def DotGeneralSimplifyPatterns : EnzymeHLOPatternOp<
    "dot_general_simplify"> {
  let patterns = ["DotGeneralSimplify"];
}

def BroadcastReshapePatterns : EnzymeHLOPatternOp<
    "broadcast_reshape"> {
  let patterns = ["BroadcastReshape"];
}

def ConcatPadPatterns : EnzymeHLOPatternOp<
    "concat_pad"> {
  let patterns = ["ConcatPad"];
}

def ApplyCSEConcatenatePatterns : EnzymeHLOPatternOp<
    "cse_concatenate"> {
  let patterns = ["CSE<stablehlo::ConcatenateOp>"];
}
def ApplySliceReshapeConcatPatterns : EnzymeHLOPatternOp<
    "slice_reshape_concat"> {
  let patterns = ["SliceReshapeConcat"];
}
def ApplySliceReshapeElementwisePatterns : EnzymeHLOPatternOp<
    "slice_reshape_elementwise"> {
  let patterns = ["SliceReshapeElementwise"];
}
def ApplySliceReshapeTransposePatterns : EnzymeHLOPatternOp<
    "slice_reshape_transpose"> {
  let patterns = ["SliceReshapeTranspose"];
}
def ApplySliceReshapeDotGeneralPatterns : EnzymeHLOPatternOp<
    "slice_reshape_dot_general"> {
  let patterns = ["SliceReshapeDotGeneral"];
}
def ApplySliceReshapeSlicePatterns : EnzymeHLOPatternOp<
    "slice_reshape_slice"> {
  let patterns = ["SliceReshapeSlice"];
}
def ConcatToBroadcastPatterns : EnzymeHLOPatternOp<
    "concat_to_broadcast"> {
  let patterns = ["ConcatToBroadcast"];
}
def DotReshapePadPatterns : EnzymeHLOPatternOp<
    "dot_reshape_pad"> {
  let patterns = ["DotReshapePad"];
}
def ZeroProductReshapePadPatterns : EnzymeHLOPatternOp<
    "zero_product_reshape_pad"> {
  let patterns = ["ZeroProductReshapePad"];
}
def PadReshapePadPatterns : EnzymeHLOPatternOp<
    "pad_reshape_pad"> {
  let patterns = ["PadReshapePad"];
}
def BinopConstReshapePadPatterns : EnzymeHLOPatternOp<
    "binop_const_reshape_pad"> {
  let patterns = ["BinopConstReshapePad"];
}
def ApplyShiftRightLogicalSimplifyPatterns : EnzymeHLOPatternOp<
    "shift_right_logical_simplify"> {
  let patterns = ["ShiftRightLogicalSimplify"];
}
def WhileDeadResultPatterns : EnzymeHLOPatternOp<
    "while_deadresult"> {
  let patterns = ["WhileDeadResults"];
}
def ApplyRemSimplifyPatterns : EnzymeHLOPatternOp<
    "rem_simplify"> {
  let patterns = ["RemSimplify"];
}
def ApplyConstPropThroughBarrierPatterns : EnzymeHLOPatternOp<
    "const_prop_through_barrier"> {
  let patterns = ["ConstPropThroughBarrier"];
}
def DynamicGatherOpIsNotDynamicPatterns : EnzymeHLOPatternOp<
    "dynamic_gather_op_is_not_dynamic"> {
  let patterns = ["DynamicGatherOpIsNotDynamic"];
}

def DivideSqrtToMultiplyRsqrtPatterns : EnzymeHLOPatternOp<
    "divide_sqrt_to_multiply_rsqrt"> {
  let patterns = ["DivideSqrtToMultiplyRsqrt"];
}

def CompareSelectSimplifyPatterns : EnzymeHLOPatternOp<
    "compare_select_simplify"> {
  let patterns = ["CompareSelectSimplify"];
}

def NotSelectSimplifyPatterns : EnzymeHLOPatternOp<
    "not_select_simplify"> {
  let patterns = ["NotSelectSimplify"];
}

def CommonCompareExpressionRewritePatterns : EnzymeHLOPatternOp<
    "common_compare_expression_rewrite"> {
  let patterns = ["CommonCompareExpressionRewrite"];
}

def ApplyScatterUpdateComputationConstPropPatterns : EnzymeHLOPatternOp<
    "scatter_update_computation_const_prop"> {
  let patterns = ["ScatterUpdateComputationConstProp"];
}

def ApplyScatterIndicesAreUniquePatterns : EnzymeHLOPatternOp<
    "scatter_indices_are_unique"> {
  let patterns = ["ScatterIndicesAreUnique"];
}

def ApplyReduceTransposeSimplifyPatterns : EnzymeHLOPatternOp<
    "reduce_transpose_simplify"> {
  let patterns = ["ReduceTransposeSimplify"];
}

def CompareIotaConstSimplifyPatterns : EnzymeHLOPatternOp<
  "compare_iota_const_simplify"
> {
  let patterns = ["CompareIotaConstSimplify"];
}

def CompareNegateConstSimplifyPatterns : EnzymeHLOPatternOp<
  "compare_negate_const_simplify"
> {
  let patterns = ["CompareNegateConstSimplify"];
}

def BroadcastIotaSimplifyPatterns : EnzymeHLOPatternOp<
  "broadcast_iota_simplify"
> {
  let patterns = ["BroadcastIotaSimplify"];
}

def BroadcastIotaPatterns : EnzymeHLOPatternOp<
  "broadcast_iota"
> {
  let patterns = ["BroadcastIota"];
}

def DUSDUSPatterns : EnzymeHLOPatternOp<
  "dus_dus"
> {
  let patterns = ["DUSDUS"];
}

def DUSDUSConcatPatterns : EnzymeHLOPatternOp<
  "dus_dus_concat"
> {
  let patterns = ["DUSDUSConcat"];
}

def TransposeDUSPatterns : EnzymeHLOPatternOp<
  "transpose_dus"
> {
  let patterns = ["TransposeDUS"];
}

def WhileOpInductionReplacementPatterns : EnzymeHLOPatternOp<
  "while_op_induction_replacement"
> {
  let patterns = ["WhileOpInductionReplacement"];
}

def TransposeWhilePatterns : EnzymeHLOPatternOp<
  "transpose_while"
> {
  let patterns = ["TransposeWhile"];
}

def TransposeSlicePatterns : EnzymeHLOPatternOp<
  "transpose_slice"
> {
  let patterns = ["TransposeSlice"];
}

def TransposeDynamicSlicePatterns : EnzymeHLOPatternOp<
  "transpose_dynamic_slice"
> {
  let patterns = ["TransposeDynamicSlice"];
}

def TransposeReshapeToBroadcastPatterns : EnzymeHLOPatternOp<
  "transpose_reshape_to_broadcast"
> {
  let patterns = ["TransposeReshapeToBroadcast"];
}

def ReshapeTransposeToBroadcastPatterns : EnzymeHLOPatternOp<
  "reshape_transpose_to_broadcast"
> {
  let patterns = ["ReshapeTransposeToBroadcast"];
}

def TransposeConcatPatterns : EnzymeHLOPatternOp<
  "transpose_concat"
> {
  let patterns = ["TransposeConcat"];
}

def TransposeIotaPatterns : EnzymeHLOPatternOp<
  "transpose_iota"
> {
  let patterns = ["TransposeIota"];
}

def TransposeReduceWindowPatterns : EnzymeHLOPatternOp<
  "transpose_reduce_window"
> {
  let patterns = ["TransposeReduceWindow"];
}

def SliceReduceWindowPatterns : EnzymeHLOPatternOp<
  "slice_reduce_window"
> {
  let patterns = ["SliceReduceWindow"];
}

def TransposeReducePatterns : EnzymeHLOPatternOp<
  "transpose_reduce"
> {
  let patterns = ["TransposeReduce"];
}

def BroadcastInDimIsReshapePatterns : EnzymeHLOPatternOp<
  "broadcastindim_is_reshape"
> {
  let patterns = ["BroadcastInDimIsReshape"];
}

def ApplyReshapeDUSPatterns : EnzymeHLOPatternOp<
    "reshape_dus"> {
  let patterns = ["ReshapeDUS"];
}

def ApplyReshapeSlicePatterns : EnzymeHLOParameterizedPatternOp<
    "reshape_slice"> {
  let arguments = (ins OptionalAttr<I64Attr>:$benefit, BoolAttr:$parameter);
  let assemblyFormat = "attr-dict";
  // TODO: this should be made better searchable.
  let extraClassDeclaration = [{
    ::llvm::SmallVector<::mlir::DictionaryAttr>
    static getPossibleAttrCombinations(::mlir::Builder &builder) {
      return {builder.getDictionaryAttr(
                  builder.getNamedAttr("parameter",
                                       builder.getBoolAttr(true)))};
    }
  }];
}

def ApplyReshapeElementwisePatterns : EnzymeHLOParameterizedPatternOp<
    "reshape_elementwise"> {
  let arguments = (ins OptionalAttr<I64Attr>:$benefit, BoolAttr:$parameter);
  let assemblyFormat = "attr-dict";
  // TODO: this should be made better searchable.
  let extraClassDeclaration = [{
    ::llvm::SmallVector<::mlir::DictionaryAttr>
    static getPossibleAttrCombinations(::mlir::Builder &builder) {
      return {builder.getDictionaryAttr(
                  builder.getNamedAttr("parameter",
                                       builder.getBoolAttr(true)))};
    }
  }];
}

def ReshapeOfConcatPatterns : EnzymeHLOPatternOp<
  "reshape_concat"
> {
  let patterns = ["ReshapeOfConcatToConcatOfReshape"];
}

def ReshapeReduceWindowPatterns : EnzymeHLOPatternOp<
  "reshape_reduce_window"
> {
  let patterns = ["ReshapeReduceWindow"];
}

def WhileDUS : EnzymeHLOPatternOp<
    "while_dus"> {
  let patterns = ["WhileDUS"];
}

def DUSSliceSimplify : EnzymeHLOPatternOp<
    "dus_slice_simplify"> {
  let patterns = ["DUSSliceSimplify"];
}

def WhileConcat : EnzymeHLOPatternOp<
    "while_concat"> {
  let patterns = ["WhileConcat"];
}

def WhileWrap : EnzymeHLOPatternOp<
    "while_wrap"> {
  let patterns = ["WhileWrap<enzymexla::WrapOp>"];
}

def WhileExtend : EnzymeHLOPatternOp<
    "while_extend"> {
  let patterns = ["WhileWrap<enzymexla::ExtendOp>"];
}

def DUSToI32 : EnzymeHLOPatternOp<
    "dus_to_i32"> {
  let patterns = ["DUSToI32"];
}

def DUSToConcat : EnzymeHLOPatternOp<
    "dus_to_concat"> {
  let patterns = ["DUSToConcat"];
}

def DUSConcat : EnzymeHLOPatternOp<
    "dus_concat"> {
  let patterns = ["DUSConcat"];
}

def DUSPad : EnzymeHLOPatternOp<
    "dus_pad"> {
  let patterns = ["DUSPad"];
}

def SliceDUSToConcat : EnzymeHLOPatternOp<
    "slice_dus_to_concat"> {
  let patterns = ["SliceDUSToConcat"];
}

def ConcatConcatToDUS : EnzymeHLOPatternOp<
    "concat_concat_to_dus"> {
  let patterns = ["ConcatConcatToDUS"];
}

def WhileInductionReduction : EnzymeHLOPatternOp<
    "while_induction_reduction"> {
  let patterns = ["WhileInductionReduction"];
}

def WhileRepeatedInductionReduction : EnzymeHLOPatternOp<
    "while_repeated_induction_reduction"> {
  let patterns = ["WhileRepeatedInductionReduction"];
}

def WhilePadInductionReduction : EnzymeHLOPatternOp<
    "while_pad_induction_reduction"> {
  let patterns = ["WhilePadInductionReduction"];
}

def SliceIf : EnzymeHLOPatternOp<
    "slice_if"> {
  let patterns = ["SliceIf"];
}

def ConstPadConcatToConcat : EnzymeHLOPatternOp<
    "const_pad_concat_to_concat"> {
  let patterns = ["ConstPadConcatToConcat"];
}

def PadConcatToConcatPad : EnzymeHLOPatternOp<
    "pad_concat_to_concat_pad"> {
  let patterns = ["PadConcatToConcatPad"];
}

def SliceSelectToSelectSlice : EnzymeHLOPatternOp<
    "slice_select_to_select_slice"> {
  let patterns = ["SliceSelect"];
}

def TransposeSelect : EnzymeHLOPatternOp<
    "transpose_select"> {
  let patterns = ["TransposeSelect"];
}

def ApplySumToConvPatterns : EnzymeHLOParameterizedPatternOp<
    "sum_to_conv"> {
  let arguments = (ins OptionalAttr<I64Attr>:$benefit, BoolAttr:$parameter);
  let assemblyFormat = "attr-dict";
  // TODO: this should be made better searchable.
  let extraClassDeclaration = [{
    ::llvm::SmallVector<::mlir::DictionaryAttr>
    static getPossibleAttrCombinations(::mlir::Builder &builder) {
      return {builder.getDictionaryAttr(
                  builder.getNamedAttr("parameter",
                                       builder.getBoolAttr(false)))};
    }
  }];
}

def ApplyXOrSimplifyPatterns : EnzymeHLOPatternOp<
    "xor_simplify"> {
  let patterns = ["XorSimplify"];
}

def SumToReduceWindow : EnzymeHLOPatternOp<
    "sum_to_reducewindow"> {
  let patterns = ["SumToReduceWindow<stablehlo::AddOp>", "SumToReduceWindow<stablehlo::SubtractOp>"];
}

// TODO: better naming for parameters requires a static interface for
// constructing them in search.

// benefit 65k + max_constant_expansion flag
def ApplyIotaSimplifyPatterns : EnzymeHLOParameterizedPatternOp<
    "iota_simplify"> {
  let arguments = (ins OptionalAttr<I64Attr>:$benefit, I64Attr:$parameter);
  let assemblyFormat = "attr-dict";
  // TODO: this should be made better searchable.
  let extraClassDeclaration = [{
    ::llvm::SmallVector<::mlir::DictionaryAttr>
    static getPossibleAttrCombinations(::mlir::Builder &builder) {
      return {builder.getDictionaryAttr(
                  builder.getNamedAttr("parameter",
                                       builder.getI64IntegerAttr(1024)))};
    }
  }];
}
def ApplyBroadcastInDimSimplifyPatterns : EnzymeHLOParameterizedPatternOp<
    "broadcast_in_dim_simplify"> {
  let arguments = (ins OptionalAttr<I64Attr>:$benefit, I64Attr:$parameter);
  let assemblyFormat = "attr-dict";
  // TODO: this should be made better searchable.
  let extraClassDeclaration = [{
    ::llvm::SmallVector<::mlir::DictionaryAttr>
    static getPossibleAttrCombinations(::mlir::Builder &builder) {
      return {builder.getDictionaryAttr(
                  builder.getNamedAttr("parameter",
                                       builder.getI64IntegerAttr(1024)))};
    }
  }];
}
def SelectOpCanonPatterns : EnzymeHLOParameterizedPatternOp<
    "select_op_canon"> {
  let arguments = (ins OptionalAttr<I64Attr>:$benefit, I64Attr:$parameter);
  let assemblyFormat = "attr-dict";
  // TODO: this should be made better searchable.
  let extraClassDeclaration = [{
    ::llvm::SmallVector<::mlir::DictionaryAttr>
    static getPossibleAttrCombinations(::mlir::Builder &builder) {
      return {builder.getDictionaryAttr(
                  builder.getNamedAttr("parameter",
                                       builder.getI64IntegerAttr(1024)))};
    }
  }];
}
def ConcatenateOpCanonPatterns : EnzymeHLOParameterizedPatternOp<
    "concatenate_op_canon"> {
  let arguments = (ins OptionalAttr<I64Attr>:$benefit, I64Attr:$parameter);
  let assemblyFormat = "attr-dict";
  // TODO: this should be made better searchable.
  let extraClassDeclaration = [{
    ::llvm::SmallVector<::mlir::DictionaryAttr>
    static getPossibleAttrCombinations(::mlir::Builder &builder) {
      return {builder.getDictionaryAttr(
                  builder.getNamedAttr("parameter",
                                       builder.getI64IntegerAttr(1024)))};
    }
  }];
}

def ApplyPadDotGeneralPatterns : EnzymeHLOParameterizedPatternOp<
    "pad_dot_general"> {
  let arguments = (ins OptionalAttr<I64Attr>:$benefit, I64Attr:$parameter);
  let assemblyFormat = "`postPad` `=` $parameter attr-dict";
  // TODO: the following can be automated by tablegen or some sort of
  // lighter-weight introspection of searchable attributes.
  let extraClassDeclaration = [{
    ::llvm::SmallVector<::mlir::DictionaryAttr>
    static getPossibleAttrCombinations(::mlir::Builder &builder) {
      return {builder.getDictionaryAttr(
                  builder.getNamedAttr("parameter", builder.getI64IntegerAttr(0))),
              builder.getDictionaryAttr(
                  builder.getNamedAttr("parameter", builder.getI64IntegerAttr(1)))};
    }
  }];
}

def GroupCommsConcat : EnzymeHLOPatternOp<
    "group_comms_concat"> {
  let patterns = ["GroupComms<stablehlo::ConcatenateOp>"];
}

def GroupCommsDUS : EnzymeHLOPatternOp<
    "group_comms_dus"> {
  let patterns = ["GroupComms<stablehlo::DynamicUpdateSliceOp>"];
}

def GroupCommsSlice : EnzymeHLOPatternOp<
    "group_comms_slice"> {
  let patterns = ["GroupComms<stablehlo::SliceOp>"];
}

def GroupCommsReshape : EnzymeHLOPatternOp<
    "group_comms_reshape"> {
  let patterns = ["GroupComms<stablehlo::ReshapeOp>"];
}

def GroupCommsRotate : EnzymeHLOPatternOp<
    "group_comms_rotate"> {
  let patterns = ["GroupComms<enzymexla::RotateOp>"];
}

def GroupCommsWrap : EnzymeHLOPatternOp<
    "group_comms_wrap"> {
  let patterns = ["GroupComms<enzymexla::WrapOp>"];
}

def GroupCommsExtend : EnzymeHLOPatternOp<
    "group_comms_extend"> {
  let patterns = ["GroupComms<enzymexla::ExtendOp>"];
}

def GroupCommsTranspose : EnzymeHLOPatternOp<
    "group_comms_transpose"> {
  let patterns = ["GroupComms<stablehlo::TransposeOp>"];
}

def LowerCommRegion : EnzymeHLOPatternOp<
    "lower_comm_region"> {
  let patterns = ["LowerCommRegion"];
}

def RecognizeRotate : EnzymeHLOPatternOp<
    "recognize_rotate"> {
  let patterns = ["RecognizeRotate"];
}

def LowerRotate : EnzymeHLOPatternOp<
    "lower_rotate"> {
  let patterns = ["LowerRotate"];
}

def RecognizeWrap : EnzymeHLOPatternOp<
    "recognize_wrap"> {
  let patterns = ["RecognizeWrap"];
}

def LowerWrap : EnzymeHLOPatternOp<
    "lower_wrap"> {
  let patterns = ["LowerWrap"];
}

def RecognizeExtend : EnzymeHLOPatternOp<
    "recognize_extend"> {
  let patterns = ["RecognizeExtend"];
}

def ExtendSplat : EnzymeHLOPatternOp<
  "extend_splat"> {
  let patterns = ["ExtendSplat"];
}

def ExtendUnaryElementwise : EnzymeHLOParameterizedPatternOp<
  "extend_unary_elementwise"> {
  let arguments = (ins OptionalAttr<I64Attr>:$benefit, BoolAttr:$parameter);
  let assemblyFormat = "attr-dict";
  // TODO: this should be made better searchable.
  let extraClassDeclaration = [{
    ::llvm::SmallVector<::mlir::DictionaryAttr>
    static getPossibleAttrCombinations(::mlir::Builder &builder) {
      return {builder.getDictionaryAttr(
                  builder.getNamedAttr("parameter",
                                       builder.getBoolAttr(true)))};
    }
  }];
}

def ExtendElementwise : EnzymeHLOPatternOp<
  "extend_elementwise"> {
  let patterns = ["ExtendElementwise<stablehlo::AddOp>",
                  "ExtendElementwise<stablehlo::SubtractOp>",
                  "ExtendElementwise<stablehlo::MulOp>",
                  "ExtendElementwise<stablehlo::DivOp>"];
}

def WrapUnaryElementwise : EnzymeHLOParameterizedPatternOp<
  "wrap_unary_elementwise"> {
  let arguments = (ins OptionalAttr<I64Attr>:$benefit, BoolAttr:$parameter);
  let assemblyFormat = "attr-dict";
  // TODO: this should be made better searchable.
  let extraClassDeclaration = [{
    ::llvm::SmallVector<::mlir::DictionaryAttr>
    static getPossibleAttrCombinations(::mlir::Builder &builder) {
      return {builder.getDictionaryAttr(
                  builder.getNamedAttr("parameter",
                                       builder.getBoolAttr(true)))};
    }
  }];
}

def WrapElementwise : EnzymeHLOPatternOp<
  "wrap_elementwise"> {
  let patterns = ["WrapElementwise<stablehlo::AddOp>",
                  "WrapElementwise<stablehlo::SubtractOp>",
                  "WrapElementwise<stablehlo::MulOp>",
                  "WrapElementwise<stablehlo::DivOp>"];
}

def SliceInternal : EnzymeHLOPatternOp<
  "slice_internal"> {
  let patterns = ["SliceInternal"];
}

def SimplifyExtend : EnzymeHLOPatternOp<
  "simplify_extend"> {
  let patterns = ["SimplifyBoundary<enzymexla::ExtendOp>"];
}

def SimplifyWrap : EnzymeHLOPatternOp<
  "simplify_wrap"> {
  let patterns = ["SimplifyBoundary<enzymexla::WrapOp>"];
}

def SimplifyRotate : EnzymeHLOPatternOp<
  "simplify_rotate"> {
  let patterns = ["SimplifyBoundary<enzymexla::RotateOp>"];
}

def SliceExtend : EnzymeHLOPatternOp<
  "slice_extend"> {
  let patterns = ["SliceExtend"];
}

def SliceRotate : EnzymeHLOPatternOp<
  "slice_rotate"> {
  let patterns = ["SliceRotate"];
}

def SliceWrap : EnzymeHLOPatternOp<
  "slice_wrap"> {
  let patterns = ["SliceWrap"];
}

def ReshapeWrap : EnzymeHLOPatternOp<
    "reshape_wrap"> {
  let patterns = ["ReshapeWrap"];
}

def ReshapeExtend : EnzymeHLOPatternOp<
    "reshape_extend"> {
  let patterns = ["ReshapeExtend"];
}

def ReshapeRotate : EnzymeHLOPatternOp<
    "reshape_rotate"> {
  let patterns = ["ReshapeRotate"];
}

def TransposeWrap : EnzymeHLOPatternOp<
    "transpose_wrap"> {
  let patterns = ["TransposeWrap"];
}

def TransposeExtend : EnzymeHLOPatternOp<
    "transpose_extend"> {
  let patterns = ["TransposeExtend"];
}

def TransposeRotate : EnzymeHLOPatternOp<
    "transpose_rotate"> {
  let patterns = ["TransposeRotate"];
}

def SelectPad : EnzymeHLOPatternOp<
    "select_pad"> {
  let patterns = ["SelectPad"];
}

def LowerExtend : EnzymeHLOPatternOp<
    "lower_extend"> {
  let patterns = ["LowerExtend"];
}

def ConcatMultiPad : EnzymeHLOPatternOp<
    "concat_multipad"> {
  let patterns = ["ConcatMultiPad"];
}

def ConcatWrap : EnzymeHLOPatternOp<
    "concat_wrap"> {
  let patterns = ["ConcatWrap"];
}

def WidenWrap : EnzymeHLOPatternOp<
    "widen_wrap"> {
  let patterns = ["WidenWrap"];
}

def WidenExtend : EnzymeHLOPatternOp<
    "widen_extend"> {
  let patterns = ["WidenExtend"];
}

def ConcatConcatAxisSwap : EnzymeHLOPatternOp<
    "concat_concat_axis_swap"> {
  let patterns = ["ConcatConcatAxisSwap"];
}

def RotatePad : EnzymeHLOPatternOp<
    "rotate_pad"> {
  let patterns = ["RotatePad"];
}

def ConcatToOneDimDUS : EnzymeHLOPatternOp<
    "concat_to_onedim_dus"> {
  let patterns = ["ConcatToOneDimDUS"];
}

def ConcatToOneDimDUSSlice : EnzymeHLOPatternOp<
    "concat_to_onedim_dusslice"> {
  let patterns = ["ConcatToOneDimDUSSlice"];
}

def ConcatReshapeToOneDimDUS : EnzymeHLOPatternOp<
    "concatreshape_to_onedim_dus"> {
  let patterns = ["ConcatReshapeToOneDimDUS"];
}

def ReshapeToBroadcast : EnzymeHLOPatternOp<
    "reshape_to_broadcast"> {
  let patterns = ["ReshapeToBroadcast"];
}

def SpeculateIfPadToSelect : EnzymeHLOPatternOp<
    "speculate_if_pad_to_select"> {
  let patterns = ["SpeculateIfPadToSelect"];
}

def CompareCleanup : EnzymeHLOPatternOp<
    "compare_cleanup"> {
  let patterns = ["CompareCleanup"];
}

def BroadcastCompare : EnzymeHLOPatternOp<
    "broadcast_compare"> {
  let patterns = ["BroadcastCompare"];
}

def NotCompare : EnzymeHLOPatternOp<
    "not_compare"> {
  let patterns = ["NotCompare"];
}

def ReshuffleAndsCompares : EnzymeHLOPatternOp<
    "reshuffle_ands_compares"> {
  let patterns = ["ReshuffleAndsCompares"];
}

def ElementwiseReshapeLike : EnzymeHLOPatternOp<
    "elementwise_reshape_like"> {
  let patterns = ["ElementwiseReshapeLike"];
}

def TransposeAllUsersSlice : EnzymeHLOPatternOp<
    "transpose_all_users_slice"> {
  let patterns = ["TransposeAllUsersSlice"];
}

def ConcatTranspose : EnzymeHLOPatternOp<
  "concat_transpose"> {
  let patterns = ["ConcatTranspose"];
}

def ReduceReduce : EnzymeHLOPatternOp<
  "reduce_reduce"> {
  let patterns = ["ReduceReduce"];
}

def ConjReal : EnzymeHLOPatternOp<
  "conj_real"> {
  let patterns = ["ConjReal"];
}

def ApplyTransposeBatchNormTrainingPatterns : EnzymeHLOPatternOp<
    "transpose_batch_norm_training"> {
  let patterns = ["TransposeBatchNormTraining"];
}

def ApplyTransposeBatchNormInferencePatterns : EnzymeHLOPatternOp<
    "transpose_batch_norm_inference"> {
  let patterns = ["TransposeBatchNormInference"];
}

def ApplyTransposeBatchNormGradPatterns : EnzymeHLOPatternOp<
    "transpose_batch_norm_grad"> {
  let patterns = ["TransposeBatchNormGrad"];
}

def ApplySelectBroadcastInDimPatterns : EnzymeHLOPatternOp<
    "select_broadcast_in_dim"> {
  let patterns = ["SelectBroadcastInDim"];
}

def ApplyTransposeIfPatterns : EnzymeHLOPatternOp<
    "transpose_if"> {
  let patterns = ["TransposeIf"];
}

def ApplyIfOpLiftCommonOpsPatterns : EnzymeHLOPatternOp<
    "if_op_lift_common_ops"> {
  let patterns = ["IfOpLiftCommonOps"];
}

def ApplyInvolutionNegSimplifyPatterns : EnzymeHLOPatternOp<
    "involution_neg_simplify"> {
  let patterns = ["InvolutionSimplify<stablehlo::NegOp>"];
}

def ApplyInvolutionConjSimplifyPatterns : EnzymeHLOPatternOp<
    "involution_conj_simplify"> {
  let patterns = ["InvolutionSimplify<chlo::ConjOp>"];
}

def ApplyInvolutionNotSimplifyPatterns : EnzymeHLOPatternOp<
    "involution_not_simplify"> {
  let patterns = ["InvolutionSimplify<stablehlo::NotOp>"];
}

def ApplyRealConjSimplifyPatterns : EnzymeHLOPatternOp<
    "real_conj_simplify"> {
  let patterns = ["RealConjSimplify"];
}

def ApplyConjComplexSimplifyPatterns : EnzymeHLOPatternOp<
    "conj_complex_simplify"> {
  let patterns = ["ConjComplexSimplify"];
}

def ApplyAddConstPropPatterns : EnzymeHLOPatternOp<
    "add_const_prop"> {
  let patterns = ["BinaryConstProp<stablehlo::AddOp, stablehlo::addOp>"];
}
def ApplyAndConstPropPatterns : EnzymeHLOPatternOp<
    "and_const_prop"> {
  let patterns = ["BinaryConstProp<stablehlo::AndOp, stablehlo::andOp>"];
}
def ApplyAtan2ConstPropPatterns : EnzymeHLOPatternOp<
    "atan2_const_prop"> {
  let patterns = ["BinaryConstProp<stablehlo::Atan2Op, stablehlo::atan2Op>"];
}
def ApplyComplexConstPropPatterns : EnzymeHLOPatternOp<
    "complex_const_prop"> {
  let patterns = ["BinaryConstProp<stablehlo::ComplexOp, stablehlo::complexOp>"];
}
def ApplyDivConstPropPatterns : EnzymeHLOPatternOp<
    "div_const_prop"> {
  let patterns = ["BinaryConstProp<stablehlo::DivOp, stablehlo::divideOp>"];
}
def ApplyMaxConstPropPatterns : EnzymeHLOPatternOp<
    "max_const_prop"> {
  let patterns = ["BinaryConstProp<stablehlo::MaxOp, stablehlo::maxOp>"];
}
def ApplyMinConstPropPatterns : EnzymeHLOPatternOp<
    "min_const_prop"> {
  let patterns = ["BinaryConstProp<stablehlo::MinOp, stablehlo::minOp>"];
}
def ApplyMulConstPropPatterns : EnzymeHLOPatternOp<
    "mul_const_prop"> {
  let patterns = ["BinaryConstProp<stablehlo::MulOp, stablehlo::multiplyOp>"];
}
def ApplyOrConstPropPatterns : EnzymeHLOPatternOp<
    "or_const_prop"> {
  let patterns = ["BinaryConstProp<stablehlo::OrOp, stablehlo::orOp>"];
}
def ApplyPowConstPropPatterns : EnzymeHLOPatternOp<
    "pow_const_prop"> {
  let patterns = ["BinaryConstProp<stablehlo::PowOp, stablehlo::powerOp>"];
}
def ApplyRemConstPropPatterns : EnzymeHLOPatternOp<
    "rem_const_prop"> {
  let patterns = ["BinaryConstProp<stablehlo::RemOp, stablehlo::remOp>"];
}
def ApplySubConstPropPatterns : EnzymeHLOPatternOp<
    "sub_const_prop"> {
  let patterns = ["BinaryConstProp<stablehlo::SubtractOp, stablehlo::subtractOp>"];
}
def ApplyXorConstPropPatterns : EnzymeHLOPatternOp<
    "xor_const_prop"> {
  let patterns = ["BinaryConstProp<stablehlo::XorOp, stablehlo::xorOp>"];
}

def ApplySplitConvolutionIntoReverseConvolution : EnzymeHLOPatternOp<
  "split_convolution_into_reverse_convolution"> {
  let patterns = [
    "SplitConvolutionIntoReverseConvolution"
  ];
}

def ApplyScatterMultiplySimplify : EnzymeHLOPatternOp<"scatter_multiply_simplify"> {
  let patterns = [
    "ScatterMultiplySimplify"
  ];
}

def ApplyGatherConstProp : EnzymeHLOPatternOp<"gather_const_prop"> {
  let patterns = [
    "GatherConstProp"
  ];
}

def ApplyUnaryElementwiseScatterSimplify : EnzymeHLOPatternOp<"unary_elementwise_scatter_simplify"> {
  let patterns = [
    "UnaryElementwiseScatterSimplify"
  ];
}

def ApplyGatherElementwise : EnzymeHLOPatternOp<"gather_elementwise"> {
  let patterns = ["GatherElementwise"];
}

def ApplyChainedMultiplyToPower : EnzymeHLOPatternOp<"chained_multiply_to_power"> {
  let patterns = ["ChainedMultiplyToPower"];
}

def ApplyPowerMultiplyToPower : EnzymeHLOPatternOp<"power_multiply_to_power"> {
  let patterns = ["PowerMultiplyToPower"];
}

def ApplyLogSimplify : EnzymeHLOPatternOp<"log_simplify"> {
  let patterns = ["LogSimplify"];
}

def ApplyNegMulConstSimplify : EnzymeHLOPatternOp<"neg_mul_const_simplify"> {
  let patterns = ["NegMulConstSimplify"];
}

def ApplyNegDivConstSimplify : EnzymeHLOPatternOp<"neg_div_const_simplify"> {
  let patterns = ["NegDivConstSimplify"];
}

def ApplyReshapeDeletionsBroadcastInDimSimplify : EnzymeHLOPatternOp<"reshape_deletions_broadcast_in_dim_simplify"> {
  let patterns = ["ReshapeDeletionsBroadcastInDimSimplify"];
}

def ApplyReshapeInsertionsBroadcastInDimSimplify : EnzymeHLOPatternOp<"reshape_insertions_broadcast_in_dim_simplify"> {
  let patterns = ["ReshapeInsertionsBroadcastInDimSimplify"];
}

def ApplyTransposeFFT : EnzymeHLOPatternOp<"transpose_fft"> {
  let patterns = ["TransposeFFT"];
}

def ApplyTransposeReshapePatterns : EnzymeHLOPatternOp<"transpose_reshape"> {
  let patterns = ["TransposeReshape"];
}

def ApplyDotGeneralReshapePatterns : EnzymeHLOPatternOp<"dot_general_reshape"> {
  let patterns = ["DotGeneralReshape"];
}

def ApplyDiagonalTensorDotGeneralRewritePatterns : EnzymeHLOPatternOp<"diagonal_tensor_dot_general_rewrite"> {
  let patterns = ["DiagonalTensorDotGeneralRewrite"];
}

def ApplySelectSimplifyPatterns : EnzymeHLOPatternOp<"select_simplify"> {
  let patterns = ["SelectSimplify"];
}

def ApplyElementwisePadPatterns : EnzymeHLOPatternOp<"elementwise_pad"> {
  let patterns = ["ElementwisePad"];
}

def ApplyConcatenateSubtractToSubtractPadPatterns : EnzymeHLOPatternOp<"concatenate_subtract_to_subtract_pad"> {
  let patterns = ["ConcatenateSubtractToSubtractPad"];
}

def ApplyConcatenateBroadcastInDimPatterns : EnzymeHLOPatternOp<"concatenate_broadcast_in_dim"> {
  let patterns = ["ConcatenateBroadcastInDim"];
}

def ApplyElementwiseRotatePatterns : EnzymeHLOPatternOp<"elementwise_rotate"> {
  let patterns = ["ElementwiseRotate"];
}

def ApplyElementwiseWrapPatterns : EnzymeHLOPatternOp<"elementwise_wrap"> {
  let patterns = ["ElementwiseWrap"];
}

def ApplyElementwiseExtendPatterns : EnzymeHLOPatternOp<"elementwise_extend"> {
  let patterns = ["ElementwiseExtend"];
}
