// StableHLO specification
def Abs : HLOInst<"AbsOp">;
def Add : HLOInst<"AddOp">;
def AfterAll : HLOInst<"AfterAllOp">;
def AllGather : HLOInst<"AllGatherOp">;
def AllReduce : HLOInst<"AllReduceOp">;
def AllToAll : HLOInst<"AllToAllOp">;
def And : HLOInst<"AndOp">;
def Atan2 : HLOInst<"Atan2Op">;
def BatchNormGrad : HLOInst<"BatchNormGradOp">;
def BatchNormInference : HLOInst<"BatchNormInferenceOp">;
def BatchNormTraining : HLOInst<"BatchNormTrainingOp">;
def BitcastConvert : HLOInst<"BitcastConvertOp">;
def BroadcastInDim : HLOInst<"BroadcastInDimOp">;
def Case : HLOInst<"CaseOp">;
def Cbrt : HLOInst<"CbrtOp">;
def Ceil : HLOInst<"CeilOp">;
def Cholesky : HLOInst<"CholeskyOp">;
def Clamp : HLOInst<"ClampOp">;
def CollectiveBroadcast : HLOInst<"CollectiveBroadcastOp">;
def CollectivePermute : HLOInst<"CollectivePermuteOp">;
def Compare : HLOInst<"CompareOp">;
def Complex : HLOInst<"ComplexOp">;
def Composite : HLOInst<"CompositeOp">;
def Concatenate : HLOInst<"ConcatenateOp">;
def Constant : HLOInst<"ConstantOp">;
def Convert : HLOInst<"ConvertOp">;
def Convolution : HLOInst<"ConvolutionOp">;
def Cos : HLOInst<"CosineOp">;
def CountLeadingZeros : HLOInst<"ClzOp">;
def CustomCall : HLOInst<"CustomCallOp">;
def Div : HLOInst<"DivOp">;
def DotGeneral : HLOInst<"DotGeneralOp", "->getResult(0)">;
def DynamicBroadcastInDim : HLOInst<"DynamicBroadcastInDimOp">;
def DynamicConv : HLOInst<"DynamicConvOp">;
def DynamicGather : HLOInst<"DynamicGatherOp">;
def DynamicIota : HLOInst<"DynamicIotaOp">;
def DynamicPad : HLOInst<"DynamicPadOp">;
def DynamicReshape : HLOInst<"DynamicReshapeOp">;
def DynamicSlice : HLOInst<"DynamicSliceOp">;
def DynamicUpdateSlice : HLOInst<"DynamicUpdateSliceOp">;
def Exp : HLOInst<"ExpOp">;
def Expm1 : HLOInst<"Expm1">;
def Fft : HLOInst<"FftOp">;
def Floor : HLOInst<"FloorOp">;
def Gather : HLOInst<"GatherOp">;
def GetDimensionSize : HLOInst<"GetDimensionSizeOp">;
def GetTupleElement : HLOInst<"GetTupleElementOp">;
def If : HLOInst<"IfOp">;
def Imag : HLOInst<"ImagOp">;
def Infeed : HLOInst<"InfeedOp">;
def Iota : HLOInst<"IotaOp">;
def IsFinite : HLOInst<"IsFiniteOp">;
def Log : HLOInst<"LogOp">;
def Log1p : HLOInst<"Log1pOp">;
def Logistic : HLOInst<"LogisticOp">;
def Map : HLOInst<"MapOp">;
def Max : HLOInst<"MaxOp">;
def Min : HLOInst<"MinOp">;
def Mul : HLOInst<"MulOp">;
def Neg : HLOInst<"NegOp">;
def Not : HLOInst<"NotOp">;
def OptimizationBarrier : HLOInst<"OptimizationBarrierOp">;
def Or : HLOInst<"OrOp">;
def Outfeed : HLOInst<"OutfeedOp">;
def Pad : HLOInst<"PadOp">;
def PartitionId : HLOInst<"PartitionIdOp">;
def PopulationCount : HLOInst<"PopcntOp">;
def Pow : HLOInst<"PowOp">;
def Real : HLOInst<"RealOp">;
def Recv : HLOInst<"RecvOp">;
def Reduce : HLOInst<"ReduceOp">;
def ReducePrecision : HLOInst<"ReducePrecisionOp">;
def ReduceScatter : HLOInst<"ReduceScatterOp">;
def ReduceWindow : HLOInst<"ReduceWindowOp">;
def Rem : HLOInst<"RemOp">;
def ReplicaId : HLOInst<"ReplicaIdOp">;
def Reshape : HLOInst<"ReshapeOp">;
def Reverse : HLOInst<"ReverseOp">;
def Rng : HLOInst<"RngOp">;
def RngBitGenerator : HLOInst<"RngBitGeneratorOp">;
def Round : HLOInst<"RoundOp">; // round_nearest_afz
def RoundNearestEven : HLOInst<"RoundNearestEvenOp">;
def Rsqrt : HLOInst<"RsqrtOp">;
def Scatter : HLOInst<"ScatterOp">;
def Select : HLOInst<"SelectOp">;
def SelectAndScatter : HLOInst<"SelectAndScatterOp">;
def Send : HLOInst<"SendOp">;
def ShiftLeft : HLOInst<"ShiftLeftOp">;
def ShiftRightArithmetic : HLOInst<"ShiftRightArithmeticOp">;
def ShiftRightLogical : HLOInst<"ShiftRightLogicalOp">;
def Sign : HLOInst<"SignOp">;
def Sin : HLOInst<"SineOp">;
def Slice : HLOInst<"SliceOp">;
def Sort : HLOInst<"SortOp">;
def Sqrt : HLOInst<"SqrtOp">;
def Sub : HLOInst<"SubtractOp">;
def Tanh : HLOInst<"TanhOp">;
def Transpose : HLOInst<"TransposeOp">;
def TriangularSolve : HLOInst<"TriangularSolveOp">;
def Tuple : HLOInst<"TupleOp">;
def UniformDequantize : HLOInst<"UniformDequantizeOp">;
def UniformQuantize : HLOInst<"UniformQuantizeOp">;
def While : HLOInst<"WhileOp">;
def Xor : HLOInst<"XorOp">;

/// StableHLO - WIP operations
def ReturnOp : HLOInst<"ReturnOp">;

// Deprecated operations from StableHLO - "Not in HLO" category
def Broadcast : HLOInst<"BroadcastOp">;
def CreateToken : HLOInst<"CreateTokenOp">;
def CrossReplicaSum : HLOInst<"CrossReplicaSumOp">;
def Dot : HLOInst<"DotOp">;
def Einsum : HLOInst<"EinsumOp">;
def TorchIndexSelect : HLOInst<"TorchIndexSelectOp">;
def UnaryEinsum : HLOInst<"UnaryEinsumOp">;

// Others
def CheckedMul : HLOInst<"MulOp">;
def CheckedDiv : HLOInst<"DivOp">;

// Helpers
def PadToSliceStart : GlobalExpr</*needsprimal*/0, /*needsshadow*/0, [{
  op.getEdgePaddingLow();
}]>;

def PadToSliceLimit : GlobalExpr</*needsprimal*/0, /*needsshadow*/0, [{
  SmallVector<int64_t> limits;
  for (auto &&[high, dim] : llvm::zip(op.getEdgePaddingHigh(), op.getType().getShape()))
    limits.push_back(to_i64(dim - high));
  getI64Attr(builder, limits);
}]>;

def PadToSliceStride : GlobalExpr</*needsprimal*/0, /*needsshadow*/0, [{
  SmallVector<int64_t> strides;
  for (auto interior : op.getInteriorPadding())
    strides.push_back(to_i64(interior + 1));
  getI64Attr(builder, strides);
}]>;

// convert

def ResultDotDim : GlobalExpr</*needsprimal*/0, /*needsshadow*/0, "op.getDotDimensionNumbersAttr()">;
def ResultDotPrec : GlobalExpr</*needsprimal*/0, /*needsshadow*/0, "op.getPrecisionConfigAttr()">;
def ResultDotAlg : GlobalExpr</*needsprimal*/0, /*needsshadow*/0, "op.getAlgorithmAttr()">;

def ShadowLHSDotDim : GlobalExpr</*needsprimal*/0, /*needsshadow*/0, [{
  auto existingattr = op.getDotDimensionNumbersAttr();

  // first result index is batching, then lhs results (aka indices not batch/contracted)
  size_t resultidx = existingattr.getLhsBatchingDimensions().size() + (op.getLhs().getType().getShape().size() - existingattr.getLhsBatchingDimensions().size() - existingattr.getLhsContractingDimensions().size());

  SmallVector<int64_t> shadowBatchingDimensions;
  for (auto en : llvm::enumerate(existingattr.getLhsBatchingDimensions()))
    shadowBatchingDimensions.push_back(en.index());

  SmallVector<int64_t> rhsContractingDimensions;
  SmallVector<int64_t> shadowResultContractingDimensions;

  for (auto en : llvm::enumerate(op.getRhs().getType().getShape())) {
     if (llvm::is_contained(existingattr.getRhsBatchingDimensions(), en.index())) continue;
     if (llvm::is_contained(existingattr.getRhsContractingDimensions(), en.index())) continue;
     rhsContractingDimensions.push_back(en.index());
     shadowResultContractingDimensions.push_back(resultidx++);
     resultidx++;
  }

  DotDimensionNumbersAttr::get(existingattr.getContext(), shadowBatchingDimensions, existingattr.getRhsBatchingDimensions(), shadowResultContractingDimensions, rhsContractingDimensions);
}]>;

def ShadowLHSDotRes : GlobalExpr</*needsprimal*/0, /*needsshadow*/0, [{
  auto existingattr = op.getDotDimensionNumbersAttr();
  auto prev = op->getResult(0).getType().cast<RankedTensorType>();
  SmallVector<int64_t> shapes;
  // Result order is batches, lhs results, rhs results  [in this case contracting dims]

  for (auto en2 : llvm::enumerate(existingattr.getLhsBatchingDimensions())) {
    shapes.push_back(op.getLhs().getType().getShape()[en2.value()]);
  }

  for (auto en : llvm::enumerate(op.getLhs().getType().getShape())) {
    if (llvm::is_contained(existingattr.getLhsBatchingDimensions(), en.index())) continue;
    if (llvm::is_contained(existingattr.getLhsContractingDimensions(), en.index())) continue;
    shapes.push_back(en.value());
  }

  for (auto en : llvm::enumerate(op.getRhs().getType().getShape())) {
    ssize_t contractidx = -1;

    for (auto en2 : llvm::enumerate(existingattr.getRhsContractingDimensions())) {
      if (en2.value() == en.index()) {
         contractidx = en2.index();
         break;
      }
    }

    if (contractidx == -1) continue;

    shapes.push_back(op.getRhs().getType().getShape()[existingattr.getRhsContractingDimensions()[contractidx]]);
  }

  RankedTensorType::get(shapes, prev.getElementType());
}]>;

def ShadowLHSTranspose : GlobalExpr</*needsprimal*/0, /*needsshadow*/0, [{

  auto existingattr = op.getDotDimensionNumbersAttr();

  size_t resultidx = existingattr.getLhsBatchingDimensions().size();

  SmallVector<int64_t> transposes;

  // Result order is batches, lhs results, rhs results  [in this case contracting dims]
  for (auto en2 : llvm::enumerate(existingattr.getLhsBatchingDimensions())) {
    transposes.push_back(en2.value());
  }

  for (auto en : llvm::enumerate(op.getLhs().getType().getShape())) {
    if (llvm::is_contained(existingattr.getLhsBatchingDimensions(), en.index())) continue;
    if (llvm::is_contained(existingattr.getLhsContractingDimensions(), en.index())) continue;
    transposes.push_back(en.index());
  }

  for (auto en : llvm::enumerate(op.getRhs().getType().getShape())) {
    if (llvm::is_contained(existingattr.getRhsBatchingDimensions(), en.index())) continue;

    ssize_t contractidx = -1;

    for (auto en2 : llvm::enumerate(existingattr.getRhsContractingDimensions())) {
      if (en2.value() == en.index()) {
         contractidx = en2.index();
         break;
      }
    }

    if (contractidx == -1) continue;

    transposes.push_back(existingattr.getLhsContractingDimensions()[contractidx]);
  }

  builder.getNamedAttr(TransposeOp::getAttributeNames()[0], builder.getDenseI64ArrayAttr(transposes));
}]>;

def ShadowRHSDotDim : GlobalExpr</*needsprimal*/0, /*needsshadow*/0, [{
  auto existingattr = op.getDotDimensionNumbersAttr();

  size_t resultidx = existingattr.getLhsBatchingDimensions().size();

  SmallVector<int64_t> shadowBatchingDimensions;
  for (auto en : llvm::enumerate(existingattr.getLhsBatchingDimensions()))
    shadowBatchingDimensions.push_back(en.index());

  SmallVector<int64_t> lhsContractingDimensions;
  SmallVector<int64_t> shadowResultContractingDimensions;

  for (auto en : llvm::enumerate(op.getLhs().getType().getShape())) {
     if (llvm::is_contained(existingattr.getLhsBatchingDimensions(), en.index())) continue;
     if (llvm::is_contained(existingattr.getLhsContractingDimensions(), en.index())) continue;
     lhsContractingDimensions.push_back(en.index());
     shadowResultContractingDimensions.push_back(resultidx++);
     resultidx++;
  }

  DotDimensionNumbersAttr::get(existingattr.getContext(), existingattr.getLhsBatchingDimensions(), shadowBatchingDimensions, lhsContractingDimensions, shadowResultContractingDimensions);
}]>;

def ShadowRHSDotRes : GlobalExpr</*needsprimal*/0, /*needsshadow*/0, [{
  auto existingattr = op.getDotDimensionNumbersAttr();
  auto prev = op->getResult(0).getType().cast<RankedTensorType>();
  SmallVector<int64_t> shapes;
  // Result order is batches, lhs results [in this case contracting dims], rhs results

  for (auto en2 : llvm::enumerate(existingattr.getLhsBatchingDimensions())) {
    shapes.push_back(op.getLhs().getType().getShape()[en2.value()]);
  }

  for (auto en : llvm::enumerate(op.getLhs().getType().getShape())) {
    ssize_t contractidx = -1;

    for (auto en2 : llvm::enumerate(existingattr.getLhsContractingDimensions())) {
      if (en2.value() == en.index()) {
         contractidx = en2.index();
         break;
      }
    }

    if (contractidx == -1) continue;

    shapes.push_back(op.getLhs().getType().getShape()[existingattr.getLhsContractingDimensions()[contractidx]]);
  }

  for (auto en : llvm::enumerate(op.getRhs().getType().getShape())) {
    if (llvm::is_contained(existingattr.getRhsBatchingDimensions(), en.index())) continue;
    if (llvm::is_contained(existingattr.getRhsContractingDimensions(), en.index())) continue;
    shapes.push_back(en.value());
  }

  RankedTensorType::get(shapes, prev.getElementType());
}]>;

def ShadowRHSTranspose : GlobalExpr</*needsprimal*/0, /*needsshadow*/0, [{

  auto existingattr = op.getDotDimensionNumbersAttr();

  size_t resultidx = existingattr.getLhsBatchingDimensions().size();

  SmallVector<int64_t> transposes;

  // Result order is batches, lhs results [in this case contracting dims], rhs results
  for (auto en2 : llvm::enumerate(existingattr.getRhsBatchingDimensions())) {
    transposes.push_back(en2.value());
  }

  for (auto en : llvm::enumerate(op.getLhs().getType().getShape())) {
    if (llvm::is_contained(existingattr.getLhsBatchingDimensions(), en.index())) continue;

    ssize_t contractidx = -1;

    for (auto en2 : llvm::enumerate(existingattr.getLhsContractingDimensions())) {
      if (en2.value() == en.index()) {
         contractidx = en2.index();
         break;
      }
    }

    if (contractidx == -1) continue;

    transposes.push_back(existingattr.getRhsContractingDimensions()[contractidx]);
  }


  for (auto en : llvm::enumerate(op.getRhs().getType().getShape())) {
    if (llvm::is_contained(existingattr.getRhsBatchingDimensions(), en.index())) continue;
    if (llvm::is_contained(existingattr.getRhsContractingDimensions(), en.index())) continue;
    transposes.push_back(en.index());
  }

  builder.getNamedAttr(TransposeOp::getAttributeNames()[0], builder.getDenseI64ArrayAttr(transposes));
}]>;

def Permutation : GlobalExpr</*needsprimal*/0, /*needsshadow*/0, [{
op.getPermutation();
}]>;

def InversePermutation : GlobalExpr</*needsprimal*/0, /*needsshadow*/0, [{
  SmallVector<int64_t> res(op.getPermutation().size(), 0);
  for (auto en : llvm::enumerate(op.getPermutation())) {
    res[to_i64(en.value())] = en.index();
  }
  getI64Attr(builder, res);
}]>;

def Dimensions : GlobalExpr</*needsprimal*/0, /*needsshadow*/0, [{
  op.getDimensions();
}]>;

def ConvWindowStrides : GlobalExpr</*needsprimal*/0, /*needsshadow*/0, [{
  op.getWindowStridesAttr();
}]>;

def ConvPadding : GlobalExpr</*needsprimal*/0, /*needsshadow*/0, [{
  op.getPaddingAttr();
}]>;

def ConvLhsDialation : GlobalExpr</*needsprimal*/0, /*needsshadow*/0, [{
  op.getLhsDilationAttr();
}]>;

def ConvRhsDialation : GlobalExpr</*needsprimal*/0, /*needsshadow*/0, [{
  op.getRhsDilationAttr();
}]>;

def ConvWindowReversal : GlobalExpr</*needsprimal*/0, /*needsshadow*/0, [{
  op.getWindowReversalAttr();
}]>;

def ConvDimensionNumbers : GlobalExpr</*needsprimal*/0, /*needsshadow*/0, [{
  op.getDimensionNumbersAttr();
}]>;

def ConvFeatureGroupCount : GlobalExpr</*needsprimal*/0, /*needsshadow*/0, [{
  op.getFeatureGroupCountAttr();
}]>;

def ConvBatchGroupCount : GlobalExpr</*needsprimal*/0, /*needsshadow*/0, [{
  op.getBatchGroupCountAttr();
}]>;

// TODO

def GradFilterConvWindowStrides : GlobalExpr</*needsprimal*/0, /*needsshadow*/0, [{
  op.getWindowStridesAttr();
}]>;

def GradFilterConvPadding : GlobalExpr</*needsprimal*/0, /*needsshadow*/0, [{
  op.getPaddingAttr();
}]>;

def GradFilterConvLhsDialation : GlobalExpr</*needsprimal*/0, /*needsshadow*/0, [{
  op.getLhsDilationAttr();
}]>;

def GradFilterConvRhsDialation : GlobalExpr</*needsprimal*/0, /*needsshadow*/0, [{
  op.getRhsDilationAttr();
}]>;

def GradFilterConvWindowReversal : GlobalExpr</*needsprimal*/0, /*needsshadow*/0, [{
  op.getWindowReversalAttr();
}]>;

def GradFilterConvDimensionNumbers : GlobalExpr</*needsprimal*/0, /*needsshadow*/0, [{
  op.getDimensionNumbersAttr();
}]>;

def GradFilterConvFeatureGroupCount : GlobalExpr</*needsprimal*/0, /*needsshadow*/0, [{
  op.getFeatureGroupCountAttr();
}]>;

def GradFilterConvBatchGroupCount : GlobalExpr</*needsprimal*/0, /*needsshadow*/0, [{
  op.getBatchGroupCountAttr();
}]>;

// TODO

def GradDataConvWindowStrides : GlobalExpr</*needsprimal*/0, /*needsshadow*/0, [{
  op.getWindowStridesAttr();
}]>;

def GradDataConvPadding : GlobalExpr</*needsprimal*/0, /*needsshadow*/0, [{
  op.getPaddingAttr();
}]>;

def GradDataConvLhsDialation : GlobalExpr</*needsprimal*/0, /*needsshadow*/0, [{
  op.getLhsDilationAttr();
}]>;

def GradDataConvRhsDialation : GlobalExpr</*needsprimal*/0, /*needsshadow*/0, [{
  op.getRhsDilationAttr();
}]>;

def GradDataConvWindowReversal : GlobalExpr</*needsprimal*/0, /*needsshadow*/0, [{
  op.getWindowReversalAttr();
}]>;

def GradDataConvDimensionNumbers : GlobalExpr</*needsprimal*/0, /*needsshadow*/0, [{
  op.getDimensionNumbersAttr();
}]>;

def GradDataConvFeatureGroupCount : GlobalExpr</*needsprimal*/0, /*needsshadow*/0, [{
  op.getFeatureGroupCountAttr();
}]>;

def GradDataConvBatchGroupCount : GlobalExpr</*needsprimal*/0, /*needsshadow*/0, [{
  op.getBatchGroupCountAttr();
}]>;

/// Helpers - Max/Min
def LT : GlobalExpr</*needsprimal*/0, /*needsshadow*/0, "ComparisonDirection::LT">;
def GT : GlobalExpr</*needsprimal*/0, /*needsshadow*/0, "ComparisonDirection::GT">;

def EinsumConfig : GlobalExpr</*needsprimal*/0, /*needsshadow*/0, [{
  op.getEinsumConfig();
}]>;

def GradUnaryEinsumConfig : GlobalExpr</*needsprimal*/0, /*needsshadow*/0, [{
  StringRef config = op.getEinsumConfig();
  auto [input, output] = config.split(StringRef("->"));
  builder.getStringAttr(output + "->" + input);
}]>;

def GradEinsumConfigLhs : GlobalExpr</*needsprimal*/0, /*needsshadow*/0, [{
  StringRef config = op.getEinsumConfig();
  auto [input, output] = config.split(StringRef("->"));
  auto [lhs, rhs] = input.split(',');
  builder.getStringAttr(output + "," + rhs + "->" + lhs);
}]>;

def GradEinsumConfigRhs : GlobalExpr</*needsprimal*/0, /*needsshadow*/0, [{
  StringRef config = op.getEinsumConfig();
  auto [input, output] = config.split(StringRef("->"));
  auto [lhs, rhs] = input.split(',');
  builder.getStringAttr(output + "," + lhs + "->" + rhs);
}]>;

def FftType : GlobalExpr</*needsprimal*/0, /*needsshadow*/0, [{
  op.getFftType();
}]>;

def FftTypeInverse : GlobalExpr</*needsprimal*/0, /*needsshadow*/0, [{
  FftType ret;
  switch (op.getFftType()) {
    case FftType::FFT: {ret = FftType::IFFT; break;}
    case FftType::IFFT: {ret = FftType::FFT; break;}
    case FftType::RFFT: {ret = FftType::IRFFT; break;}
    case FftType::IRFFT: {ret = FftType::RFFT; break;}
  }
  ret;
}]>;

def FftLength : GlobalExpr</*needsprimal*/0, /*needsshadow*/0, [{
  op.getFftLength();
}]>;

def FftMultiplier : GlobalExpr</*needsprimal*/0, /*needsshadow*/0, [{
  auto op_type = op->getResult(0).getType().cast<RankedTensorType>();
  auto lengths = op.getFftLength();
  auto N = std::accumulate(lengths.begin(), lengths.end(), llvm::APInt(64, 1, true), std::multiplies{}).getSExtValue();

  auto ret_constant = builder.create<ConstantOp>(op.getLoc(), builder.getDenseI64ArrayAttr(ArrayRef<int64_t>({N})));
  auto ret_broadcast = builder.create<BroadcastInDimOp>(op.getLoc(), op_type.clone(op_type.getShape(), builder.getI64Type()), ret_constant, builder.getI64VectorAttr(op_type.getShape()));
  builder.create<ConvertOp>(op.getLoc(), op->getResult(0).getType(), ret_broadcast);
}]>;

// Derivative rules
def : HLODerivative<"AddOp", (Op $x, $y),
                    [
                      (DiffeRet),
                      (DiffeRet),
                    ],
                    (Add (Shadow $x), (Shadow $y))
                  >;

def : HLODerivative<"Atan2Op", (Op $x, $y), 
                    [
                      (CheckedMul (DiffeRet), (Div (Neg $y), (Add (Pow $x, (HLOConstantFP<"2">)), (Pow $y, (HLOConstantFP<"2"> $y))))),
                      (CheckedMul (DiffeRet), (Div $x, (Add (Pow $x, (HLOConstantFP<"2">)), (Pow $y, (HLOConstantFP<"2"> $y)))))
                    ],
                    (Div (Sub (Mul $x, (Shadow $y)), (Mul $y, (Shadow $x))), (Add (Pow $x, (HLOConstantFP<"2">)), (Pow $y, (HLOConstantFP<"2"> $y))))
                  >;

def : HLOReadOnlyIdentityOp<"BroadcastInDimOp">;

def : HLOInactiveOp<"CeilOp">;

def : HLODerivative<"CbrtOp", (Op $x), [
  (CheckedMul (DiffeRet), (Div (Pow $x, (Div (HLOConstantFP<"-2">), (HLOConstantFP<"3">))), (HLOConstantFP<"3">))),
]>;

def : HLOInactiveOp<"CompareOp">;

def : HLODerivative<"ComplexOp", (Op $re, $im), [
  (Real (DiffeRet)),
  (Imag (DiffeRet)),
], (Complex (Shadow $re), (Shadow $im))>;

def : HLOMemoryIdentityOp<"ConcatenateOp", [], [-1]>;

def : HLODerivative<"ConvolutionOp", (Op $lhs, $rhs),
                    [
                        (Convolution (TypeOf $lhs), (DiffeRet), $lhs, (GradFilterConvWindowStrides), (GradFilterConvPadding), (GradFilterConvLhsDialation), (GradFilterConvRhsDialation), (GradFilterConvWindowReversal), (GradFilterConvDimensionNumbers), (GradFilterConvFeatureGroupCount), (GradFilterConvBatchGroupCount), (ResultDotPrec)),
                        (Convolution (TypeOf $rhs), (DiffeRet), $rhs, (GradDataConvWindowStrides), (GradDataConvPadding), (GradDataConvLhsDialation), (GradDataConvRhsDialation), (GradDataConvWindowReversal), (GradDataConvDimensionNumbers), (GradDataConvFeatureGroupCount), (GradDataConvBatchGroupCount), (ResultDotPrec))
                    ],
                    (Add (SelectIfActive $lhs, 
                      (Convolution (ResultTypes), (Shadow $lhs), $rhs, (ConvWindowStrides), (ConvPadding), (ConvLhsDialation), (ConvRhsDialation), (ConvWindowReversal), (ConvDimensionNumbers), (ConvFeatureGroupCount), (ConvBatchGroupCount), (ResultDotPrec)),
                      (HLOConstantFP<"0">)), (SelectIfActive $rhs, 
                      (Convolution (ResultTypes), $lhs, (Shadow $rhs), (ConvWindowStrides), (ConvPadding), (ConvLhsDialation), (ConvRhsDialation), (ConvWindowReversal), (ConvDimensionNumbers), (ConvFeatureGroupCount), (ConvBatchGroupCount), (ResultDotPrec)), 
                      (HLOConstantFP<"0">)))
                  >;

def : HLODerivative<"CosineOp", (Op $x), [(CheckedMul (DiffeRet), (Neg (Sin $x)))], (Neg (Mul (Sin $x), (Shadow $x)))>;

def : HLOInactiveOp<"CreateTokenOp">;

def : HLODerivative<"DivOp", (Op $x, $y), 
                    [
                      (CheckedDiv (DiffeRet), $y),
                      (Neg (Mul (CheckedDiv (DiffeRet), $y), (Div $x, $y)))
                    ]
                    // (CheckedDiv (FSub (SelectIfActive $x, (FMul (Shadow $x), $y), (Zero $x)), (SelectIfActive $y, (FMul (Shadow $y), $x), (Zero $y))), (FMul $y, $y))
                  >;

def : HLODerivative<"DotGeneralOp", (Op $lhs, $rhs),
                    [
                        (Transpose (TypeOf $lhs), (DotGeneral (ShadowLHSDotRes), (DiffeRet), $rhs, (ShadowLHSDotDim), (ResultDotPrec), (ResultDotAlg)), (ShadowLHSTranspose)),
                        (Transpose (TypeOf $rhs), (DotGeneral (ShadowRHSDotRes), $lhs, (DiffeRet), (ShadowRHSDotDim), (ResultDotPrec), (ResultDotAlg)), (ShadowRHSTranspose))
                      ],
                    (Add (SelectIfActive $lhs, (DotGeneral (ResultTypes), (Shadow $lhs), $rhs, (ResultDotDim), (ResultDotPrec), (ResultDotAlg)), (HLOConstantFP<"0">)), (SelectIfActive $rhs, (DotGeneral (ResultTypes), $lhs, (Shadow $rhs), (ResultDotDim), (ResultDotPrec), (ResultDotAlg)), (HLOConstantFP<"0">)))
                  >;

def : HLOInactiveOp<"DynamicIotaOp">;

def : HLODerivative<"ExpOp", (Op $x), [(CheckedMul (DiffeRet), (Exp $x))]>;

def : HLODerivative<"Expm1Op", (Op $x), [(CheckedMul (DiffeRet), (Exp $x))]>;

// TODO fix `rfft` and `irfft` derivatives:
// - `rfft` => divide `DiffeRet` elems by 2 except 1st elem, and last elem if `FftLength` is even
// - `irfft` => 
def : HLODerivative<"FftOp", (Op $x),
                    [
                      (Fft (DiffeRet), (FftType), (FftLength)) // TODO maybe we need to conjugate? or inverse fft + multiply by N?
                    ],
                    (Fft (Shadow $x), (FftType), (FftLength))
                  >;

def : HLOInactiveOp<"FloorOp">;

def : HLOInactiveOp<"IotaOp">;

def : HLOInactiveOp<"IsFiniteOp">;

def : HLODerivative<"LogOp", (Op $x), [(CheckedDiv (DiffeRet), $x)]>;

def : HLODerivative<"Log1pOp", (Op $x), [(CheckedDiv (DiffeRet), (Add $x, (HLOConstantFP<"1"> $x)))]>;

def : HLODerivative<"LogisticOp", (Op $x), [(CheckedMul (DiffeRet), (CheckedMul (Logistic $x), (Sub (HLOConstantFP<"1">), (Logistic $x))))]>;

def : HLODerivative<"MaxOp", (Op $x, $y),
                  [
                    (Select (Compare $x, $y, (LT)), (HLOConstantFP<"0"> $x), (DiffeRet)),
                    (Select (Compare $x, $y, (LT)), (DiffeRet), (HLOConstantFP<"0"> $y))
                  ],
                  (Select (Compare $x, $y, (LT)), (SelectIfActive $y, (Shadow $y), (HLOConstantFP<"0"> $y)), (SelectIfActive $x, (Shadow $x), (HLOConstantFP<"0"> $x)))
                  >;

def : HLODerivative<"MinOp", (Op $x, $y),
                  [
                    (Select (Compare $y, $x, (LT)), (HLOConstantFP<"0"> $x), (DiffeRet)),
                    (Select (Compare $y, $x, (LT)), (DiffeRet), (HLOConstantFP<"0"> $y))
                  ],
                  (Select (Compare $y, $x, (LT)), (SelectIfActive $y, (Shadow $y), (HLOConstantFP<"0"> $y)), (SelectIfActive $x, (Shadow $x), (HLOConstantFP<"0"> $x)))
                  >;

def : HLODerivative<"MulOp", (Op $x, $y),
                    [
                      (CheckedMul (DiffeRet), $y),
                      (CheckedMul (DiffeRet), $x)
                    ]
                  >;

def : HLODerivative<"NegOp", (Op $x), [(Neg (DiffeRet))]>;

def : HLOMemoryIdentityOp<"PadOp", [], [-1], (Op $op, $padval), [
  (Slice (TypeOf $op), (DiffeRet), (PadToSliceStart), (PadToSliceLimit), (PadToSliceStride)),
  (AssertingInactiveArg)
]>;

def : HLOInactiveOp<"PartitionIdOp">;

def : HLODerivative<"PowOp", (Op $x, $y), 
                  [
                    (CheckedMul (DiffeRet), (Mul $y, (Pow $x, (Sub $y, (HLOConstantFP<"1"> $y))))),
                    (CheckedMul (DiffeRet), (Mul (Pow $x, $y), (Log $x)))
                  ]
                  >;

def : HLOReadOnlyIdentityOp<"ReshapeOp", [0], (Op $x), [(Reshape (TypeOf $x), (DiffeRet))]>;

def : HLORegionTerminatorOp<"ReturnOp">;

def : HLODerivative<"ReverseOp", (Op $x), [(Reverse (DiffeRet), (Dimensions))], (Reverse (Shadow $x), (Dimensions))>;

def : HLOInactiveOp<"RoundOp">;

def : HLOInactiveOp<"RoundNearestEvenOp">;

def : HLOInactiveOp<"RngOp">;

def : HLOInactiveOp<"RngBitGeneratorOp">;

def : HLODerivative<"RsqrtOp", (Op $x),
                    [
                      // (Select (FCmpUEQ $x, (ConstantFP<"0"> $x)), (ConstantFP<"0"> $x), (FDiv (DiffeRet), (FMul (ConstantFP<"2"> $x), (Call<(SameFunc), [ReadNone,NoUnwind]> $x))))
                      (Div (DiffeRet), (Mul (HLOConstantFP<"2"> $x), (Mul $x, (Sqrt $x))))
                    ]
                  >;

def : HLODerivative<"SelectOp", (Op $cond, $lhs, $rhs),
                    [
                        (AssertingInactiveArg),
                        (Select $cond, (DiffeRet), (HLOConstantFP<"0">)),
                        (Select $cond, (HLOConstantFP<"0">), (DiffeRet)),
                      ],
                      (Select $cond, (SelectIfActive $lhs, (Shadow $lhs), (HLOConstantFP<"0">)), (SelectIfActive $rhs, (Shadow $rhs), (HLOConstantFP<"0">)))
                  >;

def : HLOInactiveOp<"SignOp">;

def : HLODerivative<"SineOp", (Op $x), [(CheckedMul (DiffeRet), (Cos $x))]>;

def : HLOReadOnlyIdentityOp<"SliceOp">;

def : HLODerivative<"SubtractOp", (Op $x, $y),
                    [
                      (DiffeRet),
                      (Neg (DiffeRet)),
                    ],
                    (Sub (Shadow $x), (Shadow $y))
                  >;

def : HLODerivative<"SqrtOp", (Op $x),
                    [
                      // (Select (FCmpUEQ $x, (ConstantFP<"0"> $x)), (ConstantFP<"0"> $x), (FDiv (DiffeRet), (FMul (ConstantFP<"2"> $x), (Call<(SameFunc), [ReadNone,NoUnwind]> $x))))
                      (Div (DiffeRet), (Mul (HLOConstantFP<"2"> $x), (Sqrt $x)))
                    ]
                  >;

def : HLODerivative<"TanhOp", (Op $x),
                    [
                      (CheckedMul (DiffeRet), (Sub (HLOConstantFP<"1">), (Mul (Tanh $x), (Tanh $x))))
                    ]
                  >;

def : HLOInactiveOp<"TorchIndexSelectOp">;

def : HLODerivative<"TransposeOp", (Op $x),
                    [
                        (Transpose (TypeOf $x), (DiffeRet), (InversePermutation)),
                    ],
                    (SelectIfActive $x, (Transpose (ResultTypes), (Shadow $x), (Permutation)), (HLOConstantFP<"0">))
                  >;
