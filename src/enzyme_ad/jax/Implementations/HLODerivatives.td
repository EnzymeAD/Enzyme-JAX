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
def CountLeadingZeros : HLOInst<"CountLeadingZerosOp">;
def CustomCall : HLOInst<"CustomCallOp">;
def Div : HLOInst<"DivideOp">;
def DotGeneral : HLOInst<"DotGeneralOp", "->getResult(0)">;
def DynamicBroadcastInDim : HLOInst<"DynamicBroadcastInDimOp">;
def DynamicConv : HLOInst<"DynamicConvOp">;
def DynamicGather : HLOInst<"DynamicGatherOp">;
def DynamicIota : HLOInst<"DynamicIotaOp">;
def DynamicPad : HLOInst<"DynamicPadOp">;
def DynamicReshape : HLOInst<"DynamicReshapeOp">;
def DynamicSlice : HLOInst<"DynamicSliceOp">;
def DynamicUpdateSlice : HLOInst<"DynamicUpdateSliceOp">;
def Exp : HLOInst<"ExponentialOp">;
def ExponentialMinusOne : HLOInst<"ExponentialMinusOneOp">;
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
def LogPlusOne : HLOInst<"LogPlusOneOp">;
def Logistic : HLOInst<"LogisticOp">;
def Map : HLOInst<"MapOp">;
def Maximum : HLOInst<"MaxOp">;
def Minimum : HLOInst<"MinOp">;
def Mul : HLOInst<"MulOp">;
def Neg : HLOInst<"NegateOp">;
def Not : HLOInst<"NotOp">;
def OptimizationBarrier : HLOInst<"OptimizationBarrierOp">;
def Or : HLOInst<"OrOp">;
def Outfeed : HLOInst<"OutfeedOp">;
def Pad : HLOInst<"PadOp">;
def PartitionId : HLOInst<"PartitionIdOp">;
def Popcnt : HLOInst<"PopcntOp">;
def Pow : HLOInst<"PowerOp">;
def Real : HLOInst<"RealOp">;
def Recv : HLOInst<"RecvOp">;
def Reduce : HLOInst<"ReduceOp">;
def ReducePrecision : HLOInst<"ReducePrecisionOp">;
def ReduceScatter : HLOInst<"ReduceScatterOp">;
def ReduceWindow : HLOInst<"ReduceWindowOp">;
def Rem : HLOInst<"RemainderOp">;
def ReplicaId : HLOInst<"ReplicaIdOp">;
def Reshape : HLOInst<"ReshapeOp">;
def Reverse : HLOInst<"ReverseOp">;
def Rng : HLOInst<"RngOp">;
def RngBitGenerator : HLOInst<"RngBitGeneratorOp">;
def RoundNearestAfz : HLOInst<"RoundNearestAfzOp">;
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

// CHLO
// NOTE CHLO ops are higher level and lower to StableHLO ops
// def Sinh : HLOInst<"SinhOp">;
// def Cosh : HLOInst<"CoshOp">;

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

def : HLODerivative<"AddOp", (Op $x, $y),
                    [
                      (DiffeRet),
                      (DiffeRet),
                    ]
                  >;

def : HLODerivative<"SubtractOp", (Op $x, $y),
                    [
                      (DiffeRet),
                      (Neg (DiffeRet)),
                    ]
                  >;

def : HLODerivative<"NegOp", (Op $x),
                    [
                      (Neg (DiffeRet)),
                    ]
                  >;

def : HLODerivative<"MulOp", (Op $x, $y),
                    [
                      (CheckedMul (DiffeRet), $y),
                      (CheckedMul (DiffeRet), $x)
                    ]
                  >;

def : HLODerivative<"DivOp", (Op $x, $y), 
                    [
                      (CheckedDiv (DiffeRet), $y),
                      (Neg (Mul (CheckedDiv (DiffeRet), $y), (Div $x, $y)))
                    ]
                    // (CheckedDiv (FSub (SelectIfActive $x, (FMul (Shadow $x), $y), (Zero $x)), (SelectIfActive $y, (FMul (Shadow $y), $x), (Zero $y))), (FMul $y, $y))
                  >;

def : HLODerivative<"PowOp", (Op $x, $y), 
                  [
                    (CheckedMul (DiffeRet), (Mul $y, (Pow $x, (Sub $y, (HLOConstantFP<"1"> $y))))),
                    (CheckedMul (DiffeRet), (Mul (Pow $x, $y), (Log $x)
                    ))
                  ]
                  >;

def : HLODerivative<"CosineOp", (Op $x),
                    [
                      (CheckedMul (DiffeRet), (Neg (Sin $x)))
                    ]
                  >;

def : HLODerivative<"ExpOp", (Op $x),
                    [
                      (CheckedMul (DiffeRet), (Exp $x))
                    ]
                  >;

def : HLODerivative<"SineOp", (Op $x),
                    [
                      (CheckedMul (DiffeRet), (Cos $x))
                      ]
                  >;

def : HLODerivative<"SqrtOp", (Op $x),
                    [
                      // (Select (FCmpUEQ $x, (ConstantFP<"0"> $x)), (ConstantFP<"0"> $x), (FDiv (DiffeRet), (FMul (ConstantFP<"2"> $x), (Call<(SameFunc), [ReadNone,NoUnwind]> $x))))
                      (Div (DiffeRet), (Mul (HLOConstantFP<"2"> $x), (Sqrt $x)))
                      ]
                  >;

def : HLODerivative<"RsqrtOp", (Op $x),
                    [
                      // (Select (FCmpUEQ $x, (ConstantFP<"0"> $x)), (ConstantFP<"0"> $x), (FDiv (DiffeRet), (FMul (ConstantFP<"2"> $x), (Call<(SameFunc), [ReadNone,NoUnwind]> $x))))
                      (Div (DiffeRet), (Mul (HLOConstantFP<"2"> $x), (Mul $x, (Sqrt $x))))
                      ]
                  >;

def : HLODerivative<"TanhOp", (Op $x),
                    [
                      (CheckedMul (DiffeRet), (Sub (HLOConstantFP<"1">), (Mul (Tanh $x), (Tanh $x))))
                      ]
                  >;

def : HLODerivative<"LogOp", (Op $x),
                    [
                      (CheckedDiv (DiffeRet), $x)
                      ]
                  >;

def LT : GlobalExpr</*needsprimal*/0, /*needsshadow*/0, "ComparisonDirection::LT">;

def : HLODerivative<"MaxOp", (Op $x, $y),
                  [
                    (Select (Compare $x, $y, (LT)), (HLOConstantFP<"0"> $x), (DiffeRet)),
                    (Select (Compare $x, $y, (LT)), (DiffeRet), (HLOConstantFP<"0"> $x))
                  ],
                  (Select (Compare $x, $y, (LT)), (SelectIfActive $y, (Shadow $y), (HLOConstantFP<"0"> $y)), (SelectIfActive $x, (Shadow $x), (HLOConstantFP<"0"> $x)))
                  >;

def : HLOReadOnlyIdentityOp<"ReshapeOp", [0], (Op $x), [(Reshape (TypeOf $x), (DiffeRet))]>;

def : HLOReadOnlyIdentityOp<"SliceOp">;

def : HLOReadOnlyIdentityOp<"BroadcastInDimOp">;

def : HLOMemoryIdentityOp<"ConcatenateOp", [], [-1]>;

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

def : HLOMemoryIdentityOp<"PadOp", [], [-1], (Op $op, $padval), [
  (Slice (TypeOf $op), (DiffeRet), (PadToSliceStart), (PadToSliceLimit), (PadToSliceStride)),
  (AssertingInactiveArg)
]>;

// convert

def ResultDotDim : GlobalExpr</*needsprimal*/0, /*needsshadow*/0, "op.getDotDimensionNumbersAttr()">;
def ResultDotPrec : GlobalExpr</*needsprimal*/0, /*needsshadow*/0, "op.getPrecisionConfigAttr()">;


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

def : HLODerivative<"DotGeneralOp", (Op $lhs, $rhs),
                    [
                        (Transpose (TypeOf $lhs), (DotGeneral (ShadowLHSDotRes), (DiffeRet), $rhs, (ShadowLHSDotDim), (ResultDotPrec)), (ShadowLHSTranspose)),
                        (Transpose (TypeOf $rhs), (DotGeneral (ShadowRHSDotRes), $lhs, (DiffeRet), (ShadowRHSDotDim), (ResultDotPrec)), (ShadowRHSTranspose))
                      ],
                     (Add (SelectIfActive $lhs, (DotGeneral (ResultTypes), (Shadow $lhs), $rhs, (ResultDotDim), (ResultDotPrec)), (HLOConstantFP<"0">)), (SelectIfActive $rhs, (DotGeneral (ResultTypes), $lhs, (Shadow $rhs), (ResultDotDim), (ResultDotPrec)), (HLOConstantFP<"0">)))
                  >;

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

def : HLODerivative<"TransposeOp", (Op $x),
                    [
                        (Transpose (TypeOf $x), (DiffeRet), (InversePermutation)),
                    ],
                    (SelectIfActive $x, (Transpose (ResultTypes), (Shadow $x), (Permutation)), (HLOConstantFP<"0">))
                  >;


def Convolution : HLOInst<"ConvolutionOp">;

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

def : HLODerivative<"SelectOp", (Op $cond, $lhs, $rhs),
                    [
                        (AssertingInactiveArg),
                        (Select $cond, (DiffeRet), (HLOConstantFP<"0">)),
                        (Select $cond, (HLOConstantFP<"0">), (DiffeRet)),
                      ],
                      (Select $cond, (SelectIfActive $lhs, (Shadow $lhs), (HLOConstantFP<"0">)), (SelectIfActive $rhs, (Shadow $rhs), (HLOConstantFP<"0">)))
                  >;

def : HLORegionTerminatorOp<"ReturnOp">;
