def Add : HLOInst<"AddOp">;
def Sub : HLOInst<"SubtractOp">;
def Neg : HLOInst<"NegOp">;
def Mul : HLOInst<"MulOp">;
def Div : HLOInst<"DivOp">;
def Rem : HLOInst<"RemainderOp">;
def Pow : HLOInst<"PowOp">;
def Log : HLOInst<"LogOp">;
def Cos : HLOInst<"CosineOp">;
def Sin : HLOInst<"SineOp">;
def Sqrt : HLOInst<"SqrtOp">;
def Exp : HLOInst<"ExpOp">;

def Dot : HLOInst<"DotGeneralOp", "->getResult(0)">;

def Compare : HLOInst<"CompareOp">;
def Select : HLOInst<"SelectOp">;


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

def LT : GlobalExpr</*needsprimal*/0, /*needsshadow*/0, "ComparisonDirection::LT">;
def : HLODerivative<"MaxOp", (Op $x, $y),
                  [
                    (Select (Compare $x, $y, (LT)), (HLOConstantFP<"0"> $x), (DiffeRet)),
                    (Select (Compare $x, $y, (LT)), (DiffeRet), (HLOConstantFP<"0"> $x))
                  ],
                  (Select (Compare $x, $y, (LT)), (SelectIfActive $y, (Shadow $y), (HLOConstantFP<"0"> $y)), (SelectIfActive $x, (Shadow $x), (HLOConstantFP<"0"> $x)))
                  >;

def Transpose : HLOInst<"TransposeOp">;
def Reshape : HLOInst<"ReshapeOp">;
def : HLOReadOnlyIdentityOp<"ReshapeOp", [0], (Op $x), [(Reshape (TypeOf $x), (DiffeRet))]>;

def Slice : HLOInst<"SliceOp">;
def : HLOReadOnlyIdentityOp<"SliceOp">;

def Reduce : HLOInst<"ReduceOp">;
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
                        (Transpose (TypeOf $lhs), (Dot (ShadowLHSDotRes), (DiffeRet), $rhs, (ShadowLHSDotDim), (ResultDotPrec)), (ShadowLHSTranspose)),
                        (Transpose (TypeOf $rhs), (Dot (ShadowRHSDotRes), $lhs, (DiffeRet), (ShadowRHSDotDim), (ResultDotPrec)), (ShadowRHSTranspose))
                      ],
                     (Add (SelectIfActive $lhs, (Dot (ResultTypes), (Shadow $lhs), $rhs, (ResultDotDim), (ResultDotPrec)), (HLOConstantFP<"0">)), (SelectIfActive $rhs, (Dot (ResultTypes), $lhs, (Shadow $rhs), (ResultDotDim), (ResultDotPrec)), (HLOConstantFP<"0">)))
                  >;

def : HLORegionTerminatorOp<"ReturnOp">;
