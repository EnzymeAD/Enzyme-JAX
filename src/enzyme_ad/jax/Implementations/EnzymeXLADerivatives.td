include "src/enzyme_ad/jax/Implementations/Common.td"

class EnzymeXLADerivative<string opName_, dag patternToMatch, list<dag> resultOps, dag forwardOps=(ForwardFromSummedReverse)> : MLIRDerivative<"enzymexla", opName_, patternToMatch, resultOps, forwardOps>;

class EnzymeXLAInst<string m, string postopt="", string preopt=""> : Inst<m, "enzymexla", postopt, preopt>;

class EnzymeXLAMemoryIdentityOp<string opName_, list<int> ptrargs_, list<int> storedargs_ = [], dag patternToMatch=(Unimplemented), list<dag> reverse_ = []>  : MemoryIdentityOp<"enzymexla", opName_, ptrargs_, storedargs_, patternToMatch, reverse_>;

class EnzymeXLAReadOnlyIdentityOp<string opName_, list<int> ptrargs_ = [0], dag patternToMatch=(Unimplemented), list<dag> reverse_ = []> : ReadOnlyIdentityOp<"enzymexla", opName_, ptrargs_, patternToMatch, reverse_>;

class EnzymeXLAControlFlowOp<string opName_, string impl_> : ControlFlowOp<"enzymexla", opName_, impl_>;

class EnzymeXLAConstantFP<string m> : ConstantFP<m, "enzymexla", "ConstantOp", "mlir::ElementsAttr">;

class EnzymeXLARegionTerminatorOp<string m> : RegionTerminatorOp<"enzymexla", m>;

class EnzymeXLAInactiveOp<string m> : InactiveOp<"enzymexla", m>;

def : EnzymeXLAControlFlowOp<"GPUWrapperOp", [{
  Operation *createWithShadows(Operation *op, OpBuilder &builder,
                               MGradientUtils *gutils, Operation *original,
                               ValueRange remappedOperands,
                               TypeRange rettys) const {
    return enzymexla::GPUWrapperOp::create(builder, original->getLoc(),
                                        remappedOperands);
  }
}]>;

class HLOConstantFP<string m> : ConstantFP<m, "stablehlo", "ConstantOp", "mlir::ElementsAttr">;

// Required operations from the EnzymeXLA dialect
def Gelu : EnzymeXLAInst<"GeluOp">;
def Relu : EnzymeXLAInst<"ReluOp">;

// Required operations from StableHLO/CHLO dialects
def Add : Inst<"AddOp", "stablehlo">;
def Sub : Inst<"SubtractOp", "stablehlo">;
def Mul : Inst<"MulOp", "stablehlo">;
def Div : Inst<"DivOp", "stablehlo">;
def Exp : Inst<"ExpOp", "stablehlo">;
def Tanh : Inst<"TanhOp", "stablehlo">;
def Pow : Inst<"PowOp", "stablehlo">;
def Logistic : Inst<"LogisticOp", "stablehlo">;
def Compare : Inst<"CompareOp", "stablehlo">;
def Select : Inst<"SelectOp", "stablehlo">;
def Erf : Inst<"ErfOp", "chlo">;
def Softplus : Inst<"SoftplusOp", "enzymexla">;
def Slice : Inst<"SliceOp", "stablehlo">;
def Pad : Inst<"PadOp", "stablehlo">;
def Rotate : Inst<"RotateOp", "enzymexla">;
def Extend : Inst<"ExtendOp", "enzymexla">;

def GT : GlobalExpr</*needsprimal*/0, /*needsshadow*/0, "ComparisonDirection::GT">;
def EQ : GlobalExpr</*needsprimal*/0, /*needsshadow*/0, "ComparisonDirection::EQ">;

def GeluApproximation : GlobalExpr</*needsprimal*/0, /*needsshadow*/0, [{
  op.getGeluApproximationAttr();
}]>;

def SelectIfGeluNone : StaticSelect<"cast<enzymexla::GeluOp>(op).getGeluApproximation() == enzymexla::GeluApproximation::NONE">;
def SelectIfGeluTanh : StaticSelect<"cast<enzymexla::GeluOp>(op).getGeluApproximation() == enzymexla::GeluApproximation::TANH">;

// Others
def CheckedMul : SubRoutine<(Op $diffret, $x),
                  (
                    SelectIfStrongZero
                    (Select (Compare $diffret, (HLOConstantFP<"0">):$zero, (EQ)), $zero, (Mul $diffret, $x)),
                    (Mul $diffret, $x)
                  )>;

// Derivative rules
// ReLU:
//   relu(x) = max(0, x)
//   d/dx relu(x) = 1 if x > 0 else 0
def : EnzymeXLADerivative<"ReluOp", (Op $input),
                    [
                      (Select
                        (Compare $input, (HLOConstantFP<"0"> $input), (GT)),
                        (DiffeRet),
                        (HLOConstantFP<"0"> $input)
                      )
                    ],
                    (Relu (Shadow $input))
                  >;

def : EnzymeXLADerivative<"SoftplusOp", (Op $input),
                    [
                      (CheckedMul (DiffeRet), (Logistic $input))
                    ],
                    (Softplus (Shadow $input))
                  >;

// GELU (x):
//   NONE:    gelu(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
//            d/dx = 0.5 * (1 + erf(x / sqrt(2))) + x * exp(-x^2 / 2) / sqrt(2π)
//   TANH:    gelu(x) = 0.5 * x * (1 + tanh( sqrt(2/pi) * (x + 0.044715 * x^3) ))
//            d/dx = 0.5 * (1 + tanh(u))
//                   + 0.5 * x * (1 - tanh(u)^2) * sqrt(2/pi) * (1 + 0.134145 * x^2),
//                   where u = sqrt(2/pi) * (x + 0.044715 * x^3)
//   SIGMOID: gelu(x) = x * sigmoid(1.702 * x)
//            d/dx = sigmoid(1.702*x) + 1.702*x*sigmoid(1.702*x)*(1 - sigmoid(1.702*x))
def : EnzymeXLADerivative<"GeluOp", (Op $input),
                    [
                      (CheckedMul
                        (DiffeRet),
                        (SelectIfGeluNone
                          (Add
                            (Mul
                              (HLOConstantFP<"0.5"> $input),
                              (Add
                                (HLOConstantFP<"1"> $input),
                                (Erf (Div $input, (HLOConstantFP<"1.4142135623730951"> $input)))
                              )
                            ),
                            (Mul
                              $input,
                              (Mul
                                (HLOConstantFP<"0.3989422804014327"> $input),
                                (Exp (Mul (HLOConstantFP<"-0.5"> $input), (Mul $input, $input)))
                              )
                            )
                          ),
                          (SelectIfGeluTanh
                            (Add
                              (Mul
                                (HLOConstantFP<"0.5"> $input),
                                (Add
                                  (HLOConstantFP<"1"> $input),
                                  (Tanh
                                    (Mul
                                      (HLOConstantFP<"0.7978845608028654"> $input),
                                      (Add
                                        $input,
                                        (Mul (HLOConstantFP<"0.044715"> $input), (Pow $input, (HLOConstantFP<"3"> $input)))
                                      )
                                    )
                                  )
                                )
                              ),
                              (Mul
                                (Mul (HLOConstantFP<"0.5"> $input), $input),
                                (Mul
                                  (Sub
                                    (HLOConstantFP<"1"> $input),
                                    (Mul
                                      (Tanh
                                        (Mul
                                          (HLOConstantFP<"0.7978845608028654"> $input),
                                          (Add
                                            $input,
                                            (Mul (HLOConstantFP<"0.044715"> $input), (Pow $input, (HLOConstantFP<"3"> $input)))
                                          )
                                        )
                                      ),
                                      (Tanh
                                        (Mul
                                          (HLOConstantFP<"0.7978845608028654"> $input),
                                          (Add
                                            $input,
                                            (Mul (HLOConstantFP<"0.044715"> $input), (Pow $input, (HLOConstantFP<"3"> $input)))
                                          )
                                        )
                                      )
                                    )
                                  ),
                                  (Mul
                                    (HLOConstantFP<"0.7978845608028654"> $input),
                                    (Add
                                      (HLOConstantFP<"1"> $input),
                                      (Mul (HLOConstantFP<"0.134145"> $input), (Mul $input, $input))
                                    )
                                  )
                                )
                              )
                            ),
                            (Add
                              (Logistic (Mul (HLOConstantFP<"1.702"> $input), $input)),
                              (Mul
                                (Mul (HLOConstantFP<"1.702"> $input), $input),
                                (Mul
                                  (Logistic (Mul (HLOConstantFP<"1.702"> $input), $input)),
                                  (Sub
                                    (HLOConstantFP<"1"> $input),
                                    (Logistic (Mul (HLOConstantFP<"1.702"> $input), $input))
                                  )
                                )
                              )
                            )
                          )
                        )
                      )
                    ],
                    (Gelu (Shadow $input), (GeluApproximation))
                  >;

// Helpers
def ExtendLhsType : GlobalExpr</*needsprimal*/0, /*needsshadow*/0, [{
    auto Ty = cast<RankedTensorType>(op.getResult().getType());
    SmallVector<int64_t> newShape(Ty.getShape());
    newShape[op.getDimension()] = (int64_t)op.getLhs();
    Ty.clone(newShape);
}]>;

def LhsSliceStart : GlobalExpr</*needsprimal*/0, /*needsshadow*/0, [{
    auto Ty = cast<RankedTensorType>(op.getResult().getType());
    SmallVector<int64_t> starts(Ty.getRank(), 0);
    getI64Attr(builder, starts);
}]>;
def LhsSliceLimit : GlobalExpr</*needsprimal*/0, /*needsshadow*/0, [{
    auto Ty = cast<RankedTensorType>(op.getResult().getType());
    SmallVector<int64_t> limits(Ty.getShape());
    limits[op.getDimension()] = (int64_t)op.getLhs();
    getI64Attr(builder, limits);
}]>;
def LhsSliceStride : GlobalExpr</*needsprimal*/0, /*needsshadow*/0, [{
    auto Ty = cast<RankedTensorType>(op.getResult().getType());
    SmallVector<int64_t> strides(Ty.getRank(), 1);
    getI64Attr(builder, strides);
}]>;
def LhsPaddingLow : GlobalExpr</*needsprimal*/0, /*needsshadow*/0, [{
    auto Ty = cast<RankedTensorType>(op.getResult().getType());
    SmallVector<int64_t> lows(Ty.getRank(), 0);
    getI64Attr(builder, lows);
}]>;
def LhsPaddingHigh : GlobalExpr</*needsprimal*/0, /*needsshadow*/0, [{
    auto Ty = cast<ShapedType>(op.getOperand().getType());
    int64_t N = Ty.getShape()[op.getDimension()];
    SmallVector<int64_t> highs(Ty.getRank(), 0);
    highs[op.getDimension()] = N - (int64_t)op.getLhs();
    getI64Attr(builder, highs);
}]>;
def LhsPaddingInterior : GlobalExpr</*needsprimal*/0, /*needsshadow*/0, [{
    auto Ty = cast<RankedTensorType>(op.getResult().getType());
    SmallVector<int64_t> interiors(Ty.getRank(), 0);
    getI64Attr(builder, interiors);
}]>;

def ExtendRhsType : GlobalExpr</*needsprimal*/0, /*needsshadow*/0, [{
    auto Ty = cast<RankedTensorType>(op.getResult().getType());
    SmallVector<int64_t> newShape(Ty.getShape());
    newShape[op.getDimension()] = (int64_t)op.getRhs();
    Ty.clone(newShape);
}]>;
def RhsSliceStart : GlobalExpr</*needsprimal*/0, /*needsshadow*/0, [{
    auto Ty = cast<ShapedType>(op.getOperand().getType());
    int64_t N = Ty.getShape()[op.getDimension()];
    SmallVector<int64_t> starts(Ty.getRank(), 0);
    starts[op.getDimension()] = N + (int64_t)op.getLhs();
    getI64Attr(builder, starts);
}]>;
def RhsSliceLimit : GlobalExpr</*needsprimal*/0, /*needsshadow*/0, [{
    auto Ty = cast<ShapedType>(op.getResult().getType());
    int64_t N = Ty.getShape()[op.getDimension()];
    SmallVector<int64_t> limits(Ty.getShape());
    limits[op.getDimension()] = N;
    getI64Attr(builder, limits);
}]>;
def RhsSliceStride : GlobalExpr</*needsprimal*/0, /*needsshadow*/0, [{
    auto Ty = cast<ShapedType>(op.getResult().getType());
    SmallVector<int64_t> strides(Ty.getRank(), 1);
    getI64Attr(builder, strides);
}]>;
def RhsPaddingLow : GlobalExpr</*needsprimal*/0, /*needsshadow*/0, [{
    auto Ty = cast<ShapedType>(op.getOperand().getType());
    int64_t N = Ty.getShape()[op.getDimension()];
    SmallVector<int64_t> lows(Ty.getRank(), 0);
    lows[op.getDimension()] = N - (int64_t)op.getRhs();
    getI64Attr(builder, lows);
}]>;
def RhsPaddingHigh : GlobalExpr</*needsprimal*/0, /*needsshadow*/0, [{
    auto Ty = cast<ShapedType>(op.getOperand().getType());
    SmallVector<int64_t> highs(Ty.getRank(), 0);
    getI64Attr(builder, highs);
}]>;
def RhsPaddingInterior : GlobalExpr</*needsprimal*/0, /*needsshadow*/0, [{
    auto Ty = cast<ShapedType>(op.getOperand().getType());
    SmallVector<int64_t> interior(Ty.getRank(), 0);
    getI64Attr(builder, interior);
}]>;

def SliceStart : GlobalExpr</*needsprimal*/0, /*needsshadow*/0, [{
    auto Ty = cast<ShapedType>(op.getOperand().getType());
    SmallVector<int64_t> starts(Ty.getRank(), 0);
    starts[op.getDimension()] = op.getLhs();
    getI64Attr(builder, starts);
}]>;
def SliceLimit : GlobalExpr</*needsprimal*/0, /*needsshadow*/0, [{
    auto Ty = cast<ShapedType>(op.getResult().getType());
    int64_t N = Ty.getShape()[op.getDimension()];
    SmallVector<int64_t> limits(Ty.getShape());
    limits[op.getDimension()] = N - (int64_t)op.getRhs();
    getI64Attr(builder, limits);
}]>;
def SliceStride : GlobalExpr</*needsprimal*/0, /*needsshadow*/0, [{
    auto Ty = cast<ShapedType>(op.getOperand().getType());
    SmallVector<int64_t> strides(Ty.getRank(), 1);
    getI64Attr(builder, strides);
}]>;

def ZeroVal : GlobalExpr</*needsprimal*/0, /*needsshadow*/0, [{
    auto Ty = cast<ShapedType>(op.getResult().getType()).clone(llvm::ArrayRef<int64_t>{});
    stablehlo::ConstantOp::create(builder, op.getLoc(), Ty, cast<ElementsAttr>(makeAttr(Ty, 0)));
}]>;

def ExtendLhs : GlobalExpr</*needsprimal*/0, /*needsshadow*/0, [{
    op.getLhsAttr();
}]>;
def ExtendRhs : GlobalExpr</*needsprimal*/0, /*needsshadow*/0, [{
    op.getRhsAttr();
}]>;
def ExtendDim : GlobalExpr</*needsprimal*/0, /*needsshadow*/0, [{
    op.getDimensionAttr();
}]>;

// Comms ops
def : EnzymeXLADerivative<"ExtendOp", (Op $input), [
    (Add
        (Add
            (Pad (TypeOf $input),
                 (Slice (ExtendLhsType), (DiffeRet),
                        (LhsSliceStart), (LhsSliceLimit),
                        (LhsSliceStride)),
                 (ZeroVal),
                 (LhsPaddingLow),
                 (LhsPaddingHigh),
                 (LhsPaddingInterior)),
            (Pad (TypeOf $input),
                 (Slice (ExtendRhsType), (DiffeRet),
                        (RhsSliceStart), (RhsSliceLimit),
                        (RhsSliceStride)),
                 (ZeroVal),
                 (RhsPaddingLow),
                 (RhsPaddingHigh),
                 (RhsPaddingInterior))),
        (Slice (TypeOf $input), (DiffeRet),
               (SliceStart), (SliceLimit),
               (SliceStride)))
], (Extend (Shadow $input), (ExtendLhs), (ExtendRhs), (ExtendDim))>;

def RotateRevAmount : GlobalExpr</*needsprimal*/0, /*needsshadow*/0, [{
    auto Ty = cast<ShapedType>(op.getOperand().getType());
    uint32_t N = Ty.getShape()[op.getDimension()];
    uint32_t revAmount = N - (op.getAmount() % N);
    builder.getI32IntegerAttr(revAmount);
}]>;
def RotateAmount : GlobalExpr</*needsprimal*/0, /*needsshadow*/0, [{
    op.getAmountAttr();
}]>;
def RotateDim : GlobalExpr</*needsprimal*/0, /*needsshadow*/0, [{
    op.getDimensionAttr();
}]>;
def : EnzymeXLADerivative<"RotateOp", (Op $input), [
    (Rotate (TypeOf $input), (DiffeRet), (RotateRevAmount), (RotateDim))
], (Rotate (TypeOf $input), (Shadow $input), (RotateAmount), (RotateDim))>;

def RotateLhsPaddingLow : GlobalExpr</*needsprimal*/0, /*needsshadow*/0, [{
    auto Ty = cast<ShapedType>(op.getOperand().getType());
    int64_t N = Ty.getShape()[op.getDimension()];
    SmallVector<int64_t> lows(Ty.getRank(), 0);
    lows[op.getDimension()] = N - (int64_t)op.getLhs();
    getI64Attr(builder, lows);
}]>;
def RotateLhsPaddingHigh : GlobalExpr</*needsprimal*/0, /*needsshadow*/0, [{
    auto Ty = cast<ShapedType>(op.getOperand().getType());
    SmallVector<int64_t> highs(Ty.getRank(), 0);
    getI64Attr(builder, highs);
}]>;
def RotateLhsPaddingInterior : GlobalExpr</*needsprimal*/0, /*needsshadow*/0, [{
    auto Ty = cast<ShapedType>(op.getOperand().getType());
    SmallVector<int64_t> interior(Ty.getRank(), 0);
    getI64Attr(builder, interior);
}]>;

def RotateRhsPaddingLow : GlobalExpr</*needsprimal*/0, /*needsshadow*/0, [{
    auto Ty = cast<ShapedType>(op.getOperand().getType());
    SmallVector<int64_t> lows(Ty.getRank(), 0);
    getI64Attr(builder, lows);
}]>;
def RotateRhsPaddingHigh : GlobalExpr</*needsprimal*/0, /*needsshadow*/0, [{
    auto Ty = cast<ShapedType>(op.getOperand().getType());
    int64_t N = Ty.getShape()[op.getDimension()];
    SmallVector<int64_t> highs(Ty.getRank(), 0);
    highs[op.getDimension()] = N - (int64_t)op.getRhs();
    getI64Attr(builder, highs);
}]>;
def RotateRhsPaddingInterior : GlobalExpr</*needsprimal*/0, /*needsshadow*/0, [{
    auto Ty = cast<ShapedType>(op.getOperand().getType());
    SmallVector<int64_t> interior(Ty.getRank(), 0);
    getI64Attr(builder, interior);
}]>;

def : EnzymeXLADerivative<"WrapOp", (Op $input), [
    (Add
        (Add
            (Pad (TypeOf $input),
                 (Slice (ExtendLhsType), (DiffeRet),
                        (LhsSliceStart), (LhsSliceLimit),
                        (LhsSliceStride)),
                 (ZeroVal),
                 (RotateLhsPaddingLow),
                 (RotateLhsPaddingHigh),
                 (RotateLhsPaddingInterior)),
            (Pad (TypeOf $input),
                 (Slice (ExtendRhsType), (DiffeRet),
                        (RhsSliceStart), (RhsSliceLimit),
                        (RhsSliceStride)),
                 (ZeroVal),
                 (RotateRhsPaddingLow),
                 (RotateRhsPaddingHigh),
                 (RotateRhsPaddingInterior))),
        (Slice (TypeOf $input), (DiffeRet),
               (SliceStart), (SliceLimit),
               (SliceStride)))
]>;
