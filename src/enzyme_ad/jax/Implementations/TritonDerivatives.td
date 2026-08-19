include "src/enzyme_ad/jax/Implementations/Common.td"

class TritonDerivative<string opName_, dag patternToMatch, list<dag> resultOps, dag forwardOps=(ForwardFromSummedReverse)> : MLIRDerivative<"triton", opName_, patternToMatch, resultOps, forwardOps>;

class TritonInst<string m, string postopt="", string preopt=""> : Inst<m, "triton", postopt, preopt>;

class TritonMemoryIdentityOp<string opName_, list<int> ptrargs_, list<int> storedargs_ = [], dag patternToMatch=(Unimplemented), list<dag> reverse_ = []>  : MemoryIdentityOp<"triton", opName_, ptrargs_, storedargs_, patternToMatch, reverse_>;

class TritonReadOnlyIdentityOp<string opName_, list<int> ptrargs_ = [0], dag patternToMatch=(Unimplemented), list<dag> reverse_ = []> : ReadOnlyIdentityOp<"triton", opName_, ptrargs_, patternToMatch, reverse_>;

class ArithConstantFP<string m> : ConstantFP<m, "arith", "ConstantOp", "mlir::ElementsAttr">;

class TritonInactiveOp<string m> : InactiveOp<"triton", m>;

class TritonReturnOp<string m> : ReturnOp<"triton", m>;

def FpToFp : TritonInst<"FpToFpOp">;
def PreciseDivF : TritonInst<"PreciseDivFOp">;
def MakeRange : TritonInst<"MakeRangeOp">;
def Dot : TritonInst<"DotOp", "->getResult(0)">;

def : TritonReturnOp<"ReturnOp">;

def : TritonInactiveOp<"AssertOp">;
def : TritonInactiveOp<"GetProgramIdOp">;
def : TritonInactiveOp<"MakeRangeOp">;
def : TritonInactiveOp<"PrintOp">;

def : ReadOnlyIdentityOp<"triton", "AddPtrOp", [0]>;
def : ReadOnlyIdentityOp<"triton", "BroadcastOp", [0]>;
def : ReadOnlyIdentityOp<"triton", "ExpandDimsOp", [0]>;
def : MemoryIdentityOp<"triton", "LoadOp", [0], [2]>;
def : ReadOnlyIdentityOp<"triton", "MakeTensorDescOp", [0]>;
def : ReadOnlyIdentityOp<"triton", "DescriptorLoadOp", [0]>;
def : ReadOnlyIdentityOp<"triton", "SplatOp", [0]>;
def : ReadOnlyIdentityOp<"triton", "TransOp", [0]>;
def : MemoryIdentityOp<"triton", "StoreOp", [0], [1]>;
def : MemoryIdentityOp<"triton", "DescriptorStoreOp", [0], [1]>;

def FpToFpRoundingMode : GlobalExpr</*needsprimal*/0, /*needsshadow*/0, [{
  op.getRoundingAttr();
}]>;

def DotInputPrecision : GlobalExpr</*needsprimal*/0, /*needsshadow*/0, [{
  op.getInputPrecisionAttr();
}]>;

def DotMaxNumImpreciseAcc : GlobalExpr</*needsprimal*/0, /*needsshadow*/0, [{
  op.getMaxNumImpreciseAccAttr();
}]>;

// AssertingInactiveArg would also suppress forward activity for these operands,
// so report unsupported derivatives only when reverse mode reaches this rule.
def UnimplementedDotOperandReverse
    : GlobalExpr</*needsprimal*/0, /*needsshadow*/0, [{
  op.emitError() << "reverse-mode derivatives of tt.dot matrix operands are "
                    "not implemented";
  return failure();
  mlir::Value{};
}]>;

def : TritonDerivative<"FpToFpOp", (Op $x),
                       [
                         (FpToFp (TypeOf $x), (DiffeRet), (FpToFpRoundingMode))
                       ]
                     >;

// tt.dot computes a @ b + c, so its tangent is da @ b + a @ db + dc.
// Chain both products through the accumulator to avoid a separate add.
def : TritonDerivative<"DotOp", (Op $a, $b, $c),
                       [
                         (UnimplementedDotOperandReverse),
                         (UnimplementedDotOperandReverse),
                         (DiffeRet)
                       ],
                       (SelectIfActive $b,
                         (Dot $a, (Shadow $b),
                           (SelectIfActive $a,
                             (Dot (Shadow $a), $b,
                               (SelectIfActive $c,
                                 (Shadow $c),
                                 (ArithConstantFP<"0"> $c)),
                               (DotInputPrecision),
                               (DotMaxNumImpreciseAcc)),
                             (SelectIfActive $c,
                               (Shadow $c),
                               (ArithConstantFP<"0"> $c))),
                           (DotInputPrecision),
                           (DotMaxNumImpreciseAcc)),
                         (SelectIfActive $a,
                           (Dot (Shadow $a), $b,
                             (SelectIfActive $c,
                               (Shadow $c),
                               (ArithConstantFP<"0"> $c)),
                             (DotInputPrecision),
                             (DotMaxNumImpreciseAcc)),
                           (SelectIfActive $c,
                             (Shadow $c),
                             (ArithConstantFP<"0"> $c))))
                     >;

// Note: ReduceOp is handled via custom C++ AutoDiffOpInterface in
// TritonAutoDiffOpInterfaceImpl.cpp because Triton's ReduceOp doesn't
// implement RegionBranchOpInterface required by ControlFlowOp.

def : RegionTerminatorOp<"triton", "ReduceReturnOp">;
