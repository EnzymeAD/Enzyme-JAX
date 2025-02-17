include "src/enzyme_ad/jax/Implementations/Common.td"

class HLODerivative<string opName_, dag patternToMatch, list<dag> resultOps, dag forwardOps=(ForwardFromSummedReverse)> : MLIRDerivative<"stablehlo", opName_, patternToMatch, resultOps, forwardOps>;

class HLOInst<string m, string postopt="", string preopt=""> : Inst<m, "stablehlo", postopt, preopt>;

class HLOMemoryIdentityOp<string opName_, list<int> ptrargs_, list<int> storedargs_ = [], dag patternToMatch=(Unimplemented), list<dag> reverse_ = []>  : MemoryIdentityOp<"stablehlo", opName_, ptrargs_, storedargs_, patternToMatch, reverse_>;

class HLOReadOnlyIdentityOp<string opName_, list<int> ptrargs_ = [0], dag patternToMatch=(Unimplemented), list<dag> reverse_ = []> : ReadOnlyIdentityOp<"stablehlo", opName_, ptrargs_, patternToMatch, reverse_>;

class HLOControlFlowOp<string opName_, string impl_> : ControlFlowOp<"stablehlo", opName_, impl_>;

class HLOConstantFP<string m> : ConstantFP<m, "stablehlo", "ConstantOp", "mlir::ElementsAttr">;

class HLORegionTerminatorOp<string m> : RegionTerminatorOp<"stablehlo", m>;

class HLOInactiveOp<string m> : InactiveOp<"stablehlo", m>;

include "HLODerivatives.td"

def : HLODerivative<"EinsumOp", (Op $lhs, $rhs),
                    [
                        // TODO add support for complex numbers by conjugating `$rhs` and `$lhs` (only if complex tensors)
                        (Einsum (TypeOf $lhs), (DiffeRet), $rhs, (GradEinsumConfigLhs)),
                        (Einsum (TypeOf $rhs), (DiffeRet), $lhs, (GradEinsumConfigRhs)),
                    ],
                    (Add (SelectIfActive $lhs, (Einsum (ResultTypes), (Shadow $lhs), $rhs, (EinsumConfig)), (HLOConstantFP<"0">)), (SelectIfActive $rhs, (Einsum (ResultTypes), $lhs, (Shadow $rhs), (EinsumConfig)), (HLOConstantFP<"0">)))
                  >;

def : HLODerivative<"UnaryEinsumOp", (Op $x),
                    [
                        // TODO maybe replace `UnaryEinsum` for `Reshape` + `BroadcastInDim`
                        (UnaryEinsum (TypeOf $x), (DiffeRet), (GradUnaryEinsumConfig)),
                    ],
                    (UnaryEinsum (ResultTypes), (Shadow $x), (EinsumConfig))
                  >;

def : ControlFlowOp<"stablehlo", "CaseOp", [{
  Operation *createWithShadows(Operation *op, OpBuilder &builder,
                               MGradientUtils *gutils, Operation *original,
                               ValueRange remappedOperands,
                               TypeRange rettys) const {
    return builder.create<stablehlo::CaseOp>(original->getLoc(), rettys,
                                        remappedOperands, original->getAttrs(), cast<stablehlo::CaseOp>(op).getBranches().size());
  }
}]>;
