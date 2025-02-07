include "src/enzyme_ad/jax/Implementations/Common.td"

class HLODerivative<string opName_, dag patternToMatch, list<dag> resultOps, dag forwardOps=(ForwardFromSummedReverse)> : MLIRDerivative<"mhlo", opName_, patternToMatch, resultOps, forwardOps>;

class HLOInst<string m, string postopt="", string preopt=""> : Inst<m, "mhlo", postopt, preopt>;

class HLOMemoryIdentityOp<string opName_, list<int> ptrargs_, list<int> storedargs_ = [], dag patternToMatch=(Unimplemented), list<dag> reverse_ = []>  : MemoryIdentityOp<"mhlo", opName_, ptrargs_, storedargs_, patternToMatch, reverse_>;

class HLOReadOnlyIdentityOp<string opName_, list<int> ptrargs_ = [0], dag patternToMatch=(Unimplemented), list<dag> reverse_ = []> : ReadOnlyIdentityOp<"mhlo", opName_, ptrargs_, patternToMatch, reverse_>;

class HLOControlFlowOp<string opName_, string impl_> : ControlFlowOp<"mhlo", opName_, impl_>;

class HLOConstantFP<string m> : ConstantFP<m, "mhlo", "ConstantOp", "mlir::ElementsAttr">;

class HLORegionTerminatorOp<string m> : RegionTerminatorOp<"mhlo", m>;

class HLOInactiveOp<string m> : InactiveOp<"mhlo", m>;

include "HLODerivatives.td"
