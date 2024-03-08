include "src/enzyme_ad/jax/Implementations/Common.td"

class HLODerivative<string opName_, dag patternToMatch, list<dag> resultOps, dag forwardOps=(ForwardFromSummedReverse)> : MLIRDerivative<"stablehlo", opName_, patternToMatch, resultOps, forwardOps>;

class HLOInst<string m, string postopt=""> : Inst<m, "stablehlo", postopt>;

class HLOMemoryIdentityOp<string opName_, list<int> ptrargs_, list<int> storedargs_ = [], dag patternToMatch=(Unimplemented), list<dag> reverse_ = []>  : MemoryIdentityOp<"stablehlo", opName_, ptrargs_, storedargs_, patternToMatch, reverse_>;

class HLOReadOnlyIdentityOp<string opName_, list<int> ptrargs_ = [0], dag patternToMatch=(Unimplemented), list<dag> reverse_ = []> : ReadOnlyIdentityOp<"stablehlo", opName_, ptrargs_, patternToMatch, reverse_>;

class HLOControlFlowOp<string opName_, string impl_> : ControlFlowOp<"stablehlo", opName_, impl_>;

class HLOConstantFP<string m> : ConstantFP<m, "stablehlo", "ConstantOp", "mlir::ElementsAttr">;

class HLORegionTerminatorOp<string m> : RegionTerminatorOp<"stablehlo", m>;

include "HLODerivatives.td"
