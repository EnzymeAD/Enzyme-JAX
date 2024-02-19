include "src/enzyme_ad/jax/Implementations/Common.td"

class HLODerivative<string opName_, dag patternToMatch, list<dag> resultOps, dag forwardOps=(ForwardFromSummedReverse)> : MLIRDerivative<"mhlo", opName_, patternToMatch, resultOps, forwardOps>;

class HLOInst<string m> : Inst<m, "mhlo">;

class HLOReadOnlyIdentityOp<string opName_, list<int> ptrargs_ = [0]> : ReadOnlyIdentityOp<"mhlo", opName_, ptrargs_>;

class HLOControlFlowOp<string opName_, string impl_> : ControlFlowOp<"mhlo", opName_, impl_>;

class HLOConstantFP<string m> : ConstantFP<m, "mhlo", "ConstantOp", "mlir::ElementsAttr">;

class HLORegionTerminatorOp<string m> : RegionTerminatorOp<"mhlo", m>;

include "HLODerivatives.td"
