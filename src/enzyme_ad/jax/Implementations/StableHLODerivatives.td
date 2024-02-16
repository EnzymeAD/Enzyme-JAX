include "Common.td"

class HLODerivative<string opName_, dag patternToMatch, list<dag> resultOps> : MLIRDerivative<"stablehlo", opName_, patternToMatch, resultOps>;

class HLOInst<string m> : Inst<m, "stablehlo">;

class HLOReadOnlyIdentityOp<string opName_, list<int> ptrargs_ = [0]> : ReadOnlyIdentityOp<"stablehlo", opName_, ptrargs_>;

class HLOControlFlowOp<string opName_, string impl_> : ControlFlowOp<"stablehlo", opName_, impl_>;


include "HLODerivatives.td"
