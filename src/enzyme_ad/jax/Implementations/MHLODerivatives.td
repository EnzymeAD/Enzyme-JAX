include "Common.td"

class HLODerivative<string opName_, dag patternToMatch, list<dag> resultOps> : MLIRDerivative<"mhlo", opName_, patternToMatch, resultOps>;

class HLOInst<string m> : Inst<m, "mhlo">;

class HLOReadOnlyIdentityOp<string opName_, list<int> ptrargs_ = [0]> : ReadOnlyIdentityOp<"mhlo", opName_, ptrargs_>;

class HLOControlFlowOp<string opName_, string impl_> : ControlFlowOp<"mhlo", opName_, impl_>;

class HLOConstantFP<string m> : ConstantFP<m, "mhlo", "ConstantOp">;

include "HLODerivatives.td"
