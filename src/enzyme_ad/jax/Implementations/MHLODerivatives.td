include "Common.td"

class HLODerivative<string opName_, dag patternToMatch, list<dag> resultOps> : MLIRDerivative<"mhlo", opName_, patternToMatch, resultOps>;

class HLOInst<string m> : Inst<m, "mhlo">;

class HLOReadOnlyIdentityOp<string opName_, list<int> ptrargs_ = [0]> : ReadOnlyIdentityOp<"mhlo", opName_, ptrargs_>;

include "HLODerivatives.td"
