class InactiveOp<string dialect_, string opName_> {
  string dialect = dialect_;
  string opName = opName_;
}

class AllocationOp<string dialect_, string opName_> {
  string dialect = dialect_;
  string opName = opName_;
}

class ControlFlowOp<string dialect_, string opName_, string impl_> {
  string dialect = dialect_;
  string opName = opName_;
  string impl = impl_;
}

class MemoryIdentityOp<string dialect_, string opName_, list<int> ptrargs_, list<int> storedargs_ = []> {
  string dialect = dialect_;
  string opName = opName_;
  list<int> ptrargs = ptrargs_;
  list<int> storedargs = storedargs_;
}

class ReadOnlyIdentityOp<string dialect_, string opName_, list<int> ptrargs_> : MemoryIdentityOp<dialect_, opName_, ptrargs_>;

class BranchOp<string dialect_, string opName_> {
  string dialect = dialect_;
  string opName = opName_;
}

class RegionTerminatorOp<string dialect_, string opName_> {
  string dialect = dialect_;
  string opName = opName_;
}

class ForwardFromSummedReverseInternal<int unused_> {
  int unused = unused_;
}
def ForwardFromSummedReverse : ForwardFromSummedReverseInternal<0>;


class MLIRDerivative<string dialect_, string opName_, dag patternToMatch, list<dag> resultOps, dag forwardOps=(ForwardFromSummedReverse)> {
  string dialect = dialect_;
  string opName = opName_;
  dag PatternToMatch = patternToMatch;
  list<dag> ArgDerivatives = resultOps;
  dag ArgDuals = forwardOps;
}

class Operation<bit usesPrimal_, bit usesShadow_, bit usesCustom_=0> {
  bit usesPrimal = usesPrimal_;
  bit usesShadow = usesShadow_;
  bit usesCustom = usesCustom_;
}

class DiffeRetIndex<list<int> indices_> {
  list<int> indices = indices_;
}
def DiffeRet : DiffeRetIndex<[-1]>;

def Shadow : Operation</*primal*/0, /*shadow*/1> {
}

class GlobalExpr<bit uses_primal, bit uses_shadow, string val> : Operation<uses_primal, uses_shadow>{
  string value = val;
}

class Inst<string mnemonic, string dialect_>  : Operation</*primal*/1, /*shadow*/0> {
  string name = mnemonic;
  string dialect = dialect_;
}


class ConstantFP<string val, string dialect_, string op_> : Operation</*primal*/0, /*shadow*/0> {
  string value = val;
  string dialect = dialect_;
  string opName = op_;
}

def SelectIfActive : Operation</*primal*/0, /*shadow*/0, /*custom*/1> {

}

def Op {
}

def ResultTypes : GlobalExpr</*needsprimal*/0, /*needsshadow*/0, "op->getResultTypes()">;


