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

class MLIRDerivative<string dialect_, string opName_, dag patternToMatch, list<dag> resultOps> {
  string dialect = dialect_;
  string opName = opName_;
  dag PatternToMatch = patternToMatch;
  list<dag> ArgDerivatives = resultOps;
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

class Inst<string mnemonic, string dialect_>  : Operation</*primal*/1, /*shadow*/0> {
  string name = mnemonic;
  string dialect = dialect_;
}


class ConstantFP<string val, string dialect_, string op_> : Operation</*primal*/0, /*shadow*/0> {
  string value = val;
  string dialect = dialect_;
  string opName = op_;
}

def Op {
}
