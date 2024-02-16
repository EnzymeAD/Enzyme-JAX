// Symlink-forwarded from main Enzyme.
include "src/enzyme_ad/jax/Implementations/Common.td"

class ConstantFP<string val, string dialect_, string op_> : Operation</*primal*/0, /*shadow*/0> {
  string value = val;
  string dialect = dialect_;
  string opName = op_;
}
