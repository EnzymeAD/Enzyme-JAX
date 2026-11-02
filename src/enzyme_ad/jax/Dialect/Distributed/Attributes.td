#ifndef ENZYME_AD_JAX_DIALECT_DISTRIBUTED_ATTRIBUTES_TD
#define ENZYME_AD_JAX_DIALECT_DISTRIBUTED_ATTRIBUTES_TD

include "mlir/IR/BuiltinAttributes.td"
include "shardy/dialect/sdy/ir/attrs.td"

include "Dialect.td"

// Drop-in replacement for ArrayRefParameter that can also parse the empty list
// its printer emits. Prints and parses its own `[` `]` delimiters, so the
// assembly format must not repeat them. Implemented in Dialect.cpp.
class EmptyableArrayRefParameter<string arrayOf, string desc = ""> :
    ArrayRefParameter<arrayOf, desc> {
  let parser = [{ ::mlir::enzyme::distributed::parseEmptyableArrayRef<$0>($_parser) }];
  let printer = [{ ::mlir::enzyme::distributed::printEmptyableArrayRef($_printer, $_self) }];
}

// Per-tensor sharding metadata keyed by partitioning-axis indices.
// - dim_partitioning_axes: one DenseI64ArrayAttr per tensor dim, where each
//   entry stores zero or more indices into the owning op's partitioning_axes.
// - unreduced_axes: logical axis indices that remain unreduced. (We expect all tensor reduction indices to happen locally,
//   so this only stores logical/ non-local physical axes to reduce over)
def IndexedTensorShardingAttr : AttrDef<DistributedDialect, "IndexedTensorSharding"> {
  let summary = "Tensor sharding with integer partitioning-axis indices";
  let cppNamespace = "::mlir::enzyme::distributed";
  let mnemonic = "indexed_tensor_sharding";

  let parameters = (ins
    EmptyableArrayRefParameter<"DenseI64ArrayAttr">:$dim_partitioning_axes,
    "DenseI64ArrayAttr":$unreduced_axes
  );

  let assemblyFormat = [{
    `<` `dim_partitioning_axes` `=` $dim_partitioning_axes `:`
    `unreduced_axes` `=` $unreduced_axes `>`
  }];
}

// One IndexedTensorShardingAttr per operand/result value.
def IndexedTensorShardingPerValueAttr : AttrDef<DistributedDialect, "IndexedTensorShardingPerValue"> {
  let summary = "Indexed tensor sharding for each value";
  let cppNamespace = "::mlir::enzyme::distributed";
  let mnemonic = "indexed_tensor_sharding_per_value";

  let parameters = (ins
    EmptyableArrayRefParameter<"IndexedTensorShardingAttr">:$shardings
  );

  let assemblyFormat = [{
    `<` $shardings `>`
  }];
}

// The operator axes of a grammar operator and how each of its slots reads
// them, modeled on #sdy.op_sharding_rule and printed in the same syntax. A
// 32-point DFT split as Cooley-Tukey is
//
//   <([ij])->([kl]) {i=4, j=8, k=8, l=4}>
//
// Each operator axis has an extent. Each operand and result slot has logical
// dimensions, and each lists the operator axes it is made of, major-first
// (`[ij]` is one dimension, `[i, j]` two); a dimension's extent is the product
// of theirs. A dimension with no axes has extent 1 and is written `1`
// (`[i, 1]`). The accessors below call these listed axes "digits". An axis
// may occur on operands only or results only (the DFT's input digits `i j` and
// output digits `k l`), and splitting a dimension more finely is just more
// axes. Operator axes need not be loops of any implementation.
//
// As a distributed.GrammarTerm's roles, the slots are the term's tensors,
// which have no shape: the logical dimensions are the operator's own. In the
// description of a recognized program op (see recognize()), the slots are the
// op's operands and results, and each logical dimension is one of their tensor
// dimensions.
//
// Unlike a Shardy rule, axes carry no kinds: how each axis may be sharded is
// the operator kind's base signature, which lives in a registry.
def OperatorAxesAttr : AttrDef<DistributedDialect, "OperatorAxes"> {
  let summary = "Operator axes of a grammar operator and their tensor mappings";
  let cppNamespace = "::mlir::enzyme::distributed";
  let mnemonic = "operator_axes";

  let parameters = (ins
    OptionalArrayRefParameter<"int64_t",
        "extent of each operator axis">:$axis_extents,
    OptionalArrayRefParameter<"::mlir::sdy::TensorMappingAttr",
        "operator axes of each operand dimension">:$operand_mappings,
    OptionalArrayRefParameter<"::mlir::sdy::TensorMappingAttr",
        "operator axes of each result dimension">:$result_mappings
  );

  let assemblyFormat = [{
    `<`
    custom<TensorMappings>($operand_mappings)
    `` `->` ``
    custom<TensorMappings>($result_mappings) ``
    custom<FactorSizes>($axis_extents)
    `>`
  }];

  let genVerifyDecl = 1;

  let builders = [
    // Mappings given as, per tensor, per dimension, its axis indices
    // major-first; an empty list is a digitless dimension. For example
    // `<([i, 1])->([i]) {i=4}>` is `get(ctx, {4}, {{{0}, {}}}, {{{0}}})`.
    AttrBuilder<(ins
      "::llvm::ArrayRef<int64_t>":$axisExtents,
      "::llvm::ArrayRef<::llvm::SmallVector<::llvm::SmallVector<int64_t>>>":$operandDigits,
      "::llvm::ArrayRef<::llvm::SmallVector<::llvm::SmallVector<int64_t>>>":$resultDigits)>
  ];

  let extraClassDeclaration = [{
    // Where an operator axis occurs: dimension `dim` of operand or result
    // `tensorIdx`, at digit position `pos` (0 is the major-most digit).
    struct Occurrence {
      bool isResult;
      int64_t tensorIdx;
      int64_t dim;
      int64_t pos;
    };

    int64_t getNumAxes() const { return getAxisExtents().size(); }
    int64_t getAxisExtent(int64_t axis) const { return getAxisExtents()[axis]; }

    ::llvm::ArrayRef<::mlir::sdy::TensorMappingAttr>
    getMappings(bool isResult) const {
      return isResult ? getResultMappings() : getOperandMappings();
    }

    // The operator axes of one tensor dimension, major-first.
    ::llvm::ArrayRef<int64_t> getDigits(bool isResult, int64_t tensorIdx,
                                        int64_t dim) const;

    // The stride of the digit at `pos` within its dimension: the product of
    // the extents of the digits minor to it. With the digit's extent, this is
    // the (size, stride) pair an axis factor of that dimension is written in.
    int64_t getDigitStride(bool isResult, int64_t tensorIdx, int64_t dim,
                           int64_t pos) const;

    // Every place `axis` occurs, operands first. An axis with occurrences on
    // only one side (a DFT's `j` or `k`) is consumed or produced by the
    // operator rather than carried through it.
    ::llvm::SmallVector<Occurrence> getOccurrences(int64_t axis) const;
  }];
}

#endif // ENZYME_AD_JAX_DIALECT_DISTRIBUTED_ATTRIBUTES_TD