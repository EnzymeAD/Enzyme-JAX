#ifndef ENZYME_AD_JAX_DIALECT_AXIS_UTILITIES_H
#define ENZYME_AD_JAX_DIALECT_AXIS_UTILITIES_H

#include <cstdint>
#include <utility>

#include "Dialect.h"

#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"

#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::enzyme::axis {
template <typename T> using TypedValueArrayRef = llvm::ArrayRef<TypedValue<T>>;

struct SplitExtentSlice {
  size_t extentIdx;
  uint64_t subExtent;
  uint64_t stride;
};

// Parses a type-annotated variadic operand list as:
// (%arg1 : type1, %arg2 : type2, ..., %argN : typeN) or ()
// The parenthesized empty form disambiguates an empty list from the next
// operation's result definition.
::mlir::ParseResult parseVariadicWithTypes(
    ::mlir::OpAsmParser &parser,
    llvm::SmallVectorImpl<::mlir::OpAsmParser::UnresolvedOperand> &operands,
    llvm::SmallVectorImpl<::mlir::Type> &types);

void printVariadicWithTypes(::mlir::OpAsmPrinter &printer,
                            ::mlir::Operation *op,
                            ::mlir::OperandRange operands,
                            ::mlir::TypeRange types);

// Utility for casting with better error reporting
template <typename T>
TypedValue<T> castTypedValue(::mlir::Value value, llvm::StringRef expectedType);

// Utility for casting a list of values with better error reporting
template <typename T>
llvm::SmallVector<TypedValue<T>>
castTypedValueList(ValueRange values, llvm::StringRef expectedType);

template <typename T>
TypedValue<T> castTypedValue(::mlir::Value value,
                             llvm::StringRef expectedType) {
  if (auto typed = dyn_cast<TypedValue<T>>(value)) {
    return typed;
  }
  std::string typeString;
  llvm::raw_string_ostream os(typeString);
  value.getType().print(os);
  os.flush();
  llvm::errs() << "castTypedValue failed: expected " << expectedType
               << ", got value type " << typeString << "\n";
  llvm_unreachable("invalid typed value cast");
}

template <typename T>
llvm::SmallVector<TypedValue<T>>
castTypedValueList(ValueRange values, llvm::StringRef expectedType) {
  llvm::SmallVector<TypedValue<T>> typedValues;
  typedValues.reserve(values.size());
  for (Value value : values) {
    typedValues.push_back(castTypedValue<T>(value, expectedType));
  }
  return typedValues;
}

// Returns the static extent for any canonical axis SSA value.
int getAxisExtent(::mlir::TypedValue<AxisTypeInterface> axis);

int getAxisDimIndex(::mlir::TypedValue<ShapeAxisType> axis);

// Returns the static extent for any factor SSA value.
int getFactorExtent(::mlir::TypedValue<AxisFactorType> factor);

// Returns the static stride for any factor SSA value.
int getFactorStride(::mlir::TypedValue<AxisFactorType> factor);

// Returns the defining op for a canonical axis SSA value.
::mlir::FailureOr<::mlir::Operation *> getAxisProvenanceOp(::mlir::Value axis);

// Resolves the source canonical axis used to produce a factor value.
::mlir::FailureOr<::mlir::TypedValue<AxisTypeInterface>>
getFactorProvenanceAxis(::mlir::TypedValue<AxisFactorType> factor);

// Returns the factor list used to build a factor-product SSA value.
::mlir::FailureOr<llvm::SmallVector<::mlir::TypedValue<AxisFactorType>>>
getProductProvenanceFactors(::mlir::TypedValue<FactorGroupType> factorProduct);

// Returns the product of extents for a factor-product SSA value.
::mlir::FailureOr<uint64_t>
getFactorGroupExtent(::mlir::TypedValue<FactorGroupType> factorProduct);

// Uses the type-specific axis equivalence semantics to check
// structural / logical equivalence of axes
bool areAxesEquivalent(::mlir::TypedValue<AxisTypeInterface> lhs,
                       ::mlir::TypedValue<AxisTypeInterface> rhs);

// Uses the type-specific disjointness logic to check if
// two axes may be assigned to the same space (they are disjoint)
// or if they "interfere" in some way (they are not disjoint).
// Equivalent axes may still be disjoint. Example: replication axes
// can be structurally idential / from the same IR node but can be
// reused.
bool areAxesDisjoint(::mlir::TypedValue<AxisTypeInterface> lhs,
                     ::mlir::TypedValue<AxisTypeInterface> rhs);

// Checks that factors are pairwise non-overlapping for one source axis.
bool arePairwiseFactorsDisjoint(
    ::mlir::TypedValue<AxisFactorType> lhsFactor,
    ::mlir::TypedValue<AxisFactorType> rhsFactor,
    ::mlir::TypedValue<AxisTypeInterface> lhsProvenanceAxis = nullptr,
    ::mlir::TypedValue<AxisTypeInterface> rhsProvenanceAxis = nullptr);

// Checks that factors are pairwise non-overlapping for one source axis.
bool areFactorsDisjoint(TypedValueArrayRef<AxisFactorType> factors);

// Returns true when two factor lists cover the same index space modulo
// permutation order. This checks multiset equality over factor metadata and
// provenance-axis equivalence, but does not require matching list order.
bool areFactorIndexSpacesEqual(TypedValueArrayRef<AxisFactorType> lhsFactors,
                               TypedValueArrayRef<AxisFactorType> rhsFactors);

// Returns true when two factor lists are structurally equal in-order.
// This requires equal length and pairwise equality of
// (provenance-axis equivalence, extent, stride) at each position.
bool areFactorListsStructurallyEqual(
    TypedValueArrayRef<AxisFactorType> lhsFactors,
    TypedValueArrayRef<AxisFactorType> rhsFactors,
    bool respectShapeTypes = true);

// Checks that factors cover an axis exactly and therefore are disjoint.
bool areFactorsComplete(::mlir::TypedValue<AxisTypeInterface> axis,
                        TypedValueArrayRef<AxisFactorType> factors);

// Small utlity for extracting all factors from one or more factor groups.
llvm::SmallVector<::mlir::TypedValue<AxisFactorType>>
flattenGroupsToFactors(TypedValueArrayRef<FactorGroupType> factorGroups);

// Shortcut for calling distjoint on a flattened factor list
bool areFactorGroupsDisjoint(TypedValueArrayRef<FactorGroupType> factorGroups);

// Given a shapeType with a known rank, returns a list of canonical axes for
// each dimension of that shape. Use a builder without an insertion point and
// an unknown location for ephemeral axes.
llvm::SmallVector<::mlir::TypedValue<AxisTypeInterface>>
createAxesForRankedShape(::mlir::Type shapeType, ::mlir::OpBuilder &builder,
                         ::mlir::Location loc);

// Creates a new subfactor from a given factor with the same axis,
// the given extent, and the given stride within the factor
// (total stride: product of substride and original)
::mlir::TypedValue<AxisFactorType>
createSubfactor(::mlir::TypedValue<AxisFactorType> factor, int extent,
                int strideWithinFactor, ::mlir::OpBuilder &builder,
                ::mlir::Location loc);

// Creates a single factor for each axis, with full extent and stride 1.
llvm::SmallVector<::mlir::TypedValue<AxisFactorType>>
viewAxesAsFactors(::mlir::ValueRange axes, ::mlir::OpBuilder &builder,
                  ::mlir::Location loc);
llvm::SmallVector<::mlir::TypedValue<AxisFactorType>>
viewAxesAsFactors(TypedValueArrayRef<AxisTypeInterface> axes,
                  ::mlir::OpBuilder &builder, ::mlir::Location loc);
TypedValue<AxisFactorType> viewAxisAsFactor(::mlir::Value axis,
                                            ::mlir::OpBuilder &builder,
                                            ::mlir::Location loc);
// Creates one factor-group product from an existing list of factors.
::mlir::TypedValue<FactorGroupType>
viewFactorsAsProduct(::mlir::ValueRange factors, ::mlir::OpBuilder &builder,
                     ::mlir::Location loc);
::mlir::TypedValue<FactorGroupType>
viewFactorsAsProduct(TypedValueArrayRef<AxisFactorType> factors,
                     ::mlir::OpBuilder &builder, ::mlir::Location loc);

// Rebuilds `group` with every extent-1 factor removed.
// The ressult is unchanged if there are no extent 1 factors.
// May produce an empty product.
::mlir::TypedValue<FactorGroupType>
dropUnitFactors(::mlir::TypedValue<FactorGroupType> group,
                ::mlir::OpBuilder &builder);

// Creates a full major-first factorization for one axis from major-first
// extents. Strides are inferred
llvm::SmallVector<::mlir::TypedValue<AxisFactorType>>
factorAxisByExtents(::mlir::Value axis, llvm::ArrayRef<int32_t> extents,
                    ::mlir::OpBuilder &builder, ::mlir::Location loc);

// Redirects `builder` to insert into a private scratch block for this
// guard's lifetime, and erases everything still in that block (via the
// block's own destructor) once the guard goes out of scope -- regardless of
// which exit path was taken. Call keep() on any value that should survive
// (become a detached, standalone value again, exactly as if this guard had
// never existed) before the guard is destroyed.
//
// This is for callers whose builder has no meaningful insertion point of its
// own (detached, ephemeral bookkeeping -- e.g. search-time scratch axis
// arithmetic, or a pure verifier computation): everything created here is
// meant to end up either kept as a free-floating value or discarded, never
// inserted into real IR. If a function's builder already points at a real,
// persistent destination block, don't route it through this guard -- ops
// created there are already properly owned from the moment they're built;
// track and selectively erase unconfirmed ones directly instead (see
// replaceAxisFactors for that pattern).
//
// A kept value must not depend (through its own operands) on anything else
// created in this scope that was not also kept: block teardown drops all
// internal references in one shot and doesn't know to preserve
// cross-references among discarded ops. This is a non-issue for axis-factor
// arithmetic specifically, since every AxisFactorOp's "axis" operand always
// resolves directly to its root axis (see createSubfactor), never to another
// (possibly-discarded) factor.
class TemporaryOpGuard {
  ::mlir::OpBuilder::InsertionGuard insertionGuard;
  ::mlir::Block scratch;

public:
  explicit TemporaryOpGuard(::mlir::OpBuilder &builder)
      : insertionGuard(builder) {
    builder.setInsertionPointToEnd(&scratch);
  }

  // Detaches `value`'s defining op from the scratch block (without erasing
  // it) so it survives this guard's destruction. No-op if `value` wasn't
  // actually created in this scope (e.g. a pass-through value already owned
  // elsewhere before the guard existed).
  template <typename ValT> ValT keep(ValT value) {
    if (::mlir::Operation *op = value.getDefiningOp();
        op && op->getBlock() == &scratch)
      op->remove();
    return value;
  }
};

// Subtracts subtrahend factors from a factor-group and returns the remaining
// factors in major-first order. Subtrahend factors must be representable as
// factors of the minuend index space.
::mlir::FailureOr<llvm::SmallVector<::mlir::TypedValue<AxisFactorType>>>
subtractSpace(::mlir::TypedValue<FactorGroupType> minuend,
              llvm::ArrayRef<::mlir::TypedValue<AxisFactorType>> subtrahend,
              ::mlir::OpBuilder &builder);

::mlir::FailureOr<llvm::SmallVector<::mlir::TypedValue<AxisFactorType>>>
subtractSpace(llvm::ArrayRef<::mlir::TypedValue<AxisFactorType>> minuend,
              llvm::ArrayRef<::mlir::TypedValue<AxisFactorType>> subtrahend,
              ::mlir::OpBuilder &builder);

// Computes the extent cuts used by split_divisible without materializing SSA
// factors. The returned extents are maximal one-to-one cuts where possible and
// minimal indivisible units otherwise.
llvm::SmallVector<uint64_t> computeSplits(llvm::ArrayRef<uint64_t> lhsExtents,
                                          llvm::ArrayRef<uint64_t> rhsExtents);

// Materializes one side of a split plan as, for each cut, the list of source
// extent slices contributing to that cut.
llvm::SmallVector<llvm::SmallVector<SplitExtentSlice>>
computeSplitExtentSlices(llvm::ArrayRef<uint64_t> extents,
                         llvm::ArrayRef<uint64_t> cuts);

// Splits each lhs/rhs factor-group mapping pair into maximal one-to-one
// submappings where possible, and minimal indivisible units where not possible.
// The outputs are always populated with the computed split mapping. The return
// value is true only if every produced mapping pair is fully atomic on both
// sides.
bool split_divisible(
    llvm::ArrayRef<::mlir::TypedValue<FactorGroupType>> lhs,
    llvm::ArrayRef<::mlir::TypedValue<FactorGroupType>> rhs,
    llvm::SmallVector<::mlir::TypedValue<FactorGroupType>> &lhs_out,
    llvm::SmallVector<::mlir::TypedValue<FactorGroupType>> &rhs_out,
    ::mlir::OpBuilder &builder);

// Recursively refreshes result types of user ops after a type-changing edit.
// This utility only performs in-place result-type updates. Propogation
// assumes that type changes are determined entirely by local op information
// and the types of the direct operands: no traversal to other ops is considered
// when marking users for update.
::mlir::LogicalResult
propagateResultTypeChanges(::llvm::ArrayRef<::mlir::Operation *> initialUsers);

// Replaces all uses of `from` with `to`, then propagates result-type updates
// through impacted users only when the replacement crosses a type boundary.
::mlir::LogicalResult replaceAndTypePropagate(::mlir::Value from,
                                              ::mlir::Value to);

// Finds every axis.factor op that takes `axis` as its operand, regardless of
// each factor's own extent or stride -- this does not assume `axis` is
// currently represented by a single full-extent, unit-stride factor, only
// that it has at least one. Pure query: does not mutate the IR. Call this
// before building any new axis.factor referencing `axis`, since a
// newly-built factor would itself become a use of `axis` and be picked up
// here too. Fails with a diagnostic if `axis` has no factor at all.
::mlir::FailureOr<llvm::SmallVector<AxisFactorOp>>
findAxisFactors(::mlir::TypedValue<AxisTypeInterface> axis);

// Re-projects one axis's factorization from `oldFactors` onto a new basis
// `newFactors`, rewriting every axis.product that uses an old factor in
// place. The two lists must cover the same total extent. Where their
// granularities don't align as whole units, both sides are further split
// (via computeSplits/computeSplitExtentSlices -- the same machinery
// split_divisible uses to align two independently-factored sides) to find a
// common refinement, so a single old factor may end up replaced by one or
// more new sub-factors. Each axis.product using an old factor has that
// operand spliced out and the corresponding new sub-factor(s) spliced in --
// every other operand is left untouched. `oldFactors` should come from
// findAxisFactors, called before any of `newFactors` were built (see its
// comment for why). Builds new ops at `builder`'s current insertion point,
// which the caller must choose to dominate every use. Fails with a
// diagnostic if an old factor is consumed by anything other than
// axis.product, has no axis.product use at all, or if the two sides'
// granularities can't be reconciled (an old factor would need to be merged
// with part of another to align with the new basis).
::mlir::LogicalResult
replaceAxisFactors(TypedValueArrayRef<AxisFactorType> oldFactors,
                   TypedValueArrayRef<AxisFactorType> newFactors,
                   ::mlir::OpBuilder &builder);

// some filtering / predicate utilities
template <typename T> using Predicate = std::function<bool(T)>;

template <typename T> Predicate<T> predNot(Predicate<T> pred) {
  return [pred](T val) { return !pred(val); };
}

// If respectShapeTypes is true, then identity includes
// matching shape/tensor types. Otherwise, any factors
// over different shapes but on the same rank, stride,
// and extent are considered identity. This is useful
// for certain maps between different tensor types.
Predicate<std::pair<::mlir::TypedValue<FactorGroupType>,
                    ::mlir::TypedValue<FactorGroupType>>>
predGroupPairIsIdentity(bool respectShapeTypes = true);

// From a list of factors known to be from the same axis,
// creates a list of pairs indicating the maximum factor ranges.
// Ranges are gauranteed to be return in major-first order.
llvm::SmallVector<std::pair<int, int>>
build_max_factors(TypedValueArrayRef<AxisFactorType> factors);
llvm::SmallVector<std::pair<int, int>> build_max_factors(ValueRange factors);

} // namespace mlir::enzyme::axis

#endif // ENZYME_AD_JAX_DIALECT_AXIS_UTILITIES_H
