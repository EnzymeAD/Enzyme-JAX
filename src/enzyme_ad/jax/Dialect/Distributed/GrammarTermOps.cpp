#include "DerivationIR.h"
#include "Dialect.h"
#include "GrammarRegistry.h"

#include "shardy/dialect/sdy/ir/utils.h"
#include "src/enzyme_ad/jax/Dialect/Axis/Utilities.h"
#include "llvm/ADT/BitVector.h"

namespace mlir::enzyme::distributed {

//===----------------------------------------------------------------------===//
// OperatorAxesAttr
//===----------------------------------------------------------------------===//

OperatorAxesAttr OperatorAxesAttr::get(
    MLIRContext *context, ArrayRef<int64_t> axisExtents,
    ArrayRef<SmallVector<SmallVector<int64_t>>> operandDigits,
    ArrayRef<SmallVector<SmallVector<int64_t>>> resultDigits) {
  auto toMappings = [&](ArrayRef<SmallVector<SmallVector<int64_t>>> digits) {
    return llvm::map_to_vector(
        digits, [&](ArrayRef<SmallVector<int64_t>> dims) {
          auto dimMappings =
              llvm::map_to_vector(dims, [&](ArrayRef<int64_t> axes) {
                return sdy::DimMappingAttr::get(context, axes);
              });
          return sdy::TensorMappingAttr::get(context, dimMappings);
        });
  };
  return get(context, axisExtents, toMappings(operandDigits),
             toMappings(resultDigits));
}

// The attribute's own checks: extents, axis indices, and each slot's use of
// its axes, mirroring the per-tensor checks of Shardy's
// verifyShardingRuleMapping, which isn't exported. GrammarTermOp checks the
// slot counts and extents against its inputs, outputs and operator axes.
LogicalResult
OperatorAxesAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                         ArrayRef<int64_t> axisExtents,
                         ArrayRef<sdy::TensorMappingAttr> operandMappings,
                         ArrayRef<sdy::TensorMappingAttr> resultMappings) {
  int64_t numAxes = axisExtents.size();
  for (auto [axis, extent] : llvm::enumerate(axisExtents)) {
    if (extent < 1)
      return emitError() << "operator axis " << sdy::factorSymbolString(axis)
                         << " has extent " << extent << ", expected at least 1";
  }

  llvm::BitVector usedAxes(numAxes);
  auto verifyMappings = [&](ArrayRef<sdy::TensorMappingAttr> mappings,
                            StringRef side) -> LogicalResult {
    for (auto [tensorIdx, mapping] : llvm::enumerate(mappings)) {
      llvm::BitVector usedByTensor(numAxes);
      for (auto [dim, dimMapping] : llvm::enumerate(mapping.getDimMappings())) {
        ArrayRef<int64_t> digits = dimMapping.getFactorIndices();
        for (int64_t axis : digits) {
          if (axis < 0 || axis >= numAxes)
            return emitError() << side << " " << tensorIdx << " dimension "
                               << dim << " maps operator axis index " << axis
                               << ", expected less than " << numAxes;
          if (usedByTensor.test(axis))
            return emitError()
                   << side << " " << tensorIdx << " maps operator axis "
                   << sdy::factorSymbolString(axis) << " more than once";
          if (digits.size() > 1 && axisExtents[axis] == 1)
            return emitError()
                   << side << " " << tensorIdx << " dimension " << dim
                   << " splits into several operator axes, but "
                   << sdy::factorSymbolString(axis) << " has extent 1";
          usedByTensor.set(axis);
        }
      }
      usedAxes |= usedByTensor;
    }
    return success();
  };
  if (failed(verifyMappings(operandMappings, "operand")) ||
      failed(verifyMappings(resultMappings, "result")))
    return failure();

  if (!usedAxes.all())
    return emitError() << "operator axis "
                       << sdy::factorSymbolString(usedAxes.find_first_unset())
                       << " is mapped by no operand or result";
  return success();
}

ArrayRef<int64_t> OperatorAxesAttr::getDigits(bool isResult, int64_t tensorIdx,
                                              int64_t dim) const {
  return getMappings(isResult)[tensorIdx]
      .getDimMappings()[dim]
      .getFactorIndices();
}

int64_t OperatorAxesAttr::getDigitStride(bool isResult, int64_t tensorIdx,
                                         int64_t dim, int64_t pos) const {
  int64_t stride = 1;
  for (int64_t axis : getDigits(isResult, tensorIdx, dim).drop_front(pos + 1))
    stride *= getAxisExtent(axis);
  return stride;
}

SmallVector<OperatorAxesAttr::Occurrence>
OperatorAxesAttr::getOccurrences(int64_t axis) const {
  SmallVector<Occurrence> occurrences;
  for (bool isResult : {false, true}) {
    for (auto [tensorIdx, mapping] : llvm::enumerate(getMappings(isResult))) {
      for (auto [dim, dimMapping] : llvm::enumerate(mapping.getDimMappings())) {
        for (auto [pos, digit] :
             llvm::enumerate(dimMapping.getFactorIndices())) {
          if (digit == axis)
            occurrences.push_back({isResult, static_cast<int64_t>(tensorIdx),
                                   static_cast<int64_t>(dim),
                                   static_cast<int64_t>(pos)});
        }
      }
    }
  }
  return occurrences;
}

//===----------------------------------------------------------------------===//
// Derivation IR
//===----------------------------------------------------------------------===//

// Derivation IR's digits are factors of LogicalMeshAxes roots, whose identity
// is their defining op.
static bool isDerivationFactor(Value factor) {
  auto factorOp = factor.getDefiningOp<axis::AxisFactorOp>();
  return factorOp && factorOp.getAxis().getDefiningOp<LogicalMeshAxesOp>();
}

// The first problem that makes `tensor` invalid, as a diagnostic message, or
// none. Shared by the Tensor's verifier, which reports it, and the term's,
// which leaves invalid slot tensors to their own verifier.
static std::optional<std::string> findTensorError(DerivationTensorOp tensor) {
  if (Value definition = tensor.getDefinition();
      definition && !definition.getDefiningOp<GrammarTermOp>())
    return std::string("`of` must be an output of a distributed.GrammarTerm");

  OperandRange axes = tensor.getAxes();
  for (auto [index, factor] : llvm::enumerate(axes))
    if (!isDerivationFactor(factor))
      return (Twine("axis ") + Twine(index) +
              " must be an axis.factor of a distributed.LogicalMeshAxes root")
          .str();

  SmallVector<ValueRange> perAxis;
  for (size_t index = 0; index < axes.size(); ++index)
    perAxis.push_back(axes.slice(index, 1));
  FailureOr<axis::CollectedAtoms> atoms = axis::collectAtoms(perAxis);
  if (failed(atoms))
    return std::string("has axes whose cuts on one root don't nest");
  for (size_t first = 0; first < axes.size(); ++first)
    for (size_t second = first + 1; second < axes.size(); ++second)
      if (!axis::areAtomSetsDisjoint(atoms->sets[first], atoms->sets[second]))
        return (Twine("axes ") + Twine(first) + " and " + Twine(second) +
                " share a digit")
            .str();
  return std::nullopt;
}

LogicalResult DerivationTensorOp::verify() {
  if (std::optional<std::string> error = findTensorError(*this))
    return emitOpError() << *error;
  return success();
}

std::string getTermSlotName(bool isOutput, int64_t slot) {
  return (Twine(isOutput ? "output " : "input ") + Twine(slot)).str();
}

namespace {

// The factors of each operator axis, which must be axis.products of
// derivation IR factors.
FailureOr<SmallVector<SmallVector<Value>>>
verifyOperatorAxisFactors(GrammarTermOp term) {
  SmallVector<SmallVector<Value>> factors;
  for (auto [index, product] : llvm::enumerate(term.getAxes())) {
    auto productOp = product.getDefiningOp<axis::AxisProductOp>();
    if (!productOp || !llvm::all_of(productOp.getFactors(), isDerivationFactor))
      return term.emitOpError()
             << "operator axis " << sdy::factorSymbolString(index)
             << " must be an axis.product of axis factors of "
                "distributed.LogicalMeshAxes roots";
    factors.emplace_back(productOp.getFactors());
  }
  return factors;
}

// The roles describe exactly the term's slots and operator axes.
LogicalResult verifyRolesMatchOperands(GrammarTermOp term) {
  OperatorAxesAttr roles = term.getRoles();
  for (auto [isResult, numSlots] :
       {std::pair(false, term.getInputs().size()),
        std::pair(true, term.getOutputs().size())}) {
    StringRef side = isResult ? "output" : "input";
    if (roles.getMappings(isResult).size() != numSlots)
      return term.emitOpError()
             << "has roles for " << roles.getMappings(isResult).size() << " "
             << side << " slots but " << numSlots << " " << side << "s";
  }
  if (roles.getNumAxes() != static_cast<int64_t>(term.getAxes().size()))
    return term.emitOpError()
           << "has roles over " << roles.getNumAxes() << " operator axes but "
           << term.getAxes().size() << " axis operands";
  for (auto [index, product] : llvm::enumerate(term.getAxes())) {
    uint64_t productExtent =
        cast<axis::FactorGroupType>(product.getType()).getExtent();
    if (static_cast<uint64_t>(roles.getAxisExtent(index)) != productExtent)
      return term.emitOpError()
             << "operator axis " << sdy::factorSymbolString(index)
             << " has extent " << roles.getAxisExtent(index)
             << " in the roles but its product has extent " << productExtent;
  }
  return success();
}

// The tensor node of each slot, inputs first: the Tensor an input is, and the
// one Tensor `of` each output.
FailureOr<SmallVector<DerivationTensorOp>>
verifySlotTensors(GrammarTermOp term) {
  SmallVector<DerivationTensorOp> tensors;
  for (auto [index, input] : llvm::enumerate(term.getInputs())) {
    auto tensor = input.getDefiningOp<DerivationTensorOp>();
    if (!tensor)
      return term.emitOpError() << getTermSlotName(/*isOutput=*/false, index)
                                << " must be a distributed.Tensor";
    tensors.push_back(tensor);
  }
  for (auto [index, output] : llvm::enumerate(term.getOutputs())) {
    DerivationTensorOp tensor;
    if (output.hasOneUse())
      tensor = dyn_cast<DerivationTensorOp>(*output.user_begin());
    if (!tensor)
      return term.emitOpError()
             << getTermSlotName(/*isOutput=*/true, index)
             << " must have exactly one use, a distributed.Tensor of it";
    tensors.push_back(tensor);
  }
  return tensors;
}

// The checks on digits, all decided on the common atoms of every factor in
// scope: the operator axes share no digit, each slot's tensor holds its role
// digits and no other slot's, and each output holds exactly its role digits
// and the batch digits of the inputs it depends on (the kind's dependence).
LogicalResult verifySlotDigits(GrammarTermOp term,
                               ArrayRef<SmallVector<Value>> operatorAxisFactors,
                               ArrayRef<DerivationTensorOp> tensors) {
  OperatorAxesAttr roles = term.getRoles();
  int64_t numInputs = term.getInputs().size();

  // Factor lists whose atom sets are compared below, each recorded by index.
  SmallVector<SmallVector<Value>> lists;
  auto addList = [&](SmallVector<Value> factors) {
    lists.push_back(std::move(factors));
    return lists.size() - 1;
  };

  // One list per factor of the operator axes, for pairwise disjointness. They
  // come first, so list k is factor k.
  SmallVector<int64_t> operatorAxisOfFactor;
  SmallVector<Value> allRoleFactors;
  for (auto [index, factors] : llvm::enumerate(operatorAxisFactors)) {
    for (Value factor : factors) {
      addList({factor});
      operatorAxisOfFactor.push_back(index);
    }
    llvm::append_range(allRoleFactors, factors);
  }
  size_t allRolesList = addList(allRoleFactors);

  struct SlotLists {
    bool isResult;
    int64_t slot;
    size_t roleList, tensorList;
  };
  SmallVector<SlotLists> slots;
  for (int64_t index = 0; index < static_cast<int64_t>(tensors.size());
       ++index) {
    // A copy: op accessors aren't const.
    DerivationTensorOp tensor = tensors[index];
    bool isResult = index >= numInputs;
    int64_t slot = isResult ? index - numInputs : index;
    size_t roleList =
        addList(getSlotRoleFactors(roles, operatorAxisFactors, isResult, slot));
    size_t tensorList = addList(SmallVector<Value>(tensor.getAxes()));
    slots.push_back({isResult, slot, roleList, tensorList});
  }

  SmallVector<ValueRange> ranges(lists.begin(), lists.end());
  FailureOr<axis::CollectedAtoms> collected = axis::collectAtoms(ranges);
  if (failed(collected))
    return term.emitOpError()
           << "has operator axes and tensors whose cuts on one root don't nest";
  ArrayRef<axis::AtomSet> atoms = collected->sets;

  size_t numFactors = operatorAxisOfFactor.size();
  for (size_t first = 0; first < numFactors; ++first) {
    for (size_t second = first + 1; second < numFactors; ++second) {
      if (axis::areAtomSetsDisjoint(atoms[first], atoms[second]))
        continue;
      int64_t firstAxis = operatorAxisOfFactor[first];
      int64_t secondAxis = operatorAxisOfFactor[second];
      if (firstAxis == secondAxis)
        return term.emitOpError()
               << "operator axis " << sdy::factorSymbolString(firstAxis)
               << " holds a digit twice";
      return term.emitOpError()
             << "operator axes " << sdy::factorSymbolString(firstAxis)
             << " and " << sdy::factorSymbolString(secondAxis)
             << " share a digit";
    }
  }

  const axis::AtomSet &allRoles = atoms[allRolesList];
  for (const SlotLists &slot : slots) {
    const axis::AtomSet &roleAtoms = atoms[slot.roleList];
    const axis::AtomSet &tensorAtoms = atoms[slot.tensorList];
    std::string name = getTermSlotName(slot.isResult, slot.slot);
    if (!axis::atomSetContains(tensorAtoms, roleAtoms))
      return term.emitOpError()
             << name << "'s tensor lacks some digit of its roles";
    if (!axis::areAtomSetsDisjoint(tensorAtoms,
                                   axis::atomSetMinus(allRoles, roleAtoms)))
      return term.emitOpError()
             << name << "'s tensor holds a role digit of another slot";
  }

  SmallVector<axis::AtomSet> inputBatch;
  for (const SlotLists &slot : ArrayRef(slots).take_front(numInputs))
    inputBatch.push_back(
        axis::atomSetMinus(atoms[slot.tensorList], atoms[slot.roleList]));
  SmallVector<SmallVector<int64_t>> dependence = getDependence(
      term.getContext(), term.getKind(), roles, term.getParamsAttr());
  for (auto [output, inputs] : llvm::enumerate(dependence)) {
    const SlotLists &slot = slots[numInputs + output];
    axis::AtomSet expected = atoms[slot.roleList];
    for (int64_t input : inputs)
      expected = axis::atomSetUnion(expected, inputBatch[input]);
    const axis::AtomSet &tensorAtoms = atoms[slot.tensorList];
    std::string name = getTermSlotName(/*isOutput=*/true, output);
    if (!axis::atomSetContains(tensorAtoms, expected))
      return term.emitOpError() << name
                                << "'s tensor lacks a batch digit of an input "
                                   "it depends on";
    if (tensorAtoms != expected)
      return term.emitOpError()
             << name
             << "'s tensor holds a digit that is neither its role nor the "
                "batch of an input it depends on";
  }
  return success();
}

} // namespace

LogicalResult GrammarTermOp::verify() {
  if (getKind().empty())
    return emitOpError() << "requires a non-empty kind";

  FailureOr<SmallVector<SmallVector<Value>>> operatorAxisFactors =
      verifyOperatorAxisFactors(*this);
  if (failed(operatorAxisFactors) || failed(verifyRolesMatchOperands(*this)))
    return failure();
  FailureOr<SmallVector<DerivationTensorOp>> tensors = verifySlotTensors(*this);
  if (failed(tensors))
    return failure();
  // An invalid slot tensor is reported by its own verifier, and the output
  // Tensors may not have been verified yet. The digit checks need valid
  // tensors, so they wait for them rather than report a consequence.
  for (DerivationTensorOp tensor : *tensors)
    if (findTensorError(tensor))
      return success();
  if (failed(verifySlotDigits(*this, *operatorAxisFactors, *tensors)))
    return failure();

  const GrammarKindEntry *entry =
      getGrammarRegistry(getContext()).lookup(getKind());
  if (entry && entry->verify)
    return entry->verify(*this);
  return success();
}

SmallVector<SmallVector<Value>> GrammarTermOp::getOperatorAxisFactors() {
  return llvm::map_to_vector(getAxes(), [](Value product) {
    return SmallVector<Value>(
        cast<axis::AxisProductOp>(product.getDefiningOp()).getFactors());
  });
}

DerivationTensorOp GrammarTermOp::getSlotTensor(bool isOutput, int64_t slot) {
  if (!isOutput)
    return cast<DerivationTensorOp>(getInputs()[slot].getDefiningOp());
  return cast<DerivationTensorOp>(*getOutputs()[slot].user_begin());
}

} // namespace mlir::enzyme::distributed
