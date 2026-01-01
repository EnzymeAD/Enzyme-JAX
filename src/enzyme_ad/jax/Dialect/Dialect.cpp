//===- EnzymeXLADialect.cpp - EnzymeXLA dialect -----------------------*- C++
//-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Dialect.h"
#include "Ops.h"
#include "mlir/IR/DialectImplementation.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/InterleavedRange.h"
#include "llvm/Support/LogicalResult.h"

#include "mlir/IR/Dialect.h"
#include "mlir/Transforms/InliningUtils.h"

#define GET_ATTRDEF_CLASSES
#include "src/enzyme_ad/jax/Dialect/EnzymeXLAAttrDefs.cpp.inc"
#include "src/enzyme_ad/jax/Dialect/EnzymeXLAAttrEnums.cpp.inc"

// #include "Dialect/EnzymeEnums.cpp.inc"
#include "src/enzyme_ad/jax/Dialect/EnzymeXLADialect.cpp.inc"

static llvm::ParseResult parseAsyncDependencies(
    mlir::OpAsmParser &parser, mlir::Type &asyncTokenType,
    llvm::SmallVectorImpl<mlir::OpAsmParser::UnresolvedOperand>
        &asyncDependencies) {
  using namespace mlir;
  using namespace mlir::gpu;
  auto loc = parser.getCurrentLocation();
  if (succeeded(parser.parseOptionalKeyword("async"))) {
    if (parser.getNumResults() == 0)
      return parser.emitError(loc, "needs to be named when marked 'async'");
    asyncTokenType = parser.getBuilder().getType<AsyncTokenType>();
  }
  return parser.parseOperandList(asyncDependencies,
                                 OpAsmParser::Delimiter::OptionalSquare);
}

/// Prints optional async dependencies with its leading keyword.
///   (`async`)? (`[` ssa-id-list `]`)?
// Used by the tablegen assembly format for several async ops.
static void printAsyncDependencies(mlir::OpAsmPrinter &printer,
                                   mlir::Operation *op,
                                   mlir::Type asyncTokenType,
                                   mlir::OperandRange asyncDependencies) {
  if (asyncTokenType)
    printer << "async";
  if (asyncDependencies.empty())
    return;
  if (asyncTokenType)
    printer << ' ';
  printer << llvm::interleaved_array(asyncDependencies);
}

#define GET_OP_CLASSES
#include "src/enzyme_ad/jax/Dialect/EnzymeXLAOps.cpp.inc"

// #define GET_TYPEDEF_CLASSES
// #include "Dialect/EnzymeXLAOpsTypes.cpp.inc"
//  #include "Dialect/EnzymeTypes.cpp.inc"

using namespace mlir;
using namespace mlir::enzymexla;

//===----------------------------------------------------------------------===//
// Enzyme dialect.
//===----------------------------------------------------------------------===//

namespace {
struct EnzymeXLADialectInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  // Allow all call operations to be inlined.
  bool isLegalToInline(Operation *call, Operation *callable,
                       bool wouldBeCloned) const final {
    return !isa<KernelCallOp, JITCallOp>(call);
  }
  // We don't have any special restrictions on what can be inlined into
  // destination regions (e.g. while/conditional bodies). Always allow it.
  bool isLegalToInline(Region *dest, Region *src, bool wouldBeCloned,
                       IRMapping &valueMapping) const final {
    return true;
  }
  // Operations in StableHLO dialect are always legal to inline since they are
  // pure.
  bool isLegalToInline(Operation *, Region *, bool, IRMapping &) const final {
    return true;
  }
};

} // namespace

void StructuredSparsityAttr::print(::mlir::AsmPrinter &printer) const {
  printer << "<";

  auto pattern = getPattern();
  auto kind = pattern.getKind();

  // Skip printing kind for Unknown
  bool printKind = kind != StructuredSparsityKind::Unknown;

  if (printKind) {
    printer << stringifyStructuredSparsityKind(kind);

    // Print bandwidth only for Band
    if (kind == StructuredSparsityKind::Band) {
      printer << " [" << pattern.getLowerBandwidth() << ", "
              << pattern.getUpperBandwidth() << "]";
    }
  }

  // Print value properties as a set
  auto props = getValueProperties();
  if (!props.empty()) {
    if (printKind) {
      printer << ", ";
    }
    printer << "{";
    llvm::interleaveComma(props, printer, [&](StructuredValueProperty prop) {
      printer << stringifyStructuredValueProperty(prop);
    });
    printer << "}";
  }

  printer << ">";
}

::mlir::Attribute StructuredSparsityAttr::parse(::mlir::AsmParser &parser,
                                                ::mlir::Type type) {
  if (parser.parseLess())
    return {};

  StructuredSparsityKind kind = StructuredSparsityKind::Unknown;
  int64_t lowerBandwidth = -1;
  int64_t upperBandwidth = -1;
  llvm::SmallVector<StructuredValueProperty> valueProperties;

  // Check if we start with a brace (properties only, no kind)
  if (parser.parseOptionalLBrace().failed()) {
    // Try to parse the kind
    llvm::StringRef kindStr;
    if (succeeded(parser.parseOptionalKeyword(&kindStr))) {
      auto kindOpt = symbolizeStructuredSparsityKind(kindStr);
      if (!kindOpt) {
        parser.emitError(parser.getCurrentLocation(), "invalid sparsity kind: ")
            << kindStr;
        return {};
      }
      kind = *kindOpt;

      // If Band, parse bandwidth bounds
      if (kind == StructuredSparsityKind::Band) {
        if (parser.parseLSquare() || parser.parseInteger(lowerBandwidth) ||
            parser.parseComma() || parser.parseInteger(upperBandwidth) ||
            parser.parseRSquare()) {
          return {};
        }
      }

      // Check for comma before properties
      parser.parseOptionalComma();
    }

    // Try to parse value properties set
    if (succeeded(parser.parseOptionalLBrace())) {
      // Fall through to parse properties
    } else {
      // No properties
      if (parser.parseGreater())
        return {};

      auto pattern = StructuredSparsityPatternAttr::get(
          parser.getContext(), kind, lowerBandwidth, upperBandwidth);
      return StructuredSparsityAttr::get(parser.getContext(), pattern,
                                         valueProperties);
    }
  }

  // Parse properties
  if (!parser.parseOptionalRBrace().succeeded()) {
    do {
      llvm::StringRef propStr;
      if (parser.parseKeyword(&propStr))
        return {};

      auto propOpt = symbolizeStructuredValueProperty(propStr);
      if (!propOpt) {
        parser.emitError(parser.getCurrentLocation(),
                         "invalid value property: ")
            << propStr;
        return {};
      }
      valueProperties.push_back(*propOpt);
    } while (succeeded(parser.parseOptionalComma()));

    if (parser.parseRBrace())
      return {};
  }

  if (parser.parseGreater())
    return {};

  auto pattern = StructuredSparsityPatternAttr::get(
      parser.getContext(), kind, lowerBandwidth, upperBandwidth);
  return StructuredSparsityAttr::get(parser.getContext(), pattern,
                                     valueProperties);
}

void EnzymeXLADialect::initialize() {
  addInterfaces<EnzymeXLADialectInlinerInterface>();
  addOperations<
#define GET_OP_LIST
#include "src/enzyme_ad/jax/Dialect/EnzymeXLAOps.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "src/enzyme_ad/jax/Dialect/EnzymeXLAAttrDefs.cpp.inc"
      >();
  //  addTypes<
  // #define GET_TYPEDEF_LIST
  // #include "src/enzyme_ad/jax/Dialect/EnzymeXLAOpsTypes.cpp.inc"
  //      >();
}
