//===- enzymexlamlir-tblgen.cpp - Tablegen backend for EnzymeJAX ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Signals.h"
#include "llvm/TableGen/Main.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"

enum ActionType {
  GenPopulatePatternsFuncDecl,
  GenPopulatePatternsFuncDef,
  GenPopulatePatternsInterfaceImpl,
  GenPopulateRaisingPatternsFuncDecl,
  GenPopulateRaisingPatternsFuncDef,
  GenPopulateRaisingPatternsInterfaceImpl,
};

static llvm::cl::opt<ActionType> action(
    llvm::cl::desc("action to perform"),
    llvm::cl::values(clEnumValN(GenPopulatePatternsFuncDecl,
                                "gen-populate-patterns-func-decls", "")),
    llvm::cl::values(clEnumValN(GenPopulatePatternsFuncDef,
                                "gen-populate-patterns-func-defs", "")),
    llvm::cl::values(clEnumValN(GenPopulatePatternsInterfaceImpl,
                                "gen-populate-patterns-interface-impl", "")),
    llvm::cl::values(clEnumValN(GenPopulateRaisingPatternsFuncDecl,
                                "gen-populate-raising-patterns-func-decls",
                                "")),
    llvm::cl::values(clEnumValN(GenPopulateRaisingPatternsFuncDef,
                                "gen-populate-raising-patterns-func-defs", "")),
    llvm::cl::values(clEnumValN(GenPopulateRaisingPatternsInterfaceImpl,
                                "gen-populate-raising-patterns-interface-impl",
                                "")));

llvm::StringRef getPopulateFunctionNameSuffix(const llvm::Record *rec) {
  return rec->getName().ends_with("Op") ? rec->getName().drop_back(2)
                                        : rec->getName();
}

static bool emitPopulatePatterns(llvm::raw_ostream &os,
                                 const llvm::RecordKeeper &records,
                                 llvm::StringRef patternOpStr) {
  for (const llvm::Record *rec :
       records.getAllDerivedDefinitions(patternOpStr)) {
    os << "void ";
    llvm::StringRef ns = rec->getValueAsString("cppNamespace");
    if (!ns.empty())
      os << ns << "::";
    os << rec->getName()
       << "::populatePatterns(::mlir::RewritePatternSet &patterns) {\n";
    os << "  " << ns << "::populate" << getPopulateFunctionNameSuffix(rec)
       << "(patterns, *getContext(), "
       << "getBenefit() ? PatternBenefit(*getBenefit()) : PatternBenefit(1));"
       << "\n";
    os << "}\n\n";
  }
  return false;
}

static bool emitPopulatePatternsFuncDecls(llvm::raw_ostream &os,
                                          const llvm::RecordKeeper &records,
                                          llvm::StringRef patternOpStr) {
  for (const llvm::Record *rec :
       records.getAllDerivedDefinitions(patternOpStr)) {
    llvm::StringRef ns = rec->getValueAsString("cppNamespace");
    if (ns.starts_with("::"))
      ns = ns.drop_front(2);
    os << "namespace " << ns << " {\n";
    os << "void populate" << getPopulateFunctionNameSuffix(rec)
       << "(::mlir::RewritePatternSet &patterns, ::mlir::MLIRContext "
          "&context, ::mlir::PatternBenefit benefit);\n";
    os << "} // namespace " << ns << "\n\n";
  }
  return false;
}

static bool emitPopulatePatternsFuncDefs(llvm::raw_ostream &os,
                                         const llvm::RecordKeeper &records,
                                         llvm::StringRef patternOpStr) {
  for (const llvm::Record *rec :
       records.getAllDerivedDefinitions(patternOpStr)) {
    os << "void ";
    llvm::StringRef ns = rec->getValueAsString("cppNamespace");
    if (!ns.empty())
      os << ns;
    os << "::populate" << getPopulateFunctionNameSuffix(rec)
       << "(::mlir::RewritePatternSet &patterns,\n"
       << "    ::mlir::MLIRContext &context,\n"
       << "    ::mlir::PatternBenefit benefit) {\n";

    for (llvm::StringRef pattern : rec->getValueAsListOfStrings("patterns")) {
      os << "  patterns.add<" << pattern << ">(&context);\n";
    }
    os << "}\n\n";
  }
  return false;
}

static bool tablegenMain(llvm::raw_ostream &os,
                         const llvm::RecordKeeper &records) {
  switch (action) {
  case GenPopulatePatternsFuncDecl:
    return emitPopulatePatternsFuncDecls(os, records, "EnzymeHLOPatternOp");
  case GenPopulatePatternsFuncDef:
    return emitPopulatePatternsFuncDefs(os, records, "EnzymeHLOPatternOp");
  case GenPopulatePatternsInterfaceImpl:
    return emitPopulatePatterns(os, records, "EnzymeHLOPatternOp");
  case GenPopulateRaisingPatternsFuncDecl:
    return emitPopulatePatternsFuncDecls(os, records, "RaisingPatternOp");
  case GenPopulateRaisingPatternsFuncDef:
    return emitPopulatePatternsFuncDefs(os, records, "RaisingPatternOp");
  case GenPopulateRaisingPatternsInterfaceImpl:
    return emitPopulatePatterns(os, records, "RaisingPatternOp");
  default:
    llvm::report_fatal_error("unknown action");
    return true;
  }
}

int main(int argc, char **argv) {
  llvm::sys::PrintStackTraceOnErrorSignal(argv[0]);
  llvm::PrettyStackTraceProgram X(argc, argv);
  llvm::cl::ParseCommandLineOptions(argc, argv);

  llvm::llvm_shutdown_obj Y;
  return TableGenMain(argv[0], &tablegenMain);
}
