// RUN: rm -rf %t.dir && mkdir -p %t.dir
//
// root's containing directory (%t.dir/logs) does not exist yet, so this checks
// that the pass creates it as well as the unique leaf directory.
// RUN: enzymexlamlir-opt %s --set-distributed-logging="root=%t.dir/logs module-name=mymodule commit=abc123" | FileCheck %s --check-prefix=ATTRS
// RUN: ls %t.dir/logs | FileCheck %s --check-prefix=RUNDIR
// RUN: ls %t.dir/logs/mymodule-*/ | FileCheck %s --check-prefix=SUBDIRS
//
// Re-running must mint a second, sibling run directory instead of reusing the
// first one.
// RUN: enzymexlamlir-opt %s --set-distributed-logging="root=%t.dir/logs module-name=mymodule commit=abc123"
// RUN: ls %t.dir/logs | FileCheck %s --check-prefix=TWORUNS
//
// An empty root (the default) is a no-op: no new directory, no attributes, and
// the module passes through unchanged. Reusing the same %t.dir/logs from above,
// its entry count still being the 2 minted above confirms no third directory
// was created.
// RUN: enzymexlamlir-opt %s --set-distributed-logging | FileCheck %s --check-prefix=NOOP
// RUN: ls %t.dir/logs | wc -l | FileCheck %s --check-prefix=STILLTWO

// STILLTWO: 2

// The stamped attributes appear in the module's printed "module attributes
// {...}" header, so FileCheck on this pass's output checks both that they were
// set and what they contain.
// ATTRS: module attributes {distributed.logging_commit = "abc123", distributed.logging_module_name = "mymodule", distributed.logging_root = "{{.*}}/logs/mymodule-{{[0-9]+-[0-9]+-[0-9a-f]+}}"}

// RUNDIR: mymodule-

// SUBDIRS-DAG: Candidates
// SUBDIRS-DAG: Main

// TWORUNS-COUNT-2: mymodule-
// TWORUNS-NOT: mymodule-

// NOOP-NOT: distributed.logging
// NOOP: module {
// NOOP-NEXT: func.func @main
// NOOP-NEXT: return

module {
  func.func @main() {
    return
  }
}
