// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(enzyme-hlo-generate-td{patterns=while_simplify<1>(0)})" | FileCheck %s

// CHECK: transform.apply_patterns.enzyme_hlo.while_simplify {parameter = false}

