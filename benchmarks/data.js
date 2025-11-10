window.BENCHMARK_DATA = {
  "lastUpdate": 1762812434044,
  "repoUrl": "https://github.com/EnzymeAD/Enzyme-JAX",
  "entries": {
    "EnzymeJAX Benchmarks": [
      {
        "commit": {
          "author": {
            "email": "scharfrichterq@gmail.com",
            "name": "Acake",
            "username": "sbrantq"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "f3d79b36e034cb46f7868adc1fa223a6019130f8",
          "message": "ProbProg: Static HMC (#1464)\n\n* todo\n\n* cleanup\n\n* more mh ops lowering\n\n* test\n\n* select trace/dump op\n\n* cholesky solve op\n\n* multinormal\n\n* GetFlattenedSamplesFromTraceOpConversion  LoopOpConversion UnflattenSliceOpConversion\n\n* split lower-enzyme-probprog in two\n\n* save tests\n\n* save\n\n* simplify",
          "timestamp": "2025-10-27T03:35:57-05:00",
          "tree_id": "0c283d16850205054a637775d7898e9158336be5",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/f3d79b36e034cb46f7868adc1fa223a6019130f8"
        },
        "date": 1761560091250,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.0000042890359996818,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004200187907554209,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001495036462089,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001424800402019,
            "unit": "s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "238314553+enzymead-bot[bot]@users.noreply.github.com",
            "name": "enzymead-bot[bot]",
            "username": "enzymead-bot[bot]"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "7717acb02935aa362bfe349b2287dfd5ff5357ef",
          "message": "Update EnzymeAD/Enzyme to commit 63c81877e73b019745ebff33f6dc2e13a5003a17 (#1521)\n\nDiff: https://github.com/EnzymeAD/Enzyme/compare/b0cafac31726e95279c9d0bc578a00ca5016244a...63c81877e73b019745ebff33f6dc2e13a5003a17\n\nCo-authored-by: enzymead-bot[bot] <238314553+enzymead-bot[bot]@users.noreply.github.com>",
          "timestamp": "2025-10-27T12:33:33+01:00",
          "tree_id": "50c77a2f39728a0b8b87dd546a6f6dcd4ec26c44",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/7717acb02935aa362bfe349b2287dfd5ff5357ef"
        },
        "date": 1761568601452,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004274666891433298,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004351996898185462,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001453273741994,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001480237861978,
            "unit": "s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "avikpal@mit.edu",
            "name": "Avik Pal",
            "username": "avik-pal"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "fcf54c56ea4b2b8a78b88c7ca88f1ad385916621",
          "message": "feat: support intermediate insertions/deletions while copying (#1513)\n\ntest: missing results\n\nfeat: try more generalization (partial progress)\n\nfix: uncomment\n\nfix: add comment on how to handle remaining cases\n\nfix: remove unwanted code\n\nrevert: unwanted changes\n\nfeat: handle transpose without explicit transpose op\n\nchore: run fmt\n\nchore: remove old comment",
          "timestamp": "2025-10-27T08:46:33-04:00",
          "tree_id": "0fc39fcc5d7527063e5dd5a1a7ebeae9617a17c7",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/fcf54c56ea4b2b8a78b88c7ca88f1ad385916621"
        },
        "date": 1761574023301,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004405523999594152,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004346182988956571,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001421422843006,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001422737602959,
            "unit": "s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "765740+giordano@users.noreply.github.com",
            "name": "Mosè Giordano",
            "username": "giordano"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "f0c1e1bc526fee6281db47fdeff2bd50f17b4469",
          "message": "Update JAX to commit eac1414e85f61667591b5879bbb95f80d609f23f (#1484)\n\n* Update JAX to commit eac1414e85f61667591b5879bbb95f80d609f23f\n\nDiff: https://github.com/jax-ml/jax/compare/3a22eea644237001df0f3dd42253225cc059b43c...eac1414e85f61667591b5879bbb95f80d609f23f\n\n* Disable x86 AMX support in Bazel configuration\n\n* Log compiler versions in build workflow\n\nAdded commands to display GCC, CC, and Clang versions.\n\n* Fix Bazel flags assignment in build.yml\n\n* Update build.yml\n\n* fix\n\n* fmt\n\n* Update .bazelrc\n\n* Update .bazelrc\n\n* Update JAX_COMMIT hash in workspace.bzl\n\n* Update .bazelrc\n\n* Remove compiler version checks from build.yml\n\nRemoved version checks for gcc, cc, and clang from the build workflow.\n\n* Update .bazelrc\n\n* Update JAX_COMMIT to new commit hash\n\n---------\n\nCo-authored-by: enzyme-ci-bot[bot] <78882869+enzyme-ci-bot[bot]@users.noreply.github.com>\nCo-authored-by: William Moses <wmoses@google.com>\nCo-authored-by: William S. Moses <gh@wsmoses.com>",
          "timestamp": "2025-10-27T17:27:47-05:00",
          "tree_id": "f83ee8cedecfd38343a1c34a21db8f7f6a5bbe7e",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/f0c1e1bc526fee6281db47fdeff2bd50f17b4469"
        },
        "date": 1761618411603,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000003881685994565487,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004159397992771119,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001516026142984,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001506869731936,
            "unit": "s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "avikpal@mit.edu",
            "name": "Avik Pal",
            "username": "avik-pal"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "992f348493d08666a9bd4305fdbfc35ff6910c0b",
          "message": "feat: greedy loop to batch (#1425)\n\n* feat: greedy loop to batch via loop fission\n\nfeat: support non-constant extra indices\n\nfix: only emit new ops if sure\n\nfeat: propagate bounds and eliminate noops\n\nchore: run fmt\n\nfeat: check for dynamic_slices\n\nfeat: unary elementwise working\n\nchore: comments\n\nfeat: clear out unwanted no-ops from loop body\n\nfeat: elementwise ops are completely supported\n\nchore: add a note\n\nfix: bad rebase\n\nfeat: generalize index handling\n\ntest: elementwise loop fission\n\nrefactor: support dynamicsliceop\n\nfeat: sliceinfo generalize\n\nfeat: lift ops by batch op\n\nfeat: run batching pass\n\nfeat: generalize handling for non-reshape case\n\n* chore: make a note on supporting dropdim in while is copy\n\n* feat: support while is copy with dropdims = 0\n\n* feat: add to transform ops\n\n* fix: mapping\n\n* feat: support constants by lifting them into function body\n\n* feat: special handling of reshape",
          "timestamp": "2025-10-27T19:49:19-04:00",
          "tree_id": "0253de51884ecf5ff2b226afa85c8da3ee5327b6",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/992f348493d08666a9bd4305fdbfc35ff6910c0b"
        },
        "date": 1761633458474,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004391519189812243,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004307569190859795,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001399941653944,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.000138530330907,
            "unit": "s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "wmoses@google.com",
            "name": "William Moses",
            "username": "wsmoses"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "8f6fcba733909db2ed5fe0969f7544b612357de6",
          "message": "Statically linked cuda libs (#1516)\n\n* Statically linked cuda libs\n\n* Update build-reactantjll.yml\n\n* fix\n\n* Update ML_TOOLCHAIN_COMMIT to new commit hash\n\n* nvperf\n\n* Update build-reactantjll.yml",
          "timestamp": "2025-10-28T00:51:20-05:00",
          "tree_id": "518ffa0906346414c23bf31a38e5c651cd326151",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/8f6fcba733909db2ed5fe0969f7544b612357de6"
        },
        "date": 1761650741472,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004677987098693847,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004717439110390842,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001567242303979,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001577116009895,
            "unit": "s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "avikpal@mit.edu",
            "name": "Avik Pal",
            "username": "avik-pal"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "c4ad72b48c57c448e7fe6443e1d7eb94256be40b",
          "message": "fix: batch reduce region correctly with external constants (#1526)\n\n* fix: batch constantOp correctly\n\n* feat: support cloning reduce op region with outside constants",
          "timestamp": "2025-10-28T13:34:31-04:00",
          "tree_id": "3aac18ccaad8890b16e198baf336f8d5566f838b",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/c4ad72b48c57c448e7fe6443e1d7eb94256be40b"
        },
        "date": 1761682308983,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004199580405838788,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004190525400917977,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001350029852939,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001341407105908,
            "unit": "s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "xuyuansui@outlook.com",
            "name": "xys-syx",
            "username": "xys-syx"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "8dc549db2c67dd4940743b6fbca96af2cab41de5",
          "message": "Cpuify (#1415)\n\n* Port cpuify pass from polygeist\n\n* add missing files\n\n* add cpuify tests\n\n* fmt\n\n* lit tests modified\n\n* Apply clang-format\n\n* fmt\n\n* fix\n\n* Fmt\n\n* fix\n\n* fix\n\n* fix looprecur\n\n* temp using DAG in cpuifyifsplit.mlir\n\n* fmt\n\n* fmt\n\n* fmt\n\n* fmt\n\n* fix\n\n* fix\n\n* fix\n\n* fix\n\n* fix\n\n* fix\n\n* fmt:\n\n* fmt\n\n* fix\n\n* fix\n\n* fix\n\n* fix\n\n* fix\n\n* fmt\n\n* fix\n\n* fmt\n\n* fix\n\n* macos\n\n* macos\n\n* remove\n\n* remove\n\n* Fix memory error\n\n* void\n\n* fix\n\n* fix\n\n* fmt\n\n* fmt\n\n* wrapper for LoopDistribute Specific Logic\n\n* fmt\n\n---------\n\nCo-authored-by: William S. Moses <gh@wsmoses.com>",
          "timestamp": "2025-10-28T14:05:05-05:00",
          "tree_id": "bc51ee9ff8ec53ad8ca91555493034524bc71705",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/8dc549db2c67dd4940743b6fbca96af2cab41de5"
        },
        "date": 1761691433845,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004480922990478576,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004438424995169044,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001398530440987,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001646234500105,
            "unit": "s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "238314553+enzymead-bot[bot]@users.noreply.github.com",
            "name": "enzymead-bot[bot]",
            "username": "enzymead-bot[bot]"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "03a920be9c30e11ed343f962896b77511b1baf3d",
          "message": "Update EnzymeAD/Enzyme to commit aecafad17d3f63c1c0697609ad713615af322e23 (#1524)\n\nDiff: https://github.com/EnzymeAD/Enzyme/compare/63c81877e73b019745ebff33f6dc2e13a5003a17...aecafad17d3f63c1c0697609ad713615af322e23\n\nCo-authored-by: enzymead-bot[bot] <238314553+enzymead-bot[bot]@users.noreply.github.com>",
          "timestamp": "2025-10-28T18:48:46-05:00",
          "tree_id": "d731583dc274a9258bbfd4f0aa1de0828dd56a1b",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/03a920be9c30e11ed343f962896b77511b1baf3d"
        },
        "date": 1761704648521,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.00000432896800339222,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004337205993942916,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.000133810721198,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001360229832003,
            "unit": "s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "avikpal@mit.edu",
            "name": "Avik Pal",
            "username": "avik-pal"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "fbaf8a8fee101527aeb30454651d3df06a7ea3b1",
          "message": "feat: generalize elementwise to reduction (#1528)\n\n* feat: generalize elementwise to reduction\n\n* chore: run format",
          "timestamp": "2025-10-28T19:54:54-04:00",
          "tree_id": "419538688ceca40b652cbeac75d4cb27763d046e",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/fbaf8a8fee101527aeb30454651d3df06a7ea3b1"
        },
        "date": 1761710841933,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004310710995923728,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004211899999063462,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.000149751619203,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001499467342044,
            "unit": "s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "238314553+enzymead-bot[bot]@users.noreply.github.com",
            "name": "enzymead-bot[bot]",
            "username": "enzymead-bot[bot]"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "be41edd949d05733c12b59b582b14f6c9ef462c1",
          "message": "Update EnzymeAD/Enzyme to commit fdc9f4a7266c986e0efe4dbce9d9446471f640ae (#1530)\n\nDiff: https://github.com/EnzymeAD/Enzyme/compare/aecafad17d3f63c1c0697609ad713615af322e23...fdc9f4a7266c986e0efe4dbce9d9446471f640ae\n\nCo-authored-by: enzymead-bot[bot] <238314553+enzymead-bot[bot]@users.noreply.github.com>",
          "timestamp": "2025-10-29T13:23:02+01:00",
          "tree_id": "d1709e21d05246a26712671e42c4711438d78d04",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/be41edd949d05733c12b59b582b14f6c9ef462c1"
        },
        "date": 1761755145469,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004200157907325775,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004260525992140174,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001329022263991,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001370554722961,
            "unit": "s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "238314553+enzymead-bot[bot]@users.noreply.github.com",
            "name": "enzymead-bot[bot]",
            "username": "enzymead-bot[bot]"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "54fcf287b1f2c1312116a877ef233010c172bab2",
          "message": "Update EnzymeAD/Enzyme to commit 3048141052e0d9bb497950803e74c24bd5ea8ca1 (#1531)\n\nDiff: https://github.com/EnzymeAD/Enzyme/compare/fdc9f4a7266c986e0efe4dbce9d9446471f640ae...3048141052e0d9bb497950803e74c24bd5ea8ca1\n\nCo-authored-by: enzymead-bot[bot] <238314553+enzymead-bot[bot]@users.noreply.github.com>",
          "timestamp": "2025-10-30T12:54:46Z",
          "tree_id": "bde29d185a7dc3b99c0812260cc780887675ae3b",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/54fcf287b1f2c1312116a877ef233010c172bab2"
        },
        "date": 1761832717694,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004316021304111928,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004213109402917325,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001468284287024,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001375977221992,
            "unit": "s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "34986820+EganJ@users.noreply.github.com",
            "name": "Egan",
            "username": "EganJ"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "12bca85ccbf361c5c7d3017c0b77af1a5e8cf30f",
          "message": "Bump hedron hash to fix compile commands issue (#1423)",
          "timestamp": "2025-10-30T15:11:13-05:00",
          "tree_id": "5ebb8010a494981720e1935fac94bcb7567a6b3c",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/12bca85ccbf361c5c7d3017c0b77af1a5e8cf30f"
        },
        "date": 1761868531450,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000006158093002159148,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004402495000977069,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001326389673049,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001374468563008,
            "unit": "s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "gh@wsmoses.com",
            "name": "William S. Moses",
            "username": "wsmoses"
          },
          "committer": {
            "email": "gh@wsmoses.com",
            "name": "William S. Moses",
            "username": "wsmoses"
          },
          "distinct": true,
          "id": "0a93f68f15e22ef5711857db0d050e5e66680694",
          "message": "separately consider cudnn",
          "timestamp": "2025-10-30T17:37:25-05:00",
          "tree_id": "5c03ad0eee6772a86d5d5cb605333d09a72570ee",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/0a93f68f15e22ef5711857db0d050e5e66680694"
        },
        "date": 1761886252909,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004365037300158292,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004191758390516042,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001382287155021,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001379254845087,
            "unit": "s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "wmoses@google.com",
            "name": "William Moses",
            "username": "wsmoses"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "77a8d3d05359d470c6a103e643d129a430722d2a",
          "message": "Add remove enzyme ops to pass list (#1536)",
          "timestamp": "2025-10-30T19:16:32-05:00",
          "tree_id": "974530eb996f1c910b3fe852ad8f045e671d1111",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/77a8d3d05359d470c6a103e643d129a430722d2a"
        },
        "date": 1761901041164,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004398716997820884,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004282150999642908,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001329769102972,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001306660962989,
            "unit": "s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "238314553+enzymead-bot[bot]@users.noreply.github.com",
            "name": "enzymead-bot[bot]",
            "username": "enzymead-bot[bot]"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "73ee4b92e04a17fc5ec6c2663e7fecb1ac98bd6e",
          "message": "Update jax-ml/jax to commit f6edb899a75dad482a687fe42c8a81d1eaa3aac5 (#1539)\n\nDiff: https://github.com/jax-ml/jax/compare/7d256f847561f57f290b74468b99a8b0da79cd1a...f6edb899a75dad482a687fe42c8a81d1eaa3aac5\n\nCo-authored-by: enzymead-bot[bot] <238314553+enzymead-bot[bot]@users.noreply.github.com>",
          "timestamp": "2025-11-01T13:19:18Z",
          "tree_id": "166b954f7e341ecc89a6be8d03ad421e69dbe0bf",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/73ee4b92e04a17fc5ec6c2663e7fecb1ac98bd6e"
        },
        "date": 1762015537511,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004233665904030204,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.0000041352519067004326,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001391577433096,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001395150692202,
            "unit": "s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "238314553+enzymead-bot[bot]@users.noreply.github.com",
            "name": "enzymead-bot[bot]",
            "username": "enzymead-bot[bot]"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "c432749dde4d6ca23bb358cc7dfcfd54cfefc3b1",
          "message": "Update jax-ml/jax to commit f386951c08a82781f061ef3ebb3ff28e7a617ead (#1543)\n\nDiff: https://github.com/jax-ml/jax/compare/f6edb899a75dad482a687fe42c8a81d1eaa3aac5...f386951c08a82781f061ef3ebb3ff28e7a617ead\n\nCo-authored-by: enzymead-bot[bot] <238314553+enzymead-bot[bot]@users.noreply.github.com>",
          "timestamp": "2025-11-01T21:54:11-05:00",
          "tree_id": "38739e5f995e75e6c142dc62768a698e08d240f7",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/c432749dde4d6ca23bb358cc7dfcfd54cfefc3b1"
        },
        "date": 1762076905340,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004386042291298509,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004269602289423346,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001422204643022,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001438122397055,
            "unit": "s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "wmoses@google.com",
            "name": "William Moses",
            "username": "wsmoses"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "4a22980dd3b3ffa65e81c8273b41483035f8ea02",
          "message": "Fix infinite loop ori disjoint (#1547)",
          "timestamp": "2025-11-02T09:30:18-06:00",
          "tree_id": "65f0f4954e15fc5dc09b97acc7a63e5bbde16aed",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/4a22980dd3b3ffa65e81c8273b41483035f8ea02"
        },
        "date": 1762108884746,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.0000043374820001190525,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.00000406218699990859,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001517361998001,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001501404909,
            "unit": "s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "avikpal@mit.edu",
            "name": "Avik Pal",
            "username": "avik-pal"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "b3d6fde1c3d150e8c0be3363c9e234e349aa13cd",
          "message": "feat: iota tensor detection + indirect iota indexing simplification (#1542)\n\n* feat: iota tensor detection\n\n* chore: run fmt\n\n* feat: rewrite iota ops\n\n* test: indirect indexing\n\n* feat: support more iota like ops for scatter detection",
          "timestamp": "2025-11-02T11:06:12-05:00",
          "tree_id": "608a05ee69d0011cda2f5c2d0fdb0942828d6f44",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/b3d6fde1c3d150e8c0be3363c9e234e349aa13cd"
        },
        "date": 1762120582721,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004345976980403066,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004376305988989771,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001504514841828,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001366461332887,
            "unit": "s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "avikpal@mit.edu",
            "name": "Avik Pal",
            "username": "avik-pal"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "4b250056b4c4257c5be6f10a402a1c1099b3e207",
          "message": "feat: dot_general_licm (#1550)",
          "timestamp": "2025-11-02T15:15:56-05:00",
          "tree_id": "63eba01503f1436bee5a133add791d44b5289dad",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/4b250056b4c4257c5be6f10a402a1c1099b3e207"
        },
        "date": 1762134942837,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004289858999982243,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.0000041430409000895455,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001352644156999,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001352168267003,
            "unit": "s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "avikpal@mit.edu",
            "name": "Avik Pal",
            "username": "avik-pal"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "319852ba6108e8a651f411ee816bec27a65ddc7c",
          "message": "fix: guaranteed analysis bfs queue was busted (#1538)\n\n* fix: guaranteed analysis bfs queue was busted\n\n* chore: fmt\n\n* fix: try making things deterministic\n\n* chore: bazel format\n\n* fix: address review comments\n\n* fix: revert to SmallPtrSet\n\n* chore: run fmt\n\n* fix: reuse found value",
          "timestamp": "2025-11-02T20:36:13-05:00",
          "tree_id": "a2fe7ef64c992f29e1b72db005bf2039427cb3b1",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/319852ba6108e8a651f411ee816bec27a65ddc7c"
        },
        "date": 1762149886304,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.0000043059209827333686,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004307427001185715,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001530488142045,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001533427932066,
            "unit": "s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "238314553+enzymead-bot[bot]@users.noreply.github.com",
            "name": "enzymead-bot[bot]",
            "username": "enzymead-bot[bot]"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "5c0b45a67a5b4c94d44383a1235e5d91a1b6fbfb",
          "message": "Update jax-ml/jax to commit 6537c437b48ebe8cb7696d4da693652802f8a776 (#1555)\n\nDiff: https://github.com/jax-ml/jax/compare/f386951c08a82781f061ef3ebb3ff28e7a617ead...6537c437b48ebe8cb7696d4da693652802f8a776\n\nCo-authored-by: enzymead-bot[bot] <238314553+enzymead-bot[bot]@users.noreply.github.com>",
          "timestamp": "2025-11-02T22:07:24-06:00",
          "tree_id": "a3d895eab1a8fc23385ad2403c16c5cfbabfe5e0",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/5c0b45a67a5b4c94d44383a1235e5d91a1b6fbfb"
        },
        "date": 1762154523257,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004331737999746111,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000003905613000097219,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001482329919002,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001482432529999,
            "unit": "s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "avikpal@mit.edu",
            "name": "Avik Pal",
            "username": "avik-pal"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "283c7d77ae625786387aad45537f746a77ceec42",
          "message": "feat: export enzyme_hlo_unroll via transform (#1560)",
          "timestamp": "2025-11-03T12:52:00-05:00",
          "tree_id": "2f925cb46cb7780ec4a0e836dbfa1869e661cbc7",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/283c7d77ae625786387aad45537f746a77ceec42"
        },
        "date": 1762203466429,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004386635900300462,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004332715999044012,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001515112646011,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001527759867007,
            "unit": "s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "avikpal@mit.edu",
            "name": "Avik Pal",
            "username": "avik-pal"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "4b6f35da1a180014640dc7ab1a3fd5fa881df77a",
          "message": "fix: bump enzyme commit and make linker happy (#1562)",
          "timestamp": "2025-11-03T16:21:11-05:00",
          "tree_id": "a415fa78cd4a3de078a2e2cbed25dfa4b7194a54",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/4b6f35da1a180014640dc7ab1a3fd5fa881df77a"
        },
        "date": 1762213944923,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004540896898834035,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004838709998875857,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001475114347995,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001373216477993,
            "unit": "s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "avikpal@mit.edu",
            "name": "Avik Pal",
            "username": "avik-pal"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "f729dca2048c2cfd941877cf23c1556368d04ac3",
          "message": "feat: reduce licm passes (#1564)",
          "timestamp": "2025-11-03T18:40:19-05:00",
          "tree_id": "5c674b7cf1dc3eff9f5171535c332d174a6d511e",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/f729dca2048c2cfd941877cf23c1556368d04ac3"
        },
        "date": 1762247542361,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004296338000131073,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.00000426451699968311,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001563015472005,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001416123602,
            "unit": "s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "238314553+enzymead-bot[bot]@users.noreply.github.com",
            "name": "enzymead-bot[bot]",
            "username": "enzymead-bot[bot]"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "60fe04358e8ad8be9309e87df9b286c805f019ff",
          "message": "Update jax-ml/jax to commit f59e61a2cff82c9895aab75221719002be7c17e1 (#1565)\n\nDiff: https://github.com/jax-ml/jax/compare/6537c437b48ebe8cb7696d4da693652802f8a776...f59e61a2cff82c9895aab75221719002be7c17e1\n\nCo-authored-by: enzymead-bot[bot] <238314553+enzymead-bot[bot]@users.noreply.github.com>",
          "timestamp": "2025-11-03T22:22:26-06:00",
          "tree_id": "7e2e6f3e4627aeb1b6f3576a727f398b0097f359",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/60fe04358e8ad8be9309e87df9b286c805f019ff"
        },
        "date": 1762272278255,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004435663999174722,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004404819900810253,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001602005312,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001548038672,
            "unit": "s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "avikpal@mit.edu",
            "name": "Avik Pal",
            "username": "avik-pal"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "bca1e1c04973fb10eeee51614f648df548451561",
          "message": "feat: expand coverage of noop_reverse (#1568)",
          "timestamp": "2025-11-04T09:18:20-06:00",
          "tree_id": "e597366df866e5133ae71544ff5d31c2b3969fee",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/bca1e1c04973fb10eeee51614f648df548451561"
        },
        "date": 1762284186382,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004356875998200849,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004155734999221749,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001452637479989,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001431011000007,
            "unit": "s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "avikpal@mit.edu",
            "name": "Avik Pal",
            "username": "avik-pal"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "c051c600464bf9e8936bd7b17262669d73be6e02",
          "message": "feat: enable raising more operations from loops (#1567)\n\n* feat: enable raising more operations from loops\n\n* test: add test cases",
          "timestamp": "2025-11-04T16:14:10-05:00",
          "tree_id": "e50df750ccef5af950a1008f86c72b7e41b9a35d",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/c051c600464bf9e8936bd7b17262669d73be6e02"
        },
        "date": 1762303064974,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004429023999546189,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.00000442851999978302,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001343285013004,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001343496753004,
            "unit": "s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "avikpal@mit.edu",
            "name": "Avik Pal",
            "username": "avik-pal"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "04d3efd7363e3499e4ad92c9835b2f2ec0401116",
          "message": "feat: optimized conv batching (#1569)\n\n* refactor: cleanup generic batch op interface to use batchutils\n\n* feat: better conv batching\n\n* test: conv batching\n\n* fix: stop using deprecated api\n\n* feat: exploit the batch_group_dim for batching kernel\n\n* chore: run fmt\n\n* test: update\n\n* fix: mimic jax batching\n\n* fix: layout\n\n* test: update",
          "timestamp": "2025-11-04T22:31:17-05:00",
          "tree_id": "2542f308d8143af4d3b10e9161527a03b04495e0",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/04d3efd7363e3499e4ad92c9835b2f2ec0401116"
        },
        "date": 1762319835724,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004327598999952898,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004309819897753186,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001458566345012,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001412948345008,
            "unit": "s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "gh@wsmoses.com",
            "name": "William S. Moses",
            "username": "wsmoses"
          },
          "committer": {
            "email": "gh@wsmoses.com",
            "name": "William S. Moses",
            "username": "wsmoses"
          },
          "distinct": true,
          "id": "cbde16a31c2acde41297261b3a479accb9662219",
          "message": "Rocm tmpdir",
          "timestamp": "2025-11-04T23:32:36-06:00",
          "tree_id": "e175f483360643469d884abd509b33480b94d70c",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/cbde16a31c2acde41297261b3a479accb9662219"
        },
        "date": 1762325782688,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.00000444902989838738,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004668452998157591,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.000145394529798,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001432762677985,
            "unit": "s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "gh@wsmoses.com",
            "name": "William S. Moses",
            "username": "wsmoses"
          },
          "committer": {
            "email": "gh@wsmoses.com",
            "name": "William S. Moses",
            "username": "wsmoses"
          },
          "distinct": true,
          "id": "f937a462ceaf28a0008c7aa9a03c6fc4c3a6c7c9",
          "message": "Fix rocm",
          "timestamp": "2025-11-05T00:10:19-06:00",
          "tree_id": "603551bab435d152ef56663a050df2abb37c518e",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/f937a462ceaf28a0008c7aa9a03c6fc4c3a6c7c9"
        },
        "date": 1762333943211,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004327470000134781,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004281620000256225,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001368987841997,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001384789892006,
            "unit": "s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "238314553+enzymead-bot[bot]@users.noreply.github.com",
            "name": "enzymead-bot[bot]",
            "username": "enzymead-bot[bot]"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "f38cddf4d496b40d06152c527aad8375560f7e77",
          "message": "Update EnzymeAD/Enzyme to commit 032828cbfef50bfba41443baacc39989c203534b (#1577)\n\n* Update EnzymeAD/Enzyme to commit 032828cbfef50bfba41443baacc39989c203534b\n\nDiff: https://github.com/EnzymeAD/Enzyme/compare/0ce301aedef3ca040c8703cb1b7d340ed4a58271...032828cbfef50bfba41443baacc39989c203534b\n\n* fix\n\nfix\n\n---------\n\nCo-authored-by: enzymead-bot[bot] <238314553+enzymead-bot[bot]@users.noreply.github.com>\nCo-authored-by: sbrantq <scharfrichterq@gmail.com>",
          "timestamp": "2025-11-05T13:47:13-06:00",
          "tree_id": "a73dd2e5e24890f018de2417d05101adb9bd01c6",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/f38cddf4d496b40d06152c527aad8375560f7e77"
        },
        "date": 1762385658093,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004294576001120731,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004324483900563791,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001519668891996,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001453466621984,
            "unit": "s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "238314553+enzymead-bot[bot]@users.noreply.github.com",
            "name": "enzymead-bot[bot]",
            "username": "enzymead-bot[bot]"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "b348d4ac435535038dfceb8e8903d1af12b44e69",
          "message": "Update jax-ml/jax to commit eb2d56b99ceef305933d9293d5c1715fdb333950 (#1581)\n\nDiff: https://github.com/jax-ml/jax/compare/e9609ce42f272e0a4e908b9e16ea81239e76385c...eb2d56b99ceef305933d9293d5c1715fdb333950\n\nCo-authored-by: enzymead-bot[bot] <238314553+enzymead-bot[bot]@users.noreply.github.com>",
          "timestamp": "2025-11-05T20:34:28-06:00",
          "tree_id": "fc4ebe8602bc2a1543ddcfbffd7774a64ff8e754",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/b348d4ac435535038dfceb8e8903d1af12b44e69"
        },
        "date": 1762409921201,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004288672000075167,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004136795999875176,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001309080777999,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001316320257999,
            "unit": "s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "avikpal@mit.edu",
            "name": "Avik Pal",
            "username": "avik-pal"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "4237dcbc24e14414ee6eaad08e88c88af6f06734",
          "message": "feat: better analysis for lifting copy like operations (#1582)",
          "timestamp": "2025-11-06T00:03:37-05:00",
          "tree_id": "d814204036df11e696fbed28b7817e1ff8ebb633",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/4237dcbc24e14414ee6eaad08e88c88af6f06734"
        },
        "date": 1762419628358,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.00000435075820023485,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004298714199831011,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001354317992001,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001346981126,
            "unit": "s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "238314553+enzymead-bot[bot]@users.noreply.github.com",
            "name": "enzymead-bot[bot]",
            "username": "enzymead-bot[bot]"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "0503b6153c08ad51ad4a7e1c7f8d0c1b5fad3c01",
          "message": "Update EnzymeAD/Enzyme to commit 6b1848d8582e57dd57c0bb5d0c373c5cb1c1bbfb (#1585)\n\nDiff: https://github.com/EnzymeAD/Enzyme/compare/032828cbfef50bfba41443baacc39989c203534b...6b1848d8582e57dd57c0bb5d0c373c5cb1c1bbfb\n\nCo-authored-by: enzymead-bot[bot] <238314553+enzymead-bot[bot]@users.noreply.github.com>",
          "timestamp": "2025-11-06T14:20:28+01:00",
          "tree_id": "f2d2537c743156b5c38420e2a01c8f5a6f854662",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/0503b6153c08ad51ad4a7e1c7f8d0c1b5fad3c01"
        },
        "date": 1762453760339,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004331407599966042,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.0000045653804998437406,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001360524760995,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001328413247996,
            "unit": "s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "gh@wsmoses.com",
            "name": "William S. Moses",
            "username": "wsmoses"
          },
          "committer": {
            "email": "gh@wsmoses.com",
            "name": "William S. Moses",
            "username": "wsmoses"
          },
          "distinct": true,
          "id": "edb3814fe6ab18ed5ef36a472e048c893f62bd72",
          "message": "new rocm patch",
          "timestamp": "2025-11-06T10:13:59-06:00",
          "tree_id": "17aadd9259222be66b68b03bb74b06a79ce59ad2",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/edb3814fe6ab18ed5ef36a472e048c893f62bd72"
        },
        "date": 1762457586560,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004348287099855952,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004197223200753797,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001466890640993,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001428062787992,
            "unit": "s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "gh@wsmoses.com",
            "name": "William S. Moses",
            "username": "wsmoses"
          },
          "committer": {
            "email": "gh@wsmoses.com",
            "name": "William S. Moses",
            "username": "wsmoses"
          },
          "distinct": true,
          "id": "ebf0544faa96f7eff3bee167a6c42b62f89f9c82",
          "message": "More rocm fix",
          "timestamp": "2025-11-06T16:22:38-06:00",
          "tree_id": "514529a591ad5cbddd4460d0deb8e3d3227cda95",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/ebf0544faa96f7eff3bee167a6c42b62f89f9c82"
        },
        "date": 1762477806027,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004396587099472526,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004314664100820664,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001368273229993,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001318798172011,
            "unit": "s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "avikpal@mit.edu",
            "name": "Avik Pal",
            "username": "avik-pal"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "51d94b351ff8aaf7c6f3aae1c949f3611c394c7d",
          "message": "feat: sort autodiff rules (#1584)\n\n* feat: sort forward mode AD\n\n* feat: reverse mode\n\n* refactor: move the common function\n\n* fix: derivative rule\n\n* test: update\n\n* fix: replace in cacheValues",
          "timestamp": "2025-11-06T18:55:07-05:00",
          "tree_id": "2678042ccacdf96ce90fe6cf038902e1cd435602",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/51d94b351ff8aaf7c6f3aae1c949f3611c394c7d"
        },
        "date": 1762485258317,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004135624998889398,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004203124999185093,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001434340536012,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001441685565005,
            "unit": "s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "238314553+enzymead-bot[bot]@users.noreply.github.com",
            "name": "enzymead-bot[bot]",
            "username": "enzymead-bot[bot]"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "2347ef5292dc2c2c7c3a9d3d987540ef68e3a967",
          "message": "Update EnzymeAD/Enzyme to commit 68e62fbd1b496a60490266952916112b63e62a18 (#1588)\n\nDiff: https://github.com/EnzymeAD/Enzyme/compare/6b1848d8582e57dd57c0bb5d0c373c5cb1c1bbfb...68e62fbd1b496a60490266952916112b63e62a18\n\nCo-authored-by: enzymead-bot[bot] <238314553+enzymead-bot[bot]@users.noreply.github.com>",
          "timestamp": "2025-11-07T11:58:31Z",
          "tree_id": "f802eb19af9d2a3c6d2f94d0a7d7f8b0982184b9",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/2347ef5292dc2c2c7c3a9d3d987540ef68e3a967"
        },
        "date": 1762524211386,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004325116000836715,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004328913100471254,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001335319516001,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001367936525988,
            "unit": "s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "avikpal@mit.edu",
            "name": "Avik Pal",
            "username": "avik-pal"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "8c0491d9ca738386e10d1a79aa47a7b738ce0f79",
          "message": "refactor: move common functionalities out of Ops.cpp (#1590)\n\n* refactor: move common functionalities out of Ops.cpp\n\n* fix: missing build deps",
          "timestamp": "2025-11-07T11:28:32-05:00",
          "tree_id": "f69b7c48ed31398d2849b718eed4dd6fa1396b91",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/8c0491d9ca738386e10d1a79aa47a7b738ce0f79"
        },
        "date": 1762537338016,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004395995000959374,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004284164001001045,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001479914346011,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001657945766986,
            "unit": "s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "avikpal@mit.edu",
            "name": "Avik Pal",
            "username": "avik-pal"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "6a0abebb67ddc460dabf6b57de373a2682d5493c",
          "message": "feat: support general intermediate reshape in auto-batching (#1575)\n\n* feat: exploit the batch_group_dim for batching kernel\n\n* fix: mimic jax batching\n\n* feat: support multiple dropdims intermediate\n\n* feat: support general intermediate reshapes",
          "timestamp": "2025-11-07T11:37:04-05:00",
          "tree_id": "ffafe9296a84e1dc74f0ed9751ff00883676219e",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/6a0abebb67ddc460dabf6b57de373a2682d5493c"
        },
        "date": 1762544289842,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000003918430398334749,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004305170100997202,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001320323591993,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001326188359991,
            "unit": "s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "avikpal@mit.edu",
            "name": "Avik Pal",
            "username": "avik-pal"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "32eeef933b3df77237434cc5f3ceed3de12772a4",
          "message": "feat: support partial indexing to batch (#1587)\n\n* feat: exploit the batch_group_dim for batching kernel\n\n* fix: mimic jax batching\n\n* feat: support partial indexing to batch\n\n* chore: run fmt",
          "timestamp": "2025-11-07T13:39:48-05:00",
          "tree_id": "6c383208a1e82e32602cfed2fb8e4c619a2923e0",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/32eeef933b3df77237434cc5f3ceed3de12772a4"
        },
        "date": 1762549163679,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004371509997872636,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004321837998577393,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001365762186993,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001342667606018,
            "unit": "s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "avikpal@mit.edu",
            "name": "Avik Pal",
            "username": "avik-pal"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "990b6706bcbae8fdd7d339f034bb93eda38d1223",
          "message": "feat: triton_ext dialect (#1591)\n\n* feat: triton_ext dialect\n\n* fix: remove nvgpu for now\n\n* chore: run fmt\n\n* fix: namespace\n\n* chore: run fmt\n\n* test: add simple add_kernel test",
          "timestamp": "2025-11-07T14:48:39-05:00",
          "tree_id": "eafe09eaf8f1731f2c181ffc5afb95e2f6cbe5b5",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/990b6706bcbae8fdd7d339f034bb93eda38d1223"
        },
        "date": 1762555714172,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.00000438906199997291,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004388130101142451,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001394996970979,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001417176780989,
            "unit": "s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "avikpal@mit.edu",
            "name": "Avik Pal",
            "username": "avik-pal"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "6e58b86686d04b2f803925a9ecc0021fbec1102d",
          "message": "fix: infinite raising + correct broadcast shape (#1593)",
          "timestamp": "2025-11-07T19:06:25-05:00",
          "tree_id": "b272fd5ada5ff9fd9021bf78fc923c36c99bdeda",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/6e58b86686d04b2f803925a9ecc0021fbec1102d"
        },
        "date": 1762563418795,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004448425001464784,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.00000446261799952481,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001502956085983,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001499391246004,
            "unit": "s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "238314553+enzymead-bot[bot]@users.noreply.github.com",
            "name": "enzymead-bot[bot]",
            "username": "enzymead-bot[bot]"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "24641f33ea39c03c45b07ce99e2881ad50128538",
          "message": "Update jax-ml/jax to commit 30e565311af559569b4842bddced4b461f21dd73 (#1586)\n\n* Update jax-ml/jax to commit 30e565311af559569b4842bddced4b461f21dd73\n\nDiff: https://github.com/jax-ml/jax/compare/eb2d56b99ceef305933d9293d5c1715fdb333950...30e565311af559569b4842bddced4b461f21dd73\n\n* fix\n\n* remove rocm tmpdir patch\n\n---------\n\nCo-authored-by: enzymead-bot[bot] <238314553+enzymead-bot[bot]@users.noreply.github.com>\nCo-authored-by: William S. Moses <gh@wsmoses.com>",
          "timestamp": "2025-11-07T22:51:51-06:00",
          "tree_id": "289469ea040dab7ed40380ae1a6f4377d7fa9e5c",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/24641f33ea39c03c45b07ce99e2881ad50128538"
        },
        "date": 1762585132339,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004388645000290125,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004541678001987748,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001507710175996,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001508200245996,
            "unit": "s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "scharfrichterq@gmail.com",
            "name": "Acake",
            "username": "sbrantq"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "c9ad2b9fadb847b303202e2fc5239516bb7f7a92",
          "message": "fix (#1595)",
          "timestamp": "2025-11-09T00:27:44-06:00",
          "tree_id": "c5118b4ea8659305f17c0022511e85b74b0f128e",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/c9ad2b9fadb847b303202e2fc5239516bb7f7a92"
        },
        "date": 1762684872453,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000008794522285461425,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000009516980312764643,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / gpu / Primal",
            "value": 0.00007948229289613665,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / gpu / Primal",
            "value": 0.00007941056929994375,
            "unit": "s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "avikpal@mit.edu",
            "name": "Avik Pal",
            "username": "avik-pal"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "c86285ad203b75530dcb5107c131f0820a00ab70",
          "message": "feat: support iota with offset (#1599)",
          "timestamp": "2025-11-09T23:22:30-05:00",
          "tree_id": "3af779cf0ba46380c2ed7c0ad0ed1ee56c34dab3",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/c86285ad203b75530dcb5107c131f0820a00ab70"
        },
        "date": 1762761373801,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004515824798727408,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004259957897011191,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001276962814969,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001333710766979,
            "unit": "s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "238314553+enzymead-bot[bot]@users.noreply.github.com",
            "name": "enzymead-bot[bot]",
            "username": "enzymead-bot[bot]"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "3bb71d558e5cdd7a8150fa7b0a94a600643202d5",
          "message": "Update EnzymeAD/Enzyme to commit 300e6f7913407b1216bf07b5ea8827aae963c898 (#1597)\n\nDiff: https://github.com/EnzymeAD/Enzyme/compare/68e62fbd1b496a60490266952916112b63e62a18...300e6f7913407b1216bf07b5ea8827aae963c898\n\nCo-authored-by: enzymead-bot[bot] <238314553+enzymead-bot[bot]@users.noreply.github.com>",
          "timestamp": "2025-11-10T01:22:53-06:00",
          "tree_id": "b7bc9cfcf0b203555a8f2b5082939b8778dfd618",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/3bb71d558e5cdd7a8150fa7b0a94a600643202d5"
        },
        "date": 1762768111720,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004388973099412396,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.00000432439599535428,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001340801675978,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001337204235023,
            "unit": "s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "naydex.mc+github@gmail.com",
            "name": "Paul Berg",
            "username": "Pangoraw"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "47d57c1cea7b24e210ad75aee6e7c3f93d89ff78",
          "message": "Triton forward mode AD (#1578)\n\n* Triton forward mode AD\n\n* fmt build\n\n* header\n\n* Bump enzyme\n\n* Update TritonDerivatives.td\n\n---------\n\nCo-authored-by: Mosè Giordano <765740+giordano@users.noreply.github.com>\nCo-authored-by: William Moses <wmoses@google.com>",
          "timestamp": "2025-11-10T09:26:15+01:00",
          "tree_id": "8ccf271a1cecd218e105271802b5b5bd983f7fcd",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/47d57c1cea7b24e210ad75aee6e7c3f93d89ff78"
        },
        "date": 1762785887758,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000003993225202430039,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.00000422822319669649,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001449365721025,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001456860908016,
            "unit": "s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "avikpal@mit.edu",
            "name": "Avik Pal",
            "username": "avik-pal"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "98f3f359eddb61127d85fe7e814af655e801692b",
          "message": "feat: partial support for broadcast elementwise trait (#1602)",
          "timestamp": "2025-11-10T14:33:32-05:00",
          "tree_id": "d80d63234619f36083ca084e0113bb51f541b5b6",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/98f3f359eddb61127d85fe7e814af655e801692b"
        },
        "date": 1762812432988,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004391877097077668,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004393432103097439,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001304379814013,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001292214179993,
            "unit": "s"
          }
        ]
      }
    ]
  }
}