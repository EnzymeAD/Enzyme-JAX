window.BENCHMARK_DATA = {
  "lastUpdate": 1765475613169,
  "repoUrl": "https://github.com/EnzymeAD/Enzyme-JAX",
  "entries": {
    "EnzymeJAX Benchmarks": [
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
          "id": "aded9a2821774cdfa05c6207afa3b92e16b6d583",
          "message": "fix: use lapack instead of lapacke (#1618)\n\nfeat: get_dimension_size batch interface\n\nfeat: implement jitcall batching with shlo_generic_batch_op_interface\n\nrefactor: reuse batching interface for LU factorization\n\nfix: remove old changes\n\nfeat: mark memory effects\n\nfix: update to new API\n\nfix: use correct return\n\nfix: rework svd lowering to use lapack and workspace buffers\n\nfix: use func return\n\nfix: leading dimension\n\nfix: use backend config\n\nfix: remove unused vars\n\nfix: tpu lowering\n\ntest: update",
          "timestamp": "2025-11-17T00:05:45-05:00",
          "tree_id": "ddffb8cb739571b2b0e3cf6765c4b1a2fa659fea",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/aded9a2821774cdfa05c6207afa3b92e16b6d583"
        },
        "date": 1763364371985,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004386482108384371,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004358096001669764,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001448858825955,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001450719916028,
            "unit": "s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "58750937+snonk@users.noreply.github.com",
            "name": "snonk",
            "username": "snonk"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "8c2e29e2ba68df7f492a3d4ef9b48e5f97a725c5",
          "message": "add constant int/float check for symm (#1621)\n\n* constant check\n\n* fmt\n\n* add splat and broadcast:\n\n* fix: formatting\n\n* refactor: cleanup into a single function\n\n* add test\n\n---------\n\nCo-authored-by: Avik Pal <avikpal@mit.edu>",
          "timestamp": "2025-11-18T01:46:55-06:00",
          "tree_id": "62e40d6f02428e2a18c4e740a44104a36d124a98",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/8c2e29e2ba68df7f492a3d4ef9b48e5f97a725c5"
        },
        "date": 1763474814552,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004369421099909232,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004413862099681864,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001363153289999,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001475828000999,
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
          "id": "b55ffd4f69428efae99ca2178b4b0672b97b33d3",
          "message": "feat: progressive lowering of LU and SVD via lapack ops (#1625)\n\n* feat: more lapack operations for LU/SVD\n\n* fix: lower linalg ops to lapack\n\n* chore: stub code for lowering\n\n* feat: lowering for getrf\n\n* feat: svd lapack op lowering\n\n* feat: cpu lowering\n\n* fix: missing impl in header\n\n* fix: export names\n\n* fix: update workspace buffer correctly\n\n* test: update all tests\n\n* feat: support batching",
          "timestamp": "2025-11-18T08:00:16-05:00",
          "tree_id": "b49ce610257a6818b24253418256646e4de4f917",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/b55ffd4f69428efae99ca2178b4b0672b97b33d3"
        },
        "date": 1763479651450,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004338932399696205,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004298453299998073,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001558048111997,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001522772955002,
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
          "id": "2b7c1da4ae9bef7d8aa96f9c9009aec509e510a5",
          "message": "feat: don't compute UV in SVD unless needed (#1632)\n\n* feat: don't compute UV in SVD unless needed\n\n* fix: call convention for cusolver\n\n* fix: aliasing in cpu custom_call\n\n* feat: lower to job=N when compute_uv=false",
          "timestamp": "2025-11-20T00:06:20-06:00",
          "tree_id": "a34678503a58349c91817c11c89a61ebc22fa515",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/2b7c1da4ae9bef7d8aa96f9c9009aec509e510a5"
        },
        "date": 1763639530241,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004362374899938004,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004327750999800628,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001545558818004,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001537393587001,
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
          "id": "94bdf8cd9bd5a023dbe7153ec7092db20f4a2b4c",
          "message": "Update EnzymeAD/Enzyme to commit 7c521437a9ffeea730e843d48da06f40824b0c71 (#1634)\n\nDiff: https://github.com/EnzymeAD/Enzyme/compare/6b4a73e3c71e6451c919850acf2999ee04daab12...7c521437a9ffeea730e843d48da06f40824b0c71\n\nCo-authored-by: enzymead-bot[bot] <238314553+enzymead-bot[bot]@users.noreply.github.com>",
          "timestamp": "2025-11-20T10:58:49-05:00",
          "tree_id": "ccccd8579891b436de00bd0ebaddbcb91247ff93",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/94bdf8cd9bd5a023dbe7153ec7092db20f4a2b4c"
        },
        "date": 1763670745875,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004348259000107646,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004332027999771526,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001330425719002,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001410325577999,
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
          "id": "3c38d1fd809f0f7e246bf3ba2f535195dabe78cb",
          "message": "fix: analysis should be run on Values (not Operations) (#1633)\n\n* fix: analysis should be run on Values (not Operations)\n\n* refactor: abstract away the constant check\n\n* fix: remove PENDING\n\n* fix: make naming better\n\n* fix: drop unsupported attributes\n\n* feat: C API\n\n* feat: read attrs from block args\n\n* test: more testing\n\n* fix: C API",
          "timestamp": "2025-11-20T14:44:42-05:00",
          "tree_id": "a8c08f13a1cab3ebf99b5a192e2ebdbcd6ed38a3",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/3c38d1fd809f0f7e246bf3ba2f535195dabe78cb"
        },
        "date": 1763695749000,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004347273900930304,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004293463900103233,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001352114348992,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001364999429992,
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
          "id": "3c38d1fd809f0f7e246bf3ba2f535195dabe78cb",
          "message": "fix: analysis should be run on Values (not Operations) (#1633)\n\n* fix: analysis should be run on Values (not Operations)\n\n* refactor: abstract away the constant check\n\n* fix: remove PENDING\n\n* fix: make naming better\n\n* fix: drop unsupported attributes\n\n* feat: C API\n\n* feat: read attrs from block args\n\n* test: more testing\n\n* fix: C API",
          "timestamp": "2025-11-20T14:44:42-05:00",
          "tree_id": "a8c08f13a1cab3ebf99b5a192e2ebdbcd6ed38a3",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/3c38d1fd809f0f7e246bf3ba2f535195dabe78cb"
        },
        "date": 1763705200732,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004347273900930304,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004293463900103233,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001352114348992,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001364999429992,
            "unit": "s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "39610523+vimarsh6739@users.noreply.github.com",
            "name": "Vimarsh Sathia",
            "username": "vimarsh6739"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "da7ed39b534102b1e24a1758584a98d9c79e05e5",
          "message": "Enzyme batch to StableHLO (#1556)\n\n* Add raising skeleton\n\n* Add ConcatOp lowering, fix lit test\nNeed to find another cafe to work...\n\n* fix lit tests for concatenateOp\n\n* just add conversion for Extract assuming index is 0-dim tensor\n\n* Finalize changes\n\n* fix lit",
          "timestamp": "2025-11-21T15:30:39-06:00",
          "tree_id": "c43567fed636223aed56454f7208ee76e117a5ba",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/da7ed39b534102b1e24a1758584a98d9c79e05e5"
        },
        "date": 1763789396324,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004360322000047745,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.00000430330099998173,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001381765273,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001457332371999,
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
          "id": "ea42b0233460eacd72563622ee9103711462c70e",
          "message": "Update EnzymeAD/Enzyme to commit dd29921570c549f98607f792384efffa35f8e1bc (#1637)\n\nDiff: https://github.com/EnzymeAD/Enzyme/compare/ebfdfd08ee7427f7de2f8184442d39dc8ea29b76...dd29921570c549f98607f792384efffa35f8e1bc\n\nCo-authored-by: enzymead-bot[bot] <238314553+enzymead-bot[bot]@users.noreply.github.com>",
          "timestamp": "2025-11-22T13:56:48-06:00",
          "tree_id": "c438968518ea3f2584b0319775d093811e22d50b",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/ea42b0233460eacd72563622ee9103711462c70e"
        },
        "date": 1763847963656,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004246331000467762,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004317962999630254,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001486625116012,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001533850926993,
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
          "id": "9a198274188537b5f832a9a25359f0be4b01255d",
          "message": "fix: missing patterns in transform dialect (#1639)\n\n* fix: missing patterns in transform dialect\n\n* feat: convolution_licm\n\n* chore: run format",
          "timestamp": "2025-11-22T18:53:16-05:00",
          "tree_id": "1b262845fd4e7f854effbc2598d6360b7ef022be",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/9a198274188537b5f832a9a25359f0be4b01255d"
        },
        "date": 1763872724839,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.00000435271799997281,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004273902099976112,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001496146923999,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001404514594,
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
          "id": "a2c7606121fad722e5c2acf34965afbc1af80166",
          "message": "feat: div/mul with negated operands simplify (#1640)\n\n* feat: div/mul with negated operands simplify\n\n* feat: ReplaceSubtractNegWithAdd",
          "timestamp": "2025-11-22T21:24:04-05:00",
          "tree_id": "ff0407b23e39fbaf6f8b0b44d5c74c519d92d436",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/a2c7606121fad722e5c2acf34965afbc1af80166"
        },
        "date": 1763879080840,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004406645000199205,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004316241000196896,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001385386526999,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001373059016998,
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
          "id": "764f3e4fbe6f0084353227ee3b8b66321665f529",
          "message": "fix: while induction replacement (#1641)",
          "timestamp": "2025-11-23T00:42:00-05:00",
          "tree_id": "698d1ccc58f49526f806486c3b640508b2a45c0b",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/764f3e4fbe6f0084353227ee3b8b66321665f529"
        },
        "date": 1763892843590,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004383075000077952,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.00000427507600034005,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001363426811003,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001350103551005,
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
          "id": "65b54ecea3b196567eda21dea941014d7979314b",
          "message": "Update EnzymeAD/Enzyme to commit 5db86b6fe7d77f30546627b9c483375284feb159 (#1642)\n\nDiff: https://github.com/EnzymeAD/Enzyme/compare/dd29921570c549f98607f792384efffa35f8e1bc...5db86b6fe7d77f30546627b9c483375284feb159\n\nCo-authored-by: enzymead-bot[bot] <238314553+enzymead-bot[bot]@users.noreply.github.com>",
          "timestamp": "2025-11-23T14:28:49-05:00",
          "tree_id": "3aa7658a44fc3885adc055509770c78a729712ba",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/65b54ecea3b196567eda21dea941014d7979314b"
        },
        "date": 1763944640550,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004280468999968435,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004251791000024241,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001484288595998,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001495373456,
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
          "id": "f9784e443d54e91422ca841a66f56860aeb7af92",
          "message": "Update enzyme for fwddiff (#1645)",
          "timestamp": "2025-11-23T22:52:06-06:00",
          "tree_id": "d218198ad6499354d6fb8ad358f2b17626b4c9ab",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/f9784e443d54e91422ca841a66f56860aeb7af92"
        },
        "date": 1763982351343,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004370322999966447,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004389473999981419,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001559570289999,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001467021811,
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
          "id": "e3de2244ab21c07bf1502ccb8634be0ac86ead82",
          "message": "fix: typo in pass name (#1651)",
          "timestamp": "2025-11-24T14:23:20-05:00",
          "tree_id": "3029092140f3fe6e35326873e06c1386efe12d47",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/e3de2244ab21c07bf1502ccb8634be0ac86ead82"
        },
        "date": 1764029035573,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.00000432322499982547,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004288105000159703,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001405532037999,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001445494979001,
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
          "id": "f11d5ad20f7929603bfd2b45a956fae030fccb1c",
          "message": "feat: add syrk and trmm op (#1644)\n\n* feat: add syrk op\n\n* test: syrk\n\n* fix: docs\n\n* feat: trmm op\n\n* fix: missing uplo",
          "timestamp": "2025-11-24T14:23:37-05:00",
          "tree_id": "5bf0f7eec2193ccc789003e5cfd0defc3c1a5cda",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/f11d5ad20f7929603bfd2b45a956fae030fccb1c"
        },
        "date": 1764035766460,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004226516900234856,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.0000041471548989648,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001521801919996,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001423218900003,
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
          "id": "cf16b634ef7780c681e42d9494b8d94bba5096bd",
          "message": "Update EnzymeAD/Enzyme to commit b642f02d745b5bb0425cde2a888e125b28409d19 (#1649)\n\nDiff: https://github.com/EnzymeAD/Enzyme/compare/05b1413e38c4f6651d3ca35591d34d6ec272c186...b642f02d745b5bb0425cde2a888e125b28409d19\n\nCo-authored-by: enzymead-bot[bot] <238314553+enzymead-bot[bot]@users.noreply.github.com>",
          "timestamp": "2025-11-24T22:51:19-06:00",
          "tree_id": "eab468f491326b2036506eb9138980938b263c95",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/cf16b634ef7780c681e42d9494b8d94bba5096bd"
        },
        "date": 1764061756746,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004467982599999232,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004483210600028543,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001430403397999,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001420598939999,
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
          "id": "da5e6ab8248afae73b545883fe1973b2c5644262",
          "message": "Update EnzymeAD/Enzyme to commit a13f632e2fbaed1f971de015f15e8f4b353e66cb (#1656)\n\nDiff: https://github.com/EnzymeAD/Enzyme/compare/b642f02d745b5bb0425cde2a888e125b28409d19...a13f632e2fbaed1f971de015f15e8f4b353e66cb\n\nCo-authored-by: enzymead-bot[bot] <238314553+enzymead-bot[bot]@users.noreply.github.com>",
          "timestamp": "2025-11-25T09:30:07-05:00",
          "tree_id": "0d120a9b676350bd8f870fca1c90956414907ab9",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/da5e6ab8248afae73b545883fe1973b2c5644262"
        },
        "date": 1764100658116,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.0000043999469999107535,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004439157900924329,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001253901321993,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001279873421997,
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
          "id": "88ee9750320164a5f51803b3ecb6edd211ea2f4d",
          "message": "feat: common syrk optimization patterns (#1653)\n\n* feat: common syrk optimization patterns\n\n* feat: syrk output is always symmetric\n\n* fix: address review comment\n\n* fix: docs",
          "timestamp": "2025-11-25T13:24:32-05:00",
          "tree_id": "84f7f534617a909137df3bc4bbcb852aba6582cb",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/88ee9750320164a5f51803b3ecb6edd211ea2f4d"
        },
        "date": 1764124503075,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004326902899993002,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004272585000035178,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001354923245,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001335705285999,
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
          "id": "31f39c52148a55024b464a98f15b3abc428828e2",
          "message": "feat: blas lowering for syrk (#1650)\n\n* feat: blas lowering for syrk\n\n* feat: fallback lowering for syrk\n\n* feat: lower to blas\n\n* fix: uplo and trans types\n\n* test: one more test case\n\n* fix: copy triangular\n\n* feat: add a new uplo enum entry\n\n* fix: fill\n\n* fix: debug logging\n\n* test: add lit_tests",
          "timestamp": "2025-11-25T18:09:45-05:00",
          "tree_id": "0f1063c60714835db3837d6ec6890ae835e1f8ff",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/31f39c52148a55024b464a98f15b3abc428828e2"
        },
        "date": 1764131959645,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004300762999992002,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004328533899752074,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001521289035001,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001386696216999,
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
          "id": "adb13be7e3e67dd994abfc7bdd6877346c92be11",
          "message": "Update EnzymeAD/Enzyme to commit f467ff7fab15a65a8f034017e2d9f52f9823e79d (#1658)\n\nDiff: https://github.com/EnzymeAD/Enzyme/compare/a13f632e2fbaed1f971de015f15e8f4b353e66cb...f467ff7fab15a65a8f034017e2d9f52f9823e79d\n\nCo-authored-by: enzymead-bot[bot] <238314553+enzymead-bot[bot]@users.noreply.github.com>",
          "timestamp": "2025-11-26T10:57:18-05:00",
          "tree_id": "669a3af7675f800183ddb3ba820d8ee7c735a198",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/adb13be7e3e67dd994abfc7bdd6877346c92be11"
        },
        "date": 1764187659984,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004325666999648092,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004221120000147494,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001473334281006,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001520273059999,
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
          "id": "c8ab3b72c17d50d26a3c32df11edaa165d43ab39",
          "message": "Update EnzymeAD/Enzyme to commit 9c759df0b9e137f05bbbc8212c8747effe59df56 (#1662)\n\nDiff: https://github.com/EnzymeAD/Enzyme/compare/f467ff7fab15a65a8f034017e2d9f52f9823e79d...9c759df0b9e137f05bbbc8212c8747effe59df56\n\nCo-authored-by: enzymead-bot[bot] <238314553+enzymead-bot[bot]@users.noreply.github.com>",
          "timestamp": "2025-11-27T08:34:45-05:00",
          "tree_id": "1bb62202e9d18fce7d74255084e4442079998f96",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/c8ab3b72c17d50d26a3c32df11edaa165d43ab39"
        },
        "date": 1764270090505,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004277056999853812,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004277996000018902,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001453137917997,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001477157437002,
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
          "id": "4dca341c723aa44d252140bd2b17765e1618fb22",
          "message": "fix: restrict raising dot_general to syrk to 2D tensors (#1666)",
          "timestamp": "2025-11-29T18:00:32-05:00",
          "tree_id": "231af61ab0b625d796e5b2baadc3bdb84cc3059c",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/4dca341c723aa44d252140bd2b17765e1618fb22"
        },
        "date": 1764476172094,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.00000445421799995529,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004487904000052367,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001397346443,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001396779474,
            "unit": "s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "58750937+snonk@users.noreply.github.com",
            "name": "snonk",
            "username": "snonk"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "a9caa01126d9d16da33a1b20945446cb65e60653",
          "message": "dot general case for symm (#1620)\n\n* dot general case\n\n* add test and fmt\n\n* fix: update to new version\n\n---------\n\nCo-authored-by: Avik Pal <avikpal@mit.edu>",
          "timestamp": "2025-11-30T13:09:24-05:00",
          "tree_id": "3bb8ab7140e0c3244b5613053f9640bb7758661d",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/a9caa01126d9d16da33a1b20945446cb65e60653"
        },
        "date": 1764538830589,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004310507999616675,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004355360000045039,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001534784994,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001503140465007,
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
          "id": "9571f5ecdbe0ed76955aba9cfa1c07ee163534a0",
          "message": "Update EnzymeAD/Enzyme to commit b8cfe0c70b25872ef256b07a1283f823cd7d1e5c (#1667)\n\nDiff: https://github.com/EnzymeAD/Enzyme/compare/9c759df0b9e137f05bbbc8212c8747effe59df56...b8cfe0c70b25872ef256b07a1283f823cd7d1e5c\n\nCo-authored-by: enzymead-bot[bot] <238314553+enzymead-bot[bot]@users.noreply.github.com>",
          "timestamp": "2025-11-30T13:10:25-05:00",
          "tree_id": "8115d555193f0c5a98ea4dcb11ed5e26b2b338cf",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/9571f5ecdbe0ed76955aba9cfa1c07ee163534a0"
        },
        "date": 1764545474722,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004437613900154247,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.0000044564029001776365,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001511138773999,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001473323584999,
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
          "id": "fadcc11799ed5810753997880262a79aecfc9640",
          "message": "feat: generalize existing passes (#1665)",
          "timestamp": "2025-11-30T19:21:44-05:00",
          "tree_id": "96e1d366ffe2dd058ce6fb4b0476676aa6c754a7",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/fadcc11799ed5810753997880262a79aecfc9640"
        },
        "date": 1764566940631,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004860522899980424,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004766752000432461,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001507932107,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001417516048,
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
          "id": "f603104e871db40802aba36c387576a13af52ca5",
          "message": "Reduce-Mul-Broadcast to Dot General (#1669)\n\n* Reduce-Mul-Broadcast to Dot General\n\n* unfmt",
          "timestamp": "2025-12-01T10:26:05-05:00",
          "tree_id": "e487d40d6522fed2a6541671ef4e3902da346bb5",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/f603104e871db40802aba36c387576a13af52ca5"
        },
        "date": 1764617202885,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004260189699562034,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004406429699884029,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001248841227003,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001351668211995,
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
          "id": "7054610e8cf78a56ce21365bc60b8539096c4ab0",
          "message": "Update EnzymeAD/Enzyme to commit 6c20ffc94c5abff04831f22caf46fe1b25c069be (#1670)\n\nDiff: https://github.com/EnzymeAD/Enzyme/compare/b8cfe0c70b25872ef256b07a1283f823cd7d1e5c...6c20ffc94c5abff04831f22caf46fe1b25c069be\n\nCo-authored-by: enzymead-bot[bot] <238314553+enzymead-bot[bot]@users.noreply.github.com>",
          "timestamp": "2025-12-03T16:13:56-05:00",
          "tree_id": "5295105d10e5760117a0198c5d479b01023c86bd",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/7054610e8cf78a56ce21365bc60b8539096c4ab0"
        },
        "date": 1764809226819,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004342221499973675,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004368584500025463,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001383197025999,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001325478068999,
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
          "id": "f65a4dd29e0dad1700fee99c8bb7d9ec1b0bbca5",
          "message": "RaiseToStableHLO: fix store to scatter (#1677)\n\n* RaiseToStableHLO: fix store to scatter\n\n* fix\n\n* fix\n\n* Update affine_to_stablehlo_bitcast.mlir",
          "timestamp": "2025-12-04T08:22:53-06:00",
          "tree_id": "9ba9188b5d78425de8573d9d6e2424168590d5f1",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/f65a4dd29e0dad1700fee99c8bb7d9ec1b0bbca5"
        },
        "date": 1764875097276,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004281310999795096,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004241763000027276,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001342257274001,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001461607222998,
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
          "id": "8e83bb047e01f2f0499b5a0680405ff029330436",
          "message": "Don't check cuda error on other backend (#1680)\n\n* Don't check cuda error on other backend\n\n* Fix\n\n* fix\n\n* fix\n\n* fix\n\n* fix\n\n* fix\n\n* mul to div\n\n* fix\n\n* fix",
          "timestamp": "2025-12-04T11:49:11-06:00",
          "tree_id": "e46ef81b376f671b3669d2ff14cd85258f0063d1",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/8e83bb047e01f2f0499b5a0680405ff029330436"
        },
        "date": 1764881592174,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004402163899794687,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004342059900227468,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001481182198,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001468281217999,
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
          "id": "0d4c4b951e06ac432ec19c672dcba1e2cbe10e6e",
          "message": "Glibc (#1687)\n\n* glibc sucks\n\n* fix",
          "timestamp": "2025-12-05T08:58:30-06:00",
          "tree_id": "8ffa996fb820f12411b2f15129bbb7f02ff22dc0",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/0d4c4b951e06ac432ec19c672dcba1e2cbe10e6e"
        },
        "date": 1764962518983,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004356561999884434,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004372801899444312,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001401855346994,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001373736266999,
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
          "id": "218941367add2e11c1bb5444444a30abb1e64816",
          "message": "Fix tblgen usage (#1690)",
          "timestamp": "2025-12-05T16:48:19-06:00",
          "tree_id": "486adeac89411afea0ae4c6f0398a6d79a5bbfe4",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/218941367add2e11c1bb5444444a30abb1e64816"
        },
        "date": 1764987790069,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004452517899335362,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004180978999647778,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001625410951994,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.000144374809401,
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
          "id": "3d387c5890ad3f212baa1d85f2ffccea6aba7009",
          "message": "Update jax-ml/jax to commit e45b7b654c5734ba0ebf22a771a947f9a5507859 (#1628)\n\n* Update jax-ml/jax to commit e45b7b654c5734ba0ebf22a771a947f9a5507859\n\nDiff: https://github.com/jax-ml/jax/compare/071a6a5a9a70b2447f0164e61e3e0333318422a8...e45b7b654c5734ba0ebf22a771a947f9a5507859\n\n* fix patch\n\n* fix\n\n---------\n\nCo-authored-by: enzymead-bot[bot] <238314553+enzymead-bot[bot]@users.noreply.github.com>\nCo-authored-by: William S. Moses <gh@wsmoses.com>",
          "timestamp": "2025-12-05T19:39:17-06:00",
          "tree_id": "b760630f803ff8e23a9369a909bb8b98aa0245d9",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/3d387c5890ad3f212baa1d85f2ffccea6aba7009"
        },
        "date": 1765013582463,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004378527000153554,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004377812000166159,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001487893048,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001412560908,
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
          "id": "5ca1f5fa926e4f618c67dec2a53e756821353f43",
          "message": "symbol uses",
          "timestamp": "2025-12-06T01:17:01-05:00",
          "tree_id": "f276ada78c076caad5890e7277656ef28ee0f05a",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/5ca1f5fa926e4f618c67dec2a53e756821353f43"
        },
        "date": 1765024081671,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004449392399692442,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004502534499988542,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001494022488997,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001521238961999,
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
          "id": "3c4714c2efd21ba4feb93bef2001ff01dd61eedb",
          "message": "Fix undefined behavior bugs (#1699)",
          "timestamp": "2025-12-06T12:29:53-06:00",
          "tree_id": "2ab857e6786681276a1ac8a2352c2d4d8bc39e8a",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/3c4714c2efd21ba4feb93bef2001ff01dd61eedb"
        },
        "date": 1765053462959,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004452630900050281,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004339546000119298,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001460215595994,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001359501536993,
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
          "id": "0e74a0fb5a20420419ecaa8a711269e5f0597fb4",
          "message": "fix: dus_pad inf compile (#1700)",
          "timestamp": "2025-12-06T14:58:54-05:00",
          "tree_id": "2bbcda6d1709b350be5c004eccf22db48686d48a",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/0e74a0fb5a20420419ecaa8a711269e5f0597fb4"
        },
        "date": 1765059937810,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004304463500739075,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004348599500372075,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.000137604552599,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001514616753003,
            "unit": "s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "49699333+dependabot[bot]@users.noreply.github.com",
            "name": "dependabot[bot]",
            "username": "dependabot[bot]"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "03e82068e7f4cf7830950e30510e6ef23d843949",
          "message": "build(deps): bump urllib3 from 2.2.2 to 2.6.0 in /builddeps (#1698)\n\nBumps [urllib3](https://github.com/urllib3/urllib3) from 2.2.2 to 2.6.0.\n- [Release notes](https://github.com/urllib3/urllib3/releases)\n- [Changelog](https://github.com/urllib3/urllib3/blob/main/CHANGES.rst)\n- [Commits](https://github.com/urllib3/urllib3/compare/2.2.2...2.6.0)\n\n---\nupdated-dependencies:\n- dependency-name: urllib3\n  dependency-version: 2.6.0\n  dependency-type: direct:production\n...\n\nSigned-off-by: dependabot[bot] <support@github.com>\nCo-authored-by: dependabot[bot] <49699333+dependabot[bot]@users.noreply.github.com>",
          "timestamp": "2025-12-06T16:06:52-06:00",
          "tree_id": "c40ae5d22373eff54fe31d2263df38d8efc06bde",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/03e82068e7f4cf7830950e30510e6ef23d843949"
        },
        "date": 1765075227100,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000006569129900162807,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004035204899992095,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001487978921999,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001493543142001,
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
          "id": "b6d9a7a559ef3d915c13e67901a6e60116de4fc1",
          "message": "Add rocm patch (#1704)\n\n* Add rocm patch\n\n* more\n\n* filegroup",
          "timestamp": "2025-12-07T13:04:18-06:00",
          "tree_id": "0fef94b89bdad0aab61fc2c5cf131e9d67c1e544",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/b6d9a7a559ef3d915c13e67901a6e60116de4fc1"
        },
        "date": 1765143534065,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000006150855800115096,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000003768143899833376,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001386790791999,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001336370552999,
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
          "id": "ead4414a40c594814a129adb54934720a0140c86",
          "message": "feat: add a debugging pass for removing non-splatted constants (#1706)",
          "timestamp": "2025-12-07T13:04:40-06:00",
          "tree_id": "b85852827edb95e8f0e25a8647b36aa115026cb8",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/ead4414a40c594814a129adb54934720a0140c86"
        },
        "date": 1765153413756,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004329831799987005,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000003912584799763863,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001478510524,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001495474493,
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
          "id": "5fa1fb99afb0f6ef0478a2dd2d5c0e709689441e",
          "message": "Update jax-ml/jax to commit a1c8fadb66e372bacf16c93aa6e90fa0aa6ac3af (#1702)\n\nDiff: https://github.com/jax-ml/jax/compare/3a83fa48995650b659b9200a325b7e551640958b...a1c8fadb66e372bacf16c93aa6e90fa0aa6ac3af\n\nCo-authored-by: enzymead-bot[bot] <238314553+enzymead-bot[bot]@users.noreply.github.com>",
          "timestamp": "2025-12-08T19:44:49-06:00",
          "tree_id": "2a6f9895b26fd5db7a6fc0936e0c813bdf0eeaa2",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/5fa1fb99afb0f6ef0478a2dd2d5c0e709689441e"
        },
        "date": 1765253427688,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004337968700019701,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004346270799942431,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001424101054999,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001357899519,
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
          "id": "8df9e47c6222b79d7b58d88eec9fa5ea4affd3c9",
          "message": "feat: syrk cuda lowering (#1703)\n\n* feat: syrk cuda lowering\n\n* fix: use jax convention\n\n* test: update\n\n* fix: ??\n\n* feat: lower to our custom ffi\n\n* feat: use alpha/beta attributes\n\n* test: update\n\n* chore: run fmt",
          "timestamp": "2025-12-08T21:24:20-05:00",
          "tree_id": "c537c761b9e989acefab57543a4e36683779bda7",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/8df9e47c6222b79d7b58d88eec9fa5ea4affd3c9"
        },
        "date": 1765269359317,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004262526800084742,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004347323699948902,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001549653354999,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001542208233999,
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
          "id": "0e2485cc711cee6dcd7e5346005b54c45502a757",
          "message": "Update EnzymeAD/Enzyme to commit a60b57849c63792e66f88f7b74fee9ca2f74c53d (#1705)\n\nDiff: https://github.com/EnzymeAD/Enzyme/compare/e4834164abd7367d75c98ffd90be8c406921376d...a60b57849c63792e66f88f7b74fee9ca2f74c53d\n\nCo-authored-by: enzymead-bot[bot] <238314553+enzymead-bot[bot]@users.noreply.github.com>",
          "timestamp": "2025-12-09T08:10:36-06:00",
          "tree_id": "351f9ff37de8bd7e68dd1929a858fe1107a8bd9c",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/0e2485cc711cee6dcd7e5346005b54c45502a757"
        },
        "date": 1765296122853,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004107347999888589,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.0000038343699998222295,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001313296747997,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001471130627,
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
          "id": "4f8af58e6a86f16de12d3222e5ac5da9e25c4af0",
          "message": "ci: if tests fail run them in dbg mode (#1713)",
          "timestamp": "2025-12-09T14:21:30-06:00",
          "tree_id": "fd0326889ea8893a1ccbfd6ba0baa955f839961d",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/4f8af58e6a86f16de12d3222e5ac5da9e25c4af0"
        },
        "date": 1765318751358,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.0000044047979998140366,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004377494000073057,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001294887084,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001290046253998,
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
          "id": "0341e5cc385d350b806a231b6e2d0cff0b1f1310",
          "message": "feat: correct lowering for rocm (#1715)",
          "timestamp": "2025-12-09T19:35:22-05:00",
          "tree_id": "63aa52511cd6e8ef2b932f652acc6c6022ec9e9d",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/0341e5cc385d350b806a231b6e2d0cff0b1f1310"
        },
        "date": 1765333577511,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004350405000150204,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004312506999849574,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001511090920997,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001527826641999,
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
          "id": "dbaf1f53d5926fa828df59a52afd41f95d51298e",
          "message": "Correct extend to pad indexing (#1717)",
          "timestamp": "2025-12-10T00:56:33-06:00",
          "tree_id": "6acf76486beaeda925e0a6a49b8f54188224f21e",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/dbaf1f53d5926fa828df59a52afd41f95d51298e"
        },
        "date": 1765359717688,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004252045899920632,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.00000433568690023094,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001548439926998,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001485629787999,
            "unit": "s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "31353884+jumerckx@users.noreply.github.com",
            "name": "jumerckx",
            "username": "jumerckx"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "a24b37116e34b28f857ff114aaf7fb18e49cb1a6",
          "message": "Operation definitions for Bessel functions (#1676)\n\n* bessel op definitions\n\n* get rid of redundant bessel functions\n\n* Don't use SameOperandsAndResultType\n\nThe order can be a different type (for example an integer)",
          "timestamp": "2025-12-10T10:33:50-06:00",
          "tree_id": "8e131f1e0102edf5d37abea2ebc601401dccc80c",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/a24b37116e34b28f857ff114aaf7fb18e49cb1a6"
        },
        "date": 1765391115184,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.00000431107299955329,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.0000042005948998848905,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001371250153999,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001364283852999,
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
          "id": "8d9c3b92c9ceae9944b427d00f02067ab50cb93e",
          "message": "feat: generalize reduce_slice_fusion (#1689)",
          "timestamp": "2025-12-10T11:58:04-05:00",
          "tree_id": "ebbb76a796c733eb828dd6e6012410f857fbabab",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/8d9c3b92c9ceae9944b427d00f02067ab50cb93e"
        },
        "date": 1765397328567,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004339715999958571,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004307055899698753,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001430617951999,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001465538042,
            "unit": "s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "javathunderman@users.noreply.github.com",
            "name": "Arjun Vedantham",
            "username": "javathunderman"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "0a0ee25da49bf1356e6c4e632549542b1c89360d",
          "message": "Performance Bounding Dialect (#1580)\n\n* start perfify dialect impl\n\n* Add assume op, TODO: fix roundtrip parsing test\n\n* tweak w/ clang-format\n\n* Fix AssumeOp parsing, adjust parsing test file\n\n* attempt to resolve formatting issue\n\n* format RegistryUtils?\n\n* fix bazel format issue\n\n* add check directives for parsing test\n\n---------\n\nCo-authored-by: William Moses <wmoses@google.com>",
          "timestamp": "2025-12-10T20:14:26-06:00",
          "tree_id": "22a6bf16b55cd46952b6b97aeff19695996d8b27",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/0a0ee25da49bf1356e6c4e632549542b1c89360d"
        },
        "date": 1765426083961,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004429214999981923,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004354909999892697,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001464696221999,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001298194633,
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
          "id": "cbaec252c15d5f73684dd6bb0dddf22f97cc08f3",
          "message": "fix: only raise ops with a single result (for now) (#1721)\n\n* fix: only raise ops with a single result (for now)\n\n* test: add lu as a test",
          "timestamp": "2025-12-10T23:55:14-05:00",
          "tree_id": "06e6a5f4e14625424c9dceecee67f3570d1ce7ef",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/cbaec252c15d5f73684dd6bb0dddf22f97cc08f3"
        },
        "date": 1765435182784,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004326745998696424,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004314442999020685,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001427497209006,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001392607529996,
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
          "id": "fda790d72ae45b90da783b43ab9f0e2a561387d2",
          "message": "Fix: slice fusion (#1722)\n\n* fix\n\n* format\n\n* min/max",
          "timestamp": "2025-12-11T10:02:39-06:00",
          "tree_id": "1755a2a3bbbe15d2c6e930ba60ffa870d1445362",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/fda790d72ae45b90da783b43ab9f0e2a561387d2"
        },
        "date": 1765475612088,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004338875000030385,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.00000434898189996602,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001353214798,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001321539557999,
            "unit": "s"
          }
        ]
      }
    ]
  }
}