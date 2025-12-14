window.BENCHMARK_DATA = {
  "lastUpdate": 1765708543500,
  "repoUrl": "https://github.com/EnzymeAD/Enzyme-JAX",
  "entries": {
    "EnzymeJAX Benchmarks": [
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
          "id": "779dca1bf78b03acef6b59f7af3b5075c51e1ae3",
          "message": "feat: vector mode forward for dot_general (#1724)\n\n* feat: vector mode forward for dot_general\n\n* chore: remove comment",
          "timestamp": "2025-12-11T15:47:35-05:00",
          "tree_id": "41d81b62d5062f879818f94069e39267ec042e0d",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/779dca1bf78b03acef6b59f7af3b5075c51e1ae3"
        },
        "date": 1765491190014,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004318197999964468,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.00000431587900093291,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001485531517013,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001464243666996,
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
          "id": "2308416a42c8ceb073525a68376a2d449daad77e",
          "message": "fix: dot general reshape pass (#1728)\n\n* feat: vector mode forward for dot_general\n\n* chore: remove comment\n\n* fix: dot general reshape pass",
          "timestamp": "2025-12-11T17:24:22-05:00",
          "tree_id": "2b78d35f470325f13c88b120af86480b8048af41",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/2308416a42c8ceb073525a68376a2d449daad77e"
        },
        "date": 1765500379391,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004344973900151672,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.00000424071190009272,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001438192881001,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001431801300999,
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
          "id": "5db2863c29f5e2a55255b2c66be111808ba825f3",
          "message": "Fix link error (#1727)",
          "timestamp": "2025-12-11T16:30:25-06:00",
          "tree_id": "6fe3758fe90247d2ddcae4942d6261540b68e94f",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/5db2863c29f5e2a55255b2c66be111808ba825f3"
        },
        "date": 1765509147622,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.0000044233428998268206,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004308318999392213,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001471330473999,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001468807874,
            "unit": "s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "198982749+Copilot@users.noreply.github.com",
            "name": "Copilot",
            "username": "Copilot"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "598c0a6300c0f0c176570ed76994192abe7635f1",
          "message": "Add validation to lowering passes to prevent unconverted ops (#1711)\n\n* Initial plan\n\n* Convert lowering passes to use OpConversionPattern\n\n- Convert LowerEnzymeXLABLAS to use OpConversionPattern\n- Convert LowerEnzymeXLALapack to use OpConversionPattern\n- Convert LowerEnzymeXLALinalg to use OpConversionPattern\n- Convert LowerEnzymeXLAML to use OpConversionPattern\n- Update all passes to use ConversionTarget and applyPartialConversion\n- Mark corresponding ops as illegal in conversion targets\n\nCo-authored-by: avik-pal <30564094+avik-pal@users.noreply.github.com>\n\n* Revert to OpRewritePattern and add post-lowering validation\n\nReverted the conversion pattern changes and added IR walk validation\nto check for any remaining illegal ops after pattern application.\nEach pass now emits errors if ops that should have been lowered remain.\n\nCo-authored-by: avik-pal <30564094+avik-pal@users.noreply.github.com>\n\n* Run clang-format and rewrite EnzymeBatch pass to use OpRewritePattern\n\n- Applied clang-format to all modified lowering passes\n- Converted EnzymeBatchToStableHLOPass from OpConversionPattern to OpRewritePattern\n- Added post-lowering validation to EnzymeBatch pass\n- All passes now use the same efficient pattern: OpRewritePattern with IR walk validation\n\nCo-authored-by: avik-pal <30564094+avik-pal@users.noreply.github.com>\n\n---------\n\nCo-authored-by: copilot-swe-agent[bot] <198982749+Copilot@users.noreply.github.com>\nCo-authored-by: avik-pal <30564094+avik-pal@users.noreply.github.com>",
          "timestamp": "2025-12-11T20:26:01-06:00",
          "tree_id": "2c02107bbdf60a953110dd3fa2b16e9a1fb95264",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/598c0a6300c0f0c176570ed76994192abe7635f1"
        },
        "date": 1765529974102,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.00000426514299979317,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.0000042764379002619535,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001476309046,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001477145594999,
            "unit": "s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "198982749+Copilot@users.noreply.github.com",
            "name": "copilot-swe-agent[bot]",
            "username": "Copilot"
          },
          "committer": {
            "email": "gh@wsmoses.com",
            "name": "William Moses",
            "username": "wsmoses"
          },
          "distinct": true,
          "id": "6c6972ab9808eda4a1905bed3a7ccd013c518041",
          "message": "Apply clang-format to modified files\n\nCo-authored-by: avik-pal <30564094+avik-pal@users.noreply.github.com>",
          "timestamp": "2025-12-11T22:43:52-06:00",
          "tree_id": "3f8121bed90df2fa4d492c397d871d262d55df39",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/6c6972ab9808eda4a1905bed3a7ccd013c518041"
        },
        "date": 1765542048878,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004282860999956029,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004318197000247892,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001547227711998,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001396556061998,
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
          "id": "9b1bf267de1eb1fdbcc006e316b1e4d894de2355",
          "message": "feat: generalize `slice_to_batch` and `concat_to_batch` passes (#1719)\n\n* feat: generalize slice_to_batch and concat_to_batch passes\n\n* fix: make check a bit faster\n\n* feat: generalize to support non-contiguous slices\n\n* feat: generalize to non contiguous slices\n\n* test: strided slice reduce\n\n* feat: generalize binary op to reduce\n\n* feat: generalize reduce fusion\n\n* fix: build\n\n* test: update\n\n* fix: dependency check\n\n* feat: add a optional arg for sharding attr",
          "timestamp": "2025-12-12T13:00:14-05:00",
          "tree_id": "d0377d42edbc3ee825f3c10a68a511565402fa40",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/9b1bf267de1eb1fdbcc006e316b1e4d894de2355"
        },
        "date": 1765575991153,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004463052999926731,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004374844999983907,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001605440113009,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001537183152991,
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
          "id": "ba32ea4170350ded74931de3b5cd1dadb3a36f30",
          "message": "feat: DS(DUS) simplifications (#1742)\n\n* feat: DS(DUS) simplifications\n\n* feat: more patterns",
          "timestamp": "2025-12-12T13:13:54-05:00",
          "tree_id": "637e3fa5ab043edfac331fe0a4bbd9a82406e564",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/ba32ea4170350ded74931de3b5cd1dadb3a36f30"
        },
        "date": 1765586273080,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.0000043511694006156175,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004340834400500171,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.000133705902101,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001462972663008,
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
          "id": "4f5875f4aa1b5e431d8135cb660538d135302689",
          "message": "ci: try fixing the benchmark parsing script (#1744)",
          "timestamp": "2025-12-12T19:07:06-05:00",
          "tree_id": "26cd4348841e73f59cdf5183d4baf17afbac400e",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/4f5875f4aa1b5e431d8135cb660538d135302689"
        },
        "date": 1765595000773,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "actmtch / JaX   / cpu / Primal",
            "value": 0.000004042276400014089,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / Primal",
            "value": 0.000003916199400009645,
            "unit": "s"
          },
          {
            "name": "actmtch / JaX   / cpu / Forward",
            "value": 0.000006083053700012897,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / Forward",
            "value": 0.000007343717599997035,
            "unit": "s"
          },
          {
            "name": "actmtch / JaX   / cpu / BothRev",
            "value": 0.000006368887500002529,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / PreRev",
            "value": 0.000007199492000017927,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / PostRev",
            "value": 0.000006311049599980834,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / BothRev",
            "value": 0.000007627360800006499,
            "unit": "s"
          },
          {
            "name": "actmtch / JaX   / cpu / Primal",
            "value": 0.000007267178798792884,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / Primal",
            "value": 0.000007316396397072821,
            "unit": "s"
          },
          {
            "name": "actmtch / JaX   / cpu / Forward",
            "value": 0.000011540658201556651,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / Forward",
            "value": 0.000013622546900296584,
            "unit": "s"
          },
          {
            "name": "actmtch / JaX   / cpu / BothRev",
            "value": 0.000011314728704746811,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / PreRev",
            "value": 0.000012891600100556387,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / PostRev",
            "value": 0.000011363539402373136,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / BothRev",
            "value": 0.000013984611397609115,
            "unit": "s"
          },
          {
            "name": "actmtch / JaX   / gpu / Primal",
            "value": 0.00009242747529642657,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / gpu / Primal",
            "value": 0.00007686306560062804,
            "unit": "s"
          },
          {
            "name": "actmtch / JaX   / gpu / Forward",
            "value": 0.0001070845468959,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / gpu / Forward",
            "value": 0.0001084106944035,
            "unit": "s"
          },
          {
            "name": "actmtch / JaX   / gpu / BothRev",
            "value": 0.0001049285162007,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / gpu / PreRev",
            "value": 0.0001039607186045,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / gpu / PostRev",
            "value": 0.0001040464675985,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / gpu / BothRev",
            "value": 0.0001030089301988,
            "unit": "s"
          },
          {
            "name": "actmtch / JaX   / cpu / Primal",
            "value": 0.000003946591999920201,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / Primal",
            "value": 0.0000037893138996878407,
            "unit": "s"
          },
          {
            "name": "actmtch / JaX   / cpu / Forward",
            "value": 0.000005442277900147019,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / Forward",
            "value": 0.000006189347899999121,
            "unit": "s"
          },
          {
            "name": "actmtch / JaX   / cpu / BothRev",
            "value": 0.000005278458899920224,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / PreRev",
            "value": 0.000006086759900063044,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / PostRev",
            "value": 0.000005425196899886942,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / BothRev",
            "value": 0.000006086256900016452,
            "unit": "s"
          },
          {
            "name": "actmtch / JaX   / tpu / Primal",
            "value": 0.0001592694708,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / tpu / Primal",
            "value": 0.0001507426689,
            "unit": "s"
          },
          {
            "name": "actmtch / JaX   / tpu / Forward",
            "value": 0.0002246309879999,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / tpu / Forward",
            "value": 0.0002245756369,
            "unit": "s"
          },
          {
            "name": "actmtch / JaX   / tpu / BothRev",
            "value": 0.0002285479597998,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / tpu / PreRev",
            "value": 0.0002256040049,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / tpu / PostRev",
            "value": 0.0002243553718999,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / tpu / BothRev",
            "value": 0.0002264596539,
            "unit": "s"
          },
          {
            "name": "actmtch / JaX   / cpu / Primal",
            "value": 0.00000558876990007775,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / Primal",
            "value": 0.000005315844300002937,
            "unit": "s"
          },
          {
            "name": "actmtch / JaX   / cpu / Forward",
            "value": 0.0000077687765000519,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / Forward",
            "value": 0.000009871819599993614,
            "unit": "s"
          },
          {
            "name": "actmtch / JaX   / cpu / BothRev",
            "value": 0.000008088623300045582,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / PreRev",
            "value": 0.000009030194299975846,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / PostRev",
            "value": 0.000008039371899940306,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / BothRev",
            "value": 0.000009913534899988008,
            "unit": "s"
          },
          {
            "name": "add_one / JaX   / cpu / Primal",
            "value": 0.000004249816399988049,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / Primal",
            "value": 0.000004272189900029843,
            "unit": "s"
          },
          {
            "name": "add_one / JaX   / cpu / Forward",
            "value": 0.000007593174100020406,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / Forward",
            "value": 0.000007783411599984901,
            "unit": "s"
          },
          {
            "name": "add_one / JaX   / cpu / BothRev",
            "value": 0.000007973080300007496,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / PreRev",
            "value": 0.0000077243023000392,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / PostRev",
            "value": 0.000007731054100031542,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / BothRev",
            "value": 0.000007669031400018867,
            "unit": "s"
          },
          {
            "name": "add_one / JaX   / cpu / Primal",
            "value": 0.000007646223896881566,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / Primal",
            "value": 0.000007739840098656714,
            "unit": "s"
          },
          {
            "name": "add_one / JaX   / cpu / Forward",
            "value": 0.000011806310899555685,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / Forward",
            "value": 0.0000123615862976294,
            "unit": "s"
          },
          {
            "name": "add_one / JaX   / cpu / BothRev",
            "value": 0.00001252711690030992,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / PreRev",
            "value": 0.000012572120898403228,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / PostRev",
            "value": 0.000012569695198908448,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / BothRev",
            "value": 0.000012532839097548276,
            "unit": "s"
          },
          {
            "name": "add_one / JaX   / gpu / Primal",
            "value": 0.00007735481069539674,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / gpu / Primal",
            "value": 0.00008814197140163741,
            "unit": "s"
          },
          {
            "name": "add_one / JaX   / gpu / Forward",
            "value": 0.0001075475519988,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / gpu / Forward",
            "value": 0.0001096942158997,
            "unit": "s"
          },
          {
            "name": "add_one / JaX   / gpu / BothRev",
            "value": 0.0001278089917032,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / gpu / PreRev",
            "value": 0.0001207042725989,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / gpu / PostRev",
            "value": 0.0001390853002027,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / gpu / BothRev",
            "value": 0.0001120702963031,
            "unit": "s"
          },
          {
            "name": "add_one / JaX   / cpu / Primal",
            "value": 0.0000037454478999279673,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / Primal",
            "value": 0.000003767703999983496,
            "unit": "s"
          },
          {
            "name": "add_one / JaX   / cpu / Forward",
            "value": 0.000005818007000198122,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / Forward",
            "value": 0.000006108678900272935,
            "unit": "s"
          },
          {
            "name": "add_one / JaX   / cpu / BothRev",
            "value": 0.000005800489900138928,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / PreRev",
            "value": 0.000006184972900155117,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / PostRev",
            "value": 0.000006190313900151523,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / BothRev",
            "value": 0.000006159894899974461,
            "unit": "s"
          },
          {
            "name": "add_one / JaX   / tpu / Primal",
            "value": 0.0001369392312,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / tpu / Primal",
            "value": 0.0001360472710999,
            "unit": "s"
          },
          {
            "name": "add_one / JaX   / tpu / Forward",
            "value": 0.0002379662686998,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / tpu / Forward",
            "value": 0.0002212347460001,
            "unit": "s"
          },
          {
            "name": "add_one / JaX   / tpu / BothRev",
            "value": 0.0002233373798997,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / tpu / PreRev",
            "value": 0.0002153363270997,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / tpu / PostRev",
            "value": 0.0002287923689,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / tpu / BothRev",
            "value": 0.0002227598829002,
            "unit": "s"
          },
          {
            "name": "add_one / JaX   / cpu / Primal",
            "value": 0.000005205769399981364,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / Primal",
            "value": 0.000005415437499959807,
            "unit": "s"
          },
          {
            "name": "add_one / JaX   / cpu / Forward",
            "value": 0.00000848381830001017,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / Forward",
            "value": 0.000009056562099976873,
            "unit": "s"
          },
          {
            "name": "add_one / JaX   / cpu / BothRev",
            "value": 0.000009000485700016725,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / PreRev",
            "value": 0.000009217603400065854,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / PostRev",
            "value": 0.000009187936000034824,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / BothRev",
            "value": 0.000008640141199975915,
            "unit": "s"
          },
          {
            "name": "add_two / JaX   / cpu / Primal",
            "value": 0.000004461965200016493,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / Primal",
            "value": 0.00000440150689996699,
            "unit": "s"
          },
          {
            "name": "add_two / JaX   / cpu / Forward",
            "value": 0.00000753944190000766,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / Forward",
            "value": 0.000007502525700010665,
            "unit": "s"
          },
          {
            "name": "add_two / JaX   / cpu / BothRev",
            "value": 0.000009621878699999798,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / PreRev",
            "value": 0.00000966574170001877,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / PostRev",
            "value": 0.000009659845299984226,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / BothRev",
            "value": 0.00000965334869997605,
            "unit": "s"
          },
          {
            "name": "add_two / JaX   / cpu / Primal",
            "value": 0.00000797386480262503,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / Primal",
            "value": 0.000008376738603692502,
            "unit": "s"
          },
          {
            "name": "add_two / JaX   / cpu / Forward",
            "value": 0.000012012289604172111,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / Forward",
            "value": 0.000012722084298729895,
            "unit": "s"
          },
          {
            "name": "add_two / JaX   / cpu / BothRev",
            "value": 0.000014760620996821671,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / PreRev",
            "value": 0.000014902091398835182,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / PostRev",
            "value": 0.000014903298899298534,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / BothRev",
            "value": 0.000014729966403683648,
            "unit": "s"
          },
          {
            "name": "add_two / JaX   / gpu / Primal",
            "value": 0.00007810488649993204,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / gpu / Primal",
            "value": 0.00007854466260178015,
            "unit": "s"
          },
          {
            "name": "add_two / JaX   / gpu / Forward",
            "value": 0.0001076416089024,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / gpu / Forward",
            "value": 0.0001074138785013,
            "unit": "s"
          },
          {
            "name": "add_two / JaX   / gpu / BothRev",
            "value": 0.0001499747349007,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / gpu / PreRev",
            "value": 0.0001340904692013,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / gpu / PostRev",
            "value": 0.0001284010489995,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / gpu / BothRev",
            "value": 0.0001820310273033,
            "unit": "s"
          },
          {
            "name": "add_two / JaX   / cpu / Primal",
            "value": 0.000003797406900048372,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / Primal",
            "value": 0.000003850623900143546,
            "unit": "s"
          },
          {
            "name": "add_two / JaX   / cpu / Forward",
            "value": 0.0000060268979003012645,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / Forward",
            "value": 0.000006244133899963344,
            "unit": "s"
          },
          {
            "name": "add_two / JaX   / cpu / BothRev",
            "value": 0.000006779222899785964,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / PreRev",
            "value": 0.000007409215900042909,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / PostRev",
            "value": 0.000007203996899988852,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / BothRev",
            "value": 0.000006861056900015683,
            "unit": "s"
          },
          {
            "name": "add_two / JaX   / tpu / Primal",
            "value": 0.0001417000931,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / tpu / Primal",
            "value": 0.0001411877639999,
            "unit": "s"
          },
          {
            "name": "add_two / JaX   / tpu / Forward",
            "value": 0.0002291164058002,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / tpu / Forward",
            "value": 0.0002159656220002,
            "unit": "s"
          },
          {
            "name": "add_two / JaX   / tpu / BothRev",
            "value": 0.0002289081248,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / tpu / PreRev",
            "value": 0.0002323393737999,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / tpu / PostRev",
            "value": 0.0002311962558,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / tpu / BothRev",
            "value": 0.0002457435595999,
            "unit": "s"
          },
          {
            "name": "add_two / JaX   / cpu / Primal",
            "value": 0.000005605228900003567,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / Primal",
            "value": 0.000005615957899954083,
            "unit": "s"
          },
          {
            "name": "add_two / JaX   / cpu / Forward",
            "value": 0.000008704501400006848,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / Forward",
            "value": 0.000009231029700004,
            "unit": "s"
          },
          {
            "name": "add_two / JaX   / cpu / BothRev",
            "value": 0.000010325988800013875,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / PreRev",
            "value": 0.000010171849099970132,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / PostRev",
            "value": 0.00001086309579995941,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / BothRev",
            "value": 0.0000107300755000324,
            "unit": "s"
          },
          {
            "name": "cache / JaX   / cpu / Primal",
            "value": 0.000003978495000001203,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / Primal",
            "value": 0.000003832693300000756,
            "unit": "s"
          },
          {
            "name": "cache / JaX   / cpu / Forward",
            "value": 0.000010602415499988636,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / Forward",
            "value": 0.000010562095800014504,
            "unit": "s"
          },
          {
            "name": "cache / JaX   / cpu / BothRev",
            "value": 0.00001543366649998461,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / PreRev",
            "value": 0.000011797563800018906,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / PostRev",
            "value": 0.000015810998799997834,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / BothRev",
            "value": 0.000011391727399995944,
            "unit": "s"
          },
          {
            "name": "cache / JaX   / cpu / Primal",
            "value": 0.000007623784797033295,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / Primal",
            "value": 0.00000771416379720904,
            "unit": "s"
          },
          {
            "name": "cache / JaX   / cpu / Forward",
            "value": 0.00001306924529490061,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / Forward",
            "value": 0.00001281254130299203,
            "unit": "s"
          },
          {
            "name": "cache / JaX   / cpu / BothRev",
            "value": 0.00002401710069971159,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / PreRev",
            "value": 0.00001717608330072835,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / PostRev",
            "value": 0.00002439267850131728,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / BothRev",
            "value": 0.000017580945894587785,
            "unit": "s"
          },
          {
            "name": "cache / JaX   / gpu / Primal",
            "value": 0.00009234244219842368,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / gpu / Primal",
            "value": 0.00007673824650119059,
            "unit": "s"
          },
          {
            "name": "cache / JaX   / gpu / Forward",
            "value": 0.0001157108181039,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / gpu / Forward",
            "value": 0.0001060561291,
            "unit": "s"
          },
          {
            "name": "cache / JaX   / gpu / BothRev",
            "value": 0.0001065357772982,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / gpu / PreRev",
            "value": 0.0001069570124032,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / gpu / PostRev",
            "value": 0.0001102495582017,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / gpu / BothRev",
            "value": 0.0001346565840009,
            "unit": "s"
          },
          {
            "name": "cache / JaX   / cpu / Primal",
            "value": 0.0000034485709002183285,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / Primal",
            "value": 0.0000035604699998657453,
            "unit": "s"
          },
          {
            "name": "cache / JaX   / cpu / Forward",
            "value": 0.00002432794769993052,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / Forward",
            "value": 0.000025505967600111035,
            "unit": "s"
          },
          {
            "name": "cache / JaX   / cpu / BothRev",
            "value": 0.0000335289954997279,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / PreRev",
            "value": 0.000029080436599906533,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / PostRev",
            "value": 0.00003238256449985784,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / BothRev",
            "value": 0.000029388079600175844,
            "unit": "s"
          },
          {
            "name": "cache / JaX   / tpu / Primal",
            "value": 0.0001537573538,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / tpu / Primal",
            "value": 0.0001521638328998,
            "unit": "s"
          },
          {
            "name": "cache / JaX   / tpu / Forward",
            "value": 0.0002049570301998,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / tpu / Forward",
            "value": 0.0002082783681002,
            "unit": "s"
          },
          {
            "name": "cache / JaX   / tpu / BothRev",
            "value": 0.0002169199451,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / tpu / PreRev",
            "value": 0.0002328070398998,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / tpu / PostRev",
            "value": 0.0002354366317998,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / tpu / BothRev",
            "value": 0.0002454883057002,
            "unit": "s"
          },
          {
            "name": "cache / JaX   / cpu / Primal",
            "value": 0.000005034508400058257,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / Primal",
            "value": 0.0000049609247000262254,
            "unit": "s"
          },
          {
            "name": "cache / JaX   / cpu / Forward",
            "value": 0.00000958532060003563,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / Forward",
            "value": 0.000009832659200037596,
            "unit": "s"
          },
          {
            "name": "cache / JaX   / cpu / BothRev",
            "value": 0.00002492826329998934,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / PreRev",
            "value": 0.00001347469390002516,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / PostRev",
            "value": 0.000016471001700028863,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / BothRev",
            "value": 0.00001321062470005927,
            "unit": "s"
          },
          {
            "name": "Concat / JaX   / cpu / Primal",
            "value": 0.000004152788700002929,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / Primal",
            "value": 0.000004246611700000358,
            "unit": "s"
          },
          {
            "name": "Concat / JaX   / cpu / Forward",
            "value": 0.00000724559259997477,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / Forward",
            "value": 0.000007142853000004834,
            "unit": "s"
          },
          {
            "name": "Concat / JaX   / cpu / BothRev",
            "value": 0.00000759744239999236,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / PreRev",
            "value": 0.000007904879999978221,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / PostRev",
            "value": 0.000007904899200002546,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / BothRev",
            "value": 0.000007689444099969478,
            "unit": "s"
          },
          {
            "name": "Concat / JaX   / cpu / Primal",
            "value": 0.000007940306200180203,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / Primal",
            "value": 0.000008151222002925351,
            "unit": "s"
          },
          {
            "name": "Concat / JaX   / cpu / Forward",
            "value": 0.000012682391999987885,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / Forward",
            "value": 0.000012139048299286517,
            "unit": "s"
          },
          {
            "name": "Concat / JaX   / cpu / BothRev",
            "value": 0.000012230429501505569,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / PreRev",
            "value": 0.000013741028297226876,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / PostRev",
            "value": 0.00001325814339797944,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / BothRev",
            "value": 0.000013285552297020332,
            "unit": "s"
          },
          {
            "name": "Concat / JaX   / gpu / Primal",
            "value": 0.00007979277209960856,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / gpu / Primal",
            "value": 0.00007915145299630239,
            "unit": "s"
          },
          {
            "name": "Concat / JaX   / gpu / Forward",
            "value": 0.0001063503579003,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / gpu / Forward",
            "value": 0.0001110785221972,
            "unit": "s"
          },
          {
            "name": "Concat / JaX   / gpu / BothRev",
            "value": 0.0001064024108985,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / gpu / PreRev",
            "value": 0.0001013041889003,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / gpu / PostRev",
            "value": 0.0001335859609011,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / gpu / BothRev",
            "value": 0.0001412956093961,
            "unit": "s"
          },
          {
            "name": "Concat / JaX   / cpu / Primal",
            "value": 0.000003887478999968152,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / Primal",
            "value": 0.000003788308900038828,
            "unit": "s"
          },
          {
            "name": "Concat / JaX   / cpu / Forward",
            "value": 0.000006054638900241116,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / Forward",
            "value": 0.000005834963899906143,
            "unit": "s"
          },
          {
            "name": "Concat / JaX   / cpu / BothRev",
            "value": 0.000006228923900198424,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / PreRev",
            "value": 0.000006210754900166649,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / PostRev",
            "value": 0.000006291535900163581,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / BothRev",
            "value": 0.000006245151899929624,
            "unit": "s"
          },
          {
            "name": "Concat / JaX   / tpu / Primal",
            "value": 0.0001591345608001,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / tpu / Primal",
            "value": 0.0001777192625999,
            "unit": "s"
          },
          {
            "name": "Concat / JaX   / tpu / Forward",
            "value": 0.0002358688878,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / tpu / Forward",
            "value": 0.0002288993247999,
            "unit": "s"
          },
          {
            "name": "Concat / JaX   / tpu / BothRev",
            "value": 0.0002167764359997,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / tpu / PreRev",
            "value": 0.0002356859237999,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / tpu / PostRev",
            "value": 0.0002441485726001,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / tpu / BothRev",
            "value": 0.0002441358405998,
            "unit": "s"
          },
          {
            "name": "Concat / JaX   / cpu / Primal",
            "value": 0.000005217826399984915,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / Primal",
            "value": 0.000005354281099971559,
            "unit": "s"
          },
          {
            "name": "Concat / JaX   / cpu / Forward",
            "value": 0.00000871699700001045,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / Forward",
            "value": 0.000008510659799958375,
            "unit": "s"
          },
          {
            "name": "Concat / JaX   / cpu / BothRev",
            "value": 0.000009241290199952346,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / PreRev",
            "value": 0.000009335882400046104,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / PostRev",
            "value": 0.00000918753519999882,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / BothRev",
            "value": 0.00000944589860000633,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaX   / cpu / Primal",
            "value": 0.000008376099958695705,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / cpu / Primal",
            "value": 0.000006278499995460152,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaX   / cpu / Forward",
            "value": 0.000009357199996884446,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / cpu / Forward",
            "value": 0.000010352999970564267,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaX   / cpu / Primal",
            "value": 0.000012674304889515042,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / cpu / Primal",
            "value": 0.000012132397387176752,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaX   / cpu / Forward",
            "value": 0.000016556703485548496,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / cpu / Forward",
            "value": 0.000017915799980983137,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaX   / gpu / Primal",
            "value": 0.0001277628005482,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / gpu / Primal",
            "value": 0.0001133031968493,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaX   / gpu / Forward",
            "value": 0.0001500758982729,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / gpu / Forward",
            "value": 0.0001444473979063,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaX   / cpu / Primal",
            "value": 0.000006316000144579448,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / cpu / Primal",
            "value": 0.000005187999704503454,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaX   / cpu / Forward",
            "value": 0.0000071949998527998105,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / cpu / Forward",
            "value": 0.000008556000102544203,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaX   / tpu / Primal",
            "value": 0.0001774330001353,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / tpu / Primal",
            "value": 0.0001834950002375,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaX   / tpu / Forward",
            "value": 0.0002600859999802,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / tpu / Forward",
            "value": 0.0002728700001171,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaX   / cpu / Primal",
            "value": 0.00000996959997792146,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / cpu / Primal",
            "value": 0.000008630399952380685,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaX   / cpu / Forward",
            "value": 0.000011348499992891449,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / cpu / Forward",
            "value": 0.000014455499967880317,
            "unit": "s"
          },
          {
            "name": "GenDot / JaX   / cpu / Primal",
            "value": 0.000004341393100003188,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / Primal",
            "value": 0.000004476979099990786,
            "unit": "s"
          },
          {
            "name": "GenDot / JaX   / cpu / Forward",
            "value": 0.000006552558700013833,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / Forward",
            "value": 0.000007575665200010917,
            "unit": "s"
          },
          {
            "name": "GenDot / JaX   / cpu / BothRev",
            "value": 0.000006754153599968049,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / PreRev",
            "value": 0.000007418191999977353,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / PostRev",
            "value": 0.000006352470599995286,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / BothRev",
            "value": 0.000007274764400017375,
            "unit": "s"
          },
          {
            "name": "GenDot / JaX   / cpu / Primal",
            "value": 0.000008017660200130194,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / Primal",
            "value": 0.00000840233670314774,
            "unit": "s"
          },
          {
            "name": "GenDot / JaX   / cpu / Forward",
            "value": 0.000011651724198600278,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / Forward",
            "value": 0.000014130022004246712,
            "unit": "s"
          },
          {
            "name": "GenDot / JaX   / cpu / BothRev",
            "value": 0.000011817123502260074,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / PreRev",
            "value": 0.000014127929200185465,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / PostRev",
            "value": 0.00001200449750176631,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / BothRev",
            "value": 0.000014131995302159338,
            "unit": "s"
          },
          {
            "name": "GenDot / JaX   / gpu / Primal",
            "value": 0.00007542948090122081,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / gpu / Primal",
            "value": 0.00008587434819783084,
            "unit": "s"
          },
          {
            "name": "GenDot / JaX   / gpu / Forward",
            "value": 0.0001293963116011,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / gpu / Forward",
            "value": 0.0001344898093957,
            "unit": "s"
          },
          {
            "name": "GenDot / JaX   / gpu / BothRev",
            "value": 0.0001375040979997,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / gpu / PreRev",
            "value": 0.0001206574818992,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / gpu / PostRev",
            "value": 0.0001173984160996,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / gpu / BothRev",
            "value": 0.0001039030733983,
            "unit": "s"
          },
          {
            "name": "GenDot / JaX   / cpu / Primal",
            "value": 0.000004137326999989454,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / Primal",
            "value": 0.0000041378679001354616,
            "unit": "s"
          },
          {
            "name": "GenDot / JaX   / cpu / Forward",
            "value": 0.000005576045899942983,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / Forward",
            "value": 0.00000642075690011552,
            "unit": "s"
          },
          {
            "name": "GenDot / JaX   / cpu / BothRev",
            "value": 0.000005827174899968668,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / PreRev",
            "value": 0.000006425931900230353,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / PostRev",
            "value": 0.000005949228999816114,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / BothRev",
            "value": 0.0000060808458998508285,
            "unit": "s"
          },
          {
            "name": "GenDot / JaX   / tpu / Primal",
            "value": 0.0001565142709001,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / tpu / Primal",
            "value": 0.0001587983657998,
            "unit": "s"
          },
          {
            "name": "GenDot / JaX   / tpu / Forward",
            "value": 0.0002369018667999,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / tpu / Forward",
            "value": 0.0002381662157,
            "unit": "s"
          },
          {
            "name": "GenDot / JaX   / tpu / BothRev",
            "value": 0.0002264611989001,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / tpu / PreRev",
            "value": 0.0002151042649998,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / tpu / PostRev",
            "value": 0.0002326827708002,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / tpu / BothRev",
            "value": 0.0002350176267998,
            "unit": "s"
          },
          {
            "name": "GenDot / JaX   / cpu / Primal",
            "value": 0.0000060096731999692565,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / Primal",
            "value": 0.000005763589500020317,
            "unit": "s"
          },
          {
            "name": "GenDot / JaX   / cpu / Forward",
            "value": 0.000008208799800013366,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / Forward",
            "value": 0.000009922964900033548,
            "unit": "s"
          },
          {
            "name": "GenDot / JaX   / cpu / BothRev",
            "value": 0.000008553013999971881,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / PreRev",
            "value": 0.000009490446000017983,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / PostRev",
            "value": 0.000008680555099999764,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / BothRev",
            "value": 0.00000994160889995328,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaX   / cpu / Primal",
            "value": 0.000006026293400009308,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / Primal",
            "value": 0.000005862528400029987,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaX   / cpu / Forward",
            "value": 0.000010045896599967818,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / Forward",
            "value": 0.00000998859039996205,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaX   / cpu / BothRev",
            "value": 0.000009408082200025092,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / PreRev",
            "value": 0.00000935167590000674,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / PostRev",
            "value": 0.00001011617290000686,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / BothRev",
            "value": 0.000009534606999977768,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaX   / cpu / Primal",
            "value": 0.000010197514400351791,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / Primal",
            "value": 0.000009762815898284316,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaX   / cpu / Forward",
            "value": 0.00001514410990057513,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / Forward",
            "value": 0.00001572041369508952,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaX   / cpu / BothRev",
            "value": 0.000014613362698582931,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / PreRev",
            "value": 0.00001516341289971024,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / PostRev",
            "value": 0.000015223639801843092,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / BothRev",
            "value": 0.000014753799699246884,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaX   / cpu / Primal",
            "value": 0.000005057317899991176,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / Primal",
            "value": 0.00000486440290005703,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaX   / cpu / Forward",
            "value": 0.000007447806799973478,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / Forward",
            "value": 0.000007845410899972193,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaX   / cpu / BothRev",
            "value": 0.000007249214900002698,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / PreRev",
            "value": 0.000007566724899879773,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / PostRev",
            "value": 0.000007622093899772153,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / BothRev",
            "value": 0.000007513779900182272,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaX   / tpu / Primal",
            "value": 0.000048690696399717125,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / tpu / Primal",
            "value": 0.00004764917840002454,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaX   / tpu / Forward",
            "value": 0.00007891432590004115,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / tpu / Forward",
            "value": 0.00009467764369983344,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaX   / tpu / BothRev",
            "value": 0.00009526927869992503,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / tpu / PreRev",
            "value": 0.00009010985280001478,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / tpu / PostRev",
            "value": 0.00009111265780011308,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / tpu / BothRev",
            "value": 0.00009184784870012665,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaX   / cpu / Primal",
            "value": 0.000007490049299940437,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / Primal",
            "value": 0.000007268673799990211,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaX   / cpu / Forward",
            "value": 0.000011394478800048091,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / Forward",
            "value": 0.000011848924900004933,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaX   / cpu / BothRev",
            "value": 0.000011093092999999498,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / PreRev",
            "value": 0.000011492180500044925,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / PostRev",
            "value": 0.000011606285099969685,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / BothRev",
            "value": 0.000011469892199966123,
            "unit": "s"
          },
          {
            "name": "llama / JaXPipe / cpu / Primal",
            "value": 0.000953879120002,
            "unit": "s"
          },
          {
            "name": "llama / JaX   / cpu / Primal",
            "value": 0.0009254827099994,
            "unit": "s"
          },
          {
            "name": "llama / HLOOpt / cpu / Primal",
            "value": 0.0010788851000006,
            "unit": "s"
          },
          {
            "name": "llama / PartOpt / cpu / Primal",
            "value": 0.0009278225800017,
            "unit": "s"
          },
          {
            "name": "llama / DefOpt / cpu / Primal",
            "value": 0.0010432017300036,
            "unit": "s"
          },
          {
            "name": "llama / IPartOpt / cpu / Primal",
            "value": 0.0009028872399994,
            "unit": "s"
          },
          {
            "name": "llama / IDefOpt / cpu / Primal",
            "value": 0.0010399287900008,
            "unit": "s"
          },
          {
            "name": "llama / JaXPipe / cpu / Forward",
            "value": 0.0027620586900002,
            "unit": "s"
          },
          {
            "name": "llama / JaX   / cpu / Forward",
            "value": 0.0027439073699997,
            "unit": "s"
          },
          {
            "name": "llama / HLOOpt / cpu / Forward",
            "value": 0.0028174159699983,
            "unit": "s"
          },
          {
            "name": "llama / PartOpt / cpu / Forward",
            "value": 0.0028146338600026,
            "unit": "s"
          },
          {
            "name": "llama / DefOpt / cpu / Forward",
            "value": 0.0027927575700005,
            "unit": "s"
          },
          {
            "name": "llama / IPartOpt / cpu / Forward",
            "value": 0.0027756773599958,
            "unit": "s"
          },
          {
            "name": "llama / IDefOpt / cpu / Forward",
            "value": 0.0027898585699995,
            "unit": "s"
          },
          {
            "name": "llama / JaXPipe / cpu / PreRev",
            "value": 0.0024357223199967,
            "unit": "s"
          },
          {
            "name": "llama / JaXPipe / cpu / PostRev",
            "value": 0.0024181688699991,
            "unit": "s"
          },
          {
            "name": "llama / JaXPipe / cpu / BothRev",
            "value": 0.0026116750300025,
            "unit": "s"
          },
          {
            "name": "llama / JaX   / cpu / BothRev",
            "value": 0.0025716396500001,
            "unit": "s"
          },
          {
            "name": "llama / HLOOpt / cpu / PreRev",
            "value": 0.0026205702199968,
            "unit": "s"
          },
          {
            "name": "llama / HLOOpt / cpu / PostRev",
            "value": 0.002530471949999,
            "unit": "s"
          },
          {
            "name": "llama / HLOOpt / cpu / BothRev",
            "value": 0.0026025370099978,
            "unit": "s"
          },
          {
            "name": "llama / PartOpt / cpu / PreRev",
            "value": 0.0026076413600003,
            "unit": "s"
          },
          {
            "name": "llama / PartOpt / cpu / PostRev",
            "value": 0.00259278214,
            "unit": "s"
          },
          {
            "name": "llama / PartOpt / cpu / BothRev",
            "value": 0.0026430368699993,
            "unit": "s"
          },
          {
            "name": "llama / DefOpt / cpu / PreRev",
            "value": 0.0026427476200024,
            "unit": "s"
          },
          {
            "name": "llama / DefOpt / cpu / PostRev",
            "value": 0.0025120392999997,
            "unit": "s"
          },
          {
            "name": "llama / DefOpt / cpu / BothRev",
            "value": 0.0026069907100009,
            "unit": "s"
          },
          {
            "name": "llama / IPartOpt / cpu / PreRev",
            "value": 0.0026077375799968,
            "unit": "s"
          },
          {
            "name": "llama / IPartOpt / cpu / PostRev",
            "value": 0.0025907897100023,
            "unit": "s"
          },
          {
            "name": "llama / IPartOpt / cpu / BothRev",
            "value": 0.0026240649199962,
            "unit": "s"
          },
          {
            "name": "llama / IDefOpt / cpu / PreRev",
            "value": 0.0026147797799967,
            "unit": "s"
          },
          {
            "name": "llama / IDefOpt / cpu / PostRev",
            "value": 0.0026338785300004,
            "unit": "s"
          },
          {
            "name": "llama / IDefOpt / cpu / BothRev",
            "value": 0.0026268760499988,
            "unit": "s"
          },
          {
            "name": "llama / JaXPipe / gpu / Primal",
            "value": 0.000425984277972,
            "unit": "s"
          },
          {
            "name": "llama / JaX   / gpu / Primal",
            "value": 0.0004176969539839,
            "unit": "s"
          },
          {
            "name": "llama / HLOOpt / gpu / Primal",
            "value": 0.0004351908679818,
            "unit": "s"
          },
          {
            "name": "llama / PartOpt / gpu / Primal",
            "value": 0.0004162233740789,
            "unit": "s"
          },
          {
            "name": "llama / DefOpt / gpu / Primal",
            "value": 0.0004218636179575,
            "unit": "s"
          },
          {
            "name": "llama / IPartOpt / gpu / Primal",
            "value": 0.0004261225559748,
            "unit": "s"
          },
          {
            "name": "llama / IDefOpt / gpu / Primal",
            "value": 0.0004581717819673,
            "unit": "s"
          },
          {
            "name": "llama / JaXPipe / gpu / Forward",
            "value": 0.0007449556919746,
            "unit": "s"
          },
          {
            "name": "llama / JaX   / gpu / Forward",
            "value": 0.0007245548079954,
            "unit": "s"
          },
          {
            "name": "llama / HLOOpt / gpu / Forward",
            "value": 0.0007213908600388,
            "unit": "s"
          },
          {
            "name": "llama / PartOpt / gpu / Forward",
            "value": 0.0007472802340053,
            "unit": "s"
          },
          {
            "name": "llama / DefOpt / gpu / Forward",
            "value": 0.0007360266259638,
            "unit": "s"
          },
          {
            "name": "llama / IPartOpt / gpu / Forward",
            "value": 0.0007555711059831,
            "unit": "s"
          },
          {
            "name": "llama / IDefOpt / gpu / Forward",
            "value": 0.0007041589639848,
            "unit": "s"
          },
          {
            "name": "llama / JaXPipe / gpu / PreRev",
            "value": 0.000787520772079,
            "unit": "s"
          },
          {
            "name": "llama / JaXPipe / gpu / PostRev",
            "value": 0.0008202526860404,
            "unit": "s"
          },
          {
            "name": "llama / JaXPipe / gpu / BothRev",
            "value": 0.0007864726739935,
            "unit": "s"
          },
          {
            "name": "llama / JaX   / gpu / BothRev",
            "value": 0.0008165040800813,
            "unit": "s"
          },
          {
            "name": "llama / HLOOpt / gpu / PreRev",
            "value": 0.0007784320620121,
            "unit": "s"
          },
          {
            "name": "llama / HLOOpt / gpu / PostRev",
            "value": 0.0008148124200524,
            "unit": "s"
          },
          {
            "name": "llama / HLOOpt / gpu / BothRev",
            "value": 0.0008093111700145,
            "unit": "s"
          },
          {
            "name": "llama / PartOpt / gpu / PreRev",
            "value": 0.0007854444379918,
            "unit": "s"
          },
          {
            "name": "llama / PartOpt / gpu / PostRev",
            "value": 0.0008077274359529,
            "unit": "s"
          },
          {
            "name": "llama / PartOpt / gpu / BothRev",
            "value": 0.0007809118959121,
            "unit": "s"
          },
          {
            "name": "llama / DefOpt / gpu / PreRev",
            "value": 0.0007772183580091,
            "unit": "s"
          },
          {
            "name": "llama / DefOpt / gpu / PostRev",
            "value": 0.0007386875340016,
            "unit": "s"
          },
          {
            "name": "llama / DefOpt / gpu / BothRev",
            "value": 0.0007946859078947,
            "unit": "s"
          },
          {
            "name": "llama / IPartOpt / gpu / PreRev",
            "value": 0.0008020818519871,
            "unit": "s"
          },
          {
            "name": "llama / IPartOpt / gpu / PostRev",
            "value": 0.0007821885119192,
            "unit": "s"
          },
          {
            "name": "llama / IPartOpt / gpu / BothRev",
            "value": 0.0008240658679278,
            "unit": "s"
          },
          {
            "name": "llama / IDefOpt / gpu / PreRev",
            "value": 0.0007710946319857,
            "unit": "s"
          },
          {
            "name": "llama / IDefOpt / gpu / PostRev",
            "value": 0.0007822999299969,
            "unit": "s"
          },
          {
            "name": "llama / IDefOpt / gpu / BothRev",
            "value": 0.0008158979100408,
            "unit": "s"
          },
          {
            "name": "llama / JaXPipe / tpu / Primal",
            "value": 0.0003882780139974,
            "unit": "s"
          },
          {
            "name": "llama / JaX   / tpu / Primal",
            "value": 0.0003833954939982,
            "unit": "s"
          },
          {
            "name": "llama / HLOOpt / tpu / Primal",
            "value": 0.0003783398539962,
            "unit": "s"
          },
          {
            "name": "llama / PartOpt / tpu / Primal",
            "value": 0.0003919328539996,
            "unit": "s"
          },
          {
            "name": "llama / DefOpt / tpu / Primal",
            "value": 0.0003711308140045,
            "unit": "s"
          },
          {
            "name": "llama / IPartOpt / tpu / Primal",
            "value": 0.0003926429940038,
            "unit": "s"
          },
          {
            "name": "llama / IDefOpt / tpu / Primal",
            "value": 0.000379215894005,
            "unit": "s"
          },
          {
            "name": "llama / JaXPipe / tpu / Forward",
            "value": 0.0005766092900012,
            "unit": "s"
          },
          {
            "name": "llama / JaX   / tpu / Forward",
            "value": 0.0007152709500005,
            "unit": "s"
          },
          {
            "name": "llama / HLOOpt / tpu / Forward",
            "value": 0.0005833641319986,
            "unit": "s"
          },
          {
            "name": "llama / PartOpt / tpu / Forward",
            "value": 0.000574034969999,
            "unit": "s"
          },
          {
            "name": "llama / DefOpt / tpu / Forward",
            "value": 0.0005810543519983,
            "unit": "s"
          },
          {
            "name": "llama / IPartOpt / tpu / Forward",
            "value": 0.0005738792100019,
            "unit": "s"
          },
          {
            "name": "llama / IDefOpt / tpu / Forward",
            "value": 0.0005765431320032,
            "unit": "s"
          },
          {
            "name": "llama / JaXPipe / tpu / PreRev",
            "value": 0.0004067212739973,
            "unit": "s"
          },
          {
            "name": "llama / JaXPipe / tpu / PostRev",
            "value": 0.0003582878140005,
            "unit": "s"
          },
          {
            "name": "llama / JaXPipe / tpu / BothRev",
            "value": 0.0003918005540035,
            "unit": "s"
          },
          {
            "name": "llama / JaX   / tpu / BothRev",
            "value": 0.0003582309940029,
            "unit": "s"
          },
          {
            "name": "llama / HLOOpt / tpu / PreRev",
            "value": 0.0003919378159989,
            "unit": "s"
          },
          {
            "name": "llama / HLOOpt / tpu / PostRev",
            "value": 0.000390676394003,
            "unit": "s"
          },
          {
            "name": "llama / HLOOpt / tpu / BothRev",
            "value": 0.0003919224739947,
            "unit": "s"
          },
          {
            "name": "llama / PartOpt / tpu / PreRev",
            "value": 0.0003918862540012,
            "unit": "s"
          },
          {
            "name": "llama / PartOpt / tpu / PostRev",
            "value": 0.0003720964140011,
            "unit": "s"
          },
          {
            "name": "llama / PartOpt / tpu / BothRev",
            "value": 0.000391959913999,
            "unit": "s"
          },
          {
            "name": "llama / DefOpt / tpu / PreRev",
            "value": 0.0003918760140004,
            "unit": "s"
          },
          {
            "name": "llama / DefOpt / tpu / PostRev",
            "value": 0.0003836159140046,
            "unit": "s"
          },
          {
            "name": "llama / DefOpt / tpu / BothRev",
            "value": 0.0003918232939977,
            "unit": "s"
          },
          {
            "name": "llama / IPartOpt / tpu / PreRev",
            "value": 0.0003913405740022,
            "unit": "s"
          },
          {
            "name": "llama / IPartOpt / tpu / PostRev",
            "value": 0.0003722600740002,
            "unit": "s"
          },
          {
            "name": "llama / IPartOpt / tpu / BothRev",
            "value": 0.0003919827339996,
            "unit": "s"
          },
          {
            "name": "llama / IDefOpt / tpu / PreRev",
            "value": 0.0003915870140044,
            "unit": "s"
          },
          {
            "name": "llama / IDefOpt / tpu / PostRev",
            "value": 0.0003996198140011,
            "unit": "s"
          },
          {
            "name": "llama / IDefOpt / tpu / BothRev",
            "value": 0.0003917821140057,
            "unit": "s"
          },
          {
            "name": "llama / JaXPipe / cpu / Primal",
            "value": 0.0025118638399999,
            "unit": "s"
          },
          {
            "name": "llama / JaX   / cpu / Primal",
            "value": 0.0024541628099996,
            "unit": "s"
          },
          {
            "name": "llama / HLOOpt / cpu / Primal",
            "value": 0.0024098401399987,
            "unit": "s"
          },
          {
            "name": "llama / PartOpt / cpu / Primal",
            "value": 0.0023452424600054,
            "unit": "s"
          },
          {
            "name": "llama / DefOpt / cpu / Primal",
            "value": 0.0023730191999948,
            "unit": "s"
          },
          {
            "name": "llama / IPartOpt / cpu / Primal",
            "value": 0.002410362519995,
            "unit": "s"
          },
          {
            "name": "llama / IDefOpt / cpu / Primal",
            "value": 0.0026382764800018,
            "unit": "s"
          },
          {
            "name": "llama / JaXPipe / cpu / Forward",
            "value": 0.0067970462900029,
            "unit": "s"
          },
          {
            "name": "llama / JaX   / cpu / Forward",
            "value": 0.0068298099999992,
            "unit": "s"
          },
          {
            "name": "llama / HLOOpt / cpu / Forward",
            "value": 0.0067651859800025,
            "unit": "s"
          },
          {
            "name": "llama / PartOpt / cpu / Forward",
            "value": 0.006818186649998,
            "unit": "s"
          },
          {
            "name": "llama / DefOpt / cpu / Forward",
            "value": 0.0067876143499961,
            "unit": "s"
          },
          {
            "name": "llama / IPartOpt / cpu / Forward",
            "value": 0.0064367527899958,
            "unit": "s"
          },
          {
            "name": "llama / IDefOpt / cpu / Forward",
            "value": 0.0067154070399919,
            "unit": "s"
          },
          {
            "name": "llama / JaXPipe / cpu / PreRev",
            "value": 0.0058976585599975,
            "unit": "s"
          },
          {
            "name": "llama / JaXPipe / cpu / PostRev",
            "value": 0.0053568107299997,
            "unit": "s"
          },
          {
            "name": "llama / JaXPipe / cpu / BothRev",
            "value": 0.0061718162000033,
            "unit": "s"
          },
          {
            "name": "llama / JaX   / cpu / BothRev",
            "value": 0.0051904091400047,
            "unit": "s"
          },
          {
            "name": "llama / HLOOpt / cpu / PreRev",
            "value": 0.0063451987999997,
            "unit": "s"
          },
          {
            "name": "llama / HLOOpt / cpu / PostRev",
            "value": 0.0053553890400053,
            "unit": "s"
          },
          {
            "name": "llama / HLOOpt / cpu / BothRev",
            "value": 0.0062526236299981,
            "unit": "s"
          },
          {
            "name": "llama / PartOpt / cpu / PreRev",
            "value": 0.0058145864299967,
            "unit": "s"
          },
          {
            "name": "llama / PartOpt / cpu / PostRev",
            "value": 0.0058343740899999,
            "unit": "s"
          },
          {
            "name": "llama / PartOpt / cpu / BothRev",
            "value": 0.0058421067600011,
            "unit": "s"
          },
          {
            "name": "llama / DefOpt / cpu / PreRev",
            "value": 0.0063706269000067,
            "unit": "s"
          },
          {
            "name": "llama / DefOpt / cpu / PostRev",
            "value": 0.0054638736900051,
            "unit": "s"
          },
          {
            "name": "llama / DefOpt / cpu / BothRev",
            "value": 0.0062915154500024,
            "unit": "s"
          },
          {
            "name": "llama / IPartOpt / cpu / PreRev",
            "value": 0.0058196629699978,
            "unit": "s"
          },
          {
            "name": "llama / IPartOpt / cpu / PostRev",
            "value": 0.0056631537400062,
            "unit": "s"
          },
          {
            "name": "llama / IPartOpt / cpu / BothRev",
            "value": 0.0057261384799949,
            "unit": "s"
          },
          {
            "name": "llama / IDefOpt / cpu / PreRev",
            "value": 0.0056622546699964,
            "unit": "s"
          },
          {
            "name": "llama / IDefOpt / cpu / PostRev",
            "value": 0.0059854002599968,
            "unit": "s"
          },
          {
            "name": "llama / IDefOpt / cpu / BothRev",
            "value": 0.0061599940799987,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004747398700010308,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004742438499988566,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000009105573099805044,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000008997576998081057,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / gpu / Primal",
            "value": 0.0000804016864974983,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / gpu / Primal",
            "value": 0.00008063908839831129,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.0000043399499001679945,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004309854899838683,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001523514909,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001529742769002,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000006079079099981755,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000006363962900013576,
            "unit": "s"
          },
          {
            "name": "slicing / JaX   / cpu / Primal",
            "value": 0.000003751718300009088,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / Primal",
            "value": 0.00000373822060000748,
            "unit": "s"
          },
          {
            "name": "slicing / JaX   / cpu / Forward",
            "value": 0.0000058094999000331885,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / Forward",
            "value": 0.000005969419500024742,
            "unit": "s"
          },
          {
            "name": "slicing / JaX   / cpu / BothRev",
            "value": 0.000006423055899995234,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / PreRev",
            "value": 0.000006427394399997865,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / PostRev",
            "value": 0.000006489993500008495,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / BothRev",
            "value": 0.000006449053799997273,
            "unit": "s"
          },
          {
            "name": "slicing / JaX   / cpu / Primal",
            "value": 0.000007102839899016544,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / Primal",
            "value": 0.000007423962303437292,
            "unit": "s"
          },
          {
            "name": "slicing / JaX   / cpu / Forward",
            "value": 0.000010276582703227178,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / Forward",
            "value": 0.000010852559399791062,
            "unit": "s"
          },
          {
            "name": "slicing / JaX   / cpu / BothRev",
            "value": 0.00001167701089871116,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / PreRev",
            "value": 0.000011048073100391776,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / PostRev",
            "value": 0.000011609177198261022,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / BothRev",
            "value": 0.000011624004802433774,
            "unit": "s"
          },
          {
            "name": "slicing / JaX   / gpu / Primal",
            "value": 0.00007485117940232158,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / gpu / Primal",
            "value": 0.00007555639520287514,
            "unit": "s"
          },
          {
            "name": "slicing / JaX   / gpu / Forward",
            "value": 0.0001033515762013,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / gpu / Forward",
            "value": 0.0001044832094048,
            "unit": "s"
          },
          {
            "name": "slicing / JaX   / gpu / BothRev",
            "value": 0.0001013122671982,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / gpu / PreRev",
            "value": 0.0001047621739038,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / gpu / PostRev",
            "value": 0.0001026428029988,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / gpu / BothRev",
            "value": 0.0001044709633046,
            "unit": "s"
          },
          {
            "name": "slicing / JaX   / cpu / Primal",
            "value": 0.0000035957298998255283,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / Primal",
            "value": 0.000004370602000199142,
            "unit": "s"
          },
          {
            "name": "slicing / JaX   / cpu / Forward",
            "value": 0.000004692957999941428,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / Forward",
            "value": 0.00000467520499987586,
            "unit": "s"
          },
          {
            "name": "slicing / JaX   / cpu / BothRev",
            "value": 0.0000048525959999096815,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / PreRev",
            "value": 0.000005028258899983484,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / PostRev",
            "value": 0.000005099466899991967,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / BothRev",
            "value": 0.000005694057900109329,
            "unit": "s"
          },
          {
            "name": "slicing / JaX   / tpu / Primal",
            "value": 0.0001521861168999,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / tpu / Primal",
            "value": 0.0001533395929,
            "unit": "s"
          },
          {
            "name": "slicing / JaX   / tpu / Forward",
            "value": 0.0002415839016997,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / tpu / Forward",
            "value": 0.0002261090009,
            "unit": "s"
          },
          {
            "name": "slicing / JaX   / tpu / BothRev",
            "value": 0.0002344483487999,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / tpu / PreRev",
            "value": 0.0002302942278998,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / tpu / PostRev",
            "value": 0.0002195158049999,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / tpu / BothRev",
            "value": 0.0002251412939,
            "unit": "s"
          },
          {
            "name": "slicing / JaX   / cpu / Primal",
            "value": 0.000004869463999966683,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / Primal",
            "value": 0.000005296094699951937,
            "unit": "s"
          },
          {
            "name": "slicing / JaX   / cpu / Forward",
            "value": 0.000007326353899952664,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / Forward",
            "value": 0.000007270701400011603,
            "unit": "s"
          },
          {
            "name": "slicing / JaX   / cpu / BothRev",
            "value": 0.000008321718200022588,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / PreRev",
            "value": 0.000007859259199994995,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / PostRev",
            "value": 0.000008321393299956981,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / BothRev",
            "value": 0.00000836516710005526,
            "unit": "s"
          },
          {
            "name": "sum    / JaX   / cpu / Primal",
            "value": 0.000005017851999991763,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / Primal",
            "value": 0.000005066681500011328,
            "unit": "s"
          },
          {
            "name": "sum    / JaX   / cpu / Forward",
            "value": 0.00000848556659998394,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / Forward",
            "value": 0.000008551167200039345,
            "unit": "s"
          },
          {
            "name": "sum    / JaX   / cpu / BothRev",
            "value": 0.00000769564869997339,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / PreRev",
            "value": 0.000007359192700005224,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / PostRev",
            "value": 0.000007374097499996424,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / BothRev",
            "value": 0.000007370912399983353,
            "unit": "s"
          },
          {
            "name": "sum    / JaX   / cpu / Primal",
            "value": 0.000010408044594805688,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / Primal",
            "value": 0.000009743270702892915,
            "unit": "s"
          },
          {
            "name": "sum    / JaX   / cpu / Forward",
            "value": 0.000015271548001328482,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / Forward",
            "value": 0.000014552888798061756,
            "unit": "s"
          },
          {
            "name": "sum    / JaX   / cpu / BothRev",
            "value": 0.000013468827004544436,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / PreRev",
            "value": 0.000013529903901508078,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / PostRev",
            "value": 0.00001400353719945997,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / BothRev",
            "value": 0.000013533097499748691,
            "unit": "s"
          },
          {
            "name": "sum    / JaX   / gpu / Primal",
            "value": 0.00007259774669655598,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / gpu / Primal",
            "value": 0.00007502381969825364,
            "unit": "s"
          },
          {
            "name": "sum    / JaX   / gpu / Forward",
            "value": 0.0000980319740017876,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / gpu / Forward",
            "value": 0.00009907336519681848,
            "unit": "s"
          },
          {
            "name": "sum    / JaX   / gpu / BothRev",
            "value": 0.0001031566765042,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / gpu / PreRev",
            "value": 0.0001041030914988,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / gpu / PostRev",
            "value": 0.0001029314185027,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / gpu / BothRev",
            "value": 0.0001022709308017,
            "unit": "s"
          },
          {
            "name": "sum    / JaX   / cpu / Primal",
            "value": 0.000004648657899815589,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / Primal",
            "value": 0.000004426805999901263,
            "unit": "s"
          },
          {
            "name": "sum    / JaX   / cpu / Forward",
            "value": 0.000006968069900176488,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / Forward",
            "value": 0.000006936067900096532,
            "unit": "s"
          },
          {
            "name": "sum    / JaX   / cpu / BothRev",
            "value": 0.000006382804900204064,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / PreRev",
            "value": 0.000006116585899872007,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / PostRev",
            "value": 0.000006402536900350242,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / BothRev",
            "value": 0.000006120433899923228,
            "unit": "s"
          },
          {
            "name": "sum    / JaX   / tpu / Primal",
            "value": 0.000144263709,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / tpu / Primal",
            "value": 0.0001483542870002,
            "unit": "s"
          },
          {
            "name": "sum    / JaX   / tpu / Forward",
            "value": 0.000220564811,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / tpu / Forward",
            "value": 0.0002191943331003,
            "unit": "s"
          },
          {
            "name": "sum    / JaX   / tpu / BothRev",
            "value": 0.0002308644338998,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / tpu / PreRev",
            "value": 0.0002274248169,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / tpu / PostRev",
            "value": 0.0002309231917999,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / tpu / BothRev",
            "value": 0.0002065773051999,
            "unit": "s"
          },
          {
            "name": "sum    / JaX   / cpu / Primal",
            "value": 0.000007130642100037221,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / Primal",
            "value": 0.000006949472799988144,
            "unit": "s"
          },
          {
            "name": "sum    / JaX   / cpu / Forward",
            "value": 0.000010643948300003105,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / Forward",
            "value": 0.000010678369499964902,
            "unit": "s"
          },
          {
            "name": "sum    / JaX   / cpu / BothRev",
            "value": 0.0000098578701000406,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / PreRev",
            "value": 0.00000949042369993549,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / PostRev",
            "value": 0.00000949360220001836,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / BothRev",
            "value": 0.000009533501299938507,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / JaXPipe / gpu / Primal",
            "value": 0.0010021896974649,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / JaX   / gpu / Primal",
            "value": 0.0010122301988303,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / HLOOpt / gpu / Primal",
            "value": 0.0009618734999094,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / PartOpt / gpu / Primal",
            "value": 0.0009521803993266,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / DefOpt / gpu / Primal",
            "value": 0.0006915163015946,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / IPartOpt / gpu / Primal",
            "value": 0.0009531811985652,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / IDefOpt / gpu / Primal",
            "value": 0.0007057737035211,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / JaX   / gpu / Forward",
            "value": 0.0012442076986189,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / JaXPipe / gpu / PostRev",
            "value": 0.0040186793019529,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / JaX   / gpu / BothRev",
            "value": 0.0040108970017172,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / HLOOpt / gpu / PostRev",
            "value": 0.0040236603992525,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / PartOpt / gpu / PostRev",
            "value": 0.0040931484021712,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / DefOpt / gpu / PostRev",
            "value": 0.0022164708003401,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / IPartOpt / gpu / PostRev",
            "value": 0.0040894888981711,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / IDefOpt / gpu / PostRev",
            "value": 0.0025417056982405,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / JaXPipe / tpu / Primal",
            "value": 0.00008572000006097369,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / JaX   / tpu / Primal",
            "value": 0.0000962939997407375,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / HLOOpt / tpu / Primal",
            "value": 0.00009696699999039991,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / PartOpt / tpu / Primal",
            "value": 0.00009957000002032146,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / DefOpt / tpu / Primal",
            "value": 0.0001019639999867,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / IPartOpt / tpu / Primal",
            "value": 0.0001003209999907,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / IDefOpt / tpu / Primal",
            "value": 0.0001005119000183,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / JaX   / tpu / Forward",
            "value": 0.0001793679999536,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / JaXPipe / tpu / PostRev",
            "value": 0.0001792219998606,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / JaX   / tpu / BothRev",
            "value": 0.0002025499998126,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / HLOOpt / tpu / PostRev",
            "value": 0.0002034080000157,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / PartOpt / tpu / PostRev",
            "value": 0.0002090339999995,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / DefOpt / tpu / PostRev",
            "value": 0.000201840000227,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / IPartOpt / tpu / PostRev",
            "value": 0.0002056410001387,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / IDefOpt / tpu / PostRev",
            "value": 0.0001981600002181,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / JaXPipe / cpu / Primal",
            "value": 0.00006749929998477455,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / JaX   / cpu / Primal",
            "value": 0.00004747069997392828,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / HLOOpt / cpu / Primal",
            "value": 0.00007167370004026452,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / PartOpt / cpu / Primal",
            "value": 0.00009859560004770174,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / DefOpt / cpu / Primal",
            "value": 0.00007036259994492866,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / IPartOpt / cpu / Primal",
            "value": 0.00007556290001957678,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / IDefOpt / cpu / Primal",
            "value": 0.0000739297000109218,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / JaX   / cpu / Forward",
            "value": 0.0001467305000005,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / JaXPipe / cpu / PostRev",
            "value": 0.0001261514999896,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / JaX   / cpu / BothRev",
            "value": 0.0002017273999626,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / HLOOpt / cpu / PostRev",
            "value": 0.0001452332999178,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / PartOpt / cpu / PostRev",
            "value": 0.0001535636999506,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / DefOpt / cpu / PostRev",
            "value": 0.0001419621999957,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / IPartOpt / cpu / PostRev",
            "value": 0.0001924867000525,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / IDefOpt / cpu / PostRev",
            "value": 0.0001613399000234,
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
          "id": "72661ea9f53745c4313a83cfd3f1582c19f8e278",
          "message": "feat: while dynamic_slice dynamic_update_slice simplify (#1746)\n\n* feat: while dynamic_slice dynamic_update_slice simplify\n\n* feat: generalize bounds propagation through the IR\n\n* test: add bounds annotations\n\n* test: dus ds simplify",
          "timestamp": "2025-12-13T08:16:32-05:00",
          "tree_id": "90e400238bbe85bc1c9a34e6946d7227c461df65",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/72661ea9f53745c4313a83cfd3f1582c19f8e278"
        },
        "date": 1765641294355,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "actmtch / JaX   / cpu / Primal",
            "value": 0.000004400547099976393,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / Primal",
            "value": 0.000003847257600000375,
            "unit": "s"
          },
          {
            "name": "actmtch / JaX   / cpu / Forward",
            "value": 0.000006232840500024394,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / Forward",
            "value": 0.000007316326599993772,
            "unit": "s"
          },
          {
            "name": "actmtch / JaX   / cpu / BothRev",
            "value": 0.000006366681800000151,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / PreRev",
            "value": 0.000007466115700026421,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / PostRev",
            "value": 0.00000612150520000796,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / BothRev",
            "value": 0.000007639759099993172,
            "unit": "s"
          },
          {
            "name": "actmtch / JaX   / cpu / Primal",
            "value": 0.00000717440290027298,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / Primal",
            "value": 0.000007394470198778436,
            "unit": "s"
          },
          {
            "name": "actmtch / JaX   / cpu / Forward",
            "value": 0.000010446489602327349,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / Forward",
            "value": 0.000014107218303252012,
            "unit": "s"
          },
          {
            "name": "actmtch / JaX   / cpu / BothRev",
            "value": 0.000010737801803043111,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / PreRev",
            "value": 0.000013594295299844816,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / PostRev",
            "value": 0.000010848893999354914,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / BothRev",
            "value": 0.000013441033201524989,
            "unit": "s"
          },
          {
            "name": "actmtch / JaX   / gpu / Primal",
            "value": 0.00007767562899971381,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / gpu / Primal",
            "value": 0.00007594794579781592,
            "unit": "s"
          },
          {
            "name": "actmtch / JaX   / gpu / Forward",
            "value": 0.0001049683589022,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / gpu / Forward",
            "value": 0.0001041774920013,
            "unit": "s"
          },
          {
            "name": "actmtch / JaX   / gpu / BothRev",
            "value": 0.0001057001598994,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / gpu / PreRev",
            "value": 0.0001084825240017,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / gpu / PostRev",
            "value": 0.0001060831893002,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / gpu / BothRev",
            "value": 0.0001226130085997,
            "unit": "s"
          },
          {
            "name": "actmtch / JaX   / cpu / Primal",
            "value": 0.0000036510980004095472,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / Primal",
            "value": 0.000003710966999642551,
            "unit": "s"
          },
          {
            "name": "actmtch / JaX   / cpu / Forward",
            "value": 0.00000514542290038662,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / Forward",
            "value": 0.00000640888900088612,
            "unit": "s"
          },
          {
            "name": "actmtch / JaX   / cpu / BothRev",
            "value": 0.000005483010999159887,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / PreRev",
            "value": 0.000006029347999719903,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / PostRev",
            "value": 0.000005479409899271559,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / BothRev",
            "value": 0.000006369073899986688,
            "unit": "s"
          },
          {
            "name": "actmtch / JaX   / tpu / Primal",
            "value": 0.0001596149605,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / tpu / Primal",
            "value": 0.0001531629845994,
            "unit": "s"
          },
          {
            "name": "actmtch / JaX   / tpu / Forward",
            "value": 0.0002266139339,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / tpu / Forward",
            "value": 0.0002327674528001,
            "unit": "s"
          },
          {
            "name": "actmtch / JaX   / tpu / BothRev",
            "value": 0.0002246276967998,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / tpu / PreRev",
            "value": 0.0002132159429005,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / tpu / PostRev",
            "value": 0.0002142569340008,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / tpu / BothRev",
            "value": 0.000217084993,
            "unit": "s"
          },
          {
            "name": "actmtch / JaX   / cpu / Primal",
            "value": 0.000006169318000047497,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / Primal",
            "value": 0.000006012625699986529,
            "unit": "s"
          },
          {
            "name": "actmtch / JaX   / cpu / Forward",
            "value": 0.000008609548399999767,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / Forward",
            "value": 0.000012759331000052043,
            "unit": "s"
          },
          {
            "name": "actmtch / JaX   / cpu / BothRev",
            "value": 0.000008619226399969193,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / PreRev",
            "value": 0.000012495292400035396,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / PostRev",
            "value": 0.00000934114009996847,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / BothRev",
            "value": 0.000012811163000060331,
            "unit": "s"
          },
          {
            "name": "add_one / JaX   / cpu / Primal",
            "value": 0.000004206737899994551,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / Primal",
            "value": 0.00000425670849999733,
            "unit": "s"
          },
          {
            "name": "add_one / JaX   / cpu / Forward",
            "value": 0.00000771281799998178,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / Forward",
            "value": 0.000007813527799999066,
            "unit": "s"
          },
          {
            "name": "add_one / JaX   / cpu / BothRev",
            "value": 0.000008141553799987377,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / PreRev",
            "value": 0.000007770555300021442,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / PostRev",
            "value": 0.000007923713399986809,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / BothRev",
            "value": 0.00000786075979999623,
            "unit": "s"
          },
          {
            "name": "add_one / JaX   / cpu / Primal",
            "value": 0.000007683804503176362,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / Primal",
            "value": 0.000007585405599093064,
            "unit": "s"
          },
          {
            "name": "add_one / JaX   / cpu / Forward",
            "value": 0.00001171101740328595,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / Forward",
            "value": 0.00001238468789961189,
            "unit": "s"
          },
          {
            "name": "add_one / JaX   / cpu / BothRev",
            "value": 0.000011857801303267478,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / PreRev",
            "value": 0.000011929331201827154,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / PostRev",
            "value": 0.000012685235601384191,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / BothRev",
            "value": 0.000012562349601648748,
            "unit": "s"
          },
          {
            "name": "add_one / JaX   / gpu / Primal",
            "value": 0.00007598779179970733,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / gpu / Primal",
            "value": 0.0000796766706975177,
            "unit": "s"
          },
          {
            "name": "add_one / JaX   / gpu / Forward",
            "value": 0.0001075573299021,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / gpu / Forward",
            "value": 0.0001370560679992,
            "unit": "s"
          },
          {
            "name": "add_one / JaX   / gpu / BothRev",
            "value": 0.0001099258871981,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / gpu / PreRev",
            "value": 0.0001098064329009,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / gpu / PostRev",
            "value": 0.0001077757744991,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / gpu / BothRev",
            "value": 0.0001112800272007,
            "unit": "s"
          },
          {
            "name": "add_one / JaX   / cpu / Primal",
            "value": 0.000003780662998906337,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / Primal",
            "value": 0.000003771154000423849,
            "unit": "s"
          },
          {
            "name": "add_one / JaX   / cpu / Forward",
            "value": 0.000005800796899711713,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / Forward",
            "value": 0.000006081247900146991,
            "unit": "s"
          },
          {
            "name": "add_one / JaX   / cpu / BothRev",
            "value": 0.000006133380999381188,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / PreRev",
            "value": 0.000006122940999921411,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / PostRev",
            "value": 0.0000061194878988317215,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / BothRev",
            "value": 0.000006134154000028502,
            "unit": "s"
          },
          {
            "name": "add_one / JaX   / tpu / Primal",
            "value": 0.0001397670236998,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / tpu / Primal",
            "value": 0.0001384344945996,
            "unit": "s"
          },
          {
            "name": "add_one / JaX   / tpu / Forward",
            "value": 0.0002032166901,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / tpu / Forward",
            "value": 0.0002135179000004,
            "unit": "s"
          },
          {
            "name": "add_one / JaX   / tpu / BothRev",
            "value": 0.0002238872898,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / tpu / PreRev",
            "value": 0.0002270792787996,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / tpu / PostRev",
            "value": 0.0002268739589009,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / tpu / BothRev",
            "value": 0.0002242527688009,
            "unit": "s"
          },
          {
            "name": "add_one / JaX   / cpu / Primal",
            "value": 0.000006275337599981868,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / Primal",
            "value": 0.000006356352000057086,
            "unit": "s"
          },
          {
            "name": "add_one / JaX   / cpu / Forward",
            "value": 0.000009889906199987308,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / Forward",
            "value": 0.00000989894930007722,
            "unit": "s"
          },
          {
            "name": "add_one / JaX   / cpu / BothRev",
            "value": 0.00001046865530006471,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / PreRev",
            "value": 0.000010497870200015314,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / PostRev",
            "value": 0.000013245314800042251,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / BothRev",
            "value": 0.00001047034889998031,
            "unit": "s"
          },
          {
            "name": "add_two / JaX   / cpu / Primal",
            "value": 0.000004423930900020423,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / Primal",
            "value": 0.000004419015099983881,
            "unit": "s"
          },
          {
            "name": "add_two / JaX   / cpu / Forward",
            "value": 0.000007534371200017631,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / Forward",
            "value": 0.00000759460920003221,
            "unit": "s"
          },
          {
            "name": "add_two / JaX   / cpu / BothRev",
            "value": 0.000010307918499984224,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / PreRev",
            "value": 0.000009947185100008935,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / PostRev",
            "value": 0.000009880139999995665,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / BothRev",
            "value": 0.000009652457199990749,
            "unit": "s"
          },
          {
            "name": "add_two / JaX   / cpu / Primal",
            "value": 0.000007836566300829873,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / Primal",
            "value": 0.000007989053602796048,
            "unit": "s"
          },
          {
            "name": "add_two / JaX   / cpu / Forward",
            "value": 0.000012176750198705122,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / Forward",
            "value": 0.000012651235196972266,
            "unit": "s"
          },
          {
            "name": "add_two / JaX   / cpu / BothRev",
            "value": 0.000014874262001831084,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / PreRev",
            "value": 0.000014763517101528124,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / PostRev",
            "value": 0.000014006631699157878,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / BothRev",
            "value": 0.000014821460202801972,
            "unit": "s"
          },
          {
            "name": "add_two / JaX   / gpu / Primal",
            "value": 0.00007785793200018816,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / gpu / Primal",
            "value": 0.00007639376149745658,
            "unit": "s"
          },
          {
            "name": "add_two / JaX   / gpu / Forward",
            "value": 0.0001070641455997,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / gpu / Forward",
            "value": 0.0001248084491991,
            "unit": "s"
          },
          {
            "name": "add_two / JaX   / gpu / BothRev",
            "value": 0.0001333960857999,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / gpu / PreRev",
            "value": 0.0001237886757997,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / gpu / PostRev",
            "value": 0.0001231971540022,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / gpu / BothRev",
            "value": 0.0001241640994034,
            "unit": "s"
          },
          {
            "name": "add_two / JaX   / cpu / Primal",
            "value": 0.000003883980000682641,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / Primal",
            "value": 0.000003959568998834584,
            "unit": "s"
          },
          {
            "name": "add_two / JaX   / cpu / Forward",
            "value": 0.000005965229000139516,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / Forward",
            "value": 0.000006022870900051203,
            "unit": "s"
          },
          {
            "name": "add_two / JaX   / cpu / BothRev",
            "value": 0.000007232300900795962,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / PreRev",
            "value": 0.000006924331899790559,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / PostRev",
            "value": 0.000007278672899701632,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / BothRev",
            "value": 0.000007241814899316523,
            "unit": "s"
          },
          {
            "name": "add_two / JaX   / tpu / Primal",
            "value": 0.0001388166176999,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / tpu / Primal",
            "value": 0.0001477053246009,
            "unit": "s"
          },
          {
            "name": "add_two / JaX   / tpu / Forward",
            "value": 0.0002144849339005,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / tpu / Forward",
            "value": 0.0001946795961004,
            "unit": "s"
          },
          {
            "name": "add_two / JaX   / tpu / BothRev",
            "value": 0.0002023806650991,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / tpu / PreRev",
            "value": 0.0001933179571991,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / tpu / PostRev",
            "value": 0.0001973097611,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / tpu / BothRev",
            "value": 0.0001876168702001,
            "unit": "s"
          },
          {
            "name": "add_two / JaX   / cpu / Primal",
            "value": 0.00000654602029999296,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / Primal",
            "value": 0.000006569651599966164,
            "unit": "s"
          },
          {
            "name": "add_two / JaX   / cpu / Forward",
            "value": 0.000010098877400014316,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / Forward",
            "value": 0.00001060231580004256,
            "unit": "s"
          },
          {
            "name": "add_two / JaX   / cpu / BothRev",
            "value": 0.000012398691399994278,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / PreRev",
            "value": 0.000012560004799979651,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / PostRev",
            "value": 0.00001253509779999149,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / BothRev",
            "value": 0.000012346788700051547,
            "unit": "s"
          },
          {
            "name": "cache / JaX   / cpu / Primal",
            "value": 0.000003868338199981736,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / Primal",
            "value": 0.000003946217600014279,
            "unit": "s"
          },
          {
            "name": "cache / JaX   / cpu / Forward",
            "value": 0.000010845477800012304,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / Forward",
            "value": 0.000010921989999997095,
            "unit": "s"
          },
          {
            "name": "cache / JaX   / cpu / BothRev",
            "value": 0.000015928096199968422,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / PreRev",
            "value": 0.000012396553000007771,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / PostRev",
            "value": 0.00001613206289998743,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / BothRev",
            "value": 0.000012352175800015177,
            "unit": "s"
          },
          {
            "name": "cache / JaX   / cpu / Primal",
            "value": 0.000007692726497771218,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / Primal",
            "value": 0.000008467647695215419,
            "unit": "s"
          },
          {
            "name": "cache / JaX   / cpu / Forward",
            "value": 0.00001319191989605315,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / Forward",
            "value": 0.000012646481598494574,
            "unit": "s"
          },
          {
            "name": "cache / JaX   / cpu / BothRev",
            "value": 0.00002389538189745508,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / PreRev",
            "value": 0.000017042687901994215,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / PostRev",
            "value": 0.000023872428195318207,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / BothRev",
            "value": 0.00001815648670308292,
            "unit": "s"
          },
          {
            "name": "cache / JaX   / gpu / Primal",
            "value": 0.00007524239749764092,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / gpu / Primal",
            "value": 0.00007552717790240422,
            "unit": "s"
          },
          {
            "name": "cache / JaX   / gpu / Forward",
            "value": 0.0001012045541021,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / gpu / Forward",
            "value": 0.0001033096026978,
            "unit": "s"
          },
          {
            "name": "cache / JaX   / gpu / BothRev",
            "value": 0.0001070618597965,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / gpu / PreRev",
            "value": 0.000106973509601,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / gpu / PostRev",
            "value": 0.0001040219414979,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / gpu / BothRev",
            "value": 0.0001055461128999,
            "unit": "s"
          },
          {
            "name": "cache / JaX   / cpu / Primal",
            "value": 0.000003492639900650829,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / Primal",
            "value": 0.000003513110900530592,
            "unit": "s"
          },
          {
            "name": "cache / JaX   / cpu / Forward",
            "value": 0.000006996955900103785,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / Forward",
            "value": 0.000005626710000797175,
            "unit": "s"
          },
          {
            "name": "cache / JaX   / cpu / BothRev",
            "value": 0.00002939719870046247,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / PreRev",
            "value": 0.000026958466799987944,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / PostRev",
            "value": 0.000031628981701214795,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / BothRev",
            "value": 0.00002760212769935606,
            "unit": "s"
          },
          {
            "name": "cache / JaX   / tpu / Primal",
            "value": 0.0001565434135001,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / tpu / Primal",
            "value": 0.0001335187317003,
            "unit": "s"
          },
          {
            "name": "cache / JaX   / tpu / Forward",
            "value": 0.0001909099162003,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / tpu / Forward",
            "value": 0.0001865696252993,
            "unit": "s"
          },
          {
            "name": "cache / JaX   / tpu / BothRev",
            "value": 0.0001928293521996,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / tpu / PreRev",
            "value": 0.0001936942271,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / tpu / PostRev",
            "value": 0.0001966969941,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / tpu / BothRev",
            "value": 0.0001893797251992,
            "unit": "s"
          },
          {
            "name": "cache / JaX   / cpu / Primal",
            "value": 0.000007263153599978978,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / Primal",
            "value": 0.000007660963699981948,
            "unit": "s"
          },
          {
            "name": "cache / JaX   / cpu / Forward",
            "value": 0.00001312004679994061,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / Forward",
            "value": 0.000012131845200019598,
            "unit": "s"
          },
          {
            "name": "cache / JaX   / cpu / BothRev",
            "value": 0.000022107076599968424,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / PreRev",
            "value": 0.00001654939749996629,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / PostRev",
            "value": 0.000022190686200065105,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / BothRev",
            "value": 0.000016611400500005404,
            "unit": "s"
          },
          {
            "name": "Concat / JaX   / cpu / Primal",
            "value": 0.0000041114908000054126,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / Primal",
            "value": 0.000004140261800012013,
            "unit": "s"
          },
          {
            "name": "Concat / JaX   / cpu / Forward",
            "value": 0.000007101013599958605,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / Forward",
            "value": 0.000007030118200009383,
            "unit": "s"
          },
          {
            "name": "Concat / JaX   / cpu / BothRev",
            "value": 0.000007668638800032568,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / PreRev",
            "value": 0.000007594698399998378,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / PostRev",
            "value": 0.00000787657600003513,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / BothRev",
            "value": 0.000007620093300010922,
            "unit": "s"
          },
          {
            "name": "Concat / JaX   / cpu / Primal",
            "value": 0.000007715437398292124,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / Primal",
            "value": 0.000007743358396692202,
            "unit": "s"
          },
          {
            "name": "Concat / JaX   / cpu / Forward",
            "value": 0.000011591444996884091,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / Forward",
            "value": 0.00001225108639919199,
            "unit": "s"
          },
          {
            "name": "Concat / JaX   / cpu / BothRev",
            "value": 0.000012837246304843575,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / PreRev",
            "value": 0.00001287966800155118,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / PostRev",
            "value": 0.000012816664495039733,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / BothRev",
            "value": 0.000012695630302187055,
            "unit": "s"
          },
          {
            "name": "Concat / JaX   / gpu / Primal",
            "value": 0.00008259975470136851,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / gpu / Primal",
            "value": 0.00008135784770129249,
            "unit": "s"
          },
          {
            "name": "Concat / JaX   / gpu / Forward",
            "value": 0.0001075893515022,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / gpu / Forward",
            "value": 0.0001046467869018,
            "unit": "s"
          },
          {
            "name": "Concat / JaX   / gpu / BothRev",
            "value": 0.0001001385945011,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / gpu / PreRev",
            "value": 0.0001030637247953,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / gpu / PostRev",
            "value": 0.0000995751631038729,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / gpu / BothRev",
            "value": 0.0001012936784012,
            "unit": "s"
          },
          {
            "name": "Concat / JaX   / cpu / Primal",
            "value": 0.000003755692900449504,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / Primal",
            "value": 0.000003843753899855074,
            "unit": "s"
          },
          {
            "name": "Concat / JaX   / cpu / Forward",
            "value": 0.000005732786899898201,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / Forward",
            "value": 0.000005994703999022022,
            "unit": "s"
          },
          {
            "name": "Concat / JaX   / cpu / BothRev",
            "value": 0.0000059897618994000365,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / PreRev",
            "value": 0.000006083480900269933,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / PostRev",
            "value": 0.000006175968900788575,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / BothRev",
            "value": 0.000006210520899912808,
            "unit": "s"
          },
          {
            "name": "Concat / JaX   / tpu / Primal",
            "value": 0.0001317709156995,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / tpu / Primal",
            "value": 0.0001337837167011,
            "unit": "s"
          },
          {
            "name": "Concat / JaX   / tpu / Forward",
            "value": 0.0002053089109991,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / tpu / Forward",
            "value": 0.0001945257642,
            "unit": "s"
          },
          {
            "name": "Concat / JaX   / tpu / BothRev",
            "value": 0.0001910753931995,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / tpu / PreRev",
            "value": 0.0001980634931998,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / tpu / PostRev",
            "value": 0.0001947976941999,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / tpu / BothRev",
            "value": 0.0002054399929998,
            "unit": "s"
          },
          {
            "name": "Concat / JaX   / cpu / Primal",
            "value": 0.000006576677799967001,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / Primal",
            "value": 0.000006502169700070226,
            "unit": "s"
          },
          {
            "name": "Concat / JaX   / cpu / Forward",
            "value": 0.000010230181999941124,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / Forward",
            "value": 0.000009738775000005262,
            "unit": "s"
          },
          {
            "name": "Concat / JaX   / cpu / BothRev",
            "value": 0.000010141965000002527,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / PreRev",
            "value": 0.00001078243450001537,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / PostRev",
            "value": 0.00001049496759997055,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / BothRev",
            "value": 0.000010155400199982976,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaX   / cpu / Primal",
            "value": 0.000009816400006457117,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / cpu / Primal",
            "value": 0.000009114899967244128,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaX   / cpu / Forward",
            "value": 0.000009483400026510937,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / cpu / Forward",
            "value": 0.000010349700005463092,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaX   / cpu / Primal",
            "value": 0.000013172102626413109,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / cpu / Primal",
            "value": 0.000012529600644484162,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaX   / cpu / Forward",
            "value": 0.000015766697470098733,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / cpu / Forward",
            "value": 0.00001800130121409893,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaX   / gpu / Primal",
            "value": 0.0001369419973343,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / gpu / Primal",
            "value": 0.0001201301056426,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaX   / gpu / Forward",
            "value": 0.000147876801202,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / gpu / Forward",
            "value": 0.0001477056008297,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaX   / cpu / Primal",
            "value": 0.000006335000216495246,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / cpu / Primal",
            "value": 0.000007243000436574221,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaX   / cpu / Forward",
            "value": 0.000007470000127796084,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / cpu / Forward",
            "value": 0.000008473001071251928,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaX   / tpu / Primal",
            "value": 0.0001644119998672,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / tpu / Primal",
            "value": 0.0001561040000524,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaX   / tpu / Forward",
            "value": 0.0002509390003979,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / tpu / Forward",
            "value": 0.000243151000177,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaX   / cpu / Primal",
            "value": 0.000010855300024559255,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / cpu / Primal",
            "value": 0.00001022990009005298,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaX   / cpu / Forward",
            "value": 0.000013399099952948745,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / cpu / Forward",
            "value": 0.000016705300004105085,
            "unit": "s"
          },
          {
            "name": "GenDot / JaX   / cpu / Primal",
            "value": 0.000004288850000011734,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / Primal",
            "value": 0.0000042865029000040525,
            "unit": "s"
          },
          {
            "name": "GenDot / JaX   / cpu / Forward",
            "value": 0.000006621694300019953,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / Forward",
            "value": 0.000007314420300008351,
            "unit": "s"
          },
          {
            "name": "GenDot / JaX   / cpu / BothRev",
            "value": 0.0000065707481000117696,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / PreRev",
            "value": 0.00000726144509999358,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / PostRev",
            "value": 0.000006716742600019643,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / BothRev",
            "value": 0.000007488080499979333,
            "unit": "s"
          },
          {
            "name": "GenDot / JaX   / cpu / Primal",
            "value": 0.000007950637396425008,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / Primal",
            "value": 0.000008536692702909931,
            "unit": "s"
          },
          {
            "name": "GenDot / JaX   / cpu / Forward",
            "value": 0.000012189144000876696,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / Forward",
            "value": 0.000014298559702001511,
            "unit": "s"
          },
          {
            "name": "GenDot / JaX   / cpu / BothRev",
            "value": 0.000011152991000562906,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / PreRev",
            "value": 0.000014100016094744206,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / PostRev",
            "value": 0.00001205240610288456,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / BothRev",
            "value": 0.000013991465995786713,
            "unit": "s"
          },
          {
            "name": "GenDot / JaX   / gpu / Primal",
            "value": 0.00007430664910352789,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / gpu / Primal",
            "value": 0.00007520563450525515,
            "unit": "s"
          },
          {
            "name": "GenDot / JaX   / gpu / Forward",
            "value": 0.0001029469997039,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / gpu / Forward",
            "value": 0.0001069808458036,
            "unit": "s"
          },
          {
            "name": "GenDot / JaX   / gpu / BothRev",
            "value": 0.0001038049829017,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / gpu / PreRev",
            "value": 0.0001091482757998,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / gpu / PostRev",
            "value": 0.000127933362202,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / gpu / BothRev",
            "value": 0.0001122382506029,
            "unit": "s"
          },
          {
            "name": "GenDot / JaX   / cpu / Primal",
            "value": 0.000003805467899655923,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / Primal",
            "value": 0.000006613330000254791,
            "unit": "s"
          },
          {
            "name": "GenDot / JaX   / cpu / Forward",
            "value": 0.000005057031899923459,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / Forward",
            "value": 0.000005506353899545502,
            "unit": "s"
          },
          {
            "name": "GenDot / JaX   / cpu / BothRev",
            "value": 0.000005081908899592236,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / PreRev",
            "value": 0.000005512855900451541,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / PostRev",
            "value": 0.0000052498370001558215,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / BothRev",
            "value": 0.00000628119689936284,
            "unit": "s"
          },
          {
            "name": "GenDot / JaX   / tpu / Primal",
            "value": 0.0001373556286998,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / tpu / Primal",
            "value": 0.0001348224976987,
            "unit": "s"
          },
          {
            "name": "GenDot / JaX   / tpu / Forward",
            "value": 0.0002237345068991,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / tpu / Forward",
            "value": 0.0002168323068995,
            "unit": "s"
          },
          {
            "name": "GenDot / JaX   / tpu / BothRev",
            "value": 0.0001916525421998,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / tpu / PreRev",
            "value": 0.0001953795982,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / tpu / PostRev",
            "value": 0.0001910519601995,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / tpu / BothRev",
            "value": 0.0001867825262001,
            "unit": "s"
          },
          {
            "name": "GenDot / JaX   / cpu / Primal",
            "value": 0.000006582874899959279,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / Primal",
            "value": 0.0000072101101000043855,
            "unit": "s"
          },
          {
            "name": "GenDot / JaX   / cpu / Forward",
            "value": 0.00000955548980000458,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / Forward",
            "value": 0.000012926316400080396,
            "unit": "s"
          },
          {
            "name": "GenDot / JaX   / cpu / BothRev",
            "value": 0.00001129051170000821,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / PreRev",
            "value": 0.000017040619299950778,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / PostRev",
            "value": 0.000009880929100017964,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / BothRev",
            "value": 0.000012292126800002734,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaX   / cpu / Primal",
            "value": 0.000005732394699998622,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / Primal",
            "value": 0.000005772555800012924,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaX   / cpu / Forward",
            "value": 0.000009550186900014523,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / Forward",
            "value": 0.000009573073399997156,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaX   / cpu / BothRev",
            "value": 0.000009214764099988314,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / PreRev",
            "value": 0.000009171916099967348,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / PostRev",
            "value": 0.000009329592099993531,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / BothRev",
            "value": 0.00000901915080003164,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaX   / cpu / Primal",
            "value": 0.00001079463839996606,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / Primal",
            "value": 0.000010011423303512855,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaX   / cpu / Forward",
            "value": 0.000015252258203690871,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / Forward",
            "value": 0.000015773060300853103,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaX   / cpu / BothRev",
            "value": 0.000014687184599461034,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / PreRev",
            "value": 0.000015367618994787336,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / PostRev",
            "value": 0.00001523477499722503,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / BothRev",
            "value": 0.000015625941002508625,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaX   / cpu / Primal",
            "value": 0.000004858407999563497,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / Primal",
            "value": 0.000004908778000390157,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaX   / cpu / Forward",
            "value": 0.000007539941900176928,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / Forward",
            "value": 0.000007942136899509933,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaX   / cpu / BothRev",
            "value": 0.000007292211998719722,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / PreRev",
            "value": 0.000007628533001116011,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / PostRev",
            "value": 0.000007596802999614738,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / BothRev",
            "value": 0.000007355804899998475,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaX   / tpu / Primal",
            "value": 0.00006505267839966109,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / tpu / Primal",
            "value": 0.00006713565249956446,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaX   / tpu / Forward",
            "value": 0.00009194541720062262,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / tpu / Forward",
            "value": 0.00009257244610053022,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaX   / tpu / BothRev",
            "value": 0.00009247848220111335,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / tpu / PreRev",
            "value": 0.00009161387210042447,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / tpu / PostRev",
            "value": 0.000091640951098816,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / tpu / BothRev",
            "value": 0.00009239996209944366,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaX   / cpu / Primal",
            "value": 0.00000868918220003252,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / Primal",
            "value": 0.000008295453699975041,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaX   / cpu / Forward",
            "value": 0.00001658096299997851,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / Forward",
            "value": 0.000013439955899957568,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaX   / cpu / BothRev",
            "value": 0.00001635294289999365,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / PreRev",
            "value": 0.000012976897999942594,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / PostRev",
            "value": 0.000013077046499984136,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / BothRev",
            "value": 0.000013247723800031965,
            "unit": "s"
          },
          {
            "name": "llama / JaXPipe / cpu / Primal",
            "value": 0.0009116312200012,
            "unit": "s"
          },
          {
            "name": "llama / JaX   / cpu / Primal",
            "value": 0.0009058034599956,
            "unit": "s"
          },
          {
            "name": "llama / HLOOpt / cpu / Primal",
            "value": 0.000968850879999,
            "unit": "s"
          },
          {
            "name": "llama / PartOpt / cpu / Primal",
            "value": 0.0009475656199992,
            "unit": "s"
          },
          {
            "name": "llama / DefOpt / cpu / Primal",
            "value": 0.0009603537100019,
            "unit": "s"
          },
          {
            "name": "llama / IPartOpt / cpu / Primal",
            "value": 0.000884290910003,
            "unit": "s"
          },
          {
            "name": "llama / IDefOpt / cpu / Primal",
            "value": 0.0009589604199982,
            "unit": "s"
          },
          {
            "name": "llama / JaXPipe / cpu / Forward",
            "value": 0.0024857862199996,
            "unit": "s"
          },
          {
            "name": "llama / JaX   / cpu / Forward",
            "value": 0.0026243811299991,
            "unit": "s"
          },
          {
            "name": "llama / HLOOpt / cpu / Forward",
            "value": 0.0024953370999992,
            "unit": "s"
          },
          {
            "name": "llama / PartOpt / cpu / Forward",
            "value": 0.0024911917000008,
            "unit": "s"
          },
          {
            "name": "llama / DefOpt / cpu / Forward",
            "value": 0.0027133156799982,
            "unit": "s"
          },
          {
            "name": "llama / IPartOpt / cpu / Forward",
            "value": 0.0027774407599963,
            "unit": "s"
          },
          {
            "name": "llama / IDefOpt / cpu / Forward",
            "value": 0.0024464855899987,
            "unit": "s"
          },
          {
            "name": "llama / JaXPipe / cpu / PreRev",
            "value": 0.0024722599599999,
            "unit": "s"
          },
          {
            "name": "llama / JaXPipe / cpu / PostRev",
            "value": 0.0022549781499992,
            "unit": "s"
          },
          {
            "name": "llama / JaXPipe / cpu / BothRev",
            "value": 0.0023902485999997,
            "unit": "s"
          },
          {
            "name": "llama / JaX   / cpu / BothRev",
            "value": 0.0024555513800032,
            "unit": "s"
          },
          {
            "name": "llama / HLOOpt / cpu / PreRev",
            "value": 0.002488592489999,
            "unit": "s"
          },
          {
            "name": "llama / HLOOpt / cpu / PostRev",
            "value": 0.0022813078599983,
            "unit": "s"
          },
          {
            "name": "llama / HLOOpt / cpu / BothRev",
            "value": 0.0023000072800005,
            "unit": "s"
          },
          {
            "name": "llama / PartOpt / cpu / PreRev",
            "value": 0.0023037466599998,
            "unit": "s"
          },
          {
            "name": "llama / PartOpt / cpu / PostRev",
            "value": 0.0025805363599965,
            "unit": "s"
          },
          {
            "name": "llama / PartOpt / cpu / BothRev",
            "value": 0.002506285319996,
            "unit": "s"
          },
          {
            "name": "llama / DefOpt / cpu / PreRev",
            "value": 0.0023831332500003,
            "unit": "s"
          },
          {
            "name": "llama / DefOpt / cpu / PostRev",
            "value": 0.0024076699599982,
            "unit": "s"
          },
          {
            "name": "llama / DefOpt / cpu / BothRev",
            "value": 0.0026014555099982,
            "unit": "s"
          },
          {
            "name": "llama / IPartOpt / cpu / PreRev",
            "value": 0.0023078202599981,
            "unit": "s"
          },
          {
            "name": "llama / IPartOpt / cpu / PostRev",
            "value": 0.0023213790699992,
            "unit": "s"
          },
          {
            "name": "llama / IPartOpt / cpu / BothRev",
            "value": 0.002359384209999,
            "unit": "s"
          },
          {
            "name": "llama / IDefOpt / cpu / PreRev",
            "value": 0.0024560897200035,
            "unit": "s"
          },
          {
            "name": "llama / IDefOpt / cpu / PostRev",
            "value": 0.0024125330700007,
            "unit": "s"
          },
          {
            "name": "llama / IDefOpt / cpu / BothRev",
            "value": 0.0023759166399986,
            "unit": "s"
          },
          {
            "name": "llama / JaXPipe / gpu / Primal",
            "value": 0.000417757321964,
            "unit": "s"
          },
          {
            "name": "llama / JaX   / gpu / Primal",
            "value": 0.0004251578280236,
            "unit": "s"
          },
          {
            "name": "llama / HLOOpt / gpu / Primal",
            "value": 0.0004385926000541,
            "unit": "s"
          },
          {
            "name": "llama / PartOpt / gpu / Primal",
            "value": 0.0004199412680463,
            "unit": "s"
          },
          {
            "name": "llama / DefOpt / gpu / Primal",
            "value": 0.0004280312219634,
            "unit": "s"
          },
          {
            "name": "llama / IPartOpt / gpu / Primal",
            "value": 0.0004214352220296,
            "unit": "s"
          },
          {
            "name": "llama / IDefOpt / gpu / Primal",
            "value": 0.0004484498379752,
            "unit": "s"
          },
          {
            "name": "llama / JaXPipe / gpu / Forward",
            "value": 0.0007354826759546,
            "unit": "s"
          },
          {
            "name": "llama / JaX   / gpu / Forward",
            "value": 0.0006997816599905,
            "unit": "s"
          },
          {
            "name": "llama / HLOOpt / gpu / Forward",
            "value": 0.0007096426259959,
            "unit": "s"
          },
          {
            "name": "llama / PartOpt / gpu / Forward",
            "value": 0.0007248653559945,
            "unit": "s"
          },
          {
            "name": "llama / DefOpt / gpu / Forward",
            "value": 0.0007485089600086,
            "unit": "s"
          },
          {
            "name": "llama / IPartOpt / gpu / Forward",
            "value": 0.0007184074019314,
            "unit": "s"
          },
          {
            "name": "llama / IDefOpt / gpu / Forward",
            "value": 0.0007240810838993,
            "unit": "s"
          },
          {
            "name": "llama / JaXPipe / gpu / PreRev",
            "value": 0.000779639259912,
            "unit": "s"
          },
          {
            "name": "llama / JaXPipe / gpu / PostRev",
            "value": 0.0007951593060279,
            "unit": "s"
          },
          {
            "name": "llama / JaXPipe / gpu / BothRev",
            "value": 0.0008328814220149,
            "unit": "s"
          },
          {
            "name": "llama / JaX   / gpu / BothRev",
            "value": 0.0008177384479204,
            "unit": "s"
          },
          {
            "name": "llama / HLOOpt / gpu / PreRev",
            "value": 0.0007814622500445,
            "unit": "s"
          },
          {
            "name": "llama / HLOOpt / gpu / PostRev",
            "value": 0.0007863578300457,
            "unit": "s"
          },
          {
            "name": "llama / HLOOpt / gpu / BothRev",
            "value": 0.0008322308399947,
            "unit": "s"
          },
          {
            "name": "llama / PartOpt / gpu / PreRev",
            "value": 0.0007744775040773,
            "unit": "s"
          },
          {
            "name": "llama / PartOpt / gpu / PostRev",
            "value": 0.0008087961960118,
            "unit": "s"
          },
          {
            "name": "llama / PartOpt / gpu / BothRev",
            "value": 0.0008015714079374,
            "unit": "s"
          },
          {
            "name": "llama / DefOpt / gpu / PreRev",
            "value": 0.0008217210220173,
            "unit": "s"
          },
          {
            "name": "llama / DefOpt / gpu / PostRev",
            "value": 0.0007529904500115,
            "unit": "s"
          },
          {
            "name": "llama / DefOpt / gpu / BothRev",
            "value": 0.000813793304027,
            "unit": "s"
          },
          {
            "name": "llama / IPartOpt / gpu / PreRev",
            "value": 0.0008311075459932,
            "unit": "s"
          },
          {
            "name": "llama / IPartOpt / gpu / PostRev",
            "value": 0.0007601056279381,
            "unit": "s"
          },
          {
            "name": "llama / IPartOpt / gpu / BothRev",
            "value": 0.000777988045942,
            "unit": "s"
          },
          {
            "name": "llama / IDefOpt / gpu / PreRev",
            "value": 0.0008314784159883,
            "unit": "s"
          },
          {
            "name": "llama / IDefOpt / gpu / PostRev",
            "value": 0.0007971079200506,
            "unit": "s"
          },
          {
            "name": "llama / IDefOpt / gpu / BothRev",
            "value": 0.0007642968359868,
            "unit": "s"
          },
          {
            "name": "llama / JaXPipe / tpu / Primal",
            "value": 0.0003803525159892,
            "unit": "s"
          },
          {
            "name": "llama / JaX   / tpu / Primal",
            "value": 0.0003726551159925,
            "unit": "s"
          },
          {
            "name": "llama / HLOOpt / tpu / Primal",
            "value": 0.0003678053960029,
            "unit": "s"
          },
          {
            "name": "llama / PartOpt / tpu / Primal",
            "value": 0.0003819129579933,
            "unit": "s"
          },
          {
            "name": "llama / DefOpt / tpu / Primal",
            "value": 0.0003543725959898,
            "unit": "s"
          },
          {
            "name": "llama / IPartOpt / tpu / Primal",
            "value": 0.0003733241959998,
            "unit": "s"
          },
          {
            "name": "llama / IDefOpt / tpu / Primal",
            "value": 0.0003619711959909,
            "unit": "s"
          },
          {
            "name": "llama / JaXPipe / tpu / Forward",
            "value": 0.0005743051759782,
            "unit": "s"
          },
          {
            "name": "llama / JaX   / tpu / Forward",
            "value": 0.0006943731119972,
            "unit": "s"
          },
          {
            "name": "llama / HLOOpt / tpu / Forward",
            "value": 0.00056511683602,
            "unit": "s"
          },
          {
            "name": "llama / PartOpt / tpu / Forward",
            "value": 0.000559914974001,
            "unit": "s"
          },
          {
            "name": "llama / DefOpt / tpu / Forward",
            "value": 0.0005891558739822,
            "unit": "s"
          },
          {
            "name": "llama / IPartOpt / tpu / Forward",
            "value": 0.0005706650360079,
            "unit": "s"
          },
          {
            "name": "llama / IDefOpt / tpu / Forward",
            "value": 0.0005842641739873,
            "unit": "s"
          },
          {
            "name": "llama / JaXPipe / tpu / PreRev",
            "value": 0.000407173576008,
            "unit": "s"
          },
          {
            "name": "llama / JaXPipe / tpu / PostRev",
            "value": 0.0003575073360116,
            "unit": "s"
          },
          {
            "name": "llama / JaXPipe / tpu / BothRev",
            "value": 0.0003919302760041,
            "unit": "s"
          },
          {
            "name": "llama / JaX   / tpu / BothRev",
            "value": 0.0003575019779964,
            "unit": "s"
          },
          {
            "name": "llama / HLOOpt / tpu / PreRev",
            "value": 0.0003915915960096,
            "unit": "s"
          },
          {
            "name": "llama / HLOOpt / tpu / PostRev",
            "value": 0.000390343616018,
            "unit": "s"
          },
          {
            "name": "llama / HLOOpt / tpu / BothRev",
            "value": 0.000391486537992,
            "unit": "s"
          },
          {
            "name": "llama / PartOpt / tpu / PreRev",
            "value": 0.0003916069959814,
            "unit": "s"
          },
          {
            "name": "llama / PartOpt / tpu / PostRev",
            "value": 0.0003713835180096,
            "unit": "s"
          },
          {
            "name": "llama / PartOpt / tpu / BothRev",
            "value": 0.0003913653360214,
            "unit": "s"
          },
          {
            "name": "llama / DefOpt / tpu / PreRev",
            "value": 0.0003916571960144,
            "unit": "s"
          },
          {
            "name": "llama / DefOpt / tpu / PostRev",
            "value": 0.000383015956002,
            "unit": "s"
          },
          {
            "name": "llama / DefOpt / tpu / BothRev",
            "value": 0.0003916553159942,
            "unit": "s"
          },
          {
            "name": "llama / IPartOpt / tpu / PreRev",
            "value": 0.0003917180360003,
            "unit": "s"
          },
          {
            "name": "llama / IPartOpt / tpu / PostRev",
            "value": 0.0003714232160127,
            "unit": "s"
          },
          {
            "name": "llama / IPartOpt / tpu / BothRev",
            "value": 0.0003914158160041,
            "unit": "s"
          },
          {
            "name": "llama / IDefOpt / tpu / PreRev",
            "value": 0.0003916971959988,
            "unit": "s"
          },
          {
            "name": "llama / IDefOpt / tpu / PostRev",
            "value": 0.0003991222380136,
            "unit": "s"
          },
          {
            "name": "llama / IDefOpt / tpu / BothRev",
            "value": 0.0003915888380142,
            "unit": "s"
          },
          {
            "name": "llama / JaXPipe / cpu / Primal",
            "value": 0.0018371241500062,
            "unit": "s"
          },
          {
            "name": "llama / JaX   / cpu / Primal",
            "value": 0.0018266817099993,
            "unit": "s"
          },
          {
            "name": "llama / HLOOpt / cpu / Primal",
            "value": 0.0019615575300031,
            "unit": "s"
          },
          {
            "name": "llama / PartOpt / cpu / Primal",
            "value": 0.0018836050299978,
            "unit": "s"
          },
          {
            "name": "llama / DefOpt / cpu / Primal",
            "value": 0.0019738237799992,
            "unit": "s"
          },
          {
            "name": "llama / IPartOpt / cpu / Primal",
            "value": 0.0018820348000008,
            "unit": "s"
          },
          {
            "name": "llama / IDefOpt / cpu / Primal",
            "value": 0.001901043910002,
            "unit": "s"
          },
          {
            "name": "llama / JaXPipe / cpu / Forward",
            "value": 0.0060022352899977,
            "unit": "s"
          },
          {
            "name": "llama / JaX   / cpu / Forward",
            "value": 0.0063458719200025,
            "unit": "s"
          },
          {
            "name": "llama / HLOOpt / cpu / Forward",
            "value": 0.0058494116099973,
            "unit": "s"
          },
          {
            "name": "llama / PartOpt / cpu / Forward",
            "value": 0.0059152088999962,
            "unit": "s"
          },
          {
            "name": "llama / DefOpt / cpu / Forward",
            "value": 0.0059396623300017,
            "unit": "s"
          },
          {
            "name": "llama / IPartOpt / cpu / Forward",
            "value": 0.0059749387800002,
            "unit": "s"
          },
          {
            "name": "llama / IDefOpt / cpu / Forward",
            "value": 0.0058763036799973,
            "unit": "s"
          },
          {
            "name": "llama / JaXPipe / cpu / PreRev",
            "value": 0.0056804125899998,
            "unit": "s"
          },
          {
            "name": "llama / JaXPipe / cpu / PostRev",
            "value": 0.0057985678699969,
            "unit": "s"
          },
          {
            "name": "llama / JaXPipe / cpu / BothRev",
            "value": 0.0054952634500023,
            "unit": "s"
          },
          {
            "name": "llama / JaX   / cpu / BothRev",
            "value": 0.0055707272000017,
            "unit": "s"
          },
          {
            "name": "llama / HLOOpt / cpu / PreRev",
            "value": 0.0059111222299998,
            "unit": "s"
          },
          {
            "name": "llama / HLOOpt / cpu / PostRev",
            "value": 0.00532526297,
            "unit": "s"
          },
          {
            "name": "llama / HLOOpt / cpu / BothRev",
            "value": 0.0061461415099984,
            "unit": "s"
          },
          {
            "name": "llama / PartOpt / cpu / PreRev",
            "value": 0.0056244538999999,
            "unit": "s"
          },
          {
            "name": "llama / PartOpt / cpu / PostRev",
            "value": 0.0058314764399983,
            "unit": "s"
          },
          {
            "name": "llama / PartOpt / cpu / BothRev",
            "value": 0.005548712209993,
            "unit": "s"
          },
          {
            "name": "llama / DefOpt / cpu / PreRev",
            "value": 0.0054735375299969,
            "unit": "s"
          },
          {
            "name": "llama / DefOpt / cpu / PostRev",
            "value": 0.0058514647900028,
            "unit": "s"
          },
          {
            "name": "llama / DefOpt / cpu / BothRev",
            "value": 0.0054523912500008,
            "unit": "s"
          },
          {
            "name": "llama / IPartOpt / cpu / PreRev",
            "value": 0.0060553823600002,
            "unit": "s"
          },
          {
            "name": "llama / IPartOpt / cpu / PostRev",
            "value": 0.0054994300700036,
            "unit": "s"
          },
          {
            "name": "llama / IPartOpt / cpu / BothRev",
            "value": 0.0060103528200033,
            "unit": "s"
          },
          {
            "name": "llama / IDefOpt / cpu / PreRev",
            "value": 0.005411943830004,
            "unit": "s"
          },
          {
            "name": "llama / IDefOpt / cpu / PostRev",
            "value": 0.006128978429997,
            "unit": "s"
          },
          {
            "name": "llama / IDefOpt / cpu / BothRev",
            "value": 0.0054914348300007,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004887749899990013,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.00000467760459996498,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000009174828097457069,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000009206091001396998,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / gpu / Primal",
            "value": 0.00009292014389648102,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / gpu / Primal",
            "value": 0.00007767383519676514,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004368892998900265,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.0000042255238993675445,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001317159217011,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001381257767003,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000011145293500067057,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000007547125099972618,
            "unit": "s"
          },
          {
            "name": "slicing / JaX   / cpu / Primal",
            "value": 0.000004085345299972687,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / Primal",
            "value": 0.000003823841500025082,
            "unit": "s"
          },
          {
            "name": "slicing / JaX   / cpu / Forward",
            "value": 0.000005850039500001003,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / Forward",
            "value": 0.000005945568800007095,
            "unit": "s"
          },
          {
            "name": "slicing / JaX   / cpu / BothRev",
            "value": 0.000006434162699997614,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / PreRev",
            "value": 0.000006496394099985991,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / PostRev",
            "value": 0.000006433167500017589,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / BothRev",
            "value": 0.000006594480100011424,
            "unit": "s"
          },
          {
            "name": "slicing / JaX   / cpu / Primal",
            "value": 0.000006940010003745556,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / Primal",
            "value": 0.000006985396397067234,
            "unit": "s"
          },
          {
            "name": "slicing / JaX   / cpu / Forward",
            "value": 0.0000102769911987707,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / Forward",
            "value": 0.000010831168002914637,
            "unit": "s"
          },
          {
            "name": "slicing / JaX   / cpu / BothRev",
            "value": 0.000010980097600258889,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / PreRev",
            "value": 0.00001116360820014961,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / PostRev",
            "value": 0.000012619239004561678,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / BothRev",
            "value": 0.000011091983801452442,
            "unit": "s"
          },
          {
            "name": "slicing / JaX   / gpu / Primal",
            "value": 0.0000737120355013758,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / gpu / Primal",
            "value": 0.0000733425796031952,
            "unit": "s"
          },
          {
            "name": "slicing / JaX   / gpu / Forward",
            "value": 0.0001008482995966,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / gpu / Forward",
            "value": 0.0001013499089051,
            "unit": "s"
          },
          {
            "name": "slicing / JaX   / gpu / BothRev",
            "value": 0.0001019944755011,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / gpu / PreRev",
            "value": 0.0001035950373974,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / gpu / PostRev",
            "value": 0.0001019826617965,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / gpu / BothRev",
            "value": 0.0001030729671008,
            "unit": "s"
          },
          {
            "name": "slicing / JaX   / cpu / Primal",
            "value": 0.000003692133000004105,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / Primal",
            "value": 0.000003699853000580333,
            "unit": "s"
          },
          {
            "name": "slicing / JaX   / cpu / Forward",
            "value": 0.000004937990900361911,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / Forward",
            "value": 0.000005261844900087454,
            "unit": "s"
          },
          {
            "name": "slicing / JaX   / cpu / BothRev",
            "value": 0.000005647074000444263,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / PreRev",
            "value": 0.000007099081900378223,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / PostRev",
            "value": 0.0000048148799003683964,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / BothRev",
            "value": 0.000004985927999950946,
            "unit": "s"
          },
          {
            "name": "slicing / JaX   / tpu / Primal",
            "value": 0.0001346968566998,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / tpu / Primal",
            "value": 0.0001292906017988,
            "unit": "s"
          },
          {
            "name": "slicing / JaX   / tpu / Forward",
            "value": 0.0001841045982,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / tpu / Forward",
            "value": 0.0001760742972997,
            "unit": "s"
          },
          {
            "name": "slicing / JaX   / tpu / BothRev",
            "value": 0.0001846179023006,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / tpu / PreRev",
            "value": 0.0001877291491997,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / tpu / PostRev",
            "value": 0.0001874575731999,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / tpu / BothRev",
            "value": 0.0001924586291992,
            "unit": "s"
          },
          {
            "name": "slicing / JaX   / cpu / Primal",
            "value": 0.00000580509430001257,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / Primal",
            "value": 0.000006307002100038517,
            "unit": "s"
          },
          {
            "name": "slicing / JaX   / cpu / Forward",
            "value": 0.000009993756299991218,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / Forward",
            "value": 0.000009097062700038805,
            "unit": "s"
          },
          {
            "name": "slicing / JaX   / cpu / BothRev",
            "value": 0.00000921685430002981,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / PreRev",
            "value": 0.000009246724799959338,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / PostRev",
            "value": 0.000009716418399966642,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / BothRev",
            "value": 0.000009757882000030804,
            "unit": "s"
          },
          {
            "name": "sum    / JaX   / cpu / Primal",
            "value": 0.0000054312726999796725,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / Primal",
            "value": 0.000005425862500032963,
            "unit": "s"
          },
          {
            "name": "sum    / JaX   / cpu / Forward",
            "value": 0.000008623558599992976,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / Forward",
            "value": 0.00000844751439999527,
            "unit": "s"
          },
          {
            "name": "sum    / JaX   / cpu / BothRev",
            "value": 0.000007649927600004958,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / PreRev",
            "value": 0.000007358836799994605,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / PostRev",
            "value": 0.0000073886578000383455,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / BothRev",
            "value": 0.000007380239799977062,
            "unit": "s"
          },
          {
            "name": "sum    / JaX   / cpu / Primal",
            "value": 0.000010523381998063997,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / Primal",
            "value": 0.000009914217301411554,
            "unit": "s"
          },
          {
            "name": "sum    / JaX   / cpu / Forward",
            "value": 0.000015409524802817032,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / Forward",
            "value": 0.000014691223099362103,
            "unit": "s"
          },
          {
            "name": "sum    / JaX   / cpu / BothRev",
            "value": 0.000014134842698695136,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / PreRev",
            "value": 0.000013574216497363523,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / PostRev",
            "value": 0.000013611244794446977,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / BothRev",
            "value": 0.000014081061200704426,
            "unit": "s"
          },
          {
            "name": "sum    / JaX   / gpu / Primal",
            "value": 0.00007195927860448137,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / gpu / Primal",
            "value": 0.00007215786399901844,
            "unit": "s"
          },
          {
            "name": "sum    / JaX   / gpu / Forward",
            "value": 0.00009803225649520756,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / gpu / Forward",
            "value": 0.0001004996315983,
            "unit": "s"
          },
          {
            "name": "sum    / JaX   / gpu / BothRev",
            "value": 0.0001218450932006,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / gpu / PreRev",
            "value": 0.0001052658975997,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / gpu / PostRev",
            "value": 0.0001036391525005,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / gpu / BothRev",
            "value": 0.0001033550191961,
            "unit": "s"
          },
          {
            "name": "sum    / JaX   / cpu / Primal",
            "value": 0.000004442788001324516,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / Primal",
            "value": 0.000004478355898754671,
            "unit": "s"
          },
          {
            "name": "sum    / JaX   / cpu / Forward",
            "value": 0.000006930086998909246,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / Forward",
            "value": 0.000007227165000222157,
            "unit": "s"
          },
          {
            "name": "sum    / JaX   / cpu / BothRev",
            "value": 0.000005897471000207588,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / PreRev",
            "value": 0.0000060217359990929256,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / PostRev",
            "value": 0.000006013198899745475,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / BothRev",
            "value": 0.000006082193000474945,
            "unit": "s"
          },
          {
            "name": "sum    / JaX   / tpu / Primal",
            "value": 0.0001372680087006,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / tpu / Primal",
            "value": 0.0001305529267003,
            "unit": "s"
          },
          {
            "name": "sum    / JaX   / tpu / Forward",
            "value": 0.0001787058782996,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / tpu / Forward",
            "value": 0.0001883804541997,
            "unit": "s"
          },
          {
            "name": "sum    / JaX   / tpu / BothRev",
            "value": 0.0001849994573,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / tpu / PreRev",
            "value": 0.0001804936022002,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / tpu / PostRev",
            "value": 0.0001887903281996,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / tpu / BothRev",
            "value": 0.0001882512091993,
            "unit": "s"
          },
          {
            "name": "sum    / JaX   / cpu / Primal",
            "value": 0.000009109806299966297,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / Primal",
            "value": 0.000009697455700006685,
            "unit": "s"
          },
          {
            "name": "sum    / JaX   / cpu / Forward",
            "value": 0.000013370336200023303,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / Forward",
            "value": 0.000013377153999954317,
            "unit": "s"
          },
          {
            "name": "sum    / JaX   / cpu / BothRev",
            "value": 0.00001288204680004128,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / PreRev",
            "value": 0.0000124718166999628,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / PostRev",
            "value": 0.000012528970499988644,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / BothRev",
            "value": 0.000012409789899993483,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / JaXPipe / gpu / Primal",
            "value": 0.001019851595629,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / JaX   / gpu / Primal",
            "value": 0.0010203889978583,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / HLOOpt / gpu / Primal",
            "value": 0.0009590202011168,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / PartOpt / gpu / Primal",
            "value": 0.0009513808006886,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / DefOpt / gpu / Primal",
            "value": 0.0006838892993982,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / IPartOpt / gpu / Primal",
            "value": 0.000952572398819,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / IDefOpt / gpu / Primal",
            "value": 0.0007053115987218,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / JaX   / gpu / Forward",
            "value": 0.0012643401976674,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / JaXPipe / gpu / PostRev",
            "value": 0.0040332959964871,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / JaX   / gpu / BothRev",
            "value": 0.0040380138962063,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / HLOOpt / gpu / PostRev",
            "value": 0.0040394214971456,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / PartOpt / gpu / PostRev",
            "value": 0.0041371223982423,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / DefOpt / gpu / PostRev",
            "value": 0.0022239979996811,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / IPartOpt / gpu / PostRev",
            "value": 0.0041154420003294,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / IDefOpt / gpu / PostRev",
            "value": 0.0025566152995452,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / JaXPipe / tpu / Primal",
            "value": 0.0000881089988979511,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / JaX   / tpu / Primal",
            "value": 0.00009187899995595216,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / HLOOpt / tpu / Primal",
            "value": 0.000094242999330163,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / PartOpt / tpu / Primal",
            "value": 0.00009494499972788616,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / DefOpt / tpu / Primal",
            "value": 0.00009548100060783329,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / IPartOpt / tpu / Primal",
            "value": 0.00009280099911848084,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / IDefOpt / tpu / Primal",
            "value": 0.00009143799979938194,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / JaX   / tpu / Forward",
            "value": 0.0001722250002785,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / JaXPipe / tpu / PostRev",
            "value": 0.0001794840005459,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / JaX   / tpu / BothRev",
            "value": 0.0001874980007414,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / HLOOpt / tpu / PostRev",
            "value": 0.0002003660003538,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / PartOpt / tpu / PostRev",
            "value": 0.0002072010000119,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / DefOpt / tpu / PostRev",
            "value": 0.0001888529994175,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / IPartOpt / tpu / PostRev",
            "value": 0.0001986460003536,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / IDefOpt / tpu / PostRev",
            "value": 0.0001909160011564,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / JaXPipe / cpu / Primal",
            "value": 0.00007493330003853771,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / JaX   / cpu / Primal",
            "value": 0.000058265500047127714,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / HLOOpt / cpu / Primal",
            "value": 0.00007486969998353743,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / PartOpt / cpu / Primal",
            "value": 0.00007604729999002302,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / DefOpt / cpu / Primal",
            "value": 0.00006741970000803121,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / IPartOpt / cpu / Primal",
            "value": 0.00007528819996878155,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / IDefOpt / cpu / Primal",
            "value": 0.00007447959997080034,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / JaX   / cpu / Forward",
            "value": 0.0001214075999996,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / JaXPipe / cpu / PostRev",
            "value": 0.0001421222000317,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / JaX   / cpu / BothRev",
            "value": 0.0001712215999759,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / HLOOpt / cpu / PostRev",
            "value": 0.0001531189000161,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / PartOpt / cpu / PostRev",
            "value": 0.0001759009000124,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / DefOpt / cpu / PostRev",
            "value": 0.0001393437999468,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / IPartOpt / cpu / PostRev",
            "value": 0.0001662253999711,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / IDefOpt / cpu / PostRev",
            "value": 0.0001519256999927,
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
            "name": "William Moses",
            "username": "wsmoses"
          },
          "distinct": true,
          "id": "5807b16a4ea942253c01d1148240a90f33bc72ed",
          "message": "Fix limit versin of concat2dusslice",
          "timestamp": "2025-12-13T16:47:15-06:00",
          "tree_id": "f81e6f163a64ab80682053dcb211fc458db98ec6",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/5807b16a4ea942253c01d1148240a90f33bc72ed"
        },
        "date": 1765675351080,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "actmtch / JaX   / cpu / Primal",
            "value": 0.000004077052000002368,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / Primal",
            "value": 0.000003983729900028266,
            "unit": "s"
          },
          {
            "name": "actmtch / JaX   / cpu / Forward",
            "value": 0.000006333651200020541,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / Forward",
            "value": 0.000007304315300007147,
            "unit": "s"
          },
          {
            "name": "actmtch / JaX   / cpu / BothRev",
            "value": 0.000006389050899997528,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / PreRev",
            "value": 0.000007454925799993362,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / PostRev",
            "value": 0.00000632967709998411,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / BothRev",
            "value": 0.00000736418199999207,
            "unit": "s"
          },
          {
            "name": "actmtch / JaX   / cpu / Primal",
            "value": 0.0000070230833000096025,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / Primal",
            "value": 0.000007406233300025633,
            "unit": "s"
          },
          {
            "name": "actmtch / JaX   / cpu / Forward",
            "value": 0.0000105559653000455,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / Forward",
            "value": 0.000013767179200112876,
            "unit": "s"
          },
          {
            "name": "actmtch / JaX   / cpu / BothRev",
            "value": 0.000010413583799891056,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / PreRev",
            "value": 0.000013618927300012727,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / PostRev",
            "value": 0.000011015690400017777,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / BothRev",
            "value": 0.000014105826299964974,
            "unit": "s"
          },
          {
            "name": "actmtch / JaX   / gpu / Primal",
            "value": 0.00008572420809996401,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / gpu / Primal",
            "value": 0.00007496612270006154,
            "unit": "s"
          },
          {
            "name": "actmtch / JaX   / gpu / Forward",
            "value": 0.0001092814691999,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / gpu / Forward",
            "value": 0.0001033177317,
            "unit": "s"
          },
          {
            "name": "actmtch / JaX   / gpu / BothRev",
            "value": 0.0001036159211998,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / gpu / PreRev",
            "value": 0.0001102903179998,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / gpu / PostRev",
            "value": 0.0001044380528999,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / gpu / BothRev",
            "value": 0.0001112481207999,
            "unit": "s"
          },
          {
            "name": "actmtch / JaX   / cpu / Primal",
            "value": 0.000003888893700059271,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / Primal",
            "value": 0.0000035830598004395143,
            "unit": "s"
          },
          {
            "name": "actmtch / JaX   / cpu / Forward",
            "value": 0.000005236526700173272,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / Forward",
            "value": 0.000006224521600233856,
            "unit": "s"
          },
          {
            "name": "actmtch / JaX   / cpu / BothRev",
            "value": 0.000005550141599815106,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / PreRev",
            "value": 0.000006096416700165718,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / PostRev",
            "value": 0.000005521469600353157,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / BothRev",
            "value": 0.000006435405599768273,
            "unit": "s"
          },
          {
            "name": "actmtch / JaX   / tpu / Primal",
            "value": 0.0001585435907996,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / tpu / Primal",
            "value": 0.0001541092320003,
            "unit": "s"
          },
          {
            "name": "actmtch / JaX   / tpu / Forward",
            "value": 0.0002245106158996,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / tpu / Forward",
            "value": 0.0002414041468997,
            "unit": "s"
          },
          {
            "name": "actmtch / JaX   / tpu / BothRev",
            "value": 0.0002466575215999,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / tpu / PreRev",
            "value": 0.0002274255777003,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / tpu / PostRev",
            "value": 0.0002327662272997,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / tpu / BothRev",
            "value": 0.0002310955225002,
            "unit": "s"
          },
          {
            "name": "actmtch / JaX   / cpu / Primal",
            "value": 0.000005023756900027365,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / Primal",
            "value": 0.000004939696200017352,
            "unit": "s"
          },
          {
            "name": "actmtch / JaX   / cpu / Forward",
            "value": 0.000007127565400060121,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / Forward",
            "value": 0.0000097832209000444,
            "unit": "s"
          },
          {
            "name": "actmtch / JaX   / cpu / BothRev",
            "value": 0.00000746945989994856,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / PreRev",
            "value": 0.000009220976500000689,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / PostRev",
            "value": 0.00000760794129992064,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / BothRev",
            "value": 0.00000926969170004668,
            "unit": "s"
          },
          {
            "name": "add_one / JaX   / cpu / Primal",
            "value": 0.000004221375999986776,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / Primal",
            "value": 0.000004248234299984688,
            "unit": "s"
          },
          {
            "name": "add_one / JaX   / cpu / Forward",
            "value": 0.000007514756600039618,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / Forward",
            "value": 0.000007491655400008312,
            "unit": "s"
          },
          {
            "name": "add_one / JaX   / cpu / BothRev",
            "value": 0.000007534473600026103,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / PreRev",
            "value": 0.000007696040299970264,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / PostRev",
            "value": 0.000007473608000009335,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / BothRev",
            "value": 0.000007556833199987523,
            "unit": "s"
          },
          {
            "name": "add_one / JaX   / cpu / Primal",
            "value": 0.000007333805900088919,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / Primal",
            "value": 0.000007422407400008524,
            "unit": "s"
          },
          {
            "name": "add_one / JaX   / cpu / Forward",
            "value": 0.000011781881400020213,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / Forward",
            "value": 0.000012168935099907686,
            "unit": "s"
          },
          {
            "name": "add_one / JaX   / cpu / BothRev",
            "value": 0.000011720589099968493,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / PreRev",
            "value": 0.000011663872500139405,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / PostRev",
            "value": 0.000011756568000055268,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / BothRev",
            "value": 0.000011815062599998782,
            "unit": "s"
          },
          {
            "name": "add_one / JaX   / gpu / Primal",
            "value": 0.00007682557509997423,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / gpu / Primal",
            "value": 0.00007817539320003562,
            "unit": "s"
          },
          {
            "name": "add_one / JaX   / gpu / Forward",
            "value": 0.0001271991753999,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / gpu / Forward",
            "value": 0.0001065110891,
            "unit": "s"
          },
          {
            "name": "add_one / JaX   / gpu / BothRev",
            "value": 0.0001098862513001,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / gpu / PreRev",
            "value": 0.0001095640186998,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / gpu / PostRev",
            "value": 0.0001090896196999,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / gpu / BothRev",
            "value": 0.0001090583037999,
            "unit": "s"
          },
          {
            "name": "add_one / JaX   / cpu / Primal",
            "value": 0.000003820375699433498,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / Primal",
            "value": 0.000003850745799718425,
            "unit": "s"
          },
          {
            "name": "add_one / JaX   / cpu / Forward",
            "value": 0.000005864559600013308,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / Forward",
            "value": 0.000006201351700292435,
            "unit": "s"
          },
          {
            "name": "add_one / JaX   / cpu / BothRev",
            "value": 0.000006203685599757591,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / PreRev",
            "value": 0.000006266392699762946,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / PostRev",
            "value": 0.0000062064735997410025,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / BothRev",
            "value": 0.000006255078699905425,
            "unit": "s"
          },
          {
            "name": "add_one / JaX   / tpu / Primal",
            "value": 0.0001558269189001,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / tpu / Primal",
            "value": 0.0001560672649,
            "unit": "s"
          },
          {
            "name": "add_one / JaX   / tpu / Forward",
            "value": 0.000238849361,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / tpu / Forward",
            "value": 0.000238180681,
            "unit": "s"
          },
          {
            "name": "add_one / JaX   / tpu / BothRev",
            "value": 0.0002457600515997,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / tpu / PreRev",
            "value": 0.0002196414550999,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / tpu / PostRev",
            "value": 0.0002283411046002,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / tpu / BothRev",
            "value": 0.0002264422466992,
            "unit": "s"
          },
          {
            "name": "add_one / JaX   / cpu / Primal",
            "value": 0.000005183327299982921,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / Primal",
            "value": 0.00000536612480000258,
            "unit": "s"
          },
          {
            "name": "add_one / JaX   / cpu / Forward",
            "value": 0.000008197373399980279,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / Forward",
            "value": 0.000008634414500011189,
            "unit": "s"
          },
          {
            "name": "add_one / JaX   / cpu / BothRev",
            "value": 0.000008369975099958537,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / PreRev",
            "value": 0.000008884007400047267,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / PostRev",
            "value": 0.000008856079399993177,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / BothRev",
            "value": 0.000008808596400012903,
            "unit": "s"
          },
          {
            "name": "add_two / JaX   / cpu / Primal",
            "value": 0.000004400090899980569,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / Primal",
            "value": 0.000004424161599990839,
            "unit": "s"
          },
          {
            "name": "add_two / JaX   / cpu / Forward",
            "value": 0.000007389712399981362,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / Forward",
            "value": 0.000007342772600031821,
            "unit": "s"
          },
          {
            "name": "add_two / JaX   / cpu / BothRev",
            "value": 0.000009861097500015603,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / PreRev",
            "value": 0.000009610602399970958,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / PostRev",
            "value": 0.000009470495699997628,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / BothRev",
            "value": 0.000009597478399973624,
            "unit": "s"
          },
          {
            "name": "add_two / JaX   / cpu / Primal",
            "value": 0.000007853046899981565,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / Primal",
            "value": 0.000007764462500017544,
            "unit": "s"
          },
          {
            "name": "add_two / JaX   / cpu / Forward",
            "value": 0.000011966450700128916,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / Forward",
            "value": 0.000011955080200095835,
            "unit": "s"
          },
          {
            "name": "add_two / JaX   / cpu / BothRev",
            "value": 0.00001460737679990416,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / PreRev",
            "value": 0.000014626015199974063,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / PostRev",
            "value": 0.00001432023469988053,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / BothRev",
            "value": 0.000014648092899915356,
            "unit": "s"
          },
          {
            "name": "add_two / JaX   / gpu / Primal",
            "value": 0.00007813301940004749,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / gpu / Primal",
            "value": 0.00007844890699998359,
            "unit": "s"
          },
          {
            "name": "add_two / JaX   / gpu / Forward",
            "value": 0.0001064112938,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / gpu / Forward",
            "value": 0.0001067091177999,
            "unit": "s"
          },
          {
            "name": "add_two / JaX   / gpu / BothRev",
            "value": 0.0001228723885998,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / gpu / PreRev",
            "value": 0.0001251365968999,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / gpu / PostRev",
            "value": 0.0001232139204001,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / gpu / BothRev",
            "value": 0.0001428660842999,
            "unit": "s"
          },
          {
            "name": "add_two / JaX   / cpu / Primal",
            "value": 0.0000039322336997429374,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / Primal",
            "value": 0.000004008622800029116,
            "unit": "s"
          },
          {
            "name": "add_two / JaX   / cpu / Forward",
            "value": 0.000006076181599928532,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / Forward",
            "value": 0.000006375469700287795,
            "unit": "s"
          },
          {
            "name": "add_two / JaX   / cpu / BothRev",
            "value": 0.00000739300560017,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / PreRev",
            "value": 0.000007487490600033197,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / PostRev",
            "value": 0.000007397623499855399,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / BothRev",
            "value": 0.000007432506599434419,
            "unit": "s"
          },
          {
            "name": "add_two / JaX   / tpu / Primal",
            "value": 0.0001393057717999,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / tpu / Primal",
            "value": 0.0001396214168002,
            "unit": "s"
          },
          {
            "name": "add_two / JaX   / tpu / Forward",
            "value": 0.0002332947023998,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / tpu / Forward",
            "value": 0.0002180160022995,
            "unit": "s"
          },
          {
            "name": "add_two / JaX   / tpu / BothRev",
            "value": 0.0002375591291995,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / tpu / PreRev",
            "value": 0.0002436417567994,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / tpu / PostRev",
            "value": 0.0002494659634998,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / tpu / BothRev",
            "value": 0.0002444009566999,
            "unit": "s"
          },
          {
            "name": "add_two / JaX   / cpu / Primal",
            "value": 0.000005656883099982224,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / Primal",
            "value": 0.000006004305699934775,
            "unit": "s"
          },
          {
            "name": "add_two / JaX   / cpu / Forward",
            "value": 0.000008573043199976383,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / Forward",
            "value": 0.00000887675029998718,
            "unit": "s"
          },
          {
            "name": "add_two / JaX   / cpu / BothRev",
            "value": 0.000010399737599982472,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / PreRev",
            "value": 0.000010545757100044283,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / PostRev",
            "value": 0.000010568709499966644,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / BothRev",
            "value": 0.000010440281499995763,
            "unit": "s"
          },
          {
            "name": "cache / JaX   / cpu / Primal",
            "value": 0.000004257991299982677,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / Primal",
            "value": 0.000003844736900009593,
            "unit": "s"
          },
          {
            "name": "cache / JaX   / cpu / Forward",
            "value": 0.000010476668100000096,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / Forward",
            "value": 0.000010631851300013296,
            "unit": "s"
          },
          {
            "name": "cache / JaX   / cpu / BothRev",
            "value": 0.000014376339499995083,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / PreRev",
            "value": 0.000011329544000000169,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / PostRev",
            "value": 0.000014214122499970472,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / BothRev",
            "value": 0.00001143866600000365,
            "unit": "s"
          },
          {
            "name": "cache / JaX   / cpu / Primal",
            "value": 0.000007474560599985125,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / Primal",
            "value": 0.0000074773096999706465,
            "unit": "s"
          },
          {
            "name": "cache / JaX   / cpu / Forward",
            "value": 0.00001289496479985246,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / Forward",
            "value": 0.000012986508200083336,
            "unit": "s"
          },
          {
            "name": "cache / JaX   / cpu / BothRev",
            "value": 0.00002252716250004596,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / PreRev",
            "value": 0.000017381758999908925,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / PostRev",
            "value": 0.00002375156509988301,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / BothRev",
            "value": 0.000017634953199922165,
            "unit": "s"
          },
          {
            "name": "cache / JaX   / gpu / Primal",
            "value": 0.00008125018259997888,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / gpu / Primal",
            "value": 0.00007603169419999177,
            "unit": "s"
          },
          {
            "name": "cache / JaX   / gpu / Forward",
            "value": 0.0001001671495001,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / gpu / Forward",
            "value": 0.00010074912,
            "unit": "s"
          },
          {
            "name": "cache / JaX   / gpu / BothRev",
            "value": 0.0001040311058,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / gpu / PreRev",
            "value": 0.0001043637658,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / gpu / PostRev",
            "value": 0.0001085071256,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / gpu / BothRev",
            "value": 0.0001133617831999,
            "unit": "s"
          },
          {
            "name": "cache / JaX   / cpu / Primal",
            "value": 0.000003635209800268058,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / Primal",
            "value": 0.000003623636799602537,
            "unit": "s"
          },
          {
            "name": "cache / JaX   / cpu / Forward",
            "value": 0.00001649750899960054,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / Forward",
            "value": 0.000007036248599615647,
            "unit": "s"
          },
          {
            "name": "cache / JaX   / cpu / BothRev",
            "value": 0.000031183132199657847,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / PreRev",
            "value": 0.000011077047399885489,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / PostRev",
            "value": 0.000031844339099916395,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / BothRev",
            "value": 0.000009782628399989336,
            "unit": "s"
          },
          {
            "name": "cache / JaX   / tpu / Primal",
            "value": 0.0001431915185996,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / tpu / Primal",
            "value": 0.0001409267946997,
            "unit": "s"
          },
          {
            "name": "cache / JaX   / tpu / Forward",
            "value": 0.0002235779169997,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / tpu / Forward",
            "value": 0.0002276957656999,
            "unit": "s"
          },
          {
            "name": "cache / JaX   / tpu / BothRev",
            "value": 0.0002332378573999,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / tpu / PreRev",
            "value": 0.0002316751524995,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / tpu / PostRev",
            "value": 0.0002288646747001,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / tpu / BothRev",
            "value": 0.0002285151126001,
            "unit": "s"
          },
          {
            "name": "cache / JaX   / cpu / Primal",
            "value": 0.0000050234901999829165,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / Primal",
            "value": 0.000004961079199983942,
            "unit": "s"
          },
          {
            "name": "cache / JaX   / cpu / Forward",
            "value": 0.000008903114899931098,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / Forward",
            "value": 0.000009114059999956226,
            "unit": "s"
          },
          {
            "name": "cache / JaX   / cpu / BothRev",
            "value": 0.000015200259900029778,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / PreRev",
            "value": 0.00001237679059995571,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / PostRev",
            "value": 0.000016349837800044044,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / BothRev",
            "value": 0.000012685113299994556,
            "unit": "s"
          },
          {
            "name": "Concat / JaX   / cpu / Primal",
            "value": 0.0000042229700999996565,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / Primal",
            "value": 0.0000041841809999823455,
            "unit": "s"
          },
          {
            "name": "Concat / JaX   / cpu / Forward",
            "value": 0.000006966495999995459,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / Forward",
            "value": 0.000007061789099998351,
            "unit": "s"
          },
          {
            "name": "Concat / JaX   / cpu / BothRev",
            "value": 0.000007513651099998242,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / PreRev",
            "value": 0.000007483798000021125,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / PostRev",
            "value": 0.000008010000500007663,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / BothRev",
            "value": 0.000007434162400022615,
            "unit": "s"
          },
          {
            "name": "Concat / JaX   / cpu / Primal",
            "value": 0.000007325720200060459,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / Primal",
            "value": 0.000007491727500018896,
            "unit": "s"
          },
          {
            "name": "Concat / JaX   / cpu / Forward",
            "value": 0.000012278299100034928,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / Forward",
            "value": 0.0000124513806998948,
            "unit": "s"
          },
          {
            "name": "Concat / JaX   / cpu / BothRev",
            "value": 0.000012137860899929364,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / PreRev",
            "value": 0.000012569293399974414,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / PostRev",
            "value": 0.000012664270499954,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / BothRev",
            "value": 0.000012340327099991557,
            "unit": "s"
          },
          {
            "name": "Concat / JaX   / gpu / Primal",
            "value": 0.0000781766037000125,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / gpu / Primal",
            "value": 0.00008940246439997282,
            "unit": "s"
          },
          {
            "name": "Concat / JaX   / gpu / Forward",
            "value": 0.0001067922133001,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / gpu / Forward",
            "value": 0.0001069774111998,
            "unit": "s"
          },
          {
            "name": "Concat / JaX   / gpu / BothRev",
            "value": 0.0001050804037,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / gpu / PreRev",
            "value": 0.0001099798328999,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / gpu / PostRev",
            "value": 0.00009848711129998264,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / gpu / BothRev",
            "value": 0.00009838990830012336,
            "unit": "s"
          },
          {
            "name": "Concat / JaX   / cpu / Primal",
            "value": 0.000003894566799863241,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / Primal",
            "value": 0.000003741456799616572,
            "unit": "s"
          },
          {
            "name": "Concat / JaX   / cpu / Forward",
            "value": 0.000005885957699501887,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / Forward",
            "value": 0.000006069470700458624,
            "unit": "s"
          },
          {
            "name": "Concat / JaX   / cpu / BothRev",
            "value": 0.000005945623599836836,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / PreRev",
            "value": 0.000005992460699781077,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / PostRev",
            "value": 0.000005667667699890444,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / BothRev",
            "value": 0.00000602092269982677,
            "unit": "s"
          },
          {
            "name": "Concat / JaX   / tpu / Primal",
            "value": 0.0001434232967003,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / tpu / Primal",
            "value": 0.0001399360876996,
            "unit": "s"
          },
          {
            "name": "Concat / JaX   / tpu / Forward",
            "value": 0.0002256741008,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / tpu / Forward",
            "value": 0.0002252096558004,
            "unit": "s"
          },
          {
            "name": "Concat / JaX   / tpu / BothRev",
            "value": 0.0002347539602997,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / tpu / PreRev",
            "value": 0.0002347212182998,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / tpu / PostRev",
            "value": 0.0002320430794003,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / tpu / BothRev",
            "value": 0.0002329050583997,
            "unit": "s"
          },
          {
            "name": "Concat / JaX   / cpu / Primal",
            "value": 0.000005518738299997494,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / Primal",
            "value": 0.000005337287400016066,
            "unit": "s"
          },
          {
            "name": "Concat / JaX   / cpu / Forward",
            "value": 0.000008572258100048202,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / Forward",
            "value": 0.000008571013100026904,
            "unit": "s"
          },
          {
            "name": "Concat / JaX   / cpu / BothRev",
            "value": 0.000008959626199975901,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / PreRev",
            "value": 0.000008596185500027787,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / PostRev",
            "value": 0.0000084872438000275,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / BothRev",
            "value": 0.000008524407199911366,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaX   / cpu / Primal",
            "value": 0.000007620199994562427,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / cpu / Primal",
            "value": 0.000007098800006133388,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaX   / cpu / Forward",
            "value": 0.000009316900013800478,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / cpu / Forward",
            "value": 0.000010165499998038285,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaX   / cpu / Primal",
            "value": 0.000014916400141373744,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / cpu / Primal",
            "value": 0.000012066300041624344,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaX   / cpu / Forward",
            "value": 0.000015782199989189392,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / cpu / Forward",
            "value": 0.000019401899953663816,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaX   / gpu / Primal",
            "value": 0.0001249818998985,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / gpu / Primal",
            "value": 0.0001129132000642,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaX   / gpu / Forward",
            "value": 0.0001453847999073,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / gpu / Forward",
            "value": 0.0001423464000254,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaX   / cpu / Primal",
            "value": 0.000006484999903477728,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / cpu / Primal",
            "value": 0.000005358000635169447,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaX   / cpu / Forward",
            "value": 0.00000822100046207197,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / cpu / Forward",
            "value": 0.000008420000085607171,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaX   / tpu / Primal",
            "value": 0.0001735550002194,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / tpu / Primal",
            "value": 0.000176965999708,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaX   / tpu / Forward",
            "value": 0.0002668229994014,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / tpu / Forward",
            "value": 0.0002653520001331,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaX   / cpu / Primal",
            "value": 0.000008570899990445468,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / cpu / Primal",
            "value": 0.000009101799969357672,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaX   / cpu / Forward",
            "value": 0.000010279199977958342,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / cpu / Forward",
            "value": 0.000013992300046083984,
            "unit": "s"
          },
          {
            "name": "GenDot / JaX   / cpu / Primal",
            "value": 0.000004305226599990419,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / Primal",
            "value": 0.00000431635519998963,
            "unit": "s"
          },
          {
            "name": "GenDot / JaX   / cpu / Forward",
            "value": 0.000006539181100015412,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / Forward",
            "value": 0.000007373701500000607,
            "unit": "s"
          },
          {
            "name": "GenDot / JaX   / cpu / BothRev",
            "value": 0.000006726316999993287,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / PreRev",
            "value": 0.000007370045800007574,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / PostRev",
            "value": 0.000006784010499995929,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / BothRev",
            "value": 0.000007361278600001242,
            "unit": "s"
          },
          {
            "name": "GenDot / JaX   / cpu / Primal",
            "value": 0.000008033055099986086,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / Primal",
            "value": 0.000008390950499961036,
            "unit": "s"
          },
          {
            "name": "GenDot / JaX   / cpu / Forward",
            "value": 0.000011117413099964325,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / Forward",
            "value": 0.000013905367699953786,
            "unit": "s"
          },
          {
            "name": "GenDot / JaX   / cpu / BothRev",
            "value": 0.00001075635159995727,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / PreRev",
            "value": 0.000014188015599938808,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / PostRev",
            "value": 0.000011281386900009236,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / BothRev",
            "value": 0.000014358634800009894,
            "unit": "s"
          },
          {
            "name": "GenDot / JaX   / gpu / Primal",
            "value": 0.00007917201839991322,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / gpu / Primal",
            "value": 0.00007852207989999442,
            "unit": "s"
          },
          {
            "name": "GenDot / JaX   / gpu / Forward",
            "value": 0.0001161023423001,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / gpu / Forward",
            "value": 0.0001094103704001,
            "unit": "s"
          },
          {
            "name": "GenDot / JaX   / gpu / BothRev",
            "value": 0.0001026670191,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / gpu / PreRev",
            "value": 0.000106410687,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / gpu / PostRev",
            "value": 0.0001020137320998,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / gpu / BothRev",
            "value": 0.0001150465489999,
            "unit": "s"
          },
          {
            "name": "GenDot / JaX   / cpu / Primal",
            "value": 0.0000036879227998724673,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / Primal",
            "value": 0.000004197485800250433,
            "unit": "s"
          },
          {
            "name": "GenDot / JaX   / cpu / Forward",
            "value": 0.000005905402599455556,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / Forward",
            "value": 0.000006456901600176934,
            "unit": "s"
          },
          {
            "name": "GenDot / JaX   / cpu / BothRev",
            "value": 0.000005942867599515012,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / PreRev",
            "value": 0.000006469622599979629,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / PostRev",
            "value": 0.000005636079599935329,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / BothRev",
            "value": 0.00000592213060008362,
            "unit": "s"
          },
          {
            "name": "GenDot / JaX   / tpu / Primal",
            "value": 0.0001405656597999,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / tpu / Primal",
            "value": 0.0001406256967995,
            "unit": "s"
          },
          {
            "name": "GenDot / JaX   / tpu / Forward",
            "value": 0.0002363347731996,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / tpu / Forward",
            "value": 0.0002397689259996,
            "unit": "s"
          },
          {
            "name": "GenDot / JaX   / tpu / BothRev",
            "value": 0.0002314764793998,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / tpu / PreRev",
            "value": 0.0002301387635001,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / tpu / PostRev",
            "value": 0.0002327623953999,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / tpu / BothRev",
            "value": 0.0002207230910993,
            "unit": "s"
          },
          {
            "name": "GenDot / JaX   / cpu / Primal",
            "value": 0.000005860007999945083,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / Primal",
            "value": 0.000005479336999997031,
            "unit": "s"
          },
          {
            "name": "GenDot / JaX   / cpu / Forward",
            "value": 0.000007948447499984467,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / Forward",
            "value": 0.000009511316799944324,
            "unit": "s"
          },
          {
            "name": "GenDot / JaX   / cpu / BothRev",
            "value": 0.000008006404699972337,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / PreRev",
            "value": 0.000009231253300004027,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / PostRev",
            "value": 0.000008496275299967237,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / BothRev",
            "value": 0.00000953088980004395,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaX   / cpu / Primal",
            "value": 0.000006007991599972229,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / Primal",
            "value": 0.000005891712699985874,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaX   / cpu / Forward",
            "value": 0.000009990135099997132,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / Forward",
            "value": 0.000010069322099980127,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaX   / cpu / BothRev",
            "value": 0.000009401051999975606,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / PreRev",
            "value": 0.000009514673700005006,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / PostRev",
            "value": 0.000010275410900021598,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / BothRev",
            "value": 0.000009341794300007677,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaX   / cpu / Primal",
            "value": 0.00001029326879997825,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / Primal",
            "value": 0.000009910237199983384,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaX   / cpu / Forward",
            "value": 0.000015262294300009673,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / Forward",
            "value": 0.00001589356839995162,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaX   / cpu / BothRev",
            "value": 0.000014909565999914776,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / PreRev",
            "value": 0.000014879730100074084,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / PostRev",
            "value": 0.000015360392299953673,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / BothRev",
            "value": 0.000015565025599971705,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaX   / cpu / Primal",
            "value": 0.000005132158700143919,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / Primal",
            "value": 0.000004945962800411508,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaX   / cpu / Forward",
            "value": 0.000007588141500309576,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / Forward",
            "value": 0.000007975217499915743,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaX   / cpu / BothRev",
            "value": 0.000007343821600079537,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / PreRev",
            "value": 0.000007635607500560582,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / PostRev",
            "value": 0.000007297514600213617,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / BothRev",
            "value": 0.000007616798600065522,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaX   / tpu / Primal",
            "value": 0.00004765109919972019,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / tpu / Primal",
            "value": 0.00004664374529966153,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaX   / tpu / Forward",
            "value": 0.0000632015112998488,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / tpu / Forward",
            "value": 0.00006532635120020132,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaX   / tpu / BothRev",
            "value": 0.00006411575919992174,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / tpu / PreRev",
            "value": 0.00007189345279984991,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / tpu / PostRev",
            "value": 0.00006192938829990453,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / tpu / BothRev",
            "value": 0.00006723617699972238,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaX   / cpu / Primal",
            "value": 0.000007333057300002111,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / Primal",
            "value": 0.000007372727300025872,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaX   / cpu / Forward",
            "value": 0.000011225680799998372,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / Forward",
            "value": 0.000011699876100010444,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaX   / cpu / BothRev",
            "value": 0.000011028973900010896,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / PreRev",
            "value": 0.000011553434899997228,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / PostRev",
            "value": 0.000011131199800001922,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / BothRev",
            "value": 0.000011343976100033616,
            "unit": "s"
          },
          {
            "name": "llama / JaXPipe / cpu / Primal",
            "value": 0.0009036038300018,
            "unit": "s"
          },
          {
            "name": "llama / JaX   / cpu / Primal",
            "value": 0.0008815707099984,
            "unit": "s"
          },
          {
            "name": "llama / HLOOpt / cpu / Primal",
            "value": 0.0009545046899984,
            "unit": "s"
          },
          {
            "name": "llama / PartOpt / cpu / Primal",
            "value": 0.0008596076999992,
            "unit": "s"
          },
          {
            "name": "llama / DefOpt / cpu / Primal",
            "value": 0.0009428964700009,
            "unit": "s"
          },
          {
            "name": "llama / IPartOpt / cpu / Primal",
            "value": 0.000862209490001,
            "unit": "s"
          },
          {
            "name": "llama / IDefOpt / cpu / Primal",
            "value": 0.0009406553999997,
            "unit": "s"
          },
          {
            "name": "llama / JaXPipe / cpu / Forward",
            "value": 0.0026777235899999,
            "unit": "s"
          },
          {
            "name": "llama / JaX   / cpu / Forward",
            "value": 0.0026008398599969,
            "unit": "s"
          },
          {
            "name": "llama / HLOOpt / cpu / Forward",
            "value": 0.0026227514299989,
            "unit": "s"
          },
          {
            "name": "llama / PartOpt / cpu / Forward",
            "value": 0.0026131044399971,
            "unit": "s"
          },
          {
            "name": "llama / DefOpt / cpu / Forward",
            "value": 0.002628777750001,
            "unit": "s"
          },
          {
            "name": "llama / IPartOpt / cpu / Forward",
            "value": 0.0026456083599987,
            "unit": "s"
          },
          {
            "name": "llama / IDefOpt / cpu / Forward",
            "value": 0.0026386267999987,
            "unit": "s"
          },
          {
            "name": "llama / JaXPipe / cpu / PreRev",
            "value": 0.0023864121999986,
            "unit": "s"
          },
          {
            "name": "llama / JaXPipe / cpu / PostRev",
            "value": 0.0023078690700003,
            "unit": "s"
          },
          {
            "name": "llama / JaXPipe / cpu / BothRev",
            "value": 0.0024488058399992,
            "unit": "s"
          },
          {
            "name": "llama / JaX   / cpu / BothRev",
            "value": 0.0025524685099981,
            "unit": "s"
          },
          {
            "name": "llama / HLOOpt / cpu / PreRev",
            "value": 0.0023098340199976,
            "unit": "s"
          },
          {
            "name": "llama / HLOOpt / cpu / PostRev",
            "value": 0.0026894748899985,
            "unit": "s"
          },
          {
            "name": "llama / HLOOpt / cpu / BothRev",
            "value": 0.0023117150400003,
            "unit": "s"
          },
          {
            "name": "llama / PartOpt / cpu / PreRev",
            "value": 0.0023249103199987,
            "unit": "s"
          },
          {
            "name": "llama / PartOpt / cpu / PostRev",
            "value": 0.0023054931600017,
            "unit": "s"
          },
          {
            "name": "llama / PartOpt / cpu / BothRev",
            "value": 0.0027967864899983,
            "unit": "s"
          },
          {
            "name": "llama / DefOpt / cpu / PreRev",
            "value": 0.0023714528499976,
            "unit": "s"
          },
          {
            "name": "llama / DefOpt / cpu / PostRev",
            "value": 0.0026971864500001,
            "unit": "s"
          },
          {
            "name": "llama / DefOpt / cpu / BothRev",
            "value": 0.002330606559999,
            "unit": "s"
          },
          {
            "name": "llama / IPartOpt / cpu / PreRev",
            "value": 0.0028142271800015,
            "unit": "s"
          },
          {
            "name": "llama / IPartOpt / cpu / PostRev",
            "value": 0.0023104740999997,
            "unit": "s"
          },
          {
            "name": "llama / IPartOpt / cpu / BothRev",
            "value": 0.0028174292300036,
            "unit": "s"
          },
          {
            "name": "llama / IDefOpt / cpu / PreRev",
            "value": 0.0023330732900012,
            "unit": "s"
          },
          {
            "name": "llama / IDefOpt / cpu / PostRev",
            "value": 0.0028763271700017,
            "unit": "s"
          },
          {
            "name": "llama / IDefOpt / cpu / BothRev",
            "value": 0.0023699240599989,
            "unit": "s"
          },
          {
            "name": "llama / JaXPipe / gpu / Primal",
            "value": 0.0004334704140019,
            "unit": "s"
          },
          {
            "name": "llama / JaX   / gpu / Primal",
            "value": 0.0004326484760022,
            "unit": "s"
          },
          {
            "name": "llama / HLOOpt / gpu / Primal",
            "value": 0.0004502818159999,
            "unit": "s"
          },
          {
            "name": "llama / PartOpt / gpu / Primal",
            "value": 0.000431408815999,
            "unit": "s"
          },
          {
            "name": "llama / DefOpt / gpu / Primal",
            "value": 0.0004453713379989,
            "unit": "s"
          },
          {
            "name": "llama / IPartOpt / gpu / Primal",
            "value": 0.0004493406619985,
            "unit": "s"
          },
          {
            "name": "llama / IDefOpt / gpu / Primal",
            "value": 0.0004547928200008,
            "unit": "s"
          },
          {
            "name": "llama / JaXPipe / gpu / Forward",
            "value": 0.0007431619740018,
            "unit": "s"
          },
          {
            "name": "llama / JaX   / gpu / Forward",
            "value": 0.0006885440839978,
            "unit": "s"
          },
          {
            "name": "llama / HLOOpt / gpu / Forward",
            "value": 0.0007418201320033,
            "unit": "s"
          },
          {
            "name": "llama / PartOpt / gpu / Forward",
            "value": 0.0007191103040022,
            "unit": "s"
          },
          {
            "name": "llama / DefOpt / gpu / Forward",
            "value": 0.0007279277580018,
            "unit": "s"
          },
          {
            "name": "llama / IPartOpt / gpu / Forward",
            "value": 0.0007357419239997,
            "unit": "s"
          },
          {
            "name": "llama / IDefOpt / gpu / Forward",
            "value": 0.0007442585519966,
            "unit": "s"
          },
          {
            "name": "llama / JaXPipe / gpu / PreRev",
            "value": 0.0008011282060033,
            "unit": "s"
          },
          {
            "name": "llama / JaXPipe / gpu / PostRev",
            "value": 0.0007745135199984,
            "unit": "s"
          },
          {
            "name": "llama / JaXPipe / gpu / BothRev",
            "value": 0.0008277438679979,
            "unit": "s"
          },
          {
            "name": "llama / JaX   / gpu / BothRev",
            "value": 0.0007886702680007,
            "unit": "s"
          },
          {
            "name": "llama / HLOOpt / gpu / PreRev",
            "value": 0.0007771036399972,
            "unit": "s"
          },
          {
            "name": "llama / HLOOpt / gpu / PostRev",
            "value": 0.0008106862879976,
            "unit": "s"
          },
          {
            "name": "llama / HLOOpt / gpu / BothRev",
            "value": 0.0007915957759978,
            "unit": "s"
          },
          {
            "name": "llama / PartOpt / gpu / PreRev",
            "value": 0.0008056679940018,
            "unit": "s"
          },
          {
            "name": "llama / PartOpt / gpu / PostRev",
            "value": 0.0008064865260021,
            "unit": "s"
          },
          {
            "name": "llama / PartOpt / gpu / BothRev",
            "value": 0.0008089085460014,
            "unit": "s"
          },
          {
            "name": "llama / DefOpt / gpu / PreRev",
            "value": 0.0007689638479969,
            "unit": "s"
          },
          {
            "name": "llama / DefOpt / gpu / PostRev",
            "value": 0.0007930275760008,
            "unit": "s"
          },
          {
            "name": "llama / DefOpt / gpu / BothRev",
            "value": 0.000799012544001,
            "unit": "s"
          },
          {
            "name": "llama / IPartOpt / gpu / PreRev",
            "value": 0.0007939320200021,
            "unit": "s"
          },
          {
            "name": "llama / IPartOpt / gpu / PostRev",
            "value": 0.0007959353119986,
            "unit": "s"
          },
          {
            "name": "llama / IPartOpt / gpu / BothRev",
            "value": 0.000793690112001,
            "unit": "s"
          },
          {
            "name": "llama / IDefOpt / gpu / PreRev",
            "value": 0.0007910991800017,
            "unit": "s"
          },
          {
            "name": "llama / IDefOpt / gpu / PostRev",
            "value": 0.0008137880879985,
            "unit": "s"
          },
          {
            "name": "llama / IDefOpt / gpu / BothRev",
            "value": 0.0008149435499981,
            "unit": "s"
          },
          {
            "name": "llama / JaXPipe / tpu / Primal",
            "value": 0.0003827782979933,
            "unit": "s"
          },
          {
            "name": "llama / JaX   / tpu / Primal",
            "value": 0.0003815366980124,
            "unit": "s"
          },
          {
            "name": "llama / HLOOpt / tpu / Primal",
            "value": 0.000379335798003,
            "unit": "s"
          },
          {
            "name": "llama / PartOpt / tpu / Primal",
            "value": 0.0003638919179938,
            "unit": "s"
          },
          {
            "name": "llama / DefOpt / tpu / Primal",
            "value": 0.0003311642000044,
            "unit": "s"
          },
          {
            "name": "llama / IPartOpt / tpu / Primal",
            "value": 0.0003546996400109,
            "unit": "s"
          },
          {
            "name": "llama / IDefOpt / tpu / Primal",
            "value": 0.000355125838003,
            "unit": "s"
          },
          {
            "name": "llama / JaXPipe / tpu / Forward",
            "value": 0.0005789292460103,
            "unit": "s"
          },
          {
            "name": "llama / JaX   / tpu / Forward",
            "value": 0.0007043117599969,
            "unit": "s"
          },
          {
            "name": "llama / HLOOpt / tpu / Forward",
            "value": 0.0005613966660021,
            "unit": "s"
          },
          {
            "name": "llama / PartOpt / tpu / Forward",
            "value": 0.0005685037660005,
            "unit": "s"
          },
          {
            "name": "llama / DefOpt / tpu / Forward",
            "value": 0.0005808443260029,
            "unit": "s"
          },
          {
            "name": "llama / IPartOpt / tpu / Forward",
            "value": 0.0005734330859995,
            "unit": "s"
          },
          {
            "name": "llama / IDefOpt / tpu / Forward",
            "value": 0.0005726253879984,
            "unit": "s"
          },
          {
            "name": "llama / JaXPipe / tpu / PreRev",
            "value": 0.0004065967579954,
            "unit": "s"
          },
          {
            "name": "llama / JaXPipe / tpu / PostRev",
            "value": 0.0003579140399961,
            "unit": "s"
          },
          {
            "name": "llama / JaXPipe / tpu / BothRev",
            "value": 0.0003915522780007,
            "unit": "s"
          },
          {
            "name": "llama / JaX   / tpu / BothRev",
            "value": 0.0003576681200065,
            "unit": "s"
          },
          {
            "name": "llama / HLOOpt / tpu / PreRev",
            "value": 0.0003918229980044,
            "unit": "s"
          },
          {
            "name": "llama / HLOOpt / tpu / PostRev",
            "value": 0.0003908781179925,
            "unit": "s"
          },
          {
            "name": "llama / HLOOpt / tpu / BothRev",
            "value": 0.0003915520779992,
            "unit": "s"
          },
          {
            "name": "llama / PartOpt / tpu / PreRev",
            "value": 0.0003915564759954,
            "unit": "s"
          },
          {
            "name": "llama / PartOpt / tpu / PostRev",
            "value": 0.0003723276780074,
            "unit": "s"
          },
          {
            "name": "llama / PartOpt / tpu / BothRev",
            "value": 0.0003915616579906,
            "unit": "s"
          },
          {
            "name": "llama / DefOpt / tpu / PreRev",
            "value": 0.0003916253180068,
            "unit": "s"
          },
          {
            "name": "llama / DefOpt / tpu / PostRev",
            "value": 0.0003833049380045,
            "unit": "s"
          },
          {
            "name": "llama / DefOpt / tpu / BothRev",
            "value": 0.0003914555379888,
            "unit": "s"
          },
          {
            "name": "llama / IPartOpt / tpu / PreRev",
            "value": 0.0003914821779908,
            "unit": "s"
          },
          {
            "name": "llama / IPartOpt / tpu / PostRev",
            "value": 0.0003730398379993,
            "unit": "s"
          },
          {
            "name": "llama / IPartOpt / tpu / BothRev",
            "value": 0.0003914907760045,
            "unit": "s"
          },
          {
            "name": "llama / IDefOpt / tpu / PreRev",
            "value": 0.000391751195988,
            "unit": "s"
          },
          {
            "name": "llama / IDefOpt / tpu / PostRev",
            "value": 0.0003996628360036,
            "unit": "s"
          },
          {
            "name": "llama / IDefOpt / tpu / BothRev",
            "value": 0.0003914849780121,
            "unit": "s"
          },
          {
            "name": "llama / JaXPipe / cpu / Primal",
            "value": 0.0011670381699968,
            "unit": "s"
          },
          {
            "name": "llama / JaX   / cpu / Primal",
            "value": 0.0010962365399973,
            "unit": "s"
          },
          {
            "name": "llama / HLOOpt / cpu / Primal",
            "value": 0.0011628428499989,
            "unit": "s"
          },
          {
            "name": "llama / PartOpt / cpu / Primal",
            "value": 0.0011148639200018,
            "unit": "s"
          },
          {
            "name": "llama / DefOpt / cpu / Primal",
            "value": 0.0011909537300016,
            "unit": "s"
          },
          {
            "name": "llama / IPartOpt / cpu / Primal",
            "value": 0.0011346917199989,
            "unit": "s"
          },
          {
            "name": "llama / IDefOpt / cpu / Primal",
            "value": 0.0012017925700001,
            "unit": "s"
          },
          {
            "name": "llama / JaXPipe / cpu / Forward",
            "value": 0.0037037223800052,
            "unit": "s"
          },
          {
            "name": "llama / JaX   / cpu / Forward",
            "value": 0.0035385844899974,
            "unit": "s"
          },
          {
            "name": "llama / HLOOpt / cpu / Forward",
            "value": 0.0035173198199936,
            "unit": "s"
          },
          {
            "name": "llama / PartOpt / cpu / Forward",
            "value": 0.0036813321199952,
            "unit": "s"
          },
          {
            "name": "llama / DefOpt / cpu / Forward",
            "value": 0.003670380200001,
            "unit": "s"
          },
          {
            "name": "llama / IPartOpt / cpu / Forward",
            "value": 0.0035128307600007,
            "unit": "s"
          },
          {
            "name": "llama / IDefOpt / cpu / Forward",
            "value": 0.0035756724200018,
            "unit": "s"
          },
          {
            "name": "llama / JaXPipe / cpu / PreRev",
            "value": 0.0043650554899977,
            "unit": "s"
          },
          {
            "name": "llama / JaXPipe / cpu / PostRev",
            "value": 0.0042181988099946,
            "unit": "s"
          },
          {
            "name": "llama / JaXPipe / cpu / BothRev",
            "value": 0.0042115815200031,
            "unit": "s"
          },
          {
            "name": "llama / JaX   / cpu / BothRev",
            "value": 0.0042791706499974,
            "unit": "s"
          },
          {
            "name": "llama / HLOOpt / cpu / PreRev",
            "value": 0.0042429292799988,
            "unit": "s"
          },
          {
            "name": "llama / HLOOpt / cpu / PostRev",
            "value": 0.0039379730900054,
            "unit": "s"
          },
          {
            "name": "llama / HLOOpt / cpu / BothRev",
            "value": 0.0044650751999961,
            "unit": "s"
          },
          {
            "name": "llama / PartOpt / cpu / PreRev",
            "value": 0.004621757169998,
            "unit": "s"
          },
          {
            "name": "llama / PartOpt / cpu / PostRev",
            "value": 0.0038128106800013,
            "unit": "s"
          },
          {
            "name": "llama / PartOpt / cpu / BothRev",
            "value": 0.00425144154,
            "unit": "s"
          },
          {
            "name": "llama / DefOpt / cpu / PreRev",
            "value": 0.0042016430799958,
            "unit": "s"
          },
          {
            "name": "llama / DefOpt / cpu / PostRev",
            "value": 0.0037622301899955,
            "unit": "s"
          },
          {
            "name": "llama / DefOpt / cpu / BothRev",
            "value": 0.0046494861900009,
            "unit": "s"
          },
          {
            "name": "llama / IPartOpt / cpu / PreRev",
            "value": 0.0041861027100003,
            "unit": "s"
          },
          {
            "name": "llama / IPartOpt / cpu / PostRev",
            "value": 0.0038566940200053,
            "unit": "s"
          },
          {
            "name": "llama / IPartOpt / cpu / BothRev",
            "value": 0.0045996275499965,
            "unit": "s"
          },
          {
            "name": "llama / IDefOpt / cpu / PreRev",
            "value": 0.0041370548300074,
            "unit": "s"
          },
          {
            "name": "llama / IDefOpt / cpu / PostRev",
            "value": 0.0042281408399958,
            "unit": "s"
          },
          {
            "name": "llama / IDefOpt / cpu / BothRev",
            "value": 0.0045904434699968,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004781286200022805,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004852364199996373,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000008881827300137957,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.00000887627310003154,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / gpu / Primal",
            "value": 0.00007720302430006996,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / gpu / Primal",
            "value": 0.00007867281300004834,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004444872800377197,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004391731799842091,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001361191269999,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001397103007999,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000006356748300004255,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000006204190100015694,
            "unit": "s"
          },
          {
            "name": "slicing / JaX   / cpu / Primal",
            "value": 0.0000038275482999779344,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / Primal",
            "value": 0.000003795791399988957,
            "unit": "s"
          },
          {
            "name": "slicing / JaX   / cpu / Forward",
            "value": 0.000005939210299993647,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / Forward",
            "value": 0.00000603474350000397,
            "unit": "s"
          },
          {
            "name": "slicing / JaX   / cpu / BothRev",
            "value": 0.0000064569756999844685,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / PreRev",
            "value": 0.000006422275499971874,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / PostRev",
            "value": 0.000006377462199998263,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / BothRev",
            "value": 0.000006449023799996212,
            "unit": "s"
          },
          {
            "name": "slicing / JaX   / cpu / Primal",
            "value": 0.000006842718599909858,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / Primal",
            "value": 0.000006800249799925951,
            "unit": "s"
          },
          {
            "name": "slicing / JaX   / cpu / Forward",
            "value": 0.000010032863000014912,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / Forward",
            "value": 0.00001049895199994353,
            "unit": "s"
          },
          {
            "name": "slicing / JaX   / cpu / BothRev",
            "value": 0.000010858908299996983,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / PreRev",
            "value": 0.000010888127900034305,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / PostRev",
            "value": 0.000011428525600058491,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / BothRev",
            "value": 0.000010753079900132434,
            "unit": "s"
          },
          {
            "name": "slicing / JaX   / gpu / Primal",
            "value": 0.00007418459909986268,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / gpu / Primal",
            "value": 0.00007490196130002005,
            "unit": "s"
          },
          {
            "name": "slicing / JaX   / gpu / Forward",
            "value": 0.0001017370442999,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / gpu / Forward",
            "value": 0.000100429285,
            "unit": "s"
          },
          {
            "name": "slicing / JaX   / gpu / BothRev",
            "value": 0.0001030170156998,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / gpu / PreRev",
            "value": 0.0001024978495999,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / gpu / PostRev",
            "value": 0.0001014396143,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / gpu / BothRev",
            "value": 0.0001008703892999,
            "unit": "s"
          },
          {
            "name": "slicing / JaX   / cpu / Primal",
            "value": 0.0000037400606997834985,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / Primal",
            "value": 0.000006294423600047594,
            "unit": "s"
          },
          {
            "name": "slicing / JaX   / cpu / Forward",
            "value": 0.000004757885699655162,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / Forward",
            "value": 0.000004877086800115649,
            "unit": "s"
          },
          {
            "name": "slicing / JaX   / cpu / BothRev",
            "value": 0.000004986373699648538,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / PreRev",
            "value": 0.000005012251800508238,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / PostRev",
            "value": 0.000004915261700080009,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / BothRev",
            "value": 0.000005737477599905105,
            "unit": "s"
          },
          {
            "name": "slicing / JaX   / tpu / Primal",
            "value": 0.0001431329665996,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / tpu / Primal",
            "value": 0.0001405793777004,
            "unit": "s"
          },
          {
            "name": "slicing / JaX   / tpu / Forward",
            "value": 0.0002153926234001,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / tpu / Forward",
            "value": 0.0002333329244,
            "unit": "s"
          },
          {
            "name": "slicing / JaX   / tpu / BothRev",
            "value": 0.000238331913,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / tpu / PreRev",
            "value": 0.0002370358310996,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / tpu / PostRev",
            "value": 0.0002231315370001,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / tpu / BothRev",
            "value": 0.0002277348777002,
            "unit": "s"
          },
          {
            "name": "slicing / JaX   / cpu / Primal",
            "value": 0.000004714478799996868,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / Primal",
            "value": 0.000005209389899937378,
            "unit": "s"
          },
          {
            "name": "slicing / JaX   / cpu / Forward",
            "value": 0.000007112479900024482,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / Forward",
            "value": 0.000007388717000048928,
            "unit": "s"
          },
          {
            "name": "slicing / JaX   / cpu / BothRev",
            "value": 0.000007979117299964856,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / PreRev",
            "value": 0.000007580246999987139,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / PostRev",
            "value": 0.00000771648339996318,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / BothRev",
            "value": 0.000008167079600025318,
            "unit": "s"
          },
          {
            "name": "sum    / JaX   / cpu / Primal",
            "value": 0.000005023583999991387,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / Primal",
            "value": 0.000005372708499999135,
            "unit": "s"
          },
          {
            "name": "sum    / JaX   / cpu / Forward",
            "value": 0.00000857204870003443,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / Forward",
            "value": 0.00000850996489998579,
            "unit": "s"
          },
          {
            "name": "sum    / JaX   / cpu / BothRev",
            "value": 0.000007576139799994053,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / PreRev",
            "value": 0.000007303304700008084,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / PostRev",
            "value": 0.000007272418800039304,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / BothRev",
            "value": 0.000007323511799995686,
            "unit": "s"
          },
          {
            "name": "sum    / JaX   / cpu / Primal",
            "value": 0.000009916721700028575,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / Primal",
            "value": 0.000009828059500068777,
            "unit": "s"
          },
          {
            "name": "sum    / JaX   / cpu / Forward",
            "value": 0.00001636932599994907,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / Forward",
            "value": 0.000015456843600077264,
            "unit": "s"
          },
          {
            "name": "sum    / JaX   / cpu / BothRev",
            "value": 0.00001410090150002361,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / PreRev",
            "value": 0.000013618810499974644,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / PostRev",
            "value": 0.000014028383599907102,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / BothRev",
            "value": 0.000014098836999983175,
            "unit": "s"
          },
          {
            "name": "sum    / JaX   / gpu / Primal",
            "value": 0.00007136468480002805,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / gpu / Primal",
            "value": 0.00008296686279991263,
            "unit": "s"
          },
          {
            "name": "sum    / JaX   / gpu / Forward",
            "value": 0.00009818792730002316,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / gpu / Forward",
            "value": 0.00009816430880000552,
            "unit": "s"
          },
          {
            "name": "sum    / JaX   / gpu / BothRev",
            "value": 0.0001021368894,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / gpu / PreRev",
            "value": 0.0001037522160999,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / gpu / PostRev",
            "value": 0.0001033100450998,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / gpu / BothRev",
            "value": 0.0001022641701998,
            "unit": "s"
          },
          {
            "name": "sum    / JaX   / cpu / Primal",
            "value": 0.000004773735699563986,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / Primal",
            "value": 0.000004556647699791938,
            "unit": "s"
          },
          {
            "name": "sum    / JaX   / cpu / Forward",
            "value": 0.0000070418365998193625,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / Forward",
            "value": 0.000007058710599812912,
            "unit": "s"
          },
          {
            "name": "sum    / JaX   / cpu / BothRev",
            "value": 0.000006456448700191686,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / PreRev",
            "value": 0.000006176962600147818,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / PostRev",
            "value": 0.000006197228600649396,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / BothRev",
            "value": 0.0000062193147001380565,
            "unit": "s"
          },
          {
            "name": "sum    / JaX   / tpu / Primal",
            "value": 0.0001446025065997,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / tpu / Primal",
            "value": 0.0001434415376003,
            "unit": "s"
          },
          {
            "name": "sum    / JaX   / tpu / Forward",
            "value": 0.0002228683070003,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / tpu / Forward",
            "value": 0.0002239085328998,
            "unit": "s"
          },
          {
            "name": "sum    / JaX   / tpu / BothRev",
            "value": 0.0002242962139003,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / tpu / PreRev",
            "value": 0.0002169565482996,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / tpu / PostRev",
            "value": 0.0002373330310998,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / tpu / BothRev",
            "value": 0.0002302827284998,
            "unit": "s"
          },
          {
            "name": "sum    / JaX   / cpu / Primal",
            "value": 0.00000695837080002093,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / Primal",
            "value": 0.000006898569900022267,
            "unit": "s"
          },
          {
            "name": "sum    / JaX   / cpu / Forward",
            "value": 0.000010623836600007052,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / Forward",
            "value": 0.000010618414500004292,
            "unit": "s"
          },
          {
            "name": "sum    / JaX   / cpu / BothRev",
            "value": 0.000009505044500019722,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / PreRev",
            "value": 0.000009256094800002755,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / PostRev",
            "value": 0.0000095456631000161,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / BothRev",
            "value": 0.000009615573699920788,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / JaXPipe / gpu / Primal",
            "value": 0.0010188502999881,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / JaX   / gpu / Primal",
            "value": 0.0010181756999372,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / HLOOpt / gpu / Primal",
            "value": 0.0009588386999894,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / PartOpt / gpu / Primal",
            "value": 0.0009595328998329,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / DefOpt / gpu / Primal",
            "value": 0.0006881713999973,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / IPartOpt / gpu / Primal",
            "value": 0.0009497868000835,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / IDefOpt / gpu / Primal",
            "value": 0.0007047128001431,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / JaX   / gpu / Forward",
            "value": 0.0012650197999391,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / JaXPipe / gpu / PostRev",
            "value": 0.0040265638001073,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / JaX   / gpu / BothRev",
            "value": 0.004030471400074,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / HLOOpt / gpu / PostRev",
            "value": 0.0040430701999866,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / PartOpt / gpu / PostRev",
            "value": 0.0041082302001086,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / DefOpt / gpu / PostRev",
            "value": 0.0022301827999399,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / IPartOpt / gpu / PostRev",
            "value": 0.0041123922001133,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / IDefOpt / gpu / PostRev",
            "value": 0.0025540420001561,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / JaXPipe / tpu / Primal",
            "value": 0.00008552799990866334,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / JaX   / tpu / Primal",
            "value": 0.00009704799958853982,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / HLOOpt / tpu / Primal",
            "value": 0.00009828399997786618,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / PartOpt / tpu / Primal",
            "value": 0.0001040369999827,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / DefOpt / tpu / Primal",
            "value": 0.00009933100009220652,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / IPartOpt / tpu / Primal",
            "value": 0.0000997780000034254,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / IDefOpt / tpu / Primal",
            "value": 0.00009857900004135444,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / JaX   / tpu / Forward",
            "value": 0.0001789469999494,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / JaXPipe / tpu / PostRev",
            "value": 0.0001996730003156,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / JaX   / tpu / BothRev",
            "value": 0.0002029668998147,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / HLOOpt / tpu / PostRev",
            "value": 0.0002013620003708,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / PartOpt / tpu / PostRev",
            "value": 0.0002020419997279,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / DefOpt / tpu / PostRev",
            "value": 0.0001945800002431,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / IPartOpt / tpu / PostRev",
            "value": 0.000194637000095,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / IDefOpt / tpu / PostRev",
            "value": 0.0001989140000659,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / JaXPipe / cpu / Primal",
            "value": 0.0000745606999771553,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / JaX   / cpu / Primal",
            "value": 0.00004693930004577851,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / HLOOpt / cpu / Primal",
            "value": 0.00006560740002896637,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / PartOpt / cpu / Primal",
            "value": 0.0000610910000432341,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / DefOpt / cpu / Primal",
            "value": 0.000056112699985533256,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / IPartOpt / cpu / Primal",
            "value": 0.00007039359998088912,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / IDefOpt / cpu / Primal",
            "value": 0.00006217230002221185,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / JaX   / cpu / Forward",
            "value": 0.0001253925000128,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / JaXPipe / cpu / PostRev",
            "value": 0.0001088297999558,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / JaX   / cpu / BothRev",
            "value": 0.0001564086000144,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / HLOOpt / cpu / PostRev",
            "value": 0.0001211591000355,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / PartOpt / cpu / PostRev",
            "value": 0.0001418984000338,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / DefOpt / cpu / PostRev",
            "value": 0.0001165472999673,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / IPartOpt / cpu / PostRev",
            "value": 0.0001367823999316,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / IDefOpt / cpu / PostRev",
            "value": 0.000118978699993,
            "unit": "s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "198982749+Copilot@users.noreply.github.com",
            "name": "copilot-swe-agent[bot]",
            "username": "Copilot"
          },
          "committer": {
            "email": "gh@wsmoses.com",
            "name": "William Moses",
            "username": "wsmoses"
          },
          "distinct": true,
          "id": "bb3983720271c8b91efe56af88a121d8c47c59cf",
          "message": "Replace template create patterns with Op::create pattern\n\nCo-authored-by: wsmoses <1260124+wsmoses@users.noreply.github.com>",
          "timestamp": "2025-12-13T21:50:12-06:00",
          "tree_id": "827c5397a8c3cd0bf4314b2b6d2b8517c853ba00",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/bb3983720271c8b91efe56af88a121d8c47c59cf"
        },
        "date": 1765695748041,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "actmtch / JaX   / cpu / Primal",
            "value": 0.000003935989899991909,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / Primal",
            "value": 0.0000038969108999936,
            "unit": "s"
          },
          {
            "name": "actmtch / JaX   / cpu / Forward",
            "value": 0.000006244940499982477,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / Forward",
            "value": 0.000007268571099984911,
            "unit": "s"
          },
          {
            "name": "actmtch / JaX   / cpu / BothRev",
            "value": 0.000006411144000003333,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / PreRev",
            "value": 0.000007325726799990661,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / PostRev",
            "value": 0.000006191166399958092,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / BothRev",
            "value": 0.000007468332500002361,
            "unit": "s"
          },
          {
            "name": "actmtch / JaX   / cpu / Primal",
            "value": 0.00000740782049979316,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / Primal",
            "value": 0.000007516916500026128,
            "unit": "s"
          },
          {
            "name": "actmtch / JaX   / cpu / Forward",
            "value": 0.000010705241400137312,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / Forward",
            "value": 0.00001426556549995439,
            "unit": "s"
          },
          {
            "name": "actmtch / JaX   / cpu / BothRev",
            "value": 0.000011346812099873203,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / PreRev",
            "value": 0.000013784571499854792,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / PostRev",
            "value": 0.00001082232250009838,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / BothRev",
            "value": 0.000014315120199898957,
            "unit": "s"
          },
          {
            "name": "actmtch / JaX   / gpu / Primal",
            "value": 0.0000810803175001638,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / gpu / Primal",
            "value": 0.00007633166480009095,
            "unit": "s"
          },
          {
            "name": "actmtch / JaX   / gpu / Forward",
            "value": 0.0001084318463999,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / gpu / Forward",
            "value": 0.0001030053077,
            "unit": "s"
          },
          {
            "name": "actmtch / JaX   / gpu / BothRev",
            "value": 0.0001041229152,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / gpu / PreRev",
            "value": 0.0001028877947999,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / gpu / PostRev",
            "value": 0.0001032997990998,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / gpu / BothRev",
            "value": 0.0001066338190998,
            "unit": "s"
          },
          {
            "name": "actmtch / JaX   / cpu / Primal",
            "value": 0.0000035030474999985017,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / Primal",
            "value": 0.0000036955794999812497,
            "unit": "s"
          },
          {
            "name": "actmtch / JaX   / cpu / Forward",
            "value": 0.000005486394699983066,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / Forward",
            "value": 0.000006343584799924429,
            "unit": "s"
          },
          {
            "name": "actmtch / JaX   / cpu / BothRev",
            "value": 0.000005185161699955643,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / PreRev",
            "value": 0.000006052387799991266,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / PostRev",
            "value": 0.00000549643960002868,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / BothRev",
            "value": 0.000006355960800010507,
            "unit": "s"
          },
          {
            "name": "actmtch / JaX   / tpu / Primal",
            "value": 0.0001555292342999,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / tpu / Primal",
            "value": 0.0001526204189,
            "unit": "s"
          },
          {
            "name": "actmtch / JaX   / tpu / Forward",
            "value": 0.0002265897162,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / tpu / Forward",
            "value": 0.0002180169631,
            "unit": "s"
          },
          {
            "name": "actmtch / JaX   / tpu / BothRev",
            "value": 0.0002117369663999,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / tpu / PreRev",
            "value": 0.0002130754095,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / tpu / PostRev",
            "value": 0.0002221445816999,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / tpu / BothRev",
            "value": 0.0002263404201999,
            "unit": "s"
          },
          {
            "name": "actmtch / JaX   / cpu / Primal",
            "value": 0.000005234974499944656,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / Primal",
            "value": 0.000005387769600019965,
            "unit": "s"
          },
          {
            "name": "actmtch / JaX   / cpu / Forward",
            "value": 0.000007545677000052819,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / Forward",
            "value": 0.000011407275600049615,
            "unit": "s"
          },
          {
            "name": "actmtch / JaX   / cpu / BothRev",
            "value": 0.000008036667000033048,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / PreRev",
            "value": 0.000009319593100008205,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / PostRev",
            "value": 0.000007491046300037851,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / BothRev",
            "value": 0.00000981074669998634,
            "unit": "s"
          },
          {
            "name": "add_one / JaX   / cpu / Primal",
            "value": 0.000004259401199988133,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / Primal",
            "value": 0.000004180993999989369,
            "unit": "s"
          },
          {
            "name": "add_one / JaX   / cpu / Forward",
            "value": 0.000007150962300011088,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / Forward",
            "value": 0.000007210154899985355,
            "unit": "s"
          },
          {
            "name": "add_one / JaX   / cpu / BothRev",
            "value": 0.000007578011399982642,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / PreRev",
            "value": 0.000007285204900017561,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / PostRev",
            "value": 0.000007180361299970172,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / BothRev",
            "value": 0.000007173504299998967,
            "unit": "s"
          },
          {
            "name": "add_one / JaX   / cpu / Primal",
            "value": 0.000007729755800028215,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / Primal",
            "value": 0.000007810098200206994,
            "unit": "s"
          },
          {
            "name": "add_one / JaX   / cpu / Forward",
            "value": 0.000011962959699667409,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / Forward",
            "value": 0.00001246550540017779,
            "unit": "s"
          },
          {
            "name": "add_one / JaX   / cpu / BothRev",
            "value": 0.000013059870899814995,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / PreRev",
            "value": 0.000013039436400140403,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / PostRev",
            "value": 0.000012848378000126103,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / BothRev",
            "value": 0.00001281376660008391,
            "unit": "s"
          },
          {
            "name": "add_one / JaX   / gpu / Primal",
            "value": 0.00007877420449985949,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / gpu / Primal",
            "value": 0.00008143914259999292,
            "unit": "s"
          },
          {
            "name": "add_one / JaX   / gpu / Forward",
            "value": 0.0001138067293999,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / gpu / Forward",
            "value": 0.0001059984526,
            "unit": "s"
          },
          {
            "name": "add_one / JaX   / gpu / BothRev",
            "value": 0.0001093323073997,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / gpu / PreRev",
            "value": 0.0001087607269,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / gpu / PostRev",
            "value": 0.0001096278380999,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / gpu / BothRev",
            "value": 0.0001135328707001,
            "unit": "s"
          },
          {
            "name": "add_one / JaX   / cpu / Primal",
            "value": 0.0000037815435000084106,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / Primal",
            "value": 0.000003789228500045283,
            "unit": "s"
          },
          {
            "name": "add_one / JaX   / cpu / Forward",
            "value": 0.000005841887699989457,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / Forward",
            "value": 0.000006122690699976374,
            "unit": "s"
          },
          {
            "name": "add_one / JaX   / cpu / BothRev",
            "value": 0.000006240361799973471,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / PreRev",
            "value": 0.00000616803679995428,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / PostRev",
            "value": 0.00000627595879996079,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / BothRev",
            "value": 0.000006217034800010878,
            "unit": "s"
          },
          {
            "name": "add_one / JaX   / tpu / Primal",
            "value": 0.0001418831986,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / tpu / Primal",
            "value": 0.0001511724827999,
            "unit": "s"
          },
          {
            "name": "add_one / JaX   / tpu / Forward",
            "value": 0.0002120691074,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / tpu / Forward",
            "value": 0.0002201888763,
            "unit": "s"
          },
          {
            "name": "add_one / JaX   / tpu / BothRev",
            "value": 0.0002156163268,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / tpu / PreRev",
            "value": 0.0002144280976999,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / tpu / PostRev",
            "value": 0.0002177218051,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / tpu / BothRev",
            "value": 0.0002146545687,
            "unit": "s"
          },
          {
            "name": "add_one / JaX   / cpu / Primal",
            "value": 0.000005280082000081166,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / Primal",
            "value": 0.000005407637500047713,
            "unit": "s"
          },
          {
            "name": "add_one / JaX   / cpu / Forward",
            "value": 0.000008426858799975889,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / Forward",
            "value": 0.000008471397900029842,
            "unit": "s"
          },
          {
            "name": "add_one / JaX   / cpu / BothRev",
            "value": 0.000008408516800045619,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / PreRev",
            "value": 0.000008982991900029446,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / PostRev",
            "value": 0.000008993429700058186,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / BothRev",
            "value": 0.000008873756800039701,
            "unit": "s"
          },
          {
            "name": "add_two / JaX   / cpu / Primal",
            "value": 0.000004405232499993872,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / Primal",
            "value": 0.000004453076300023895,
            "unit": "s"
          },
          {
            "name": "add_two / JaX   / cpu / Forward",
            "value": 0.000007228910599997107,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / Forward",
            "value": 0.000007291983100003562,
            "unit": "s"
          },
          {
            "name": "add_two / JaX   / cpu / BothRev",
            "value": 0.000009160446700025204,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / PreRev",
            "value": 0.000008848932099999729,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / PostRev",
            "value": 0.00000895968790000552,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / BothRev",
            "value": 0.000009247785399975328,
            "unit": "s"
          },
          {
            "name": "add_two / JaX   / cpu / Primal",
            "value": 0.000007915507399957279,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / Primal",
            "value": 0.000007761010900139809,
            "unit": "s"
          },
          {
            "name": "add_two / JaX   / cpu / Forward",
            "value": 0.00001226260870025726,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / Forward",
            "value": 0.000013019072899987804,
            "unit": "s"
          },
          {
            "name": "add_two / JaX   / cpu / BothRev",
            "value": 0.000015081487399947943,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / PreRev",
            "value": 0.000015102540800216958,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / PostRev",
            "value": 0.000015349834799781092,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / BothRev",
            "value": 0.000015375475399923743,
            "unit": "s"
          },
          {
            "name": "add_two / JaX   / gpu / Primal",
            "value": 0.00007805660719968727,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / gpu / Primal",
            "value": 0.00007877889899973525,
            "unit": "s"
          },
          {
            "name": "add_two / JaX   / gpu / Forward",
            "value": 0.0001060963617997,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / gpu / Forward",
            "value": 0.0001073848211999,
            "unit": "s"
          },
          {
            "name": "add_two / JaX   / gpu / BothRev",
            "value": 0.0001213753555999,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / gpu / PreRev",
            "value": 0.0001263721109997,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / gpu / PostRev",
            "value": 0.0001254233076997,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / gpu / BothRev",
            "value": 0.0001268099798002,
            "unit": "s"
          },
          {
            "name": "add_two / JaX   / cpu / Primal",
            "value": 0.000003876230499918165,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / Primal",
            "value": 0.000003827583499969478,
            "unit": "s"
          },
          {
            "name": "add_two / JaX   / cpu / Forward",
            "value": 0.000006025397800021892,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / Forward",
            "value": 0.00000630160579994481,
            "unit": "s"
          },
          {
            "name": "add_two / JaX   / cpu / BothRev",
            "value": 0.0000072142218999943,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / PreRev",
            "value": 0.000006926993800061609,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / PostRev",
            "value": 0.000007207089900020946,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / BothRev",
            "value": 0.000007175409899991791,
            "unit": "s"
          },
          {
            "name": "add_two / JaX   / tpu / Primal",
            "value": 0.0001338290885999,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / tpu / Primal",
            "value": 0.0001348636847,
            "unit": "s"
          },
          {
            "name": "add_two / JaX   / tpu / Forward",
            "value": 0.0002049803875,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / tpu / Forward",
            "value": 0.0001998201119,
            "unit": "s"
          },
          {
            "name": "add_two / JaX   / tpu / BothRev",
            "value": 0.00020882078,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / tpu / PreRev",
            "value": 0.0002083972499999,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / tpu / PostRev",
            "value": 0.0002099474250999,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / tpu / BothRev",
            "value": 0.0002127120403999,
            "unit": "s"
          },
          {
            "name": "add_two / JaX   / cpu / Primal",
            "value": 0.000005617902900030458,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / Primal",
            "value": 0.000006003769200015086,
            "unit": "s"
          },
          {
            "name": "add_two / JaX   / cpu / Forward",
            "value": 0.000008681744600016827,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / Forward",
            "value": 0.000011620221300017874,
            "unit": "s"
          },
          {
            "name": "add_two / JaX   / cpu / BothRev",
            "value": 0.000010907571500047196,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / PreRev",
            "value": 0.000011069568999937474,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / PostRev",
            "value": 0.000011191211400000612,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / BothRev",
            "value": 0.000010609101399950304,
            "unit": "s"
          },
          {
            "name": "cache / JaX   / cpu / Primal",
            "value": 0.000003922753300003024,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / Primal",
            "value": 0.000004232568399993397,
            "unit": "s"
          },
          {
            "name": "cache / JaX   / cpu / Forward",
            "value": 0.000010339387399972113,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / Forward",
            "value": 0.000010178420099964567,
            "unit": "s"
          },
          {
            "name": "cache / JaX   / cpu / BothRev",
            "value": 0.000015630533500007004,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / PreRev",
            "value": 0.000011397483199971247,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / PostRev",
            "value": 0.000014657190699972488,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / BothRev",
            "value": 0.000011851312299995698,
            "unit": "s"
          },
          {
            "name": "cache / JaX   / cpu / Primal",
            "value": 0.000007865725999727146,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / Primal",
            "value": 0.000008406263399956515,
            "unit": "s"
          },
          {
            "name": "cache / JaX   / cpu / Forward",
            "value": 0.000013655810199998089,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / Forward",
            "value": 0.000012928149599974858,
            "unit": "s"
          },
          {
            "name": "cache / JaX   / cpu / BothRev",
            "value": 0.000023126076700282283,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / PreRev",
            "value": 0.0000176461566999933,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / PostRev",
            "value": 0.00002419318120009848,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / BothRev",
            "value": 0.000017463980899992747,
            "unit": "s"
          },
          {
            "name": "cache / JaX   / gpu / Primal",
            "value": 0.00008121045510015392,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / gpu / Primal",
            "value": 0.000074185534299977,
            "unit": "s"
          },
          {
            "name": "cache / JaX   / gpu / Forward",
            "value": 0.0001133440797002,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / gpu / Forward",
            "value": 0.0001049986611997,
            "unit": "s"
          },
          {
            "name": "cache / JaX   / gpu / BothRev",
            "value": 0.0001047593428,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / gpu / PreRev",
            "value": 0.0001044106232999,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / gpu / PostRev",
            "value": 0.0001111918259997,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / gpu / BothRev",
            "value": 0.0001062530514001,
            "unit": "s"
          },
          {
            "name": "cache / JaX   / cpu / Primal",
            "value": 0.000003417496499969274,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / Primal",
            "value": 0.0000034936585000650664,
            "unit": "s"
          },
          {
            "name": "cache / JaX   / cpu / Forward",
            "value": 0.00002339880190002077,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / Forward",
            "value": 0.000022553927799981465,
            "unit": "s"
          },
          {
            "name": "cache / JaX   / cpu / BothRev",
            "value": 0.00003169753399997717,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / PreRev",
            "value": 0.00002650847229997453,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / PostRev",
            "value": 0.00003238875400002144,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / BothRev",
            "value": 0.00002747296939996886,
            "unit": "s"
          },
          {
            "name": "cache / JaX   / tpu / Primal",
            "value": 0.0001321666083999,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / tpu / Primal",
            "value": 0.0001360331939,
            "unit": "s"
          },
          {
            "name": "cache / JaX   / tpu / Forward",
            "value": 0.0002280197692999,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / tpu / Forward",
            "value": 0.0002261243141999,
            "unit": "s"
          },
          {
            "name": "cache / JaX   / tpu / BothRev",
            "value": 0.0002291170544999,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / tpu / PreRev",
            "value": 0.0002283014343999,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / tpu / PostRev",
            "value": 0.0002263779111,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / tpu / BothRev",
            "value": 0.0002250137319999,
            "unit": "s"
          },
          {
            "name": "cache / JaX   / cpu / Primal",
            "value": 0.000005003984799986938,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / Primal",
            "value": 0.000005036349400052132,
            "unit": "s"
          },
          {
            "name": "cache / JaX   / cpu / Forward",
            "value": 0.00002447639389993128,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / Forward",
            "value": 0.00002478361600005883,
            "unit": "s"
          },
          {
            "name": "cache / JaX   / cpu / BothRev",
            "value": 0.000020478015000026065,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / PreRev",
            "value": 0.000014891087999967566,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / PostRev",
            "value": 0.000023839686199971766,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / BothRev",
            "value": 0.000023557134799921183,
            "unit": "s"
          },
          {
            "name": "Concat / JaX   / cpu / Primal",
            "value": 0.00000423354740000832,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / Primal",
            "value": 0.0000042152592000093135,
            "unit": "s"
          },
          {
            "name": "Concat / JaX   / cpu / Forward",
            "value": 0.000007006618499963224,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / Forward",
            "value": 0.000007023140100000091,
            "unit": "s"
          },
          {
            "name": "Concat / JaX   / cpu / BothRev",
            "value": 0.000007308724799986521,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / PreRev",
            "value": 0.000007338694200007012,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / PostRev",
            "value": 0.000007232438099981663,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / BothRev",
            "value": 0.00000742403330000343,
            "unit": "s"
          },
          {
            "name": "Concat / JaX   / cpu / Primal",
            "value": 0.000007666090700149652,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / Primal",
            "value": 0.000007737706500120113,
            "unit": "s"
          },
          {
            "name": "Concat / JaX   / cpu / Forward",
            "value": 0.000012396286999864967,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / Forward",
            "value": 0.000012652058200183092,
            "unit": "s"
          },
          {
            "name": "Concat / JaX   / cpu / BothRev",
            "value": 0.000012213893799707876,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / PreRev",
            "value": 0.000012761617900105194,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / PostRev",
            "value": 0.000012848489899988636,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / BothRev",
            "value": 0.000012710217399944669,
            "unit": "s"
          },
          {
            "name": "Concat / JaX   / gpu / Primal",
            "value": 0.00007754830550002225,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / gpu / Primal",
            "value": 0.00007704054939968046,
            "unit": "s"
          },
          {
            "name": "Concat / JaX   / gpu / Forward",
            "value": 0.0001045616368999,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / gpu / Forward",
            "value": 0.0001087932423997,
            "unit": "s"
          },
          {
            "name": "Concat / JaX   / gpu / BothRev",
            "value": 0.00009878375319967744,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / gpu / PreRev",
            "value": 0.00009861480079998728,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / gpu / PostRev",
            "value": 0.000098672245799753,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / gpu / BothRev",
            "value": 0.00009705555799991998,
            "unit": "s"
          },
          {
            "name": "Concat / JaX   / cpu / Primal",
            "value": 0.000003768266499992024,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / Primal",
            "value": 0.000003719634400022187,
            "unit": "s"
          },
          {
            "name": "Concat / JaX   / cpu / Forward",
            "value": 0.000005642627699944569,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / Forward",
            "value": 0.000005959253699984401,
            "unit": "s"
          },
          {
            "name": "Concat / JaX   / cpu / BothRev",
            "value": 0.00000588942770000358,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / PreRev",
            "value": 0.000006279228800030978,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / PostRev",
            "value": 0.000005783654700007901,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / BothRev",
            "value": 0.000006182870700013154,
            "unit": "s"
          },
          {
            "name": "Concat / JaX   / tpu / Primal",
            "value": 0.0001541618401999,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / tpu / Primal",
            "value": 0.0001550240233,
            "unit": "s"
          },
          {
            "name": "Concat / JaX   / tpu / Forward",
            "value": 0.0002256862709999,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / tpu / Forward",
            "value": 0.0002109014681999,
            "unit": "s"
          },
          {
            "name": "Concat / JaX   / tpu / BothRev",
            "value": 0.0002156640397999,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / tpu / PreRev",
            "value": 0.0002218912525999,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / tpu / PostRev",
            "value": 0.0002344614611999,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / tpu / BothRev",
            "value": 0.0002114943692,
            "unit": "s"
          },
          {
            "name": "Concat / JaX   / cpu / Primal",
            "value": 0.000005337549899923033,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / Primal",
            "value": 0.000005214129099931597,
            "unit": "s"
          },
          {
            "name": "Concat / JaX   / cpu / Forward",
            "value": 0.000008887079200030713,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / Forward",
            "value": 0.000008784269300031155,
            "unit": "s"
          },
          {
            "name": "Concat / JaX   / cpu / BothRev",
            "value": 0.000008692721599982178,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / PreRev",
            "value": 0.000009189378799965195,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / PostRev",
            "value": 0.000009356790800029557,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / BothRev",
            "value": 0.000009122414400007985,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaX   / cpu / Primal",
            "value": 0.000008356499984074617,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / cpu / Primal",
            "value": 0.000007680199996684677,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaX   / cpu / Forward",
            "value": 0.000009154899998975452,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / cpu / Forward",
            "value": 0.000009877999991658724,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaX   / cpu / Primal",
            "value": 0.000012535900168586522,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / cpu / Primal",
            "value": 0.000011970600098720752,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaX   / cpu / Forward",
            "value": 0.000014952699712011962,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / cpu / Forward",
            "value": 0.000017646500054979697,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaX   / gpu / Primal",
            "value": 0.0001251323999895,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / gpu / Primal",
            "value": 0.0001095126001018,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaX   / gpu / Forward",
            "value": 0.0001464335000491,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / gpu / Forward",
            "value": 0.0001428445000783,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaX   / cpu / Primal",
            "value": 0.00000713499994162703,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / cpu / Primal",
            "value": 0.000005313000019668834,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaX   / cpu / Forward",
            "value": 0.000007328999981837114,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / cpu / Forward",
            "value": 0.000008338000043295324,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaX   / tpu / Primal",
            "value": 0.0001706009999907,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / tpu / Primal",
            "value": 0.0001750239999637,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaX   / tpu / Forward",
            "value": 0.0002552540999204,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / tpu / Forward",
            "value": 0.0002512879999812,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaX   / cpu / Primal",
            "value": 0.00001076260004992946,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / cpu / Primal",
            "value": 0.000009598599990567892,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaX   / cpu / Forward",
            "value": 0.000010861800001293888,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / cpu / Forward",
            "value": 0.000012402000083966414,
            "unit": "s"
          },
          {
            "name": "GenDot / JaX   / cpu / Primal",
            "value": 0.000004686706800021057,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / Primal",
            "value": 0.000004451252699982433,
            "unit": "s"
          },
          {
            "name": "GenDot / JaX   / cpu / Forward",
            "value": 0.000006631589200014787,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / Forward",
            "value": 0.000007169623299978412,
            "unit": "s"
          },
          {
            "name": "GenDot / JaX   / cpu / BothRev",
            "value": 0.000006586491100006242,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / PreRev",
            "value": 0.0000073758642000029796,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / PostRev",
            "value": 0.000006608066800026791,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / BothRev",
            "value": 0.000007345668400012073,
            "unit": "s"
          },
          {
            "name": "GenDot / JaX   / cpu / Primal",
            "value": 0.00000798068570002215,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / Primal",
            "value": 0.000008463510900037364,
            "unit": "s"
          },
          {
            "name": "GenDot / JaX   / cpu / Forward",
            "value": 0.000011215387899937924,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / Forward",
            "value": 0.000014310497000042233,
            "unit": "s"
          },
          {
            "name": "GenDot / JaX   / cpu / BothRev",
            "value": 0.00001215368560006027,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / PreRev",
            "value": 0.00001427645060030045,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / PostRev",
            "value": 0.000012367568400077287,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / BothRev",
            "value": 0.000014295722799943178,
            "unit": "s"
          },
          {
            "name": "GenDot / JaX   / gpu / Primal",
            "value": 0.00007463315240020166,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / gpu / Primal",
            "value": 0.00007414261119993171,
            "unit": "s"
          },
          {
            "name": "GenDot / JaX   / gpu / Forward",
            "value": 0.0001045971755997,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / gpu / Forward",
            "value": 0.0001059973037998,
            "unit": "s"
          },
          {
            "name": "GenDot / JaX   / gpu / BothRev",
            "value": 0.0001039953690997,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / gpu / PreRev",
            "value": 0.0001074619289,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / gpu / PostRev",
            "value": 0.0001020258289998,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / gpu / BothRev",
            "value": 0.0001263560584,
            "unit": "s"
          },
          {
            "name": "GenDot / JaX   / cpu / Primal",
            "value": 0.000004104306499993982,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / Primal",
            "value": 0.000003918636500020512,
            "unit": "s"
          },
          {
            "name": "GenDot / JaX   / cpu / Forward",
            "value": 0.000005226478700024018,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / Forward",
            "value": 0.000006376141800046753,
            "unit": "s"
          },
          {
            "name": "GenDot / JaX   / cpu / BothRev",
            "value": 0.000005629822699938813,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / PreRev",
            "value": 0.000006064300700018066,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / PostRev",
            "value": 0.000005707127699952253,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / BothRev",
            "value": 0.000006373491800059128,
            "unit": "s"
          },
          {
            "name": "GenDot / JaX   / tpu / Primal",
            "value": 0.0001597905448999,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / tpu / Primal",
            "value": 0.0001485523084999,
            "unit": "s"
          },
          {
            "name": "GenDot / JaX   / tpu / Forward",
            "value": 0.0002235678838,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / tpu / Forward",
            "value": 0.0002147153756999,
            "unit": "s"
          },
          {
            "name": "GenDot / JaX   / tpu / BothRev",
            "value": 0.0002253641629999,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / tpu / PreRev",
            "value": 0.0002185973532,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / tpu / PostRev",
            "value": 0.0002162538219,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / tpu / BothRev",
            "value": 0.0002104169191999,
            "unit": "s"
          },
          {
            "name": "GenDot / JaX   / cpu / Primal",
            "value": 0.0000054452003999358565,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / Primal",
            "value": 0.000005981080299989117,
            "unit": "s"
          },
          {
            "name": "GenDot / JaX   / cpu / Forward",
            "value": 0.00000847384829994553,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / Forward",
            "value": 0.000009912036399964564,
            "unit": "s"
          },
          {
            "name": "GenDot / JaX   / cpu / BothRev",
            "value": 0.00000847558260002188,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / PreRev",
            "value": 0.00000988996070000212,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / PostRev",
            "value": 0.000008181467200029147,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / BothRev",
            "value": 0.000009827009299988276,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaX   / cpu / Primal",
            "value": 0.000006168208300005062,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / Primal",
            "value": 0.000005936786099982783,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaX   / cpu / Forward",
            "value": 0.000009950193199983914,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / Forward",
            "value": 0.000009914378799976477,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaX   / cpu / BothRev",
            "value": 0.000009365203799961818,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / PreRev",
            "value": 0.000009397579599999516,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / PostRev",
            "value": 0.000009701209800005016,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / BothRev",
            "value": 0.00000943634039999779,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaX   / cpu / Primal",
            "value": 0.000010633199799849537,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / Primal",
            "value": 0.000010156823800207348,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaX   / cpu / Forward",
            "value": 0.000015378811199843766,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / Forward",
            "value": 0.00001594945840006403,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaX   / cpu / BothRev",
            "value": 0.00001500100260018371,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / PreRev",
            "value": 0.000015503992799858678,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / PostRev",
            "value": 0.00001502164020021155,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / BothRev",
            "value": 0.000015578925799854916,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaX   / cpu / Primal",
            "value": 0.000005121234499983984,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / Primal",
            "value": 0.000004887398400023813,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaX   / cpu / Forward",
            "value": 0.000007625346699933289,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / Forward",
            "value": 0.000007901028800006316,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaX   / cpu / BothRev",
            "value": 0.000007335956699989765,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / PreRev",
            "value": 0.00000764955770000597,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / PostRev",
            "value": 0.000007684465799957251,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / BothRev",
            "value": 0.000007655510699987644,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaX   / tpu / Primal",
            "value": 0.00006580474359998334,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / tpu / Primal",
            "value": 0.00006641348810007912,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaX   / tpu / Forward",
            "value": 0.00009053783159997691,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / tpu / Forward",
            "value": 0.00009167972369996278,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaX   / tpu / BothRev",
            "value": 0.00009009743650003656,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / tpu / PreRev",
            "value": 0.00008869690040000933,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / tpu / PostRev",
            "value": 0.00008962293450003926,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / tpu / BothRev",
            "value": 0.00008754988629998479,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaX   / cpu / Primal",
            "value": 0.00000765354949999164,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / Primal",
            "value": 0.000007279041200035863,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaX   / cpu / Forward",
            "value": 0.000011270057899946551,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / Forward",
            "value": 0.000011930867099999887,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaX   / cpu / BothRev",
            "value": 0.00001106904049993318,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / PreRev",
            "value": 0.000014027972900021269,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / PostRev",
            "value": 0.000011356376300045668,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / BothRev",
            "value": 0.000011303273100020303,
            "unit": "s"
          },
          {
            "name": "llama / JaXPipe / cpu / Primal",
            "value": 0.0009264197899983,
            "unit": "s"
          },
          {
            "name": "llama / JaX   / cpu / Primal",
            "value": 0.000919607779997,
            "unit": "s"
          },
          {
            "name": "llama / HLOOpt / cpu / Primal",
            "value": 0.0010412843999984,
            "unit": "s"
          },
          {
            "name": "llama / PartOpt / cpu / Primal",
            "value": 0.0009601453499999,
            "unit": "s"
          },
          {
            "name": "llama / DefOpt / cpu / Primal",
            "value": 0.0010416875600003,
            "unit": "s"
          },
          {
            "name": "llama / IPartOpt / cpu / Primal",
            "value": 0.000936633010001,
            "unit": "s"
          },
          {
            "name": "llama / IDefOpt / cpu / Primal",
            "value": 0.0010555406699995,
            "unit": "s"
          },
          {
            "name": "llama / JaXPipe / cpu / Forward",
            "value": 0.0028179615499993,
            "unit": "s"
          },
          {
            "name": "llama / JaX   / cpu / Forward",
            "value": 0.0027775640299978,
            "unit": "s"
          },
          {
            "name": "llama / HLOOpt / cpu / Forward",
            "value": 0.0026907549499992,
            "unit": "s"
          },
          {
            "name": "llama / PartOpt / cpu / Forward",
            "value": 0.0026706383999999,
            "unit": "s"
          },
          {
            "name": "llama / DefOpt / cpu / Forward",
            "value": 0.0026408617200013,
            "unit": "s"
          },
          {
            "name": "llama / IPartOpt / cpu / Forward",
            "value": 0.0027876785399985,
            "unit": "s"
          },
          {
            "name": "llama / IDefOpt / cpu / Forward",
            "value": 0.0028345087200023,
            "unit": "s"
          },
          {
            "name": "llama / JaXPipe / cpu / PreRev",
            "value": 0.0024003109099976,
            "unit": "s"
          },
          {
            "name": "llama / JaXPipe / cpu / PostRev",
            "value": 0.0023600027499969,
            "unit": "s"
          },
          {
            "name": "llama / JaXPipe / cpu / BothRev",
            "value": 0.002369193290001,
            "unit": "s"
          },
          {
            "name": "llama / JaX   / cpu / BothRev",
            "value": 0.002427909009998,
            "unit": "s"
          },
          {
            "name": "llama / HLOOpt / cpu / PreRev",
            "value": 0.0025309068000024,
            "unit": "s"
          },
          {
            "name": "llama / HLOOpt / cpu / PostRev",
            "value": 0.0024625289799996,
            "unit": "s"
          },
          {
            "name": "llama / HLOOpt / cpu / BothRev",
            "value": 0.0025146137000001,
            "unit": "s"
          },
          {
            "name": "llama / PartOpt / cpu / PreRev",
            "value": 0.0024628951500017,
            "unit": "s"
          },
          {
            "name": "llama / PartOpt / cpu / PostRev",
            "value": 0.00251218203,
            "unit": "s"
          },
          {
            "name": "llama / PartOpt / cpu / BothRev",
            "value": 0.0025367441799971,
            "unit": "s"
          },
          {
            "name": "llama / DefOpt / cpu / PreRev",
            "value": 0.0024606511699994,
            "unit": "s"
          },
          {
            "name": "llama / DefOpt / cpu / PostRev",
            "value": 0.0024741650099986,
            "unit": "s"
          },
          {
            "name": "llama / DefOpt / cpu / BothRev",
            "value": 0.0023797439899999,
            "unit": "s"
          },
          {
            "name": "llama / IPartOpt / cpu / PreRev",
            "value": 0.0027227103200038,
            "unit": "s"
          },
          {
            "name": "llama / IPartOpt / cpu / PostRev",
            "value": 0.002558178600002,
            "unit": "s"
          },
          {
            "name": "llama / IPartOpt / cpu / BothRev",
            "value": 0.0025605320099975,
            "unit": "s"
          },
          {
            "name": "llama / IDefOpt / cpu / PreRev",
            "value": 0.0025464439899997,
            "unit": "s"
          },
          {
            "name": "llama / IDefOpt / cpu / PostRev",
            "value": 0.0023908290499957,
            "unit": "s"
          },
          {
            "name": "llama / IDefOpt / cpu / BothRev",
            "value": 0.0026756317399986,
            "unit": "s"
          },
          {
            "name": "llama / JaXPipe / gpu / Primal",
            "value": 0.0004380538639961,
            "unit": "s"
          },
          {
            "name": "llama / JaX   / gpu / Primal",
            "value": 0.0004299214799975,
            "unit": "s"
          },
          {
            "name": "llama / HLOOpt / gpu / Primal",
            "value": 0.0004359674079969,
            "unit": "s"
          },
          {
            "name": "llama / PartOpt / gpu / Primal",
            "value": 0.0004390834199948,
            "unit": "s"
          },
          {
            "name": "llama / DefOpt / gpu / Primal",
            "value": 0.0004289257620039,
            "unit": "s"
          },
          {
            "name": "llama / IPartOpt / gpu / Primal",
            "value": 0.0004415484880009,
            "unit": "s"
          },
          {
            "name": "llama / IDefOpt / gpu / Primal",
            "value": 0.0004380797959966,
            "unit": "s"
          },
          {
            "name": "llama / JaXPipe / gpu / Forward",
            "value": 0.0007373580979983,
            "unit": "s"
          },
          {
            "name": "llama / JaX   / gpu / Forward",
            "value": 0.0007310638100025,
            "unit": "s"
          },
          {
            "name": "llama / HLOOpt / gpu / Forward",
            "value": 0.0007371923160026,
            "unit": "s"
          },
          {
            "name": "llama / PartOpt / gpu / Forward",
            "value": 0.0007162623940021,
            "unit": "s"
          },
          {
            "name": "llama / DefOpt / gpu / Forward",
            "value": 0.0007209243199977,
            "unit": "s"
          },
          {
            "name": "llama / IPartOpt / gpu / Forward",
            "value": 0.0007385722699982,
            "unit": "s"
          },
          {
            "name": "llama / IDefOpt / gpu / Forward",
            "value": 0.000732064797994,
            "unit": "s"
          },
          {
            "name": "llama / JaXPipe / gpu / PreRev",
            "value": 0.0007991179300006,
            "unit": "s"
          },
          {
            "name": "llama / JaXPipe / gpu / PostRev",
            "value": 0.000809604646005,
            "unit": "s"
          },
          {
            "name": "llama / JaXPipe / gpu / BothRev",
            "value": 0.0007722584500006,
            "unit": "s"
          },
          {
            "name": "llama / JaX   / gpu / BothRev",
            "value": 0.0008287035539979,
            "unit": "s"
          },
          {
            "name": "llama / HLOOpt / gpu / PreRev",
            "value": 0.0008256826900033,
            "unit": "s"
          },
          {
            "name": "llama / HLOOpt / gpu / PostRev",
            "value": 0.000807207470003,
            "unit": "s"
          },
          {
            "name": "llama / HLOOpt / gpu / BothRev",
            "value": 0.0008022522800019,
            "unit": "s"
          },
          {
            "name": "llama / PartOpt / gpu / PreRev",
            "value": 0.0008008463959995,
            "unit": "s"
          },
          {
            "name": "llama / PartOpt / gpu / PostRev",
            "value": 0.0008141954620004,
            "unit": "s"
          },
          {
            "name": "llama / PartOpt / gpu / BothRev",
            "value": 0.0008156599499998,
            "unit": "s"
          },
          {
            "name": "llama / DefOpt / gpu / PreRev",
            "value": 0.0007698629740043,
            "unit": "s"
          },
          {
            "name": "llama / DefOpt / gpu / PostRev",
            "value": 0.000776568053996,
            "unit": "s"
          },
          {
            "name": "llama / DefOpt / gpu / BothRev",
            "value": 0.0007719280520032,
            "unit": "s"
          },
          {
            "name": "llama / IPartOpt / gpu / PreRev",
            "value": 0.000770981734,
            "unit": "s"
          },
          {
            "name": "llama / IPartOpt / gpu / PostRev",
            "value": 0.0008255853360024,
            "unit": "s"
          },
          {
            "name": "llama / IPartOpt / gpu / BothRev",
            "value": 0.000767006946,
            "unit": "s"
          },
          {
            "name": "llama / IDefOpt / gpu / PreRev",
            "value": 0.0008029240659961,
            "unit": "s"
          },
          {
            "name": "llama / IDefOpt / gpu / PostRev",
            "value": 0.0008346296180025,
            "unit": "s"
          },
          {
            "name": "llama / IDefOpt / gpu / BothRev",
            "value": 0.0008307905940018,
            "unit": "s"
          },
          {
            "name": "llama / JaXPipe / tpu / Primal",
            "value": 0.0003648689239998,
            "unit": "s"
          },
          {
            "name": "llama / JaX   / tpu / Primal",
            "value": 0.0003477619619989,
            "unit": "s"
          },
          {
            "name": "llama / HLOOpt / tpu / Primal",
            "value": 0.000343355424,
            "unit": "s"
          },
          {
            "name": "llama / PartOpt / tpu / Primal",
            "value": 0.000356040224,
            "unit": "s"
          },
          {
            "name": "llama / DefOpt / tpu / Primal",
            "value": 0.0003420255419987,
            "unit": "s"
          },
          {
            "name": "llama / IPartOpt / tpu / Primal",
            "value": 0.0003718146480005,
            "unit": "s"
          },
          {
            "name": "llama / IDefOpt / tpu / Primal",
            "value": 0.0003460656419993,
            "unit": "s"
          },
          {
            "name": "llama / JaXPipe / tpu / Forward",
            "value": 0.0005609599300005,
            "unit": "s"
          },
          {
            "name": "llama / JaX   / tpu / Forward",
            "value": 0.0006866480660009,
            "unit": "s"
          },
          {
            "name": "llama / HLOOpt / tpu / Forward",
            "value": 0.0005524864479993,
            "unit": "s"
          },
          {
            "name": "llama / PartOpt / tpu / Forward",
            "value": 0.0005470957479992,
            "unit": "s"
          },
          {
            "name": "llama / DefOpt / tpu / Forward",
            "value": 0.0005570634279993,
            "unit": "s"
          },
          {
            "name": "llama / IPartOpt / tpu / Forward",
            "value": 0.0005393770480004,
            "unit": "s"
          },
          {
            "name": "llama / IDefOpt / tpu / Forward",
            "value": 0.0005375789080007,
            "unit": "s"
          },
          {
            "name": "llama / JaXPipe / tpu / PreRev",
            "value": 0.0004067289319991,
            "unit": "s"
          },
          {
            "name": "llama / JaXPipe / tpu / PostRev",
            "value": 0.000357656606,
            "unit": "s"
          },
          {
            "name": "llama / JaXPipe / tpu / BothRev",
            "value": 0.0003918577479998,
            "unit": "s"
          },
          {
            "name": "llama / JaX   / tpu / BothRev",
            "value": 0.0003578938259997,
            "unit": "s"
          },
          {
            "name": "llama / HLOOpt / tpu / PreRev",
            "value": 0.0003915822080016,
            "unit": "s"
          },
          {
            "name": "llama / HLOOpt / tpu / PostRev",
            "value": 0.0003904554279997,
            "unit": "s"
          },
          {
            "name": "llama / HLOOpt / tpu / BothRev",
            "value": 0.0003918835700005,
            "unit": "s"
          },
          {
            "name": "llama / PartOpt / tpu / PreRev",
            "value": 0.000391715407999,
            "unit": "s"
          },
          {
            "name": "llama / PartOpt / tpu / PostRev",
            "value": 0.0003720656460009,
            "unit": "s"
          },
          {
            "name": "llama / PartOpt / tpu / BothRev",
            "value": 0.0003909968280004,
            "unit": "s"
          },
          {
            "name": "llama / DefOpt / tpu / PreRev",
            "value": 0.0003914394700004,
            "unit": "s"
          },
          {
            "name": "llama / DefOpt / tpu / PostRev",
            "value": 0.0003830314879996,
            "unit": "s"
          },
          {
            "name": "llama / DefOpt / tpu / BothRev",
            "value": 0.0003915101080001,
            "unit": "s"
          },
          {
            "name": "llama / IPartOpt / tpu / PreRev",
            "value": 0.000391859629999,
            "unit": "s"
          },
          {
            "name": "llama / IPartOpt / tpu / PostRev",
            "value": 0.0003717591460008,
            "unit": "s"
          },
          {
            "name": "llama / IPartOpt / tpu / BothRev",
            "value": 0.0003919172679998,
            "unit": "s"
          },
          {
            "name": "llama / IDefOpt / tpu / PreRev",
            "value": 0.0003915879879987,
            "unit": "s"
          },
          {
            "name": "llama / IDefOpt / tpu / PostRev",
            "value": 0.0003991450699995,
            "unit": "s"
          },
          {
            "name": "llama / IDefOpt / tpu / BothRev",
            "value": 0.0003917922679993,
            "unit": "s"
          },
          {
            "name": "llama / JaXPipe / cpu / Primal",
            "value": 0.0024976179399982,
            "unit": "s"
          },
          {
            "name": "llama / JaX   / cpu / Primal",
            "value": 0.0024023491200023,
            "unit": "s"
          },
          {
            "name": "llama / HLOOpt / cpu / Primal",
            "value": 0.0024423539800045,
            "unit": "s"
          },
          {
            "name": "llama / PartOpt / cpu / Primal",
            "value": 0.0024161131099936,
            "unit": "s"
          },
          {
            "name": "llama / DefOpt / cpu / Primal",
            "value": 0.0024066981299984,
            "unit": "s"
          },
          {
            "name": "llama / IPartOpt / cpu / Primal",
            "value": 0.0024245429700022,
            "unit": "s"
          },
          {
            "name": "llama / IDefOpt / cpu / Primal",
            "value": 0.0023978968900064,
            "unit": "s"
          },
          {
            "name": "llama / JaXPipe / cpu / Forward",
            "value": 0.0072334919900004,
            "unit": "s"
          },
          {
            "name": "llama / JaX   / cpu / Forward",
            "value": 0.0074719863800055,
            "unit": "s"
          },
          {
            "name": "llama / HLOOpt / cpu / Forward",
            "value": 0.0074336547500024,
            "unit": "s"
          },
          {
            "name": "llama / PartOpt / cpu / Forward",
            "value": 0.0073996124099994,
            "unit": "s"
          },
          {
            "name": "llama / DefOpt / cpu / Forward",
            "value": 0.0073527198599913,
            "unit": "s"
          },
          {
            "name": "llama / IPartOpt / cpu / Forward",
            "value": 0.0074811655699977,
            "unit": "s"
          },
          {
            "name": "llama / IDefOpt / cpu / Forward",
            "value": 0.0074471851999987,
            "unit": "s"
          },
          {
            "name": "llama / JaXPipe / cpu / PreRev",
            "value": 0.0063640083700011,
            "unit": "s"
          },
          {
            "name": "llama / JaXPipe / cpu / PostRev",
            "value": 0.0063373358300032,
            "unit": "s"
          },
          {
            "name": "llama / JaXPipe / cpu / BothRev",
            "value": 0.006767499739999,
            "unit": "s"
          },
          {
            "name": "llama / JaX   / cpu / BothRev",
            "value": 0.0062543021300007,
            "unit": "s"
          },
          {
            "name": "llama / HLOOpt / cpu / PreRev",
            "value": 0.0064941424700009,
            "unit": "s"
          },
          {
            "name": "llama / HLOOpt / cpu / PostRev",
            "value": 0.0062977001600029,
            "unit": "s"
          },
          {
            "name": "llama / HLOOpt / cpu / BothRev",
            "value": 0.0063270228100009,
            "unit": "s"
          },
          {
            "name": "llama / PartOpt / cpu / PreRev",
            "value": 0.006731196669998,
            "unit": "s"
          },
          {
            "name": "llama / PartOpt / cpu / PostRev",
            "value": 0.0064767904399923,
            "unit": "s"
          },
          {
            "name": "llama / PartOpt / cpu / BothRev",
            "value": 0.006617894029996,
            "unit": "s"
          },
          {
            "name": "llama / DefOpt / cpu / PreRev",
            "value": 0.0063734385400039,
            "unit": "s"
          },
          {
            "name": "llama / DefOpt / cpu / PostRev",
            "value": 0.0064469466300033,
            "unit": "s"
          },
          {
            "name": "llama / DefOpt / cpu / BothRev",
            "value": 0.0065783934399951,
            "unit": "s"
          },
          {
            "name": "llama / IPartOpt / cpu / PreRev",
            "value": 0.0066822847700041,
            "unit": "s"
          },
          {
            "name": "llama / IPartOpt / cpu / PostRev",
            "value": 0.0062377234300038,
            "unit": "s"
          },
          {
            "name": "llama / IPartOpt / cpu / BothRev",
            "value": 0.00672303714,
            "unit": "s"
          },
          {
            "name": "llama / IDefOpt / cpu / PreRev",
            "value": 0.0063416430500001,
            "unit": "s"
          },
          {
            "name": "llama / IDefOpt / cpu / PostRev",
            "value": 0.0066902855399985,
            "unit": "s"
          },
          {
            "name": "llama / IDefOpt / cpu / BothRev",
            "value": 0.0064977127200018,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004813113000000158,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004827188699982799,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000009535172799951397,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000008999622100236593,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / gpu / Primal",
            "value": 0.00007819565889985825,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / gpu / Primal",
            "value": 0.00008032242319968646,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004407082500074466,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004422867499943095,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001594897448999,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001551899883,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000006278184999973746,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000006323931499991886,
            "unit": "s"
          },
          {
            "name": "slicing / JaX   / cpu / Primal",
            "value": 0.000003808293399970353,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / Primal",
            "value": 0.000003816485300012573,
            "unit": "s"
          },
          {
            "name": "slicing / JaX   / cpu / Forward",
            "value": 0.000005888684399997146,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / Forward",
            "value": 0.000005881901900011144,
            "unit": "s"
          },
          {
            "name": "slicing / JaX   / cpu / BothRev",
            "value": 0.000006323026199970627,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / PreRev",
            "value": 0.000006329546899996785,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / PostRev",
            "value": 0.000006361815499985824,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / BothRev",
            "value": 0.0000063229329000023424,
            "unit": "s"
          },
          {
            "name": "slicing / JaX   / cpu / Primal",
            "value": 0.000007082442799946875,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / Primal",
            "value": 0.000007398777200069162,
            "unit": "s"
          },
          {
            "name": "slicing / JaX   / cpu / Forward",
            "value": 0.000010247711500051085,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / Forward",
            "value": 0.000010982851299922914,
            "unit": "s"
          },
          {
            "name": "slicing / JaX   / cpu / BothRev",
            "value": 0.000011171678200116733,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / PreRev",
            "value": 0.000011203958800251712,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / PostRev",
            "value": 0.000011187692399835214,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / BothRev",
            "value": 0.000011098366599981093,
            "unit": "s"
          },
          {
            "name": "slicing / JaX   / gpu / Primal",
            "value": 0.00007416690430000016,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / gpu / Primal",
            "value": 0.00007883164939994458,
            "unit": "s"
          },
          {
            "name": "slicing / JaX   / gpu / Forward",
            "value": 0.0001022122555998,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / gpu / Forward",
            "value": 0.0001013894181,
            "unit": "s"
          },
          {
            "name": "slicing / JaX   / gpu / BothRev",
            "value": 0.0001039356283999,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / gpu / PreRev",
            "value": 0.0001078292715999,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / gpu / PostRev",
            "value": 0.0001027468459,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / gpu / BothRev",
            "value": 0.0001296970685001,
            "unit": "s"
          },
          {
            "name": "slicing / JaX   / cpu / Primal",
            "value": 0.000003341891400032182,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / Primal",
            "value": 0.000003695954500017251,
            "unit": "s"
          },
          {
            "name": "slicing / JaX   / cpu / Forward",
            "value": 0.000004908864600020024,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / Forward",
            "value": 0.000005220084699976724,
            "unit": "s"
          },
          {
            "name": "slicing / JaX   / cpu / BothRev",
            "value": 0.00000562946070003818,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / PreRev",
            "value": 0.000005249302700030967,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / PostRev",
            "value": 0.000005630648700025631,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / BothRev",
            "value": 0.000005627748699953372,
            "unit": "s"
          },
          {
            "name": "slicing / JaX   / tpu / Primal",
            "value": 0.0001509427648,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / tpu / Primal",
            "value": 0.0001454291060999,
            "unit": "s"
          },
          {
            "name": "slicing / JaX   / tpu / Forward",
            "value": 0.0001980444285999,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / tpu / Forward",
            "value": 0.0001969822445,
            "unit": "s"
          },
          {
            "name": "slicing / JaX   / tpu / BothRev",
            "value": 0.0001968386834999,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / tpu / PreRev",
            "value": 0.0001988925467999,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / tpu / PostRev",
            "value": 0.0001942631900999,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / tpu / BothRev",
            "value": 0.0001933476600999,
            "unit": "s"
          },
          {
            "name": "slicing / JaX   / cpu / Primal",
            "value": 0.000004892243599988433,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / Primal",
            "value": 0.000004901276100008545,
            "unit": "s"
          },
          {
            "name": "slicing / JaX   / cpu / Forward",
            "value": 0.000007240539199938212,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / Forward",
            "value": 0.00000731692560002557,
            "unit": "s"
          },
          {
            "name": "slicing / JaX   / cpu / BothRev",
            "value": 0.000007668731400008255,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / PreRev",
            "value": 0.000007683428200016351,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / PostRev",
            "value": 0.000007795620599972608,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / BothRev",
            "value": 0.000008251582299999427,
            "unit": "s"
          },
          {
            "name": "sum    / JaX   / cpu / Primal",
            "value": 0.000004996084300000803,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / Primal",
            "value": 0.000005020378600011099,
            "unit": "s"
          },
          {
            "name": "sum    / JaX   / cpu / Forward",
            "value": 0.000008238468199988346,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / Forward",
            "value": 0.000008300342400025328,
            "unit": "s"
          },
          {
            "name": "sum    / JaX   / cpu / BothRev",
            "value": 0.000007551606200013338,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / PreRev",
            "value": 0.000007239456399975097,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / PostRev",
            "value": 0.000007268517100010285,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / BothRev",
            "value": 0.00000722197629997936,
            "unit": "s"
          },
          {
            "name": "sum    / JaX   / cpu / Primal",
            "value": 0.000010077771800206392,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / Primal",
            "value": 0.000010036578099970938,
            "unit": "s"
          },
          {
            "name": "sum    / JaX   / cpu / Forward",
            "value": 0.000015792964399952326,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / Forward",
            "value": 0.000015674774799845182,
            "unit": "s"
          },
          {
            "name": "sum    / JaX   / cpu / BothRev",
            "value": 0.0000142391850000422,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / PreRev",
            "value": 0.000013806967999698828,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / PostRev",
            "value": 0.000014457539100112628,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / BothRev",
            "value": 0.000013722311500168872,
            "unit": "s"
          },
          {
            "name": "sum    / JaX   / gpu / Primal",
            "value": 0.00008294505870035209,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / gpu / Primal",
            "value": 0.0000795090635001543,
            "unit": "s"
          },
          {
            "name": "sum    / JaX   / gpu / Forward",
            "value": 0.0000985774502998538,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / gpu / Forward",
            "value": 0.000099367179699766,
            "unit": "s"
          },
          {
            "name": "sum    / JaX   / gpu / BothRev",
            "value": 0.0001014185667001,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / gpu / PreRev",
            "value": 0.0001032010003,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / gpu / PostRev",
            "value": 0.0001026419748002,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / gpu / BothRev",
            "value": 0.0001073224997002,
            "unit": "s"
          },
          {
            "name": "sum    / JaX   / cpu / Primal",
            "value": 0.000004272487500020361,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / Primal",
            "value": 0.000004458338499989623,
            "unit": "s"
          },
          {
            "name": "sum    / JaX   / cpu / Forward",
            "value": 0.000006853161900016858,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / Forward",
            "value": 0.000006862103899948124,
            "unit": "s"
          },
          {
            "name": "sum    / JaX   / cpu / BothRev",
            "value": 0.000006303996800033928,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / PreRev",
            "value": 0.000006141809700056911,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / PostRev",
            "value": 0.000006348948800041398,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / BothRev",
            "value": 0.000006375791800019215,
            "unit": "s"
          },
          {
            "name": "sum    / JaX   / tpu / Primal",
            "value": 0.0001296193850999,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / tpu / Primal",
            "value": 0.0001297850531,
            "unit": "s"
          },
          {
            "name": "sum    / JaX   / tpu / Forward",
            "value": 0.0002014630620999,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / tpu / Forward",
            "value": 0.0001996795618999,
            "unit": "s"
          },
          {
            "name": "sum    / JaX   / tpu / BothRev",
            "value": 0.0001976506936,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / tpu / PreRev",
            "value": 0.0002418499401,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / tpu / PostRev",
            "value": 0.0002381269995999,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / tpu / BothRev",
            "value": 0.000240870296,
            "unit": "s"
          },
          {
            "name": "sum    / JaX   / cpu / Primal",
            "value": 0.000006818363399997906,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / Primal",
            "value": 0.000006808848900072917,
            "unit": "s"
          },
          {
            "name": "sum    / JaX   / cpu / Forward",
            "value": 0.000010714350499983993,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / Forward",
            "value": 0.000010769604400047685,
            "unit": "s"
          },
          {
            "name": "sum    / JaX   / cpu / BothRev",
            "value": 0.000009869949299991276,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / PreRev",
            "value": 0.000009521072400002594,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / PostRev",
            "value": 0.00000975717599994823,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / BothRev",
            "value": 0.000009390763900046295,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / JaXPipe / gpu / Primal",
            "value": 0.0010167173000809,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / JaX   / gpu / Primal",
            "value": 0.0010026357002061,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / HLOOpt / gpu / Primal",
            "value": 0.0009447530002944,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / PartOpt / gpu / Primal",
            "value": 0.0009382888001709,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / DefOpt / gpu / Primal",
            "value": 0.0006840113997895,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / IPartOpt / gpu / Primal",
            "value": 0.0010092287000588,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / IDefOpt / gpu / Primal",
            "value": 0.0007018871998297,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / JaX   / gpu / Forward",
            "value": 0.001240917099858,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / JaXPipe / gpu / PostRev",
            "value": 0.0039422001002094,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / JaX   / gpu / BothRev",
            "value": 0.0039440286000171,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / HLOOpt / gpu / PostRev",
            "value": 0.0039627198999369,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / PartOpt / gpu / PostRev",
            "value": 0.0040122591999534,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / DefOpt / gpu / PostRev",
            "value": 0.0022106971002358,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / IPartOpt / gpu / PostRev",
            "value": 0.0040136405998055,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / IDefOpt / gpu / PostRev",
            "value": 0.0025069694002013,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / JaXPipe / tpu / Primal",
            "value": 0.00008921499993448378,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / JaX   / tpu / Primal",
            "value": 0.00009383400001752306,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / HLOOpt / tpu / Primal",
            "value": 0.00009190099999614176,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / PartOpt / tpu / Primal",
            "value": 0.0001170989999991,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / DefOpt / tpu / Primal",
            "value": 0.0001221330000589,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / IPartOpt / tpu / Primal",
            "value": 0.00009093199996641488,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / IDefOpt / tpu / Primal",
            "value": 0.00008704499996383674,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / JaX   / tpu / Forward",
            "value": 0.0001787270000022,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / JaXPipe / tpu / PostRev",
            "value": 0.0001921551000123,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / JaX   / tpu / BothRev",
            "value": 0.0002040299999862,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / HLOOpt / tpu / PostRev",
            "value": 0.0002074209999591,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / PartOpt / tpu / PostRev",
            "value": 0.0002050889999736,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / DefOpt / tpu / PostRev",
            "value": 0.0001977531000193,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / IPartOpt / tpu / PostRev",
            "value": 0.0002038329999777,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / IDefOpt / tpu / PostRev",
            "value": 0.0001993651000702,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / JaXPipe / cpu / Primal",
            "value": 0.00007419180001306813,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / JaX   / cpu / Primal",
            "value": 0.0000476203000289388,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / HLOOpt / cpu / Primal",
            "value": 0.00006997930004217778,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / PartOpt / cpu / Primal",
            "value": 0.00008144330004142831,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / DefOpt / cpu / Primal",
            "value": 0.00007021159999567316,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / IPartOpt / cpu / Primal",
            "value": 0.000076058499962528,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / IDefOpt / cpu / Primal",
            "value": 0.00007893579995652545,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / JaX   / cpu / Forward",
            "value": 0.0001624477000405,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / JaXPipe / cpu / PostRev",
            "value": 0.0001491279999754,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / JaX   / cpu / BothRev",
            "value": 0.0001850342000579,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / HLOOpt / cpu / PostRev",
            "value": 0.000154961499993,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / PartOpt / cpu / PostRev",
            "value": 0.0002345742000215,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / DefOpt / cpu / PostRev",
            "value": 0.0001429568000276,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / IPartOpt / cpu / PostRev",
            "value": 0.0001807009000003,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / IDefOpt / cpu / PostRev",
            "value": 0.0001646601000174,
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
            "email": "gh@wsmoses.com",
            "name": "William Moses",
            "username": "wsmoses"
          },
          "distinct": true,
          "id": "58a6dd321665a740e1fdab2d980d2c78131092c6",
          "message": "test: more tests",
          "timestamp": "2025-12-14T00:08:02-06:00",
          "tree_id": "087d9de5dbc3b8acd5e3884776c1b430547f958f",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/58a6dd321665a740e1fdab2d980d2c78131092c6"
        },
        "date": 1765708542833,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "actmtch / JaX   / cpu / Primal",
            "value": 0.000003921564600022975,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / Primal",
            "value": 0.000003845813199995973,
            "unit": "s"
          },
          {
            "name": "actmtch / JaX   / cpu / Forward",
            "value": 0.000006483998800013069,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / Forward",
            "value": 0.000007843436200028008,
            "unit": "s"
          },
          {
            "name": "actmtch / JaX   / cpu / BothRev",
            "value": 0.000006455199600031847,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / PreRev",
            "value": 0.000007359487500025352,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / PostRev",
            "value": 0.000005890227999998387,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / BothRev",
            "value": 0.000007404597300001115,
            "unit": "s"
          },
          {
            "name": "actmtch / JaX   / cpu / Primal",
            "value": 0.000007207351499528158,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / Primal",
            "value": 0.000007386805799615104,
            "unit": "s"
          },
          {
            "name": "actmtch / JaX   / cpu / Forward",
            "value": 0.000011305273800098804,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / Forward",
            "value": 0.000014323599699127954,
            "unit": "s"
          },
          {
            "name": "actmtch / JaX   / cpu / BothRev",
            "value": 0.00001135532580083236,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / PreRev",
            "value": 0.00001377144349971786,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / PostRev",
            "value": 0.000012840120799955912,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / BothRev",
            "value": 0.000014111195800069254,
            "unit": "s"
          },
          {
            "name": "actmtch / JaX   / gpu / Primal",
            "value": 0.00007688760340097361,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / gpu / Primal",
            "value": 0.00007878601039992646,
            "unit": "s"
          },
          {
            "name": "actmtch / JaX   / gpu / Forward",
            "value": 0.0001092637152003,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / gpu / Forward",
            "value": 0.0001081586214,
            "unit": "s"
          },
          {
            "name": "actmtch / JaX   / gpu / BothRev",
            "value": 0.0001054493777002,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / gpu / PreRev",
            "value": 0.0001071451074996,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / gpu / PostRev",
            "value": 0.0001113502189997,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / gpu / BothRev",
            "value": 0.0001050373338002,
            "unit": "s"
          },
          {
            "name": "actmtch / JaX   / cpu / Primal",
            "value": 0.0000036029749004228504,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / Primal",
            "value": 0.000003662745999463368,
            "unit": "s"
          },
          {
            "name": "actmtch / JaX   / cpu / Forward",
            "value": 0.000005553300899919122,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / Forward",
            "value": 0.000006150249000347685,
            "unit": "s"
          },
          {
            "name": "actmtch / JaX   / cpu / BothRev",
            "value": 0.000005493537900474621,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / PreRev",
            "value": 0.00000615797400023439,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / PostRev",
            "value": 0.000005482535999908578,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / BothRev",
            "value": 0.000006354146899684565,
            "unit": "s"
          },
          {
            "name": "actmtch / JaX   / tpu / Primal",
            "value": 0.0001522168622999,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / tpu / Primal",
            "value": 0.0001374857094,
            "unit": "s"
          },
          {
            "name": "actmtch / JaX   / tpu / Forward",
            "value": 0.0002148543359995,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / tpu / Forward",
            "value": 0.0002088025980003,
            "unit": "s"
          },
          {
            "name": "actmtch / JaX   / tpu / BothRev",
            "value": 0.0002033665061004,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / tpu / PreRev",
            "value": 0.0002224648238996,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / tpu / PostRev",
            "value": 0.0002160167479996,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / tpu / BothRev",
            "value": 0.0002227855498997,
            "unit": "s"
          },
          {
            "name": "actmtch / JaX   / cpu / Primal",
            "value": 0.000005341018600029201,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / Primal",
            "value": 0.000007316592899951502,
            "unit": "s"
          },
          {
            "name": "actmtch / JaX   / cpu / Forward",
            "value": 0.000007276487499984796,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / Forward",
            "value": 0.000010191768999993655,
            "unit": "s"
          },
          {
            "name": "actmtch / JaX   / cpu / BothRev",
            "value": 0.000008197276400005649,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / PreRev",
            "value": 0.000009028872700037028,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / PostRev",
            "value": 0.000007389501399939036,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / BothRev",
            "value": 0.000009485565300019516,
            "unit": "s"
          },
          {
            "name": "actmtch / JaX   / cpu / Primal",
            "value": 0.00000365054579997377,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / Primal",
            "value": 0.000003635441699952935,
            "unit": "s"
          },
          {
            "name": "actmtch / JaX   / cpu / Forward",
            "value": 0.000005508716599979379,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / Forward",
            "value": 0.000006890399999974761,
            "unit": "s"
          },
          {
            "name": "actmtch / JaX   / cpu / BothRev",
            "value": 0.000006120520900003612,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / PreRev",
            "value": 0.000007367758399959712,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / PostRev",
            "value": 0.000005868649999865739,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / BothRev",
            "value": 0.000006541316699986055,
            "unit": "s"
          },
          {
            "name": "add_one / JaX   / cpu / Primal",
            "value": 0.000004238353499977165,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / Primal",
            "value": 0.000004281596200007698,
            "unit": "s"
          },
          {
            "name": "add_one / JaX   / cpu / Forward",
            "value": 0.000007857112599958782,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / Forward",
            "value": 0.000007876014600014969,
            "unit": "s"
          },
          {
            "name": "add_one / JaX   / cpu / BothRev",
            "value": 0.00000821751840003344,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / PreRev",
            "value": 0.000007755668400022841,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / PostRev",
            "value": 0.00000793305719998898,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / BothRev",
            "value": 0.000007865629199977774,
            "unit": "s"
          },
          {
            "name": "add_one / JaX   / cpu / Primal",
            "value": 0.000007707188499625772,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / Primal",
            "value": 0.00000757509730028687,
            "unit": "s"
          },
          {
            "name": "add_one / JaX   / cpu / Forward",
            "value": 0.000012110247599775903,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / Forward",
            "value": 0.00001417758819879964,
            "unit": "s"
          },
          {
            "name": "add_one / JaX   / cpu / BothRev",
            "value": 0.000012552390899509193,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / PreRev",
            "value": 0.00001243222799967043,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / PostRev",
            "value": 0.0000124866641999688,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / BothRev",
            "value": 0.000012454364000586793,
            "unit": "s"
          },
          {
            "name": "add_one / JaX   / gpu / Primal",
            "value": 0.00007741707410023081,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / gpu / Primal",
            "value": 0.00007777027249976527,
            "unit": "s"
          },
          {
            "name": "add_one / JaX   / gpu / Forward",
            "value": 0.0001088907058001,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / gpu / Forward",
            "value": 0.0001092670547994,
            "unit": "s"
          },
          {
            "name": "add_one / JaX   / gpu / BothRev",
            "value": 0.0001130498055004,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / gpu / PreRev",
            "value": 0.0001135505355996,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / gpu / PostRev",
            "value": 0.0001139645657996,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / gpu / BothRev",
            "value": 0.0001126461626001,
            "unit": "s"
          },
          {
            "name": "add_one / JaX   / cpu / Primal",
            "value": 0.000003803349000372691,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / Primal",
            "value": 0.000003783854999346658,
            "unit": "s"
          },
          {
            "name": "add_one / JaX   / cpu / Forward",
            "value": 0.000005905685000470839,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / Forward",
            "value": 0.00000613846799969906,
            "unit": "s"
          },
          {
            "name": "add_one / JaX   / cpu / BothRev",
            "value": 0.000006145850900065853,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / PreRev",
            "value": 0.000006137385000329232,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / PostRev",
            "value": 0.000006194932899961713,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / BothRev",
            "value": 0.000006138853999436833,
            "unit": "s"
          },
          {
            "name": "add_one / JaX   / tpu / Primal",
            "value": 0.0001498981953001,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / tpu / Primal",
            "value": 0.0001483538853994,
            "unit": "s"
          },
          {
            "name": "add_one / JaX   / tpu / Forward",
            "value": 0.0002319359499,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / tpu / Forward",
            "value": 0.0002331666908998,
            "unit": "s"
          },
          {
            "name": "add_one / JaX   / tpu / BothRev",
            "value": 0.0002337815869999,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / tpu / PreRev",
            "value": 0.0002288968810004,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / tpu / PostRev",
            "value": 0.0002136755879997,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / tpu / BothRev",
            "value": 0.0002255472980003,
            "unit": "s"
          },
          {
            "name": "add_one / JaX   / cpu / Primal",
            "value": 0.000005349261599985766,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / Primal",
            "value": 0.000005491581500064058,
            "unit": "s"
          },
          {
            "name": "add_one / JaX   / cpu / Forward",
            "value": 0.000008629256199947121,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / Forward",
            "value": 0.000008579168899996149,
            "unit": "s"
          },
          {
            "name": "add_one / JaX   / cpu / BothRev",
            "value": 0.000008617740099998628,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / PreRev",
            "value": 0.000009041264000006775,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / PostRev",
            "value": 0.000008655772200017963,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / BothRev",
            "value": 0.00000909314069995162,
            "unit": "s"
          },
          {
            "name": "add_one / JaX   / cpu / Primal",
            "value": 0.000003510412500145321,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / Primal",
            "value": 0.00000457139999998617,
            "unit": "s"
          },
          {
            "name": "add_one / JaX   / cpu / Forward",
            "value": 0.000005901841599916224,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / Forward",
            "value": 0.0000056339249998927695,
            "unit": "s"
          },
          {
            "name": "add_one / JaX   / cpu / BothRev",
            "value": 0.000006803779099936946,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / PreRev",
            "value": 0.000006365295800060266,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / PostRev",
            "value": 0.000006366650000018126,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / BothRev",
            "value": 0.000006292050000047311,
            "unit": "s"
          },
          {
            "name": "add_two / JaX   / cpu / Primal",
            "value": 0.000004439629000034983,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / Primal",
            "value": 0.000004363252800021655,
            "unit": "s"
          },
          {
            "name": "add_two / JaX   / cpu / Forward",
            "value": 0.000007747113999994326,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / Forward",
            "value": 0.000007719281399977262,
            "unit": "s"
          },
          {
            "name": "add_two / JaX   / cpu / BothRev",
            "value": 0.000009599743100034174,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / PreRev",
            "value": 0.000009580201100015985,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / PostRev",
            "value": 0.000009501530900024591,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / BothRev",
            "value": 0.000009632188699970356,
            "unit": "s"
          },
          {
            "name": "add_two / JaX   / cpu / Primal",
            "value": 0.000007982453401200474,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / Primal",
            "value": 0.000008047861100931187,
            "unit": "s"
          },
          {
            "name": "add_two / JaX   / cpu / Forward",
            "value": 0.000012074471100640948,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / Forward",
            "value": 0.000012638317000528332,
            "unit": "s"
          },
          {
            "name": "add_two / JaX   / cpu / BothRev",
            "value": 0.000014903987900470383,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / PreRev",
            "value": 0.000014801577599428128,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / PostRev",
            "value": 0.000014715902699390429,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / BothRev",
            "value": 0.000015528397301386575,
            "unit": "s"
          },
          {
            "name": "add_two / JaX   / gpu / Primal",
            "value": 0.00008023529810016043,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / gpu / Primal",
            "value": 0.00008724298809975153,
            "unit": "s"
          },
          {
            "name": "add_two / JaX   / gpu / Forward",
            "value": 0.0001081960641997,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / gpu / Forward",
            "value": 0.0001057026514987,
            "unit": "s"
          },
          {
            "name": "add_two / JaX   / gpu / BothRev",
            "value": 0.0001227994611006,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / gpu / PreRev",
            "value": 0.0001270741697997,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / gpu / PostRev",
            "value": 0.0001241995717005,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / gpu / BothRev",
            "value": 0.0001323667405988,
            "unit": "s"
          },
          {
            "name": "add_two / JaX   / cpu / Primal",
            "value": 0.00000388022999977693,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / Primal",
            "value": 0.000003902675899735186,
            "unit": "s"
          },
          {
            "name": "add_two / JaX   / cpu / Forward",
            "value": 0.000006042410000372911,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / Forward",
            "value": 0.000006252257000596728,
            "unit": "s"
          },
          {
            "name": "add_two / JaX   / cpu / BothRev",
            "value": 0.000007231529900309397,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / PreRev",
            "value": 0.00000726670899966848,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / PostRev",
            "value": 0.000007233767899742815,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / BothRev",
            "value": 0.0000072490578997530975,
            "unit": "s"
          },
          {
            "name": "add_two / JaX   / tpu / Primal",
            "value": 0.0001353645294002,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / tpu / Primal",
            "value": 0.0001497547812999,
            "unit": "s"
          },
          {
            "name": "add_two / JaX   / tpu / Forward",
            "value": 0.000223919724,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / tpu / Forward",
            "value": 0.0002055709899999,
            "unit": "s"
          },
          {
            "name": "add_two / JaX   / tpu / BothRev",
            "value": 0.0002308868919993,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / tpu / PreRev",
            "value": 0.0002402805929996,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / tpu / PostRev",
            "value": 0.0002319690030002,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / tpu / BothRev",
            "value": 0.0002397160808999,
            "unit": "s"
          },
          {
            "name": "add_two / JaX   / cpu / Primal",
            "value": 0.000005755408299955888,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / Primal",
            "value": 0.000005736746700040385,
            "unit": "s"
          },
          {
            "name": "add_two / JaX   / cpu / Forward",
            "value": 0.0000110537382000075,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / Forward",
            "value": 0.000009167960499962649,
            "unit": "s"
          },
          {
            "name": "add_two / JaX   / cpu / BothRev",
            "value": 0.000012864492699918629,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / PreRev",
            "value": 0.00001030779739994614,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / PostRev",
            "value": 0.000010680399099965144,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / BothRev",
            "value": 0.000012480110500018782,
            "unit": "s"
          },
          {
            "name": "add_two / JaX   / cpu / Primal",
            "value": 0.0000035753374999330844,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / Primal",
            "value": 0.000003727700000126788,
            "unit": "s"
          },
          {
            "name": "add_two / JaX   / cpu / Forward",
            "value": 0.000006016641700080072,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / Forward",
            "value": 0.000006586358300046413,
            "unit": "s"
          },
          {
            "name": "add_two / JaX   / cpu / BothRev",
            "value": 0.0000074236209000446255,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / PreRev",
            "value": 0.00000780657080013043,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / PostRev",
            "value": 0.000007628875000045809,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / BothRev",
            "value": 0.000007652237499860348,
            "unit": "s"
          },
          {
            "name": "cache / JaX   / cpu / Primal",
            "value": 0.0000038379609999992685,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / Primal",
            "value": 0.000004191734899995936,
            "unit": "s"
          },
          {
            "name": "cache / JaX   / cpu / Forward",
            "value": 0.000010165292100009538,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / Forward",
            "value": 0.000010023599199985255,
            "unit": "s"
          },
          {
            "name": "cache / JaX   / cpu / BothRev",
            "value": 0.000013827765199994246,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / PreRev",
            "value": 0.000010889822199987976,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / PostRev",
            "value": 0.000014659246500013978,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / BothRev",
            "value": 0.000010952607499984878,
            "unit": "s"
          },
          {
            "name": "cache / JaX   / cpu / Primal",
            "value": 0.00000796771409950452,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / Primal",
            "value": 0.000007937584900355433,
            "unit": "s"
          },
          {
            "name": "cache / JaX   / cpu / Forward",
            "value": 0.000013119718300004023,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / Forward",
            "value": 0.000012974730400310365,
            "unit": "s"
          },
          {
            "name": "cache / JaX   / cpu / BothRev",
            "value": 0.00002317615680076415,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / PreRev",
            "value": 0.000017547203801223076,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / PostRev",
            "value": 0.000024142596300225703,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / BothRev",
            "value": 0.000017651054600719363,
            "unit": "s"
          },
          {
            "name": "cache / JaX   / gpu / Primal",
            "value": 0.00007551079279946861,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / gpu / Primal",
            "value": 0.00007553912879957351,
            "unit": "s"
          },
          {
            "name": "cache / JaX   / gpu / Forward",
            "value": 0.0000980166534005548,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / gpu / Forward",
            "value": 0.0001015004483997,
            "unit": "s"
          },
          {
            "name": "cache / JaX   / gpu / BothRev",
            "value": 0.0001175972883997,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / gpu / PreRev",
            "value": 0.0001119294449003,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / gpu / PostRev",
            "value": 0.0001076381531995,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / gpu / BothRev",
            "value": 0.0001101016738,
            "unit": "s"
          },
          {
            "name": "cache / JaX   / cpu / Primal",
            "value": 0.000003812030999688432,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / Primal",
            "value": 0.000003577171000506496,
            "unit": "s"
          },
          {
            "name": "cache / JaX   / cpu / Forward",
            "value": 0.000006726671000069473,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / Forward",
            "value": 0.000005948529900342692,
            "unit": "s"
          },
          {
            "name": "cache / JaX   / cpu / BothRev",
            "value": 0.000032416021899553015,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / PreRev",
            "value": 0.000012122006899880944,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / PostRev",
            "value": 0.0000255690609003068,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / BothRev",
            "value": 0.000021920404900447467,
            "unit": "s"
          },
          {
            "name": "cache / JaX   / tpu / Primal",
            "value": 0.0001509684074,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / tpu / Primal",
            "value": 0.0001355307263998,
            "unit": "s"
          },
          {
            "name": "cache / JaX   / tpu / Forward",
            "value": 0.0002178112400004,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / tpu / Forward",
            "value": 0.0002187209059993,
            "unit": "s"
          },
          {
            "name": "cache / JaX   / tpu / BothRev",
            "value": 0.0002238176110004,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / tpu / PreRev",
            "value": 0.0002225815420002,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / tpu / PostRev",
            "value": 0.0002230406028997,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / tpu / BothRev",
            "value": 0.0002069479790996,
            "unit": "s"
          },
          {
            "name": "cache / JaX   / cpu / Primal",
            "value": 0.000005164968200006115,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / Primal",
            "value": 0.000005702998000015213,
            "unit": "s"
          },
          {
            "name": "cache / JaX   / cpu / Forward",
            "value": 0.000023743001599996204,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / Forward",
            "value": 0.000022353226900031588,
            "unit": "s"
          },
          {
            "name": "cache / JaX   / cpu / BothRev",
            "value": 0.00003238162850002482,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / PreRev",
            "value": 0.00002784308059999603,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / PostRev",
            "value": 0.000032757576699987113,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / BothRev",
            "value": 0.000027215169800001604,
            "unit": "s"
          },
          {
            "name": "cache / JaX   / cpu / Primal",
            "value": 0.0000031707791998996983,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / Primal",
            "value": 0.000003289904200028104,
            "unit": "s"
          },
          {
            "name": "cache / JaX   / cpu / Forward",
            "value": 0.00000636390420004318,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / Forward",
            "value": 0.000006432008299998415,
            "unit": "s"
          },
          {
            "name": "cache / JaX   / cpu / BothRev",
            "value": 0.000007592079099958937,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / PreRev",
            "value": 0.000006488862500009418,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / PostRev",
            "value": 0.000007409262500004843,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / BothRev",
            "value": 0.000010857029100043292,
            "unit": "s"
          },
          {
            "name": "Concat / JaX   / cpu / Primal",
            "value": 0.000004082074400002966,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / Primal",
            "value": 0.000004142072399963581,
            "unit": "s"
          },
          {
            "name": "Concat / JaX   / cpu / Forward",
            "value": 0.000007425710100005744,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / Forward",
            "value": 0.000007260632600036843,
            "unit": "s"
          },
          {
            "name": "Concat / JaX   / cpu / BothRev",
            "value": 0.000007687794599996777,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / PreRev",
            "value": 0.00000764028209996468,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / PostRev",
            "value": 0.000007646879900039494,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / BothRev",
            "value": 0.000007613118700010091,
            "unit": "s"
          },
          {
            "name": "Concat / JaX   / cpu / Primal",
            "value": 0.000007698948700272012,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / Primal",
            "value": 0.000007856241799890995,
            "unit": "s"
          },
          {
            "name": "Concat / JaX   / cpu / Forward",
            "value": 0.00001220212670014007,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / Forward",
            "value": 0.000012219415100116748,
            "unit": "s"
          },
          {
            "name": "Concat / JaX   / cpu / BothRev",
            "value": 0.00001213907680066768,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / PreRev",
            "value": 0.00001289466239977628,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / PostRev",
            "value": 0.000012806844800070394,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / BothRev",
            "value": 0.000012636134099739138,
            "unit": "s"
          },
          {
            "name": "Concat / JaX   / gpu / Primal",
            "value": 0.00007751117779989727,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / gpu / Primal",
            "value": 0.00007725569129979703,
            "unit": "s"
          },
          {
            "name": "Concat / JaX   / gpu / Forward",
            "value": 0.0001079242381994,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / gpu / Forward",
            "value": 0.0001122185992004,
            "unit": "s"
          },
          {
            "name": "Concat / JaX   / gpu / BothRev",
            "value": 0.0001039703596004,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / gpu / PreRev",
            "value": 0.0001044355953999,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / gpu / PostRev",
            "value": 0.0001029672024,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / gpu / BothRev",
            "value": 0.0001409916261996,
            "unit": "s"
          },
          {
            "name": "Concat / JaX   / cpu / Primal",
            "value": 0.000003767692999826977,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / Primal",
            "value": 0.0000045303440005227455,
            "unit": "s"
          },
          {
            "name": "Concat / JaX   / cpu / Forward",
            "value": 0.000005974665999383433,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / Forward",
            "value": 0.000006090410899923882,
            "unit": "s"
          },
          {
            "name": "Concat / JaX   / cpu / BothRev",
            "value": 0.000006015489999845159,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / PreRev",
            "value": 0.000006334579000395024,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / PostRev",
            "value": 0.000006283346999407513,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / BothRev",
            "value": 0.0000063470209999650255,
            "unit": "s"
          },
          {
            "name": "Concat / JaX   / tpu / Primal",
            "value": 0.0001413948973,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / tpu / Primal",
            "value": 0.0001512813162997,
            "unit": "s"
          },
          {
            "name": "Concat / JaX   / tpu / Forward",
            "value": 0.0002148676689997,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / tpu / Forward",
            "value": 0.0002195608250003,
            "unit": "s"
          },
          {
            "name": "Concat / JaX   / tpu / BothRev",
            "value": 0.0002218354030002,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / tpu / PreRev",
            "value": 0.0002183276299998,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / tpu / PostRev",
            "value": 0.000221908468,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / tpu / BothRev",
            "value": 0.0002291133489001,
            "unit": "s"
          },
          {
            "name": "Concat / JaX   / cpu / Primal",
            "value": 0.000005455328400057624,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / Primal",
            "value": 0.00000532030800004577,
            "unit": "s"
          },
          {
            "name": "Concat / JaX   / cpu / Forward",
            "value": 0.000008245991000057984,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / Forward",
            "value": 0.000008237428200027352,
            "unit": "s"
          },
          {
            "name": "Concat / JaX   / cpu / BothRev",
            "value": 0.000008559027499995864,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / PreRev",
            "value": 0.000009625010899981135,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / PostRev",
            "value": 0.000009289023500059556,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / BothRev",
            "value": 0.000008778968200022064,
            "unit": "s"
          },
          {
            "name": "Concat / JaX   / cpu / Primal",
            "value": 0.000003569633299957786,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / Primal",
            "value": 0.0000036291332999098814,
            "unit": "s"
          },
          {
            "name": "Concat / JaX   / cpu / Forward",
            "value": 0.000005747154200071236,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / Forward",
            "value": 0.000005712137500086101,
            "unit": "s"
          },
          {
            "name": "Concat / JaX   / cpu / BothRev",
            "value": 0.000006152420800026448,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / PreRev",
            "value": 0.000006501608300095541,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / PostRev",
            "value": 0.000006082729199988535,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / BothRev",
            "value": 0.000006491300000016054,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaX   / cpu / Primal",
            "value": 0.000007657799960725242,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / cpu / Primal",
            "value": 0.000007318500001929351,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaX   / cpu / Forward",
            "value": 0.00000906179998310108,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / cpu / Forward",
            "value": 0.000010463099988555768,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaX   / cpu / Primal",
            "value": 0.00001741860032780096,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / cpu / Primal",
            "value": 0.000013964799290988594,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaX   / cpu / Forward",
            "value": 0.00001554280024720356,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / cpu / Forward",
            "value": 0.00001921929942909628,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaX   / gpu / Primal",
            "value": 0.0001225590996909,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / gpu / Primal",
            "value": 0.0001125908005633,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaX   / gpu / Forward",
            "value": 0.0001450365001801,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / gpu / Forward",
            "value": 0.0001505619991803,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaX   / cpu / Primal",
            "value": 0.000006683000538032502,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / cpu / Primal",
            "value": 0.000005755000165663659,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaX   / cpu / Forward",
            "value": 0.000007561999518657103,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / cpu / Forward",
            "value": 0.000009923000470735131,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaX   / tpu / Primal",
            "value": 0.0002380619000177,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / tpu / Primal",
            "value": 0.0001831219997256,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaX   / tpu / Forward",
            "value": 0.0002720019998378,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / tpu / Forward",
            "value": 0.0002687420004804,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaX   / cpu / Primal",
            "value": 0.000009651000073063188,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / cpu / Primal",
            "value": 0.000008705300024303142,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaX   / cpu / Forward",
            "value": 0.00001349929998468724,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / cpu / Forward",
            "value": 0.000012685500041698106,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaX   / cpu / Primal",
            "value": 0.000006179099909786601,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / cpu / Primal",
            "value": 0.000005108400000608526,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaX   / cpu / Forward",
            "value": 0.000007004100007179659,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / cpu / Forward",
            "value": 0.00000849170010042144,
            "unit": "s"
          },
          {
            "name": "GenDot / JaX   / cpu / Primal",
            "value": 0.000004283868599986818,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / Primal",
            "value": 0.000004443865299981553,
            "unit": "s"
          },
          {
            "name": "GenDot / JaX   / cpu / Forward",
            "value": 0.00000660724049998862,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / Forward",
            "value": 0.000007397160199980135,
            "unit": "s"
          },
          {
            "name": "GenDot / JaX   / cpu / BothRev",
            "value": 0.000006872402700037128,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / PreRev",
            "value": 0.000007685212300020795,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / PostRev",
            "value": 0.000006625085599989689,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / BothRev",
            "value": 0.000007353765500010923,
            "unit": "s"
          },
          {
            "name": "GenDot / JaX   / cpu / Primal",
            "value": 0.000009016189699468668,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / Primal",
            "value": 0.000008624273199529853,
            "unit": "s"
          },
          {
            "name": "GenDot / JaX   / cpu / Forward",
            "value": 0.000011281070599216036,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / Forward",
            "value": 0.000014466385899868328,
            "unit": "s"
          },
          {
            "name": "GenDot / JaX   / cpu / BothRev",
            "value": 0.000011546395099139772,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / PreRev",
            "value": 0.000013822862799861468,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / PostRev",
            "value": 0.000012255935999564826,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / BothRev",
            "value": 0.000014849487399624197,
            "unit": "s"
          },
          {
            "name": "GenDot / JaX   / gpu / Primal",
            "value": 0.00008194280519965105,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / gpu / Primal",
            "value": 0.00007519147130078637,
            "unit": "s"
          },
          {
            "name": "GenDot / JaX   / gpu / Forward",
            "value": 0.0001055878393002,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / gpu / Forward",
            "value": 0.0001045679963004,
            "unit": "s"
          },
          {
            "name": "GenDot / JaX   / gpu / BothRev",
            "value": 0.0001030107231999,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / gpu / PreRev",
            "value": 0.0001075848307998,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / gpu / PostRev",
            "value": 0.000106439821601,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / gpu / BothRev",
            "value": 0.0001215008053011,
            "unit": "s"
          },
          {
            "name": "GenDot / JaX   / cpu / Primal",
            "value": 0.000004044511900428915,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / Primal",
            "value": 0.000003953633000492119,
            "unit": "s"
          },
          {
            "name": "GenDot / JaX   / cpu / Forward",
            "value": 0.0000054695859995263165,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / Forward",
            "value": 0.000005516165000153706,
            "unit": "s"
          },
          {
            "name": "GenDot / JaX   / cpu / BothRev",
            "value": 0.000005085540000436595,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / PreRev",
            "value": 0.000005570104000071297,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / PostRev",
            "value": 0.000005129583000234561,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / BothRev",
            "value": 0.00000633987100009108,
            "unit": "s"
          },
          {
            "name": "GenDot / JaX   / tpu / Primal",
            "value": 0.0001585112852997,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / tpu / Primal",
            "value": 0.0001529706752997,
            "unit": "s"
          },
          {
            "name": "GenDot / JaX   / tpu / Forward",
            "value": 0.0002328496759,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / tpu / Forward",
            "value": 0.0002283283119999,
            "unit": "s"
          },
          {
            "name": "GenDot / JaX   / tpu / BothRev",
            "value": 0.0002309583409005,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / tpu / PreRev",
            "value": 0.0002384318799005,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / tpu / PostRev",
            "value": 0.0002380794488999,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / tpu / BothRev",
            "value": 0.000230550663,
            "unit": "s"
          },
          {
            "name": "GenDot / JaX   / cpu / Primal",
            "value": 0.000005626422199929948,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / Primal",
            "value": 0.0000060470812999483314,
            "unit": "s"
          },
          {
            "name": "GenDot / JaX   / cpu / Forward",
            "value": 0.000007878270699984568,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / Forward",
            "value": 0.00001012336020003204,
            "unit": "s"
          },
          {
            "name": "GenDot / JaX   / cpu / BothRev",
            "value": 0.000008338413499950548,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / PreRev",
            "value": 0.00001015556980000838,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / PostRev",
            "value": 0.00000824944179994418,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / BothRev",
            "value": 0.00001035303959997691,
            "unit": "s"
          },
          {
            "name": "GenDot / JaX   / cpu / Primal",
            "value": 0.0000036205958000209646,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / Primal",
            "value": 0.000004038795800079243,
            "unit": "s"
          },
          {
            "name": "GenDot / JaX   / cpu / Forward",
            "value": 0.000006818641700010631,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / Forward",
            "value": 0.000006357112500154471,
            "unit": "s"
          },
          {
            "name": "GenDot / JaX   / cpu / BothRev",
            "value": 0.000005586574999870208,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / PreRev",
            "value": 0.000006232237500080373,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / PostRev",
            "value": 0.000005721958299909602,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / BothRev",
            "value": 0.000006609054199907405,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaX   / cpu / Primal",
            "value": 0.00000585785550001674,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / Primal",
            "value": 0.000005772714599970641,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaX   / cpu / Forward",
            "value": 0.000009535359200026503,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / Forward",
            "value": 0.000009582002400020428,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaX   / cpu / BothRev",
            "value": 0.000009245713700011036,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / PreRev",
            "value": 0.000009256986399987,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / PostRev",
            "value": 0.000009691316399994322,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / BothRev",
            "value": 0.000009292709000010293,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaX   / cpu / Primal",
            "value": 0.000010509710600308609,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / Primal",
            "value": 0.000010020754499419126,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaX   / cpu / Forward",
            "value": 0.00001559038870036602,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / Forward",
            "value": 0.00001604407910053851,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaX   / cpu / BothRev",
            "value": 0.000015657136600930243,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / PreRev",
            "value": 0.000015655730399885215,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / PostRev",
            "value": 0.000015582097599690314,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / BothRev",
            "value": 0.000015648292300465984,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaX   / cpu / Primal",
            "value": 0.000004990376000205288,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / Primal",
            "value": 0.000005036665999796242,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaX   / cpu / Forward",
            "value": 0.000007654927999828942,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / Forward",
            "value": 0.000007969786000467138,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaX   / cpu / BothRev",
            "value": 0.000007351582899718778,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / PreRev",
            "value": 0.000007671221000055084,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / PostRev",
            "value": 0.0000073946459997387135,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / BothRev",
            "value": 0.000007663809999939986,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaX   / tpu / Primal",
            "value": 0.00005172377080016304,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / tpu / Primal",
            "value": 0.00006730162069943617,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaX   / tpu / Forward",
            "value": 0.0000923369926000305,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / tpu / Forward",
            "value": 0.00009380668660014636,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaX   / tpu / BothRev",
            "value": 0.00009413366459993994,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / tpu / PreRev",
            "value": 0.00009191799359978176,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / tpu / PostRev",
            "value": 0.00009140644960061764,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / tpu / BothRev",
            "value": 0.0000917381955005112,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaX   / cpu / Primal",
            "value": 0.000007576715800041711,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / Primal",
            "value": 0.000007289722499990603,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaX   / cpu / Forward",
            "value": 0.0000111554526000873,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / Forward",
            "value": 0.000011459271900002931,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaX   / cpu / BothRev",
            "value": 0.000010920708400044532,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / PreRev",
            "value": 0.000011286616399956984,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / PostRev",
            "value": 0.000011229298600028416,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / BothRev",
            "value": 0.000011287977099982529,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaX   / cpu / Primal",
            "value": 0.000007864316599989251,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / Primal",
            "value": 0.000005599612500009243,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaX   / cpu / Forward",
            "value": 0.000009173687499969674,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / Forward",
            "value": 0.00001018502080005419,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaX   / cpu / BothRev",
            "value": 0.000007665908399940236,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / PreRev",
            "value": 0.00000862057089998416,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / PostRev",
            "value": 0.000007897233299991058,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / BothRev",
            "value": 0.000007895258299868146,
            "unit": "s"
          },
          {
            "name": "llama / JaXPipe / cpu / Primal",
            "value": 0.000875317760001,
            "unit": "s"
          },
          {
            "name": "llama / JaX   / cpu / Primal",
            "value": 0.000848846230001,
            "unit": "s"
          },
          {
            "name": "llama / HLOOpt / cpu / Primal",
            "value": 0.0009508277900022,
            "unit": "s"
          },
          {
            "name": "llama / PartOpt / cpu / Primal",
            "value": 0.0008461020100003,
            "unit": "s"
          },
          {
            "name": "llama / DefOpt / cpu / Primal",
            "value": 0.0009268839800006,
            "unit": "s"
          },
          {
            "name": "llama / IPartOpt / cpu / Primal",
            "value": 0.0008505649900007,
            "unit": "s"
          },
          {
            "name": "llama / IDefOpt / cpu / Primal",
            "value": 0.000946404780002,
            "unit": "s"
          },
          {
            "name": "llama / JaXPipe / cpu / Forward",
            "value": 0.00237775091,
            "unit": "s"
          },
          {
            "name": "llama / JaX   / cpu / Forward",
            "value": 0.0024262123200014,
            "unit": "s"
          },
          {
            "name": "llama / HLOOpt / cpu / Forward",
            "value": 0.0023481404600033,
            "unit": "s"
          },
          {
            "name": "llama / PartOpt / cpu / Forward",
            "value": 0.0023777744700009,
            "unit": "s"
          },
          {
            "name": "llama / DefOpt / cpu / Forward",
            "value": 0.0023446018000004,
            "unit": "s"
          },
          {
            "name": "llama / IPartOpt / cpu / Forward",
            "value": 0.0023391960900016,
            "unit": "s"
          },
          {
            "name": "llama / IDefOpt / cpu / Forward",
            "value": 0.0023517205600001,
            "unit": "s"
          },
          {
            "name": "llama / JaXPipe / cpu / PreRev",
            "value": 0.0021960026700025,
            "unit": "s"
          },
          {
            "name": "llama / JaXPipe / cpu / PostRev",
            "value": 0.0021825938799975,
            "unit": "s"
          },
          {
            "name": "llama / JaXPipe / cpu / BothRev",
            "value": 0.0021510245000035,
            "unit": "s"
          },
          {
            "name": "llama / JaX   / cpu / BothRev",
            "value": 0.0021465607800018,
            "unit": "s"
          },
          {
            "name": "llama / HLOOpt / cpu / PreRev",
            "value": 0.0021519097500004,
            "unit": "s"
          },
          {
            "name": "llama / HLOOpt / cpu / PostRev",
            "value": 0.0020716996600003,
            "unit": "s"
          },
          {
            "name": "llama / HLOOpt / cpu / BothRev",
            "value": 0.0021646580700007,
            "unit": "s"
          },
          {
            "name": "llama / PartOpt / cpu / PreRev",
            "value": 0.0021576972900038,
            "unit": "s"
          },
          {
            "name": "llama / PartOpt / cpu / PostRev",
            "value": 0.002141910790001,
            "unit": "s"
          },
          {
            "name": "llama / PartOpt / cpu / BothRev",
            "value": 0.0021637656400025,
            "unit": "s"
          },
          {
            "name": "llama / DefOpt / cpu / PreRev",
            "value": 0.0021685835499965,
            "unit": "s"
          },
          {
            "name": "llama / DefOpt / cpu / PostRev",
            "value": 0.0020386241199958,
            "unit": "s"
          },
          {
            "name": "llama / DefOpt / cpu / BothRev",
            "value": 0.0021715412599996,
            "unit": "s"
          },
          {
            "name": "llama / IPartOpt / cpu / PreRev",
            "value": 0.0021492072599994,
            "unit": "s"
          },
          {
            "name": "llama / IPartOpt / cpu / PostRev",
            "value": 0.002143050300001,
            "unit": "s"
          },
          {
            "name": "llama / IPartOpt / cpu / BothRev",
            "value": 0.002169131569999,
            "unit": "s"
          },
          {
            "name": "llama / IDefOpt / cpu / PreRev",
            "value": 0.0021638065999968,
            "unit": "s"
          },
          {
            "name": "llama / IDefOpt / cpu / PostRev",
            "value": 0.0021905070999991,
            "unit": "s"
          },
          {
            "name": "llama / IDefOpt / cpu / BothRev",
            "value": 0.0021979462499984,
            "unit": "s"
          },
          {
            "name": "llama / JaXPipe / gpu / Primal",
            "value": 0.0004734230859903,
            "unit": "s"
          },
          {
            "name": "llama / JaX   / gpu / Primal",
            "value": 0.0004739554699917,
            "unit": "s"
          },
          {
            "name": "llama / HLOOpt / gpu / Primal",
            "value": 0.0004733618399768,
            "unit": "s"
          },
          {
            "name": "llama / PartOpt / gpu / Primal",
            "value": 0.0004779653199948,
            "unit": "s"
          },
          {
            "name": "llama / DefOpt / gpu / Primal",
            "value": 0.0004329766999871,
            "unit": "s"
          },
          {
            "name": "llama / IPartOpt / gpu / Primal",
            "value": 0.000442989935982,
            "unit": "s"
          },
          {
            "name": "llama / IDefOpt / gpu / Primal",
            "value": 0.0004452833819959,
            "unit": "s"
          },
          {
            "name": "llama / JaXPipe / gpu / Forward",
            "value": 0.0007279538759903,
            "unit": "s"
          },
          {
            "name": "llama / JaX   / gpu / Forward",
            "value": 0.0007558644480013,
            "unit": "s"
          },
          {
            "name": "llama / HLOOpt / gpu / Forward",
            "value": 0.0007625300040235,
            "unit": "s"
          },
          {
            "name": "llama / PartOpt / gpu / Forward",
            "value": 0.0008009813679964,
            "unit": "s"
          },
          {
            "name": "llama / DefOpt / gpu / Forward",
            "value": 0.0007744016679935,
            "unit": "s"
          },
          {
            "name": "llama / IPartOpt / gpu / Forward",
            "value": 0.0007404444319836,
            "unit": "s"
          },
          {
            "name": "llama / IDefOpt / gpu / Forward",
            "value": 0.0007906445740081,
            "unit": "s"
          },
          {
            "name": "llama / JaXPipe / gpu / PreRev",
            "value": 0.000856824641989,
            "unit": "s"
          },
          {
            "name": "llama / JaXPipe / gpu / PostRev",
            "value": 0.000810599999997,
            "unit": "s"
          },
          {
            "name": "llama / JaXPipe / gpu / BothRev",
            "value": 0.0008007707659853,
            "unit": "s"
          },
          {
            "name": "llama / JaX   / gpu / BothRev",
            "value": 0.0008413018559804,
            "unit": "s"
          },
          {
            "name": "llama / HLOOpt / gpu / PreRev",
            "value": 0.000795593207993,
            "unit": "s"
          },
          {
            "name": "llama / HLOOpt / gpu / PostRev",
            "value": 0.0008367176059982,
            "unit": "s"
          },
          {
            "name": "llama / HLOOpt / gpu / BothRev",
            "value": 0.0008361503420164,
            "unit": "s"
          },
          {
            "name": "llama / PartOpt / gpu / PreRev",
            "value": 0.0008082720020029,
            "unit": "s"
          },
          {
            "name": "llama / PartOpt / gpu / PostRev",
            "value": 0.0008355694400088,
            "unit": "s"
          },
          {
            "name": "llama / PartOpt / gpu / BothRev",
            "value": 0.0008270790099923,
            "unit": "s"
          },
          {
            "name": "llama / DefOpt / gpu / PreRev",
            "value": 0.0007909286239882,
            "unit": "s"
          },
          {
            "name": "llama / DefOpt / gpu / PostRev",
            "value": 0.0007614632839977,
            "unit": "s"
          },
          {
            "name": "llama / DefOpt / gpu / BothRev",
            "value": 0.0008108141659758,
            "unit": "s"
          },
          {
            "name": "llama / IPartOpt / gpu / PreRev",
            "value": 0.0007977655840222,
            "unit": "s"
          },
          {
            "name": "llama / IPartOpt / gpu / PostRev",
            "value": 0.0008050945180002,
            "unit": "s"
          },
          {
            "name": "llama / IPartOpt / gpu / BothRev",
            "value": 0.000829977655987,
            "unit": "s"
          },
          {
            "name": "llama / IDefOpt / gpu / PreRev",
            "value": 0.000843174330017,
            "unit": "s"
          },
          {
            "name": "llama / IDefOpt / gpu / PostRev",
            "value": 0.0008576854499988,
            "unit": "s"
          },
          {
            "name": "llama / IDefOpt / gpu / BothRev",
            "value": 0.0008273077000048,
            "unit": "s"
          },
          {
            "name": "llama / JaXPipe / tpu / Primal",
            "value": 0.0003669029180018,
            "unit": "s"
          },
          {
            "name": "llama / JaX   / tpu / Primal",
            "value": 0.0003630636580055,
            "unit": "s"
          },
          {
            "name": "llama / HLOOpt / tpu / Primal",
            "value": 0.0003561747379862,
            "unit": "s"
          },
          {
            "name": "llama / PartOpt / tpu / Primal",
            "value": 0.000368330918005,
            "unit": "s"
          },
          {
            "name": "llama / DefOpt / tpu / Primal",
            "value": 0.0003426556580088,
            "unit": "s"
          },
          {
            "name": "llama / IPartOpt / tpu / Primal",
            "value": 0.0003706853580079,
            "unit": "s"
          },
          {
            "name": "llama / IDefOpt / tpu / Primal",
            "value": 0.0003686552579893,
            "unit": "s"
          },
          {
            "name": "llama / JaXPipe / tpu / Forward",
            "value": 0.0005803040760074,
            "unit": "s"
          },
          {
            "name": "llama / JaX   / tpu / Forward",
            "value": 0.0006976176160096,
            "unit": "s"
          },
          {
            "name": "llama / HLOOpt / tpu / Forward",
            "value": 0.0005658263160003,
            "unit": "s"
          },
          {
            "name": "llama / PartOpt / tpu / Forward",
            "value": 0.0005590071379992,
            "unit": "s"
          },
          {
            "name": "llama / DefOpt / tpu / Forward",
            "value": 0.0005809512980049,
            "unit": "s"
          },
          {
            "name": "llama / IPartOpt / tpu / Forward",
            "value": 0.0005602008380083,
            "unit": "s"
          },
          {
            "name": "llama / IDefOpt / tpu / Forward",
            "value": 0.0005639691579999,
            "unit": "s"
          },
          {
            "name": "llama / JaXPipe / tpu / PreRev",
            "value": 0.0004067248379869,
            "unit": "s"
          },
          {
            "name": "llama / JaXPipe / tpu / PostRev",
            "value": 0.0003576812400133,
            "unit": "s"
          },
          {
            "name": "llama / JaXPipe / tpu / BothRev",
            "value": 0.0003914951579936,
            "unit": "s"
          },
          {
            "name": "llama / JaX   / tpu / BothRev",
            "value": 0.0003573898180038,
            "unit": "s"
          },
          {
            "name": "llama / HLOOpt / tpu / PreRev",
            "value": 0.0003914461179956,
            "unit": "s"
          },
          {
            "name": "llama / HLOOpt / tpu / PostRev",
            "value": 0.0003900264600088,
            "unit": "s"
          },
          {
            "name": "llama / HLOOpt / tpu / BothRev",
            "value": 0.0003915486579935,
            "unit": "s"
          },
          {
            "name": "llama / PartOpt / tpu / PreRev",
            "value": 0.0003913379779987,
            "unit": "s"
          },
          {
            "name": "llama / PartOpt / tpu / PostRev",
            "value": 0.0003722094179975,
            "unit": "s"
          },
          {
            "name": "llama / PartOpt / tpu / BothRev",
            "value": 0.000391454498007,
            "unit": "s"
          },
          {
            "name": "llama / DefOpt / tpu / PreRev",
            "value": 0.0003915161779877,
            "unit": "s"
          },
          {
            "name": "llama / DefOpt / tpu / PostRev",
            "value": 0.0003828403379884,
            "unit": "s"
          },
          {
            "name": "llama / DefOpt / tpu / BothRev",
            "value": 0.0003917327580129,
            "unit": "s"
          },
          {
            "name": "llama / IPartOpt / tpu / PreRev",
            "value": 0.0003913836380088,
            "unit": "s"
          },
          {
            "name": "llama / IPartOpt / tpu / PostRev",
            "value": 0.0003721787579997,
            "unit": "s"
          },
          {
            "name": "llama / IPartOpt / tpu / BothRev",
            "value": 0.0003914667580102,
            "unit": "s"
          },
          {
            "name": "llama / IDefOpt / tpu / PreRev",
            "value": 0.0003917037379869,
            "unit": "s"
          },
          {
            "name": "llama / IDefOpt / tpu / PostRev",
            "value": 0.0003991957980033,
            "unit": "s"
          },
          {
            "name": "llama / IDefOpt / tpu / BothRev",
            "value": 0.0003915916580008,
            "unit": "s"
          },
          {
            "name": "llama / JaXPipe / cpu / Primal",
            "value": 0.0024898157400002,
            "unit": "s"
          },
          {
            "name": "llama / JaX   / cpu / Primal",
            "value": 0.0023771349799972,
            "unit": "s"
          },
          {
            "name": "llama / HLOOpt / cpu / Primal",
            "value": 0.0022154570900056,
            "unit": "s"
          },
          {
            "name": "llama / PartOpt / cpu / Primal",
            "value": 0.0021952235699973,
            "unit": "s"
          },
          {
            "name": "llama / DefOpt / cpu / Primal",
            "value": 0.0024513488199954,
            "unit": "s"
          },
          {
            "name": "llama / IPartOpt / cpu / Primal",
            "value": 0.0020983698100008,
            "unit": "s"
          },
          {
            "name": "llama / IDefOpt / cpu / Primal",
            "value": 0.0024916109600053,
            "unit": "s"
          },
          {
            "name": "llama / JaXPipe / cpu / Forward",
            "value": 0.0061849582600007,
            "unit": "s"
          },
          {
            "name": "llama / JaX   / cpu / Forward",
            "value": 0.0063810917800037,
            "unit": "s"
          },
          {
            "name": "llama / HLOOpt / cpu / Forward",
            "value": 0.0064787494400025,
            "unit": "s"
          },
          {
            "name": "llama / PartOpt / cpu / Forward",
            "value": 0.0063954231800016,
            "unit": "s"
          },
          {
            "name": "llama / DefOpt / cpu / Forward",
            "value": 0.0066075791499952,
            "unit": "s"
          },
          {
            "name": "llama / IPartOpt / cpu / Forward",
            "value": 0.0064581263800027,
            "unit": "s"
          },
          {
            "name": "llama / IDefOpt / cpu / Forward",
            "value": 0.006675883959997,
            "unit": "s"
          },
          {
            "name": "llama / JaXPipe / cpu / PreRev",
            "value": 0.0057965760799925,
            "unit": "s"
          },
          {
            "name": "llama / JaXPipe / cpu / PostRev",
            "value": 0.0059266901699993,
            "unit": "s"
          },
          {
            "name": "llama / JaXPipe / cpu / BothRev",
            "value": 0.0057145269000011,
            "unit": "s"
          },
          {
            "name": "llama / JaX   / cpu / BothRev",
            "value": 0.0054524416599997,
            "unit": "s"
          },
          {
            "name": "llama / HLOOpt / cpu / PreRev",
            "value": 0.0063360658100009,
            "unit": "s"
          },
          {
            "name": "llama / HLOOpt / cpu / PostRev",
            "value": 0.0052895589900072,
            "unit": "s"
          },
          {
            "name": "llama / HLOOpt / cpu / BothRev",
            "value": 0.0064189597400036,
            "unit": "s"
          },
          {
            "name": "llama / PartOpt / cpu / PreRev",
            "value": 0.0057521545599956,
            "unit": "s"
          },
          {
            "name": "llama / PartOpt / cpu / PostRev",
            "value": 0.0047957364300054,
            "unit": "s"
          },
          {
            "name": "llama / PartOpt / cpu / BothRev",
            "value": 0.0058596395499989,
            "unit": "s"
          },
          {
            "name": "llama / DefOpt / cpu / PreRev",
            "value": 0.0052668772800006,
            "unit": "s"
          },
          {
            "name": "llama / DefOpt / cpu / PostRev",
            "value": 0.0054896592199929,
            "unit": "s"
          },
          {
            "name": "llama / DefOpt / cpu / BothRev",
            "value": 0.0054978379699969,
            "unit": "s"
          },
          {
            "name": "llama / IPartOpt / cpu / PreRev",
            "value": 0.0064899603799949,
            "unit": "s"
          },
          {
            "name": "llama / IPartOpt / cpu / PostRev",
            "value": 0.0049984392500027,
            "unit": "s"
          },
          {
            "name": "llama / IPartOpt / cpu / BothRev",
            "value": 0.006257398990001,
            "unit": "s"
          },
          {
            "name": "llama / IDefOpt / cpu / PreRev",
            "value": 0.0057198707899988,
            "unit": "s"
          },
          {
            "name": "llama / IDefOpt / cpu / PostRev",
            "value": 0.0064488095399974,
            "unit": "s"
          },
          {
            "name": "llama / IDefOpt / cpu / BothRev",
            "value": 0.0055958025000018,
            "unit": "s"
          },
          {
            "name": "llama / JaXPipe / cpu / Primal",
            "value": 0.0016710741699898,
            "unit": "s"
          },
          {
            "name": "llama / JaX   / cpu / Primal",
            "value": 0.001699542919996,
            "unit": "s"
          },
          {
            "name": "llama / HLOOpt / cpu / Primal",
            "value": 0.0016913504100011,
            "unit": "s"
          },
          {
            "name": "llama / PartOpt / cpu / Primal",
            "value": 0.0017010541600029,
            "unit": "s"
          },
          {
            "name": "llama / DefOpt / cpu / Primal",
            "value": 0.0017505604199868,
            "unit": "s"
          },
          {
            "name": "llama / IPartOpt / cpu / Primal",
            "value": 0.0017948504199921,
            "unit": "s"
          },
          {
            "name": "llama / IDefOpt / cpu / Primal",
            "value": 0.0017075004199978,
            "unit": "s"
          },
          {
            "name": "llama / JaXPipe / cpu / Forward",
            "value": 0.0045039512500079,
            "unit": "s"
          },
          {
            "name": "llama / JaX   / cpu / Forward",
            "value": 0.004766165839992,
            "unit": "s"
          },
          {
            "name": "llama / HLOOpt / cpu / Forward",
            "value": 0.0045421933400029,
            "unit": "s"
          },
          {
            "name": "llama / PartOpt / cpu / Forward",
            "value": 0.0047433216699937,
            "unit": "s"
          },
          {
            "name": "llama / DefOpt / cpu / Forward",
            "value": 0.004382832500014,
            "unit": "s"
          },
          {
            "name": "llama / IPartOpt / cpu / Forward",
            "value": 0.0047151124999982,
            "unit": "s"
          },
          {
            "name": "llama / IDefOpt / cpu / Forward",
            "value": 0.0046351474999937,
            "unit": "s"
          },
          {
            "name": "llama / JaXPipe / cpu / PreRev",
            "value": 0.0069531716700112,
            "unit": "s"
          },
          {
            "name": "llama / JaXPipe / cpu / PostRev",
            "value": 0.0092743641699962,
            "unit": "s"
          },
          {
            "name": "llama / JaXPipe / cpu / BothRev",
            "value": 0.0058029483300015,
            "unit": "s"
          },
          {
            "name": "llama / JaX   / cpu / BothRev",
            "value": 0.0071942654099984,
            "unit": "s"
          },
          {
            "name": "llama / HLOOpt / cpu / PreRev",
            "value": 0.007205220830001,
            "unit": "s"
          },
          {
            "name": "llama / HLOOpt / cpu / PostRev",
            "value": 0.0052121395799986,
            "unit": "s"
          },
          {
            "name": "llama / HLOOpt / cpu / BothRev",
            "value": 0.0068570566599919,
            "unit": "s"
          },
          {
            "name": "llama / PartOpt / cpu / PreRev",
            "value": 0.0066600762499911,
            "unit": "s"
          },
          {
            "name": "llama / PartOpt / cpu / PostRev",
            "value": 0.0073194458299985,
            "unit": "s"
          },
          {
            "name": "llama / PartOpt / cpu / BothRev",
            "value": 0.0077074199999879,
            "unit": "s"
          },
          {
            "name": "llama / DefOpt / cpu / PreRev",
            "value": 0.0065348020799865,
            "unit": "s"
          },
          {
            "name": "llama / DefOpt / cpu / PostRev",
            "value": 0.0055175404200053,
            "unit": "s"
          },
          {
            "name": "llama / DefOpt / cpu / BothRev",
            "value": 0.0059383075000005,
            "unit": "s"
          },
          {
            "name": "llama / IPartOpt / cpu / PreRev",
            "value": 0.0060204737499952,
            "unit": "s"
          },
          {
            "name": "llama / IPartOpt / cpu / PostRev",
            "value": 0.0097610979099954,
            "unit": "s"
          },
          {
            "name": "llama / IPartOpt / cpu / BothRev",
            "value": 0.0058494204199996,
            "unit": "s"
          },
          {
            "name": "llama / IDefOpt / cpu / PreRev",
            "value": 0.0054790049999974,
            "unit": "s"
          },
          {
            "name": "llama / IDefOpt / cpu / PostRev",
            "value": 0.0113908925000032,
            "unit": "s"
          },
          {
            "name": "llama / IDefOpt / cpu / BothRev",
            "value": 0.0062457662500128,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004921021200016185,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004905543899985787,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000009337979300471487,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000009291691800171976,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / gpu / Primal",
            "value": 0.00008844819939986337,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / gpu / Primal",
            "value": 0.00009611055590066826,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004312023999955273,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004359792000468588,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001577131582998,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001581669382998,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000006865173000005597,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000006248296100056905,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004068800000095507,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004663770899969677,
            "unit": "s"
          },
          {
            "name": "slicing / JaX   / cpu / Primal",
            "value": 0.0000037464245000137454,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / Primal",
            "value": 0.000003752261900035592,
            "unit": "s"
          },
          {
            "name": "slicing / JaX   / cpu / Forward",
            "value": 0.000006112209699995219,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / Forward",
            "value": 0.000005968349000022499,
            "unit": "s"
          },
          {
            "name": "slicing / JaX   / cpu / BothRev",
            "value": 0.000006554401600033089,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / PreRev",
            "value": 0.000006622399700017922,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / PostRev",
            "value": 0.000006502300200008904,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / BothRev",
            "value": 0.0000064975537999998776,
            "unit": "s"
          },
          {
            "name": "slicing / JaX   / cpu / Primal",
            "value": 0.000007148238300578668,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / Primal",
            "value": 0.000007086030399659648,
            "unit": "s"
          },
          {
            "name": "slicing / JaX   / cpu / Forward",
            "value": 0.000011321523599326611,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / Forward",
            "value": 0.000011236860300414265,
            "unit": "s"
          },
          {
            "name": "slicing / JaX   / cpu / BothRev",
            "value": 0.000011396960201091133,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / PreRev",
            "value": 0.000011444578299415298,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / PostRev",
            "value": 0.000011495444100000897,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / BothRev",
            "value": 0.00001152646789996652,
            "unit": "s"
          },
          {
            "name": "slicing / JaX   / gpu / Primal",
            "value": 0.0001063942456996,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / gpu / Primal",
            "value": 0.0000954202038992662,
            "unit": "s"
          },
          {
            "name": "slicing / JaX   / gpu / Forward",
            "value": 0.0001040513929998,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / gpu / Forward",
            "value": 0.0001065859643,
            "unit": "s"
          },
          {
            "name": "slicing / JaX   / gpu / BothRev",
            "value": 0.0001055226078999,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / gpu / PreRev",
            "value": 0.0001114417225995,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / gpu / PostRev",
            "value": 0.0001065377810999,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / gpu / BothRev",
            "value": 0.0001061524710006,
            "unit": "s"
          },
          {
            "name": "slicing / JaX   / cpu / Primal",
            "value": 0.000003712004000408342,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / Primal",
            "value": 0.000003492096900299657,
            "unit": "s"
          },
          {
            "name": "slicing / JaX   / cpu / Forward",
            "value": 0.000004984095999679994,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / Forward",
            "value": 0.000005277221999858739,
            "unit": "s"
          },
          {
            "name": "slicing / JaX   / cpu / BothRev",
            "value": 0.000005706018899945775,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / PreRev",
            "value": 0.000005307306999748107,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / PostRev",
            "value": 0.0000056710470002144575,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / BothRev",
            "value": 0.000005675778900331352,
            "unit": "s"
          },
          {
            "name": "slicing / JaX   / tpu / Primal",
            "value": 0.0001556763282998,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / tpu / Primal",
            "value": 0.0001558257613003,
            "unit": "s"
          },
          {
            "name": "slicing / JaX   / tpu / Forward",
            "value": 0.0002342938569003,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / tpu / Forward",
            "value": 0.0002090604340999,
            "unit": "s"
          },
          {
            "name": "slicing / JaX   / tpu / BothRev",
            "value": 0.0002113666021003,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / tpu / PreRev",
            "value": 0.0002082768601001,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / tpu / PostRev",
            "value": 0.000215686301,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / tpu / BothRev",
            "value": 0.0002361931440005,
            "unit": "s"
          },
          {
            "name": "slicing / JaX   / cpu / Primal",
            "value": 0.0000053622080999957685,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / Primal",
            "value": 0.0000053820653000002495,
            "unit": "s"
          },
          {
            "name": "slicing / JaX   / cpu / Forward",
            "value": 0.000007395265600007405,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / Forward",
            "value": 0.000007376680000015767,
            "unit": "s"
          },
          {
            "name": "slicing / JaX   / cpu / BothRev",
            "value": 0.000007764177500030201,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / PreRev",
            "value": 0.00000780138359996272,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / PostRev",
            "value": 0.00000861557009993703,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / BothRev",
            "value": 0.000007768558400039182,
            "unit": "s"
          },
          {
            "name": "slicing / JaX   / cpu / Primal",
            "value": 0.0000034368417000223417,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / Primal",
            "value": 0.000003257233400108817,
            "unit": "s"
          },
          {
            "name": "slicing / JaX   / cpu / Forward",
            "value": 0.000005998695900052553,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / Forward",
            "value": 0.000005179987500014249,
            "unit": "s"
          },
          {
            "name": "slicing / JaX   / cpu / BothRev",
            "value": 0.000005434470900036103,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / PreRev",
            "value": 0.000005328195800029789,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / PostRev",
            "value": 0.000006080983399988327,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / BothRev",
            "value": 0.000005454937499962398,
            "unit": "s"
          },
          {
            "name": "sum    / JaX   / cpu / Primal",
            "value": 0.000005028060100039511,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / Primal",
            "value": 0.0000051917343999775765,
            "unit": "s"
          },
          {
            "name": "sum    / JaX   / cpu / Forward",
            "value": 0.000008924673299998176,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / Forward",
            "value": 0.000008960439699967538,
            "unit": "s"
          },
          {
            "name": "sum    / JaX   / cpu / BothRev",
            "value": 0.000007599694400005319,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / PreRev",
            "value": 0.000007424699100010912,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / PostRev",
            "value": 0.000007511679500021273,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / BothRev",
            "value": 0.000007494298899973728,
            "unit": "s"
          },
          {
            "name": "sum    / JaX   / cpu / Primal",
            "value": 0.00001086116270016646,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / Primal",
            "value": 0.000010287866900034716,
            "unit": "s"
          },
          {
            "name": "sum    / JaX   / cpu / Forward",
            "value": 0.00001603231670014793,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / Forward",
            "value": 0.00001596198000042932,
            "unit": "s"
          },
          {
            "name": "sum    / JaX   / cpu / BothRev",
            "value": 0.000014465044399548789,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / PreRev",
            "value": 0.000015375638600380625,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / PostRev",
            "value": 0.00001450329129875172,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / BothRev",
            "value": 0.000014577108899538871,
            "unit": "s"
          },
          {
            "name": "sum    / JaX   / gpu / Primal",
            "value": 0.00007810165539995069,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / gpu / Primal",
            "value": 0.00007442570220009657,
            "unit": "s"
          },
          {
            "name": "sum    / JaX   / gpu / Forward",
            "value": 0.0001005508109999,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / gpu / Forward",
            "value": 0.0000989883848000318,
            "unit": "s"
          },
          {
            "name": "sum    / JaX   / gpu / BothRev",
            "value": 0.0001032921389996,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / gpu / PreRev",
            "value": 0.0001199377919998,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / gpu / PostRev",
            "value": 0.0001041434875005,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / gpu / BothRev",
            "value": 0.0001052776102005,
            "unit": "s"
          },
          {
            "name": "sum    / JaX   / cpu / Primal",
            "value": 0.000004742906000319635,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / Primal",
            "value": 0.000004496113899949705,
            "unit": "s"
          },
          {
            "name": "sum    / JaX   / cpu / Forward",
            "value": 0.000008915306999551831,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / Forward",
            "value": 0.000006137958000181243,
            "unit": "s"
          },
          {
            "name": "sum    / JaX   / cpu / BothRev",
            "value": 0.000005913611999858404,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / PreRev",
            "value": 0.00000612112899980275,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / PostRev",
            "value": 0.000006050510999921244,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / BothRev",
            "value": 0.000006135057000210509,
            "unit": "s"
          },
          {
            "name": "sum    / JaX   / tpu / Primal",
            "value": 0.0001542791972999,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / tpu / Primal",
            "value": 0.0001495201333003,
            "unit": "s"
          },
          {
            "name": "sum    / JaX   / tpu / Forward",
            "value": 0.0002194713789998,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / tpu / Forward",
            "value": 0.0002152614580001,
            "unit": "s"
          },
          {
            "name": "sum    / JaX   / tpu / BothRev",
            "value": 0.0002150233119995,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / tpu / PreRev",
            "value": 0.000218504897,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / tpu / PostRev",
            "value": 0.0002194729640003,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / tpu / BothRev",
            "value": 0.0002056697840002,
            "unit": "s"
          },
          {
            "name": "sum    / JaX   / cpu / Primal",
            "value": 0.000006866329599961318,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / Primal",
            "value": 0.000006922589600071661,
            "unit": "s"
          },
          {
            "name": "sum    / JaX   / cpu / Forward",
            "value": 0.000010603489299955982,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / Forward",
            "value": 0.000010641032100011216,
            "unit": "s"
          },
          {
            "name": "sum    / JaX   / cpu / BothRev",
            "value": 0.000009782088399970236,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / PreRev",
            "value": 0.000009483931900012977,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / PostRev",
            "value": 0.000009372859300037816,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / BothRev",
            "value": 0.00000972276079992298,
            "unit": "s"
          },
          {
            "name": "sum    / JaX   / cpu / Primal",
            "value": 0.000004420666599980904,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / Primal",
            "value": 0.0000045057499999529685,
            "unit": "s"
          },
          {
            "name": "sum    / JaX   / cpu / Forward",
            "value": 0.000007459033399936743,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / Forward",
            "value": 0.000007891762499821198,
            "unit": "s"
          },
          {
            "name": "sum    / JaX   / cpu / BothRev",
            "value": 0.0000062954249999165765,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / PreRev",
            "value": 0.000006153520899897558,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / PostRev",
            "value": 0.000006345404199964832,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / BothRev",
            "value": 0.0000062641790998895885,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / JaXPipe / gpu / Primal",
            "value": 0.0010599539004033,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / JaX   / gpu / Primal",
            "value": 0.0010454086004756,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / HLOOpt / gpu / Primal",
            "value": 0.0009718950008391,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / PartOpt / gpu / Primal",
            "value": 0.0009905122991767,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / DefOpt / gpu / Primal",
            "value": 0.0006882260000566,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / IPartOpt / gpu / Primal",
            "value": 0.0009922832003212,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / IDefOpt / gpu / Primal",
            "value": 0.0007042857992928,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / JaX   / gpu / Forward",
            "value": 0.0014091449993429,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / JaXPipe / gpu / PostRev",
            "value": 0.0043419046007329,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / JaX   / gpu / BothRev",
            "value": 0.0043388513993704,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / HLOOpt / gpu / PostRev",
            "value": 0.0043327013001544,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / PartOpt / gpu / PostRev",
            "value": 0.0043929897001362,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / DefOpt / gpu / PostRev",
            "value": 0.0022839921002741,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / IPartOpt / gpu / PostRev",
            "value": 0.0043506029003765,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / IDefOpt / gpu / PostRev",
            "value": 0.0026334634996601,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / JaXPipe / tpu / Primal",
            "value": 0.0000903940002899617,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / JaX   / tpu / Primal",
            "value": 0.0001003730001684,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / HLOOpt / tpu / Primal",
            "value": 0.0001020600000629,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / PartOpt / tpu / Primal",
            "value": 0.0001003199999104,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / DefOpt / tpu / Primal",
            "value": 0.0001027930004056,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / IPartOpt / tpu / Primal",
            "value": 0.000104167000245,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / IDefOpt / tpu / Primal",
            "value": 0.0001010339998174,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / JaX   / tpu / Forward",
            "value": 0.000181128999975,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / JaXPipe / tpu / PostRev",
            "value": 0.0001919200003612,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / JaX   / tpu / BothRev",
            "value": 0.0002057589998003,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / HLOOpt / tpu / PostRev",
            "value": 0.0002082619997963,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / PartOpt / tpu / PostRev",
            "value": 0.000192737999896,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / DefOpt / tpu / PostRev",
            "value": 0.0002036300000327,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / IPartOpt / tpu / PostRev",
            "value": 0.0002129671003785,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / IDefOpt / tpu / PostRev",
            "value": 0.0001997679995838,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / JaXPipe / cpu / Primal",
            "value": 0.00005864360000487068,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / JaX   / cpu / Primal",
            "value": 0.00004913610000585322,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / HLOOpt / cpu / Primal",
            "value": 0.00008935850000852952,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / PartOpt / cpu / Primal",
            "value": 0.00008571960006520385,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / DefOpt / cpu / Primal",
            "value": 0.000076712900045095,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / IPartOpt / cpu / Primal",
            "value": 0.0000868738000463054,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / IDefOpt / cpu / Primal",
            "value": 0.0000836377999803517,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / JaX   / cpu / Forward",
            "value": 0.0001349326999843,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / JaXPipe / cpu / PostRev",
            "value": 0.0001436748999367,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / JaX   / cpu / BothRev",
            "value": 0.0001945138000337,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / HLOOpt / cpu / PostRev",
            "value": 0.0001597988999492,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / PartOpt / cpu / PostRev",
            "value": 0.0002198013999986,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / DefOpt / cpu / PostRev",
            "value": 0.0001216000000567,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / IPartOpt / cpu / PostRev",
            "value": 0.0001668513000367,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / IDefOpt / cpu / PostRev",
            "value": 0.0001676359000157,
            "unit": "s"
          }
        ]
      }
    ]
  }
}