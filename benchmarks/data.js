window.BENCHMARK_DATA = {
  "lastUpdate": 1766728975202,
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
          "id": "66133dbf44eaada5e8f6f18d888c1d0ec4f51667",
          "message": "feat: benchmarking infrastructure using xprof (#1831)\n\n* feat: benchmarking infrastructure using xprof\n\n* debug: set LD_DEBUG and TF_CPP_MAX_VLOG_LEVEL\n\n* test: use recent checkpoints for neuralgcm\n\n* ci: remove ld_debug\n\n* test: reduce number of llama evals\n\n* ci: print for debugging\n\n* fix: backends\n\n* fix: use op_profile to extract TPU timings\n\n* fix: path to cuda libs\n\n* chore: fmt\n\n* fix: attempt to fix cuda paths\n\n* debug: print contents of symlinked dir\n\n* chore: restore\n\n* fix: test\n\n* chore: restore",
          "timestamp": "2025-12-25T09:50:04-05:00",
          "tree_id": "d43f4801d7fed2431cba904645174eefa9d3af13",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/66133dbf44eaada5e8f6f18d888c1d0ec4f51667"
        },
        "date": 1766686759133,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "actmtch / JaXPipe / cpu / Primal",
            "value": 0.00000713281997377635,
            "unit": "s"
          },
          {
            "name": "actmtch / Jax / cpu / Primal",
            "value": 0.000006339999999909196,
            "unit": "s"
          },
          {
            "name": "actmtch / HLOOpt / cpu / Primal",
            "value": 0.000007280080017153523,
            "unit": "s"
          },
          {
            "name": "actmtch / PartOpt / cpu / Primal",
            "value": 0.000006684480013063876,
            "unit": "s"
          },
          {
            "name": "actmtch / IPartOpt / cpu / Primal",
            "value": 0.000006549319987243508,
            "unit": "s"
          },
          {
            "name": "actmtch / DefOpt / cpu / Primal",
            "value": 0.000007395840020762989,
            "unit": "s"
          },
          {
            "name": "actmtch / IDefOpt / cpu / Primal",
            "value": 0.000007337020006161765,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / Forward",
            "value": 0.00001092381994567404,
            "unit": "s"
          },
          {
            "name": "actmtch / Jax / cpu / Forward",
            "value": 0.000009964540013243094,
            "unit": "s"
          },
          {
            "name": "actmtch / HLOOpt / cpu / Forward",
            "value": 0.000010882579999815787,
            "unit": "s"
          },
          {
            "name": "actmtch / PartOpt / cpu / Forward",
            "value": 0.000010533580052651814,
            "unit": "s"
          },
          {
            "name": "actmtch / IPartOpt / cpu / Forward",
            "value": 0.000010967720008920878,
            "unit": "s"
          },
          {
            "name": "actmtch / DefOpt / cpu / Forward",
            "value": 0.00001072210001439089,
            "unit": "s"
          },
          {
            "name": "actmtch / IDefOpt / cpu / Forward",
            "value": 0.00001106340000660566,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / PreRev",
            "value": 0.000011070179989474126,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / PostRev",
            "value": 0.000009969440015993314,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / BothRev",
            "value": 0.00001166578003903851,
            "unit": "s"
          },
          {
            "name": "actmtch / Jax / cpu / BothRev",
            "value": 0.0000100311000005604,
            "unit": "s"
          },
          {
            "name": "actmtch / HLOOpt / cpu / PreRev",
            "value": 0.000011173419934493725,
            "unit": "s"
          },
          {
            "name": "actmtch / HLOOpt / cpu / PostRev",
            "value": 0.00001262077998035238,
            "unit": "s"
          },
          {
            "name": "actmtch / HLOOpt / cpu / BothRev",
            "value": 0.000010960199988403474,
            "unit": "s"
          },
          {
            "name": "actmtch / PartOpt / cpu / PreRev",
            "value": 0.000010591719992589788,
            "unit": "s"
          },
          {
            "name": "actmtch / PartOpt / cpu / PostRev",
            "value": 0.000010557359992162674,
            "unit": "s"
          },
          {
            "name": "actmtch / PartOpt / cpu / BothRev",
            "value": 0.00001126127997849835,
            "unit": "s"
          },
          {
            "name": "actmtch / IPartOpt / cpu / PreRev",
            "value": 0.000010461360016051911,
            "unit": "s"
          },
          {
            "name": "actmtch / IPartOpt / cpu / PostRev",
            "value": 0.000009731580012157791,
            "unit": "s"
          },
          {
            "name": "actmtch / IPartOpt / cpu / BothRev",
            "value": 0.000010840979966815211,
            "unit": "s"
          },
          {
            "name": "actmtch / DefOpt / cpu / PreRev",
            "value": 0.00001129805999880773,
            "unit": "s"
          },
          {
            "name": "actmtch / DefOpt / cpu / PostRev",
            "value": 0.000010375859956184286,
            "unit": "s"
          },
          {
            "name": "actmtch / DefOpt / cpu / BothRev",
            "value": 0.000010885640003834853,
            "unit": "s"
          },
          {
            "name": "actmtch / IDefOpt / cpu / PreRev",
            "value": 0.000010767719995783407,
            "unit": "s"
          },
          {
            "name": "actmtch / IDefOpt / cpu / PostRev",
            "value": 0.000011479120003059506,
            "unit": "s"
          },
          {
            "name": "actmtch / IDefOpt / cpu / BothRev",
            "value": 0.000010819939989232807,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cuda / Primal",
            "value": 0.000002015,
            "unit": "s"
          },
          {
            "name": "actmtch / Jax / cuda / Primal",
            "value": 0.000002015,
            "unit": "s"
          },
          {
            "name": "actmtch / HLOOpt / cuda / Primal",
            "value": 0.000002015,
            "unit": "s"
          },
          {
            "name": "actmtch / PartOpt / cuda / Primal",
            "value": 0.000002015,
            "unit": "s"
          },
          {
            "name": "actmtch / IPartOpt / cuda / Primal",
            "value": 0.000002015,
            "unit": "s"
          },
          {
            "name": "actmtch / DefOpt / cuda / Primal",
            "value": 0.000002015,
            "unit": "s"
          },
          {
            "name": "actmtch / IDefOpt / cuda / Primal",
            "value": 0.000002015,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cuda / Forward",
            "value": 0.000009856,
            "unit": "s"
          },
          {
            "name": "actmtch / Jax / cuda / Forward",
            "value": 0.000009856,
            "unit": "s"
          },
          {
            "name": "actmtch / HLOOpt / cuda / Forward",
            "value": 0.000009696,
            "unit": "s"
          },
          {
            "name": "actmtch / PartOpt / cuda / Forward",
            "value": 0.000009952,
            "unit": "s"
          },
          {
            "name": "actmtch / IPartOpt / cuda / Forward",
            "value": 0.000009824,
            "unit": "s"
          },
          {
            "name": "actmtch / DefOpt / cuda / Forward",
            "value": 0.000010112,
            "unit": "s"
          },
          {
            "name": "actmtch / IDefOpt / cuda / Forward",
            "value": 0.00001008,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cuda / PreRev",
            "value": 0.00000992,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cuda / PostRev",
            "value": 0.000009985,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cuda / BothRev",
            "value": 0.000010144,
            "unit": "s"
          },
          {
            "name": "actmtch / Jax / cuda / BothRev",
            "value": 0.000010272,
            "unit": "s"
          },
          {
            "name": "actmtch / HLOOpt / cuda / PreRev",
            "value": 0.000010432,
            "unit": "s"
          },
          {
            "name": "actmtch / HLOOpt / cuda / PostRev",
            "value": 0.000010016,
            "unit": "s"
          },
          {
            "name": "actmtch / HLOOpt / cuda / BothRev",
            "value": 0.000010368,
            "unit": "s"
          },
          {
            "name": "actmtch / PartOpt / cuda / PreRev",
            "value": 0.000009888,
            "unit": "s"
          },
          {
            "name": "actmtch / PartOpt / cuda / PostRev",
            "value": 0.00001008,
            "unit": "s"
          },
          {
            "name": "actmtch / PartOpt / cuda / BothRev",
            "value": 0.000010209,
            "unit": "s"
          },
          {
            "name": "actmtch / IPartOpt / cuda / PreRev",
            "value": 0.000010079,
            "unit": "s"
          },
          {
            "name": "actmtch / IPartOpt / cuda / PostRev",
            "value": 0.000010208,
            "unit": "s"
          },
          {
            "name": "actmtch / IPartOpt / cuda / BothRev",
            "value": 0.00000992,
            "unit": "s"
          },
          {
            "name": "actmtch / DefOpt / cuda / PreRev",
            "value": 0.000010303,
            "unit": "s"
          },
          {
            "name": "actmtch / DefOpt / cuda / PostRev",
            "value": 0.000009888,
            "unit": "s"
          },
          {
            "name": "actmtch / DefOpt / cuda / BothRev",
            "value": 0.000009953,
            "unit": "s"
          },
          {
            "name": "actmtch / IDefOpt / cuda / PreRev",
            "value": 0.00001056,
            "unit": "s"
          },
          {
            "name": "actmtch / IDefOpt / cuda / PostRev",
            "value": 0.000010272,
            "unit": "s"
          },
          {
            "name": "actmtch / IDefOpt / cuda / BothRev",
            "value": 0.000010336,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / tpu / Primal",
            "value": 5.632000000000001e-7,
            "unit": "s"
          },
          {
            "name": "actmtch / Jax / tpu / Primal",
            "value": 5.97e-7,
            "unit": "s"
          },
          {
            "name": "actmtch / HLOOpt / tpu / Primal",
            "value": 0.000002100575,
            "unit": "s"
          },
          {
            "name": "actmtch / PartOpt / tpu / Primal",
            "value": 5.964250000000001e-7,
            "unit": "s"
          },
          {
            "name": "actmtch / IPartOpt / tpu / Primal",
            "value": 5.525e-7,
            "unit": "s"
          },
          {
            "name": "actmtch / DefOpt / tpu / Primal",
            "value": 0.0000021614,
            "unit": "s"
          },
          {
            "name": "actmtch / IDefOpt / tpu / Primal",
            "value": 0.000002094925,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / tpu / Forward",
            "value": 0.000003830450000000001,
            "unit": "s"
          },
          {
            "name": "actmtch / Jax / tpu / Forward",
            "value": 0.000001206,
            "unit": "s"
          },
          {
            "name": "actmtch / HLOOpt / tpu / Forward",
            "value": 0.0000039277000000000005,
            "unit": "s"
          },
          {
            "name": "actmtch / PartOpt / tpu / Forward",
            "value": 0.0000039145500000000005,
            "unit": "s"
          },
          {
            "name": "actmtch / IPartOpt / tpu / Forward",
            "value": 0.000003941225,
            "unit": "s"
          },
          {
            "name": "actmtch / DefOpt / tpu / Forward",
            "value": 0.0000039124,
            "unit": "s"
          },
          {
            "name": "actmtch / IDefOpt / tpu / Forward",
            "value": 0.0000039432,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / tpu / PreRev",
            "value": 0.0000034772,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / tpu / PostRev",
            "value": 0.000001644325,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / tpu / BothRev",
            "value": 0.00000347105,
            "unit": "s"
          },
          {
            "name": "actmtch / Jax / tpu / BothRev",
            "value": 0.0000016378999999999998,
            "unit": "s"
          },
          {
            "name": "actmtch / HLOOpt / tpu / PreRev",
            "value": 0.000003480525,
            "unit": "s"
          },
          {
            "name": "actmtch / HLOOpt / tpu / PostRev",
            "value": 0.0000034096,
            "unit": "s"
          },
          {
            "name": "actmtch / HLOOpt / tpu / BothRev",
            "value": 0.00000346665,
            "unit": "s"
          },
          {
            "name": "actmtch / PartOpt / tpu / PreRev",
            "value": 0.0000034179,
            "unit": "s"
          },
          {
            "name": "actmtch / PartOpt / tpu / PostRev",
            "value": 0.0000015853,
            "unit": "s"
          },
          {
            "name": "actmtch / PartOpt / tpu / BothRev",
            "value": 0.000003403625,
            "unit": "s"
          },
          {
            "name": "actmtch / IPartOpt / tpu / PreRev",
            "value": 0.0000034651749999999994,
            "unit": "s"
          },
          {
            "name": "actmtch / IPartOpt / tpu / PostRev",
            "value": 0.00000163315,
            "unit": "s"
          },
          {
            "name": "actmtch / IPartOpt / tpu / BothRev",
            "value": 0.000003482425,
            "unit": "s"
          },
          {
            "name": "actmtch / DefOpt / tpu / PreRev",
            "value": 0.000003414375,
            "unit": "s"
          },
          {
            "name": "actmtch / DefOpt / tpu / PostRev",
            "value": 0.0000034209750000000003,
            "unit": "s"
          },
          {
            "name": "actmtch / DefOpt / tpu / BothRev",
            "value": 0.0000034067750000000005,
            "unit": "s"
          },
          {
            "name": "actmtch / IDefOpt / tpu / PreRev",
            "value": 0.00000347925,
            "unit": "s"
          },
          {
            "name": "actmtch / IDefOpt / tpu / PostRev",
            "value": 0.000003409275,
            "unit": "s"
          },
          {
            "name": "actmtch / IDefOpt / tpu / BothRev",
            "value": 0.0000034686,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / Primal",
            "value": 0.000016341,
            "unit": "s"
          },
          {
            "name": "actmtch / Jax / cpu / Primal",
            "value": 0.000016444,
            "unit": "s"
          },
          {
            "name": "actmtch / HLOOpt / cpu / Primal",
            "value": 0.000017334,
            "unit": "s"
          },
          {
            "name": "actmtch / PartOpt / cpu / Primal",
            "value": 0.000016272,
            "unit": "s"
          },
          {
            "name": "actmtch / IPartOpt / cpu / Primal",
            "value": 0.000016595,
            "unit": "s"
          },
          {
            "name": "actmtch / DefOpt / cpu / Primal",
            "value": 0.000017528,
            "unit": "s"
          },
          {
            "name": "actmtch / IDefOpt / cpu / Primal",
            "value": 0.00001717,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / Forward",
            "value": 0.000023761,
            "unit": "s"
          },
          {
            "name": "actmtch / Jax / cpu / Forward",
            "value": 0.000022194,
            "unit": "s"
          },
          {
            "name": "actmtch / HLOOpt / cpu / Forward",
            "value": 0.000023296,
            "unit": "s"
          },
          {
            "name": "actmtch / PartOpt / cpu / Forward",
            "value": 0.000023328,
            "unit": "s"
          },
          {
            "name": "actmtch / IPartOpt / cpu / Forward",
            "value": 0.000023496,
            "unit": "s"
          },
          {
            "name": "actmtch / DefOpt / cpu / Forward",
            "value": 0.000023855,
            "unit": "s"
          },
          {
            "name": "actmtch / IDefOpt / cpu / Forward",
            "value": 0.000023374,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / PreRev",
            "value": 0.000023763,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / PostRev",
            "value": 0.000021665,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / BothRev",
            "value": 0.000023871,
            "unit": "s"
          },
          {
            "name": "actmtch / Jax / cpu / BothRev",
            "value": 0.000021271,
            "unit": "s"
          },
          {
            "name": "actmtch / HLOOpt / cpu / PreRev",
            "value": 0.000023685,
            "unit": "s"
          },
          {
            "name": "actmtch / HLOOpt / cpu / PostRev",
            "value": 0.000023462,
            "unit": "s"
          },
          {
            "name": "actmtch / HLOOpt / cpu / BothRev",
            "value": 0.00002343,
            "unit": "s"
          },
          {
            "name": "actmtch / PartOpt / cpu / PreRev",
            "value": 0.000023694,
            "unit": "s"
          },
          {
            "name": "actmtch / PartOpt / cpu / PostRev",
            "value": 0.000021353,
            "unit": "s"
          },
          {
            "name": "actmtch / PartOpt / cpu / BothRev",
            "value": 0.000023772,
            "unit": "s"
          },
          {
            "name": "actmtch / IPartOpt / cpu / PreRev",
            "value": 0.000023578,
            "unit": "s"
          },
          {
            "name": "actmtch / IPartOpt / cpu / PostRev",
            "value": 0.000021537,
            "unit": "s"
          },
          {
            "name": "actmtch / IPartOpt / cpu / BothRev",
            "value": 0.000023783,
            "unit": "s"
          },
          {
            "name": "actmtch / DefOpt / cpu / PreRev",
            "value": 0.000023848000000000003,
            "unit": "s"
          },
          {
            "name": "actmtch / DefOpt / cpu / PostRev",
            "value": 0.000029312,
            "unit": "s"
          },
          {
            "name": "actmtch / DefOpt / cpu / BothRev",
            "value": 0.000023879,
            "unit": "s"
          },
          {
            "name": "actmtch / IDefOpt / cpu / PreRev",
            "value": 0.000023418,
            "unit": "s"
          },
          {
            "name": "actmtch / IDefOpt / cpu / PostRev",
            "value": 0.000023511,
            "unit": "s"
          },
          {
            "name": "actmtch / IDefOpt / cpu / BothRev",
            "value": 0.000023625,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / Primal",
            "value": 0.000008999999999999999,
            "unit": "s"
          },
          {
            "name": "actmtch / Jax / cpu / Primal",
            "value": 0.00001,
            "unit": "s"
          },
          {
            "name": "actmtch / HLOOpt / cpu / Primal",
            "value": 0.00001,
            "unit": "s"
          },
          {
            "name": "actmtch / PartOpt / cpu / Primal",
            "value": 0.000008999999999999999,
            "unit": "s"
          },
          {
            "name": "actmtch / IPartOpt / cpu / Primal",
            "value": 0.000008999999999999999,
            "unit": "s"
          },
          {
            "name": "actmtch / DefOpt / cpu / Primal",
            "value": 0.000008999999999999999,
            "unit": "s"
          },
          {
            "name": "actmtch / IDefOpt / cpu / Primal",
            "value": 0.000008999999999999999,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / Forward",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "actmtch / Jax / cpu / Forward",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "actmtch / HLOOpt / cpu / Forward",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "actmtch / PartOpt / cpu / Forward",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "actmtch / IPartOpt / cpu / Forward",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "actmtch / DefOpt / cpu / Forward",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "actmtch / IDefOpt / cpu / Forward",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / PreRev",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / PostRev",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / BothRev",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "actmtch / Jax / cpu / BothRev",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "actmtch / HLOOpt / cpu / PreRev",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "actmtch / HLOOpt / cpu / PostRev",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "actmtch / HLOOpt / cpu / BothRev",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "actmtch / PartOpt / cpu / PreRev",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "actmtch / PartOpt / cpu / PostRev",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "actmtch / PartOpt / cpu / BothRev",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "actmtch / IPartOpt / cpu / PreRev",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "actmtch / IPartOpt / cpu / PostRev",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "actmtch / IPartOpt / cpu / BothRev",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "actmtch / DefOpt / cpu / PreRev",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "actmtch / DefOpt / cpu / PostRev",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "actmtch / DefOpt / cpu / BothRev",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "actmtch / IDefOpt / cpu / PreRev",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "actmtch / IDefOpt / cpu / PostRev",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "actmtch / IDefOpt / cpu / BothRev",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / Primal",
            "value": 0.000006659000000581727,
            "unit": "s"
          },
          {
            "name": "add_one / Jax / cpu / Primal",
            "value": 0.000006576479972864036,
            "unit": "s"
          },
          {
            "name": "add_one / HLOOpt / cpu / Primal",
            "value": 0.0000066848000187746944,
            "unit": "s"
          },
          {
            "name": "add_one / PartOpt / cpu / Primal",
            "value": 0.000006537359977301093,
            "unit": "s"
          },
          {
            "name": "add_one / IPartOpt / cpu / Primal",
            "value": 0.000006783579992770683,
            "unit": "s"
          },
          {
            "name": "add_one / DefOpt / cpu / Primal",
            "value": 0.000006492540005638148,
            "unit": "s"
          },
          {
            "name": "add_one / IDefOpt / cpu / Primal",
            "value": 0.000006682360044578673,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / Forward",
            "value": 0.000009964399996533756,
            "unit": "s"
          },
          {
            "name": "add_one / Jax / cpu / Forward",
            "value": 0.000010458340038894676,
            "unit": "s"
          },
          {
            "name": "add_one / HLOOpt / cpu / Forward",
            "value": 0.000010503659987080028,
            "unit": "s"
          },
          {
            "name": "add_one / PartOpt / cpu / Forward",
            "value": 0.000009928980007316569,
            "unit": "s"
          },
          {
            "name": "add_one / IPartOpt / cpu / Forward",
            "value": 0.0000104195800031448,
            "unit": "s"
          },
          {
            "name": "add_one / DefOpt / cpu / Forward",
            "value": 0.000010610659956000744,
            "unit": "s"
          },
          {
            "name": "add_one / IDefOpt / cpu / Forward",
            "value": 0.000010212440029135906,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / PreRev",
            "value": 0.000011650119986370556,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / PostRev",
            "value": 0.00001176739995571552,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / BothRev",
            "value": 0.000012446319997252431,
            "unit": "s"
          },
          {
            "name": "add_one / Jax / cpu / BothRev",
            "value": 0.000011605139998209778,
            "unit": "s"
          },
          {
            "name": "add_one / HLOOpt / cpu / PreRev",
            "value": 0.00001167976000942872,
            "unit": "s"
          },
          {
            "name": "add_one / HLOOpt / cpu / PostRev",
            "value": 0.000013972540000395385,
            "unit": "s"
          },
          {
            "name": "add_one / HLOOpt / cpu / BothRev",
            "value": 0.000011715100017681834,
            "unit": "s"
          },
          {
            "name": "add_one / PartOpt / cpu / PreRev",
            "value": 0.000011190120021637997,
            "unit": "s"
          },
          {
            "name": "add_one / PartOpt / cpu / PostRev",
            "value": 0.000011890860023413551,
            "unit": "s"
          },
          {
            "name": "add_one / PartOpt / cpu / BothRev",
            "value": 0.00001255587994819507,
            "unit": "s"
          },
          {
            "name": "add_one / IPartOpt / cpu / PreRev",
            "value": 0.000011687179994623876,
            "unit": "s"
          },
          {
            "name": "add_one / IPartOpt / cpu / PostRev",
            "value": 0.000011444279998613638,
            "unit": "s"
          },
          {
            "name": "add_one / IPartOpt / cpu / BothRev",
            "value": 0.000011512159999256254,
            "unit": "s"
          },
          {
            "name": "add_one / DefOpt / cpu / PreRev",
            "value": 0.000011918979998881695,
            "unit": "s"
          },
          {
            "name": "add_one / DefOpt / cpu / PostRev",
            "value": 0.000011530939973454224,
            "unit": "s"
          },
          {
            "name": "add_one / DefOpt / cpu / BothRev",
            "value": 0.000011958420000155456,
            "unit": "s"
          },
          {
            "name": "add_one / IDefOpt / cpu / PreRev",
            "value": 0.000011380919968360104,
            "unit": "s"
          },
          {
            "name": "add_one / IDefOpt / cpu / PostRev",
            "value": 0.00001230832002875104,
            "unit": "s"
          },
          {
            "name": "add_one / IDefOpt / cpu / BothRev",
            "value": 0.000011778740017689416,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cuda / Primal",
            "value": 0.000001919,
            "unit": "s"
          },
          {
            "name": "add_one / Jax / cuda / Primal",
            "value": 0.0000019200000000000003,
            "unit": "s"
          },
          {
            "name": "add_one / HLOOpt / cuda / Primal",
            "value": 0.0000019200000000000003,
            "unit": "s"
          },
          {
            "name": "add_one / PartOpt / cuda / Primal",
            "value": 0.0000019200000000000003,
            "unit": "s"
          },
          {
            "name": "add_one / IPartOpt / cuda / Primal",
            "value": 0.0000019200000000000003,
            "unit": "s"
          },
          {
            "name": "add_one / DefOpt / cuda / Primal",
            "value": 0.0000019200000000000003,
            "unit": "s"
          },
          {
            "name": "add_one / IDefOpt / cuda / Primal",
            "value": 0.0000019200000000000003,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cuda / Forward",
            "value": 0.00001056,
            "unit": "s"
          },
          {
            "name": "add_one / Jax / cuda / Forward",
            "value": 0.000010144,
            "unit": "s"
          },
          {
            "name": "add_one / HLOOpt / cuda / Forward",
            "value": 0.000009888,
            "unit": "s"
          },
          {
            "name": "add_one / PartOpt / cuda / Forward",
            "value": 0.000009952,
            "unit": "s"
          },
          {
            "name": "add_one / IPartOpt / cuda / Forward",
            "value": 0.000010016,
            "unit": "s"
          },
          {
            "name": "add_one / DefOpt / cuda / Forward",
            "value": 0.000010144,
            "unit": "s"
          },
          {
            "name": "add_one / IDefOpt / cuda / Forward",
            "value": 0.000010175,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cuda / PreRev",
            "value": 0.000025216,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cuda / PostRev",
            "value": 0.000024896,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cuda / BothRev",
            "value": 0.000025088,
            "unit": "s"
          },
          {
            "name": "add_one / Jax / cuda / BothRev",
            "value": 0.00002608,
            "unit": "s"
          },
          {
            "name": "add_one / HLOOpt / cuda / PreRev",
            "value": 0.000025184,
            "unit": "s"
          },
          {
            "name": "add_one / HLOOpt / cuda / PostRev",
            "value": 0.00002464,
            "unit": "s"
          },
          {
            "name": "add_one / HLOOpt / cuda / BothRev",
            "value": 0.00002464,
            "unit": "s"
          },
          {
            "name": "add_one / PartOpt / cuda / PreRev",
            "value": 0.000024992,
            "unit": "s"
          },
          {
            "name": "add_one / PartOpt / cuda / PostRev",
            "value": 0.000025472000000000003,
            "unit": "s"
          },
          {
            "name": "add_one / PartOpt / cuda / BothRev",
            "value": 0.00002528,
            "unit": "s"
          },
          {
            "name": "add_one / IPartOpt / cuda / PreRev",
            "value": 0.000025023,
            "unit": "s"
          },
          {
            "name": "add_one / IPartOpt / cuda / PostRev",
            "value": 0.00002512,
            "unit": "s"
          },
          {
            "name": "add_one / IPartOpt / cuda / BothRev",
            "value": 0.000025408,
            "unit": "s"
          },
          {
            "name": "add_one / DefOpt / cuda / PreRev",
            "value": 0.000025216,
            "unit": "s"
          },
          {
            "name": "add_one / DefOpt / cuda / PostRev",
            "value": 0.000024928,
            "unit": "s"
          },
          {
            "name": "add_one / DefOpt / cuda / BothRev",
            "value": 0.000025216,
            "unit": "s"
          },
          {
            "name": "add_one / IDefOpt / cuda / PreRev",
            "value": 0.000025248,
            "unit": "s"
          },
          {
            "name": "add_one / IDefOpt / cuda / PostRev",
            "value": 0.000025408,
            "unit": "s"
          },
          {
            "name": "add_one / IDefOpt / cuda / BothRev",
            "value": 0.000024992,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / tpu / Primal",
            "value": 0.0000014259250000000002,
            "unit": "s"
          },
          {
            "name": "add_one / Jax / tpu / Primal",
            "value": 0.0000014046,
            "unit": "s"
          },
          {
            "name": "add_one / HLOOpt / tpu / Primal",
            "value": 0.0000014235999999999998,
            "unit": "s"
          },
          {
            "name": "add_one / PartOpt / tpu / Primal",
            "value": 0.0000014038999999999998,
            "unit": "s"
          },
          {
            "name": "add_one / IPartOpt / tpu / Primal",
            "value": 0.00000142825,
            "unit": "s"
          },
          {
            "name": "add_one / DefOpt / tpu / Primal",
            "value": 0.0000014073749999999998,
            "unit": "s"
          },
          {
            "name": "add_one / IDefOpt / tpu / Primal",
            "value": 0.0000014244,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / tpu / Forward",
            "value": 0.00000185515,
            "unit": "s"
          },
          {
            "name": "add_one / Jax / tpu / Forward",
            "value": 0.0000018396,
            "unit": "s"
          },
          {
            "name": "add_one / HLOOpt / tpu / Forward",
            "value": 0.0000018493,
            "unit": "s"
          },
          {
            "name": "add_one / PartOpt / tpu / Forward",
            "value": 0.00000183585,
            "unit": "s"
          },
          {
            "name": "add_one / IPartOpt / tpu / Forward",
            "value": 0.000001853275,
            "unit": "s"
          },
          {
            "name": "add_one / DefOpt / tpu / Forward",
            "value": 0.0000018479,
            "unit": "s"
          },
          {
            "name": "add_one / IDefOpt / tpu / Forward",
            "value": 0.000001846525,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / tpu / PreRev",
            "value": 0.0000022328,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / tpu / PostRev",
            "value": 0.0000022363750000000003,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / tpu / BothRev",
            "value": 0.000002242525,
            "unit": "s"
          },
          {
            "name": "add_one / Jax / tpu / BothRev",
            "value": 0.0000022507,
            "unit": "s"
          },
          {
            "name": "add_one / HLOOpt / tpu / PreRev",
            "value": 0.00000225035,
            "unit": "s"
          },
          {
            "name": "add_one / HLOOpt / tpu / PostRev",
            "value": 0.0000022378,
            "unit": "s"
          },
          {
            "name": "add_one / HLOOpt / tpu / BothRev",
            "value": 0.0000022373,
            "unit": "s"
          },
          {
            "name": "add_one / PartOpt / tpu / PreRev",
            "value": 0.000002240775,
            "unit": "s"
          },
          {
            "name": "add_one / PartOpt / tpu / PostRev",
            "value": 0.0000022342500000000003,
            "unit": "s"
          },
          {
            "name": "add_one / PartOpt / tpu / BothRev",
            "value": 0.000002242075,
            "unit": "s"
          },
          {
            "name": "add_one / IPartOpt / tpu / PreRev",
            "value": 0.0000022358500000000004,
            "unit": "s"
          },
          {
            "name": "add_one / IPartOpt / tpu / PostRev",
            "value": 0.000002243625,
            "unit": "s"
          },
          {
            "name": "add_one / IPartOpt / tpu / BothRev",
            "value": 0.0000022312,
            "unit": "s"
          },
          {
            "name": "add_one / DefOpt / tpu / PreRev",
            "value": 0.0000022413,
            "unit": "s"
          },
          {
            "name": "add_one / DefOpt / tpu / PostRev",
            "value": 0.0000022313,
            "unit": "s"
          },
          {
            "name": "add_one / DefOpt / tpu / BothRev",
            "value": 0.00000223185,
            "unit": "s"
          },
          {
            "name": "add_one / IDefOpt / tpu / PreRev",
            "value": 0.00000224095,
            "unit": "s"
          },
          {
            "name": "add_one / IDefOpt / tpu / PostRev",
            "value": 0.00000224465,
            "unit": "s"
          },
          {
            "name": "add_one / IDefOpt / tpu / BothRev",
            "value": 0.000002237925,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / Primal",
            "value": 0.000015945,
            "unit": "s"
          },
          {
            "name": "add_one / Jax / cpu / Primal",
            "value": 0.000015866,
            "unit": "s"
          },
          {
            "name": "add_one / HLOOpt / cpu / Primal",
            "value": 0.000015913,
            "unit": "s"
          },
          {
            "name": "add_one / PartOpt / cpu / Primal",
            "value": 0.000015995,
            "unit": "s"
          },
          {
            "name": "add_one / IPartOpt / cpu / Primal",
            "value": 0.000015932000000000003,
            "unit": "s"
          },
          {
            "name": "add_one / DefOpt / cpu / Primal",
            "value": 0.000016053,
            "unit": "s"
          },
          {
            "name": "add_one / IDefOpt / cpu / Primal",
            "value": 0.000016149,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / Forward",
            "value": 0.000021977,
            "unit": "s"
          },
          {
            "name": "add_one / Jax / cpu / Forward",
            "value": 0.000021241,
            "unit": "s"
          },
          {
            "name": "add_one / HLOOpt / cpu / Forward",
            "value": 0.000021205,
            "unit": "s"
          },
          {
            "name": "add_one / PartOpt / cpu / Forward",
            "value": 0.000021394,
            "unit": "s"
          },
          {
            "name": "add_one / IPartOpt / cpu / Forward",
            "value": 0.000021451,
            "unit": "s"
          },
          {
            "name": "add_one / DefOpt / cpu / Forward",
            "value": 0.000021657,
            "unit": "s"
          },
          {
            "name": "add_one / IDefOpt / cpu / Forward",
            "value": 0.000021295,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / PreRev",
            "value": 0.000029895,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / PostRev",
            "value": 0.000029777000000000003,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / BothRev",
            "value": 0.000023593,
            "unit": "s"
          },
          {
            "name": "add_one / Jax / cpu / BothRev",
            "value": 0.00002343,
            "unit": "s"
          },
          {
            "name": "add_one / HLOOpt / cpu / PreRev",
            "value": 0.000023988,
            "unit": "s"
          },
          {
            "name": "add_one / HLOOpt / cpu / PostRev",
            "value": 0.000024156,
            "unit": "s"
          },
          {
            "name": "add_one / HLOOpt / cpu / BothRev",
            "value": 0.000023569,
            "unit": "s"
          },
          {
            "name": "add_one / PartOpt / cpu / PreRev",
            "value": 0.000023824,
            "unit": "s"
          },
          {
            "name": "add_one / PartOpt / cpu / PostRev",
            "value": 0.000023663,
            "unit": "s"
          },
          {
            "name": "add_one / PartOpt / cpu / BothRev",
            "value": 0.000023797,
            "unit": "s"
          },
          {
            "name": "add_one / IPartOpt / cpu / PreRev",
            "value": 0.000023716,
            "unit": "s"
          },
          {
            "name": "add_one / IPartOpt / cpu / PostRev",
            "value": 0.000024071,
            "unit": "s"
          },
          {
            "name": "add_one / IPartOpt / cpu / BothRev",
            "value": 0.00002373,
            "unit": "s"
          },
          {
            "name": "add_one / DefOpt / cpu / PreRev",
            "value": 0.000024038,
            "unit": "s"
          },
          {
            "name": "add_one / DefOpt / cpu / PostRev",
            "value": 0.000024259,
            "unit": "s"
          },
          {
            "name": "add_one / DefOpt / cpu / BothRev",
            "value": 0.000024098,
            "unit": "s"
          },
          {
            "name": "add_one / IDefOpt / cpu / PreRev",
            "value": 0.000023777,
            "unit": "s"
          },
          {
            "name": "add_one / IDefOpt / cpu / PostRev",
            "value": 0.000023975,
            "unit": "s"
          },
          {
            "name": "add_one / IDefOpt / cpu / BothRev",
            "value": 0.000023648,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / Primal",
            "value": 0.000008,
            "unit": "s"
          },
          {
            "name": "add_one / Jax / cpu / Primal",
            "value": 0.000008,
            "unit": "s"
          },
          {
            "name": "add_one / HLOOpt / cpu / Primal",
            "value": 0.000008999999999999999,
            "unit": "s"
          },
          {
            "name": "add_one / PartOpt / cpu / Primal",
            "value": 0.000008999999999999999,
            "unit": "s"
          },
          {
            "name": "add_one / IPartOpt / cpu / Primal",
            "value": 0.000008,
            "unit": "s"
          },
          {
            "name": "add_one / DefOpt / cpu / Primal",
            "value": 0.000008999999999999999,
            "unit": "s"
          },
          {
            "name": "add_one / IDefOpt / cpu / Primal",
            "value": 0.000008999999999999999,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / Forward",
            "value": 0.000011,
            "unit": "s"
          },
          {
            "name": "add_one / Jax / cpu / Forward",
            "value": 0.000011,
            "unit": "s"
          },
          {
            "name": "add_one / HLOOpt / cpu / Forward",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "add_one / PartOpt / cpu / Forward",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "add_one / IPartOpt / cpu / Forward",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "add_one / DefOpt / cpu / Forward",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "add_one / IDefOpt / cpu / Forward",
            "value": 0.000011,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / PreRev",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / PostRev",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / BothRev",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "add_one / Jax / cpu / BothRev",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "add_one / HLOOpt / cpu / PreRev",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "add_one / HLOOpt / cpu / PostRev",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "add_one / HLOOpt / cpu / BothRev",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "add_one / PartOpt / cpu / PreRev",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "add_one / PartOpt / cpu / PostRev",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "add_one / PartOpt / cpu / BothRev",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "add_one / IPartOpt / cpu / PreRev",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "add_one / IPartOpt / cpu / PostRev",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "add_one / IPartOpt / cpu / BothRev",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "add_one / DefOpt / cpu / PreRev",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "add_one / DefOpt / cpu / PostRev",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "add_one / DefOpt / cpu / BothRev",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "add_one / IDefOpt / cpu / PreRev",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "add_one / IDefOpt / cpu / PostRev",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "add_one / IDefOpt / cpu / BothRev",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / Primal",
            "value": 0.000006892639967190917,
            "unit": "s"
          },
          {
            "name": "add_two / Jax / cpu / Primal",
            "value": 0.000006933860004210146,
            "unit": "s"
          },
          {
            "name": "add_two / HLOOpt / cpu / Primal",
            "value": 0.000007509379993280163,
            "unit": "s"
          },
          {
            "name": "add_two / PartOpt / cpu / Primal",
            "value": 0.000006904719966769335,
            "unit": "s"
          },
          {
            "name": "add_two / IPartOpt / cpu / Primal",
            "value": 0.000007135020032364992,
            "unit": "s"
          },
          {
            "name": "add_two / DefOpt / cpu / Primal",
            "value": 0.000006958459998713806,
            "unit": "s"
          },
          {
            "name": "add_two / IDefOpt / cpu / Primal",
            "value": 0.0000065641800119919935,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / Forward",
            "value": 0.000010514580008020858,
            "unit": "s"
          },
          {
            "name": "add_two / Jax / cpu / Forward",
            "value": 0.000010020859963333351,
            "unit": "s"
          },
          {
            "name": "add_two / HLOOpt / cpu / Forward",
            "value": 0.000010757099989859853,
            "unit": "s"
          },
          {
            "name": "add_two / PartOpt / cpu / Forward",
            "value": 0.000010200499991697142,
            "unit": "s"
          },
          {
            "name": "add_two / IPartOpt / cpu / Forward",
            "value": 0.000010306379981557256,
            "unit": "s"
          },
          {
            "name": "add_two / DefOpt / cpu / Forward",
            "value": 0.00001038984000842902,
            "unit": "s"
          },
          {
            "name": "add_two / IDefOpt / cpu / Forward",
            "value": 0.000010215420024906052,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / PreRev",
            "value": 0.000013841239997418598,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / PostRev",
            "value": 0.000014024039992364124,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / BothRev",
            "value": 0.000014294040029199096,
            "unit": "s"
          },
          {
            "name": "add_two / Jax / cpu / BothRev",
            "value": 0.000013598579989775315,
            "unit": "s"
          },
          {
            "name": "add_two / HLOOpt / cpu / PreRev",
            "value": 0.00001361135996376106,
            "unit": "s"
          },
          {
            "name": "add_two / HLOOpt / cpu / PostRev",
            "value": 0.000016257539991784144,
            "unit": "s"
          },
          {
            "name": "add_two / HLOOpt / cpu / BothRev",
            "value": 0.000014164219992380822,
            "unit": "s"
          },
          {
            "name": "add_two / PartOpt / cpu / PreRev",
            "value": 0.000013886520009691596,
            "unit": "s"
          },
          {
            "name": "add_two / PartOpt / cpu / PostRev",
            "value": 0.00001393359996654908,
            "unit": "s"
          },
          {
            "name": "add_two / PartOpt / cpu / BothRev",
            "value": 0.00001496692000728217,
            "unit": "s"
          },
          {
            "name": "add_two / IPartOpt / cpu / PreRev",
            "value": 0.000013847800000803544,
            "unit": "s"
          },
          {
            "name": "add_two / IPartOpt / cpu / PostRev",
            "value": 0.000014076920015213546,
            "unit": "s"
          },
          {
            "name": "add_two / IPartOpt / cpu / BothRev",
            "value": 0.000014492899990727893,
            "unit": "s"
          },
          {
            "name": "add_two / DefOpt / cpu / PreRev",
            "value": 0.00001408632003403909,
            "unit": "s"
          },
          {
            "name": "add_two / DefOpt / cpu / PostRev",
            "value": 0.000014134579969322658,
            "unit": "s"
          },
          {
            "name": "add_two / DefOpt / cpu / BothRev",
            "value": 0.000014213100002962163,
            "unit": "s"
          },
          {
            "name": "add_two / IDefOpt / cpu / PreRev",
            "value": 0.000013673720004589996,
            "unit": "s"
          },
          {
            "name": "add_two / IDefOpt / cpu / PostRev",
            "value": 0.000014411659985853476,
            "unit": "s"
          },
          {
            "name": "add_two / IDefOpt / cpu / BothRev",
            "value": 0.0000148106800406822,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cuda / Primal",
            "value": 0.000001887,
            "unit": "s"
          },
          {
            "name": "add_two / Jax / cuda / Primal",
            "value": 0.000001887,
            "unit": "s"
          },
          {
            "name": "add_two / HLOOpt / cuda / Primal",
            "value": 0.000001887,
            "unit": "s"
          },
          {
            "name": "add_two / PartOpt / cuda / Primal",
            "value": 0.000001888,
            "unit": "s"
          },
          {
            "name": "add_two / IPartOpt / cuda / Primal",
            "value": 0.000001887,
            "unit": "s"
          },
          {
            "name": "add_two / DefOpt / cuda / Primal",
            "value": 0.000001887,
            "unit": "s"
          },
          {
            "name": "add_two / IDefOpt / cuda / Primal",
            "value": 0.000001887,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cuda / Forward",
            "value": 0.000010048,
            "unit": "s"
          },
          {
            "name": "add_two / Jax / cuda / Forward",
            "value": 0.000009824,
            "unit": "s"
          },
          {
            "name": "add_two / HLOOpt / cuda / Forward",
            "value": 0.0000096,
            "unit": "s"
          },
          {
            "name": "add_two / PartOpt / cuda / Forward",
            "value": 0.000009696,
            "unit": "s"
          },
          {
            "name": "add_two / IPartOpt / cuda / Forward",
            "value": 0.000009985,
            "unit": "s"
          },
          {
            "name": "add_two / DefOpt / cuda / Forward",
            "value": 0.000009792,
            "unit": "s"
          },
          {
            "name": "add_two / IDefOpt / cuda / Forward",
            "value": 0.000009727,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cuda / PreRev",
            "value": 0.000031519,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cuda / PostRev",
            "value": 0.000032127000000000006,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cuda / BothRev",
            "value": 0.000032736,
            "unit": "s"
          },
          {
            "name": "add_two / Jax / cuda / BothRev",
            "value": 0.000032191,
            "unit": "s"
          },
          {
            "name": "add_two / HLOOpt / cuda / PreRev",
            "value": 0.000032384,
            "unit": "s"
          },
          {
            "name": "add_two / HLOOpt / cuda / PostRev",
            "value": 0.000032256,
            "unit": "s"
          },
          {
            "name": "add_two / HLOOpt / cuda / BothRev",
            "value": 0.000032384,
            "unit": "s"
          },
          {
            "name": "add_two / PartOpt / cuda / PreRev",
            "value": 0.000032255,
            "unit": "s"
          },
          {
            "name": "add_two / PartOpt / cuda / PostRev",
            "value": 0.000032384,
            "unit": "s"
          },
          {
            "name": "add_two / PartOpt / cuda / BothRev",
            "value": 0.000032032,
            "unit": "s"
          },
          {
            "name": "add_two / IPartOpt / cuda / PreRev",
            "value": 0.000033087,
            "unit": "s"
          },
          {
            "name": "add_two / IPartOpt / cuda / PostRev",
            "value": 0.00003152,
            "unit": "s"
          },
          {
            "name": "add_two / IPartOpt / cuda / BothRev",
            "value": 0.000032575,
            "unit": "s"
          },
          {
            "name": "add_two / DefOpt / cuda / PreRev",
            "value": 0.000032384,
            "unit": "s"
          },
          {
            "name": "add_two / DefOpt / cuda / PostRev",
            "value": 0.000031808000000000004,
            "unit": "s"
          },
          {
            "name": "add_two / DefOpt / cuda / BothRev",
            "value": 0.000032448,
            "unit": "s"
          },
          {
            "name": "add_two / IDefOpt / cuda / PreRev",
            "value": 0.000032256,
            "unit": "s"
          },
          {
            "name": "add_two / IDefOpt / cuda / PostRev",
            "value": 0.00003184,
            "unit": "s"
          },
          {
            "name": "add_two / IDefOpt / cuda / BothRev",
            "value": 0.000031776,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / tpu / Primal",
            "value": 0.0000014347750000000002,
            "unit": "s"
          },
          {
            "name": "add_two / Jax / tpu / Primal",
            "value": 0.000001478,
            "unit": "s"
          },
          {
            "name": "add_two / HLOOpt / tpu / Primal",
            "value": 0.0000014378750000000002,
            "unit": "s"
          },
          {
            "name": "add_two / PartOpt / tpu / Primal",
            "value": 0.0000014793249999999998,
            "unit": "s"
          },
          {
            "name": "add_two / IPartOpt / tpu / Primal",
            "value": 0.0000014442749999999998,
            "unit": "s"
          },
          {
            "name": "add_two / DefOpt / tpu / Primal",
            "value": 0.000001471275,
            "unit": "s"
          },
          {
            "name": "add_two / IDefOpt / tpu / Primal",
            "value": 0.000001440075,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / tpu / Forward",
            "value": 0.00000183065,
            "unit": "s"
          },
          {
            "name": "add_two / Jax / tpu / Forward",
            "value": 0.000001831625,
            "unit": "s"
          },
          {
            "name": "add_two / HLOOpt / tpu / Forward",
            "value": 0.000001825575,
            "unit": "s"
          },
          {
            "name": "add_two / PartOpt / tpu / Forward",
            "value": 0.000001826125,
            "unit": "s"
          },
          {
            "name": "add_two / IPartOpt / tpu / Forward",
            "value": 0.0000018336,
            "unit": "s"
          },
          {
            "name": "add_two / DefOpt / tpu / Forward",
            "value": 0.000001827625,
            "unit": "s"
          },
          {
            "name": "add_two / IDefOpt / tpu / Forward",
            "value": 0.0000018318,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / tpu / PreRev",
            "value": 0.000002840625,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / tpu / PostRev",
            "value": 0.0000027612500000000005,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / tpu / BothRev",
            "value": 0.0000028416,
            "unit": "s"
          },
          {
            "name": "add_two / Jax / tpu / BothRev",
            "value": 0.000002758925,
            "unit": "s"
          },
          {
            "name": "add_two / HLOOpt / tpu / PreRev",
            "value": 0.000002828875,
            "unit": "s"
          },
          {
            "name": "add_two / HLOOpt / tpu / PostRev",
            "value": 0.0000027550500000000004,
            "unit": "s"
          },
          {
            "name": "add_two / HLOOpt / tpu / BothRev",
            "value": 0.000002834525,
            "unit": "s"
          },
          {
            "name": "add_two / PartOpt / tpu / PreRev",
            "value": 0.0000027605000000000005,
            "unit": "s"
          },
          {
            "name": "add_two / PartOpt / tpu / PostRev",
            "value": 0.000002838175,
            "unit": "s"
          },
          {
            "name": "add_two / PartOpt / tpu / BothRev",
            "value": 0.000002754175,
            "unit": "s"
          },
          {
            "name": "add_two / IPartOpt / tpu / PreRev",
            "value": 0.0000028504000000000003,
            "unit": "s"
          },
          {
            "name": "add_two / IPartOpt / tpu / PostRev",
            "value": 0.000002743325,
            "unit": "s"
          },
          {
            "name": "add_two / IPartOpt / tpu / BothRev",
            "value": 0.000002840525,
            "unit": "s"
          },
          {
            "name": "add_two / DefOpt / tpu / PreRev",
            "value": 0.0000027612500000000005,
            "unit": "s"
          },
          {
            "name": "add_two / DefOpt / tpu / PostRev",
            "value": 0.000002837125,
            "unit": "s"
          },
          {
            "name": "add_two / DefOpt / tpu / BothRev",
            "value": 0.000002749525,
            "unit": "s"
          },
          {
            "name": "add_two / IDefOpt / tpu / PreRev",
            "value": 0.000002838625,
            "unit": "s"
          },
          {
            "name": "add_two / IDefOpt / tpu / PostRev",
            "value": 0.0000027566,
            "unit": "s"
          },
          {
            "name": "add_two / IDefOpt / tpu / BothRev",
            "value": 0.000002840325,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / Primal",
            "value": 0.000017015,
            "unit": "s"
          },
          {
            "name": "add_two / Jax / cpu / Primal",
            "value": 0.000016556999999999998,
            "unit": "s"
          },
          {
            "name": "add_two / HLOOpt / cpu / Primal",
            "value": 0.000016389999999999997,
            "unit": "s"
          },
          {
            "name": "add_two / PartOpt / cpu / Primal",
            "value": 0.000016462999999999998,
            "unit": "s"
          },
          {
            "name": "add_two / IPartOpt / cpu / Primal",
            "value": 0.000016492,
            "unit": "s"
          },
          {
            "name": "add_two / DefOpt / cpu / Primal",
            "value": 0.000016155,
            "unit": "s"
          },
          {
            "name": "add_two / IDefOpt / cpu / Primal",
            "value": 0.000016219000000000002,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / Forward",
            "value": 0.000022391,
            "unit": "s"
          },
          {
            "name": "add_two / Jax / cpu / Forward",
            "value": 0.000021814,
            "unit": "s"
          },
          {
            "name": "add_two / HLOOpt / cpu / Forward",
            "value": 0.000021874,
            "unit": "s"
          },
          {
            "name": "add_two / PartOpt / cpu / Forward",
            "value": 0.000022175,
            "unit": "s"
          },
          {
            "name": "add_two / IPartOpt / cpu / Forward",
            "value": 0.00002181,
            "unit": "s"
          },
          {
            "name": "add_two / DefOpt / cpu / Forward",
            "value": 0.000021978,
            "unit": "s"
          },
          {
            "name": "add_two / IDefOpt / cpu / Forward",
            "value": 0.000022321,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / PreRev",
            "value": 0.000028118,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / PostRev",
            "value": 0.000028093,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / BothRev",
            "value": 0.000028071,
            "unit": "s"
          },
          {
            "name": "add_two / Jax / cpu / BothRev",
            "value": 0.000028266,
            "unit": "s"
          },
          {
            "name": "add_two / HLOOpt / cpu / PreRev",
            "value": 0.000027768,
            "unit": "s"
          },
          {
            "name": "add_two / HLOOpt / cpu / PostRev",
            "value": 0.000027831,
            "unit": "s"
          },
          {
            "name": "add_two / HLOOpt / cpu / BothRev",
            "value": 0.000028114,
            "unit": "s"
          },
          {
            "name": "add_two / PartOpt / cpu / PreRev",
            "value": 0.000028145,
            "unit": "s"
          },
          {
            "name": "add_two / PartOpt / cpu / PostRev",
            "value": 0.000028245,
            "unit": "s"
          },
          {
            "name": "add_two / PartOpt / cpu / BothRev",
            "value": 0.000027868,
            "unit": "s"
          },
          {
            "name": "add_two / IPartOpt / cpu / PreRev",
            "value": 0.000027679,
            "unit": "s"
          },
          {
            "name": "add_two / IPartOpt / cpu / PostRev",
            "value": 0.000028069,
            "unit": "s"
          },
          {
            "name": "add_two / IPartOpt / cpu / BothRev",
            "value": 0.000027883,
            "unit": "s"
          },
          {
            "name": "add_two / DefOpt / cpu / PreRev",
            "value": 0.000027499,
            "unit": "s"
          },
          {
            "name": "add_two / DefOpt / cpu / PostRev",
            "value": 0.000028011,
            "unit": "s"
          },
          {
            "name": "add_two / DefOpt / cpu / BothRev",
            "value": 0.000027736,
            "unit": "s"
          },
          {
            "name": "add_two / IDefOpt / cpu / PreRev",
            "value": 0.000028068,
            "unit": "s"
          },
          {
            "name": "add_two / IDefOpt / cpu / PostRev",
            "value": 0.000028195,
            "unit": "s"
          },
          {
            "name": "add_two / IDefOpt / cpu / BothRev",
            "value": 0.000027894,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / Primal",
            "value": 0.000008999999999999999,
            "unit": "s"
          },
          {
            "name": "add_two / Jax / cpu / Primal",
            "value": 0.000008999999999999999,
            "unit": "s"
          },
          {
            "name": "add_two / HLOOpt / cpu / Primal",
            "value": 0.000008999999999999999,
            "unit": "s"
          },
          {
            "name": "add_two / PartOpt / cpu / Primal",
            "value": 0.000008999999999999999,
            "unit": "s"
          },
          {
            "name": "add_two / IPartOpt / cpu / Primal",
            "value": 0.000008999999999999999,
            "unit": "s"
          },
          {
            "name": "add_two / DefOpt / cpu / Primal",
            "value": 0.000008999999999999999,
            "unit": "s"
          },
          {
            "name": "add_two / IDefOpt / cpu / Primal",
            "value": 0.000008999999999999999,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / Forward",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "add_two / Jax / cpu / Forward",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "add_two / HLOOpt / cpu / Forward",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "add_two / PartOpt / cpu / Forward",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "add_two / IPartOpt / cpu / Forward",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "add_two / DefOpt / cpu / Forward",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "add_two / IDefOpt / cpu / Forward",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / PreRev",
            "value": 0.000016,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / PostRev",
            "value": 0.000016,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / BothRev",
            "value": 0.000016,
            "unit": "s"
          },
          {
            "name": "add_two / Jax / cpu / BothRev",
            "value": 0.000017,
            "unit": "s"
          },
          {
            "name": "add_two / HLOOpt / cpu / PreRev",
            "value": 0.000016,
            "unit": "s"
          },
          {
            "name": "add_two / HLOOpt / cpu / PostRev",
            "value": 0.000017,
            "unit": "s"
          },
          {
            "name": "add_two / HLOOpt / cpu / BothRev",
            "value": 0.000016,
            "unit": "s"
          },
          {
            "name": "add_two / PartOpt / cpu / PreRev",
            "value": 0.000016,
            "unit": "s"
          },
          {
            "name": "add_two / PartOpt / cpu / PostRev",
            "value": 0.000017,
            "unit": "s"
          },
          {
            "name": "add_two / PartOpt / cpu / BothRev",
            "value": 0.000016,
            "unit": "s"
          },
          {
            "name": "add_two / IPartOpt / cpu / PreRev",
            "value": 0.000016,
            "unit": "s"
          },
          {
            "name": "add_two / IPartOpt / cpu / PostRev",
            "value": 0.000016,
            "unit": "s"
          },
          {
            "name": "add_two / IPartOpt / cpu / BothRev",
            "value": 0.000015,
            "unit": "s"
          },
          {
            "name": "add_two / DefOpt / cpu / PreRev",
            "value": 0.000016,
            "unit": "s"
          },
          {
            "name": "add_two / DefOpt / cpu / PostRev",
            "value": 0.000016,
            "unit": "s"
          },
          {
            "name": "add_two / DefOpt / cpu / BothRev",
            "value": 0.000017,
            "unit": "s"
          },
          {
            "name": "add_two / IDefOpt / cpu / PreRev",
            "value": 0.000017,
            "unit": "s"
          },
          {
            "name": "add_two / IDefOpt / cpu / PostRev",
            "value": 0.000016,
            "unit": "s"
          },
          {
            "name": "add_two / IDefOpt / cpu / BothRev",
            "value": 0.000017,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / Primal",
            "value": 0.000006208280001374078,
            "unit": "s"
          },
          {
            "name": "cache / Jax / cpu / Primal",
            "value": 0.0000063180999859469015,
            "unit": "s"
          },
          {
            "name": "cache / HLOOpt / cpu / Primal",
            "value": 0.0000059947600129817145,
            "unit": "s"
          },
          {
            "name": "cache / PartOpt / cpu / Primal",
            "value": 0.000005927159972998197,
            "unit": "s"
          },
          {
            "name": "cache / IPartOpt / cpu / Primal",
            "value": 0.000006284619985308382,
            "unit": "s"
          },
          {
            "name": "cache / DefOpt / cpu / Primal",
            "value": 0.000006352519994834438,
            "unit": "s"
          },
          {
            "name": "cache / IDefOpt / cpu / Primal",
            "value": 0.000006789879980715341,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / Forward",
            "value": 0.000014292039995780216,
            "unit": "s"
          },
          {
            "name": "cache / Jax / cpu / Forward",
            "value": 0.000014286380037447087,
            "unit": "s"
          },
          {
            "name": "cache / HLOOpt / cpu / Forward",
            "value": 0.000016100380007628702,
            "unit": "s"
          },
          {
            "name": "cache / PartOpt / cpu / Forward",
            "value": 0.00001493561997449433,
            "unit": "s"
          },
          {
            "name": "cache / IPartOpt / cpu / Forward",
            "value": 0.000015857279977353754,
            "unit": "s"
          },
          {
            "name": "cache / DefOpt / cpu / Forward",
            "value": 0.00001508203996309021,
            "unit": "s"
          },
          {
            "name": "cache / IDefOpt / cpu / Forward",
            "value": 0.000015270519970727036,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / PreRev",
            "value": 0.0000160701800177776,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / PostRev",
            "value": 0.00002165167999010009,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / BothRev",
            "value": 0.000016645660007270636,
            "unit": "s"
          },
          {
            "name": "cache / Jax / cpu / BothRev",
            "value": 0.00002096367997182824,
            "unit": "s"
          },
          {
            "name": "cache / HLOOpt / cpu / PreRev",
            "value": 0.000016826599967316725,
            "unit": "s"
          },
          {
            "name": "cache / HLOOpt / cpu / PostRev",
            "value": 0.000017899080003189738,
            "unit": "s"
          },
          {
            "name": "cache / HLOOpt / cpu / BothRev",
            "value": 0.00001644926000153646,
            "unit": "s"
          },
          {
            "name": "cache / PartOpt / cpu / PreRev",
            "value": 0.00001535305999823322,
            "unit": "s"
          },
          {
            "name": "cache / PartOpt / cpu / PostRev",
            "value": 0.000020587000026353053,
            "unit": "s"
          },
          {
            "name": "cache / PartOpt / cpu / BothRev",
            "value": 0.000015736519981146557,
            "unit": "s"
          },
          {
            "name": "cache / IPartOpt / cpu / PreRev",
            "value": 0.00001527264000287687,
            "unit": "s"
          },
          {
            "name": "cache / IPartOpt / cpu / PostRev",
            "value": 0.00002052838002782664,
            "unit": "s"
          },
          {
            "name": "cache / IPartOpt / cpu / BothRev",
            "value": 0.000015178359981291576,
            "unit": "s"
          },
          {
            "name": "cache / DefOpt / cpu / PreRev",
            "value": 0.000015391139995699633,
            "unit": "s"
          },
          {
            "name": "cache / DefOpt / cpu / PostRev",
            "value": 0.000015205459985736523,
            "unit": "s"
          },
          {
            "name": "cache / DefOpt / cpu / BothRev",
            "value": 0.000015585520031891065,
            "unit": "s"
          },
          {
            "name": "cache / IDefOpt / cpu / PreRev",
            "value": 0.000015139099969019298,
            "unit": "s"
          },
          {
            "name": "cache / IDefOpt / cpu / PostRev",
            "value": 0.000015227000030790805,
            "unit": "s"
          },
          {
            "name": "cache / IDefOpt / cpu / BothRev",
            "value": 0.000015294940003514056,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cuda / Primal",
            "value": 0.000002335,
            "unit": "s"
          },
          {
            "name": "cache / Jax / cuda / Primal",
            "value": 0.000002336,
            "unit": "s"
          },
          {
            "name": "cache / HLOOpt / cuda / Primal",
            "value": 0.000002273,
            "unit": "s"
          },
          {
            "name": "cache / PartOpt / cuda / Primal",
            "value": 0.000002304,
            "unit": "s"
          },
          {
            "name": "cache / IPartOpt / cuda / Primal",
            "value": 0.000002335,
            "unit": "s"
          },
          {
            "name": "cache / DefOpt / cuda / Primal",
            "value": 0.000002272,
            "unit": "s"
          },
          {
            "name": "cache / IDefOpt / cuda / Primal",
            "value": 0.000002272,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cuda / Forward",
            "value": 0.000002336,
            "unit": "s"
          },
          {
            "name": "cache / Jax / cuda / Forward",
            "value": 0.0000023670000000000004,
            "unit": "s"
          },
          {
            "name": "cache / HLOOpt / cuda / Forward",
            "value": 0.000002336,
            "unit": "s"
          },
          {
            "name": "cache / PartOpt / cuda / Forward",
            "value": 0.0000023670000000000004,
            "unit": "s"
          },
          {
            "name": "cache / IPartOpt / cuda / Forward",
            "value": 0.000002336,
            "unit": "s"
          },
          {
            "name": "cache / DefOpt / cuda / Forward",
            "value": 0.000002272,
            "unit": "s"
          },
          {
            "name": "cache / IDefOpt / cuda / Forward",
            "value": 0.000002336,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cuda / PreRev",
            "value": 0.000010687,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cuda / PostRev",
            "value": 0.000010847,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cuda / BothRev",
            "value": 0.00001072,
            "unit": "s"
          },
          {
            "name": "cache / Jax / cuda / BothRev",
            "value": 0.00001104,
            "unit": "s"
          },
          {
            "name": "cache / HLOOpt / cuda / PreRev",
            "value": 0.00001312,
            "unit": "s"
          },
          {
            "name": "cache / HLOOpt / cuda / PostRev",
            "value": 0.000013088,
            "unit": "s"
          },
          {
            "name": "cache / HLOOpt / cuda / BothRev",
            "value": 0.000013088,
            "unit": "s"
          },
          {
            "name": "cache / PartOpt / cuda / PreRev",
            "value": 0.000010689,
            "unit": "s"
          },
          {
            "name": "cache / PartOpt / cuda / PostRev",
            "value": 0.000010752,
            "unit": "s"
          },
          {
            "name": "cache / PartOpt / cuda / BothRev",
            "value": 0.00001088,
            "unit": "s"
          },
          {
            "name": "cache / IPartOpt / cuda / PreRev",
            "value": 0.000010752,
            "unit": "s"
          },
          {
            "name": "cache / IPartOpt / cuda / PostRev",
            "value": 0.000010464,
            "unit": "s"
          },
          {
            "name": "cache / IPartOpt / cuda / BothRev",
            "value": 0.000011968,
            "unit": "s"
          },
          {
            "name": "cache / DefOpt / cuda / PreRev",
            "value": 0.000012736,
            "unit": "s"
          },
          {
            "name": "cache / DefOpt / cuda / PostRev",
            "value": 0.000010976,
            "unit": "s"
          },
          {
            "name": "cache / DefOpt / cuda / BothRev",
            "value": 0.000011007,
            "unit": "s"
          },
          {
            "name": "cache / IDefOpt / cuda / PreRev",
            "value": 0.00001072,
            "unit": "s"
          },
          {
            "name": "cache / IDefOpt / cuda / PostRev",
            "value": 0.000010432,
            "unit": "s"
          },
          {
            "name": "cache / IDefOpt / cuda / BothRev",
            "value": 0.000010272,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / tpu / Primal",
            "value": 0.00000247245,
            "unit": "s"
          },
          {
            "name": "cache / Jax / tpu / Primal",
            "value": 0.00000248475,
            "unit": "s"
          },
          {
            "name": "cache / HLOOpt / tpu / Primal",
            "value": 0.0000024782000000000003,
            "unit": "s"
          },
          {
            "name": "cache / PartOpt / tpu / Primal",
            "value": 0.00000247335,
            "unit": "s"
          },
          {
            "name": "cache / IPartOpt / tpu / Primal",
            "value": 0.000002474875,
            "unit": "s"
          },
          {
            "name": "cache / DefOpt / tpu / Primal",
            "value": 0.0000024653,
            "unit": "s"
          },
          {
            "name": "cache / IDefOpt / tpu / Primal",
            "value": 0.00000247745,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / tpu / Forward",
            "value": 0.00000353595,
            "unit": "s"
          },
          {
            "name": "cache / Jax / tpu / Forward",
            "value": 0.0000035281,
            "unit": "s"
          },
          {
            "name": "cache / HLOOpt / tpu / Forward",
            "value": 0.00000354365,
            "unit": "s"
          },
          {
            "name": "cache / PartOpt / tpu / Forward",
            "value": 0.00000353045,
            "unit": "s"
          },
          {
            "name": "cache / IPartOpt / tpu / Forward",
            "value": 0.000003554525,
            "unit": "s"
          },
          {
            "name": "cache / DefOpt / tpu / Forward",
            "value": 0.00000352865,
            "unit": "s"
          },
          {
            "name": "cache / IDefOpt / tpu / Forward",
            "value": 0.000003546525,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / tpu / PreRev",
            "value": 0.000004956274999999999,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / tpu / PostRev",
            "value": 0.0000049525250000000005,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / tpu / BothRev",
            "value": 0.0000049765,
            "unit": "s"
          },
          {
            "name": "cache / Jax / tpu / BothRev",
            "value": 0.00000501655,
            "unit": "s"
          },
          {
            "name": "cache / HLOOpt / tpu / PreRev",
            "value": 0.000003950025,
            "unit": "s"
          },
          {
            "name": "cache / HLOOpt / tpu / PostRev",
            "value": 0.000004126874999999999,
            "unit": "s"
          },
          {
            "name": "cache / HLOOpt / tpu / BothRev",
            "value": 0.0000039442,
            "unit": "s"
          },
          {
            "name": "cache / PartOpt / tpu / PreRev",
            "value": 0.000004969825,
            "unit": "s"
          },
          {
            "name": "cache / PartOpt / tpu / PostRev",
            "value": 0.000004958925,
            "unit": "s"
          },
          {
            "name": "cache / PartOpt / tpu / BothRev",
            "value": 0.0000049805,
            "unit": "s"
          },
          {
            "name": "cache / IPartOpt / tpu / PreRev",
            "value": 0.000004979225000000001,
            "unit": "s"
          },
          {
            "name": "cache / IPartOpt / tpu / PostRev",
            "value": 0.000004973025,
            "unit": "s"
          },
          {
            "name": "cache / IPartOpt / tpu / BothRev",
            "value": 0.0000049724,
            "unit": "s"
          },
          {
            "name": "cache / DefOpt / tpu / PreRev",
            "value": 0.000005000974999999999,
            "unit": "s"
          },
          {
            "name": "cache / DefOpt / tpu / PostRev",
            "value": 0.000004967875,
            "unit": "s"
          },
          {
            "name": "cache / DefOpt / tpu / BothRev",
            "value": 0.000004958125,
            "unit": "s"
          },
          {
            "name": "cache / IDefOpt / tpu / PreRev",
            "value": 0.00000496475,
            "unit": "s"
          },
          {
            "name": "cache / IDefOpt / tpu / PostRev",
            "value": 0.000004959200000000001,
            "unit": "s"
          },
          {
            "name": "cache / IDefOpt / tpu / BothRev",
            "value": 0.0000049626000000000006,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / Primal",
            "value": 0.000018722,
            "unit": "s"
          },
          {
            "name": "cache / Jax / cpu / Primal",
            "value": 0.000018335,
            "unit": "s"
          },
          {
            "name": "cache / HLOOpt / cpu / Primal",
            "value": 0.000018541,
            "unit": "s"
          },
          {
            "name": "cache / PartOpt / cpu / Primal",
            "value": 0.000018377,
            "unit": "s"
          },
          {
            "name": "cache / IPartOpt / cpu / Primal",
            "value": 0.00002911,
            "unit": "s"
          },
          {
            "name": "cache / DefOpt / cpu / Primal",
            "value": 0.000018167,
            "unit": "s"
          },
          {
            "name": "cache / IDefOpt / cpu / Primal",
            "value": 0.000018643,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / Forward",
            "value": 0.000020901,
            "unit": "s"
          },
          {
            "name": "cache / Jax / cpu / Forward",
            "value": 0.00002118,
            "unit": "s"
          },
          {
            "name": "cache / HLOOpt / cpu / Forward",
            "value": 0.000021282,
            "unit": "s"
          },
          {
            "name": "cache / PartOpt / cpu / Forward",
            "value": 0.000021185,
            "unit": "s"
          },
          {
            "name": "cache / IPartOpt / cpu / Forward",
            "value": 0.00002103,
            "unit": "s"
          },
          {
            "name": "cache / DefOpt / cpu / Forward",
            "value": 0.000021335,
            "unit": "s"
          },
          {
            "name": "cache / IDefOpt / cpu / Forward",
            "value": 0.000020767,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / PreRev",
            "value": 0.000021584,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / PostRev",
            "value": 0.000023284,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / BothRev",
            "value": 0.000021783,
            "unit": "s"
          },
          {
            "name": "cache / Jax / cpu / BothRev",
            "value": 0.000029595,
            "unit": "s"
          },
          {
            "name": "cache / HLOOpt / cpu / PreRev",
            "value": 0.000021659,
            "unit": "s"
          },
          {
            "name": "cache / HLOOpt / cpu / PostRev",
            "value": 0.000021187,
            "unit": "s"
          },
          {
            "name": "cache / HLOOpt / cpu / BothRev",
            "value": 0.000021513,
            "unit": "s"
          },
          {
            "name": "cache / PartOpt / cpu / PreRev",
            "value": 0.000021382000000000003,
            "unit": "s"
          },
          {
            "name": "cache / PartOpt / cpu / PostRev",
            "value": 0.00002737,
            "unit": "s"
          },
          {
            "name": "cache / PartOpt / cpu / BothRev",
            "value": 0.00002137,
            "unit": "s"
          },
          {
            "name": "cache / IPartOpt / cpu / PreRev",
            "value": 0.000021485,
            "unit": "s"
          },
          {
            "name": "cache / IPartOpt / cpu / PostRev",
            "value": 0.000023948,
            "unit": "s"
          },
          {
            "name": "cache / IPartOpt / cpu / BothRev",
            "value": 0.000022828,
            "unit": "s"
          },
          {
            "name": "cache / DefOpt / cpu / PreRev",
            "value": 0.0000221,
            "unit": "s"
          },
          {
            "name": "cache / DefOpt / cpu / PostRev",
            "value": 0.000022502,
            "unit": "s"
          },
          {
            "name": "cache / DefOpt / cpu / BothRev",
            "value": 0.000021013,
            "unit": "s"
          },
          {
            "name": "cache / IDefOpt / cpu / PreRev",
            "value": 0.00002206,
            "unit": "s"
          },
          {
            "name": "cache / IDefOpt / cpu / PostRev",
            "value": 0.000022125,
            "unit": "s"
          },
          {
            "name": "cache / IDefOpt / cpu / BothRev",
            "value": 0.000021393,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / Primal",
            "value": 0.000008,
            "unit": "s"
          },
          {
            "name": "cache / Jax / cpu / Primal",
            "value": 0.000008,
            "unit": "s"
          },
          {
            "name": "cache / HLOOpt / cpu / Primal",
            "value": 0.000008999999999999999,
            "unit": "s"
          },
          {
            "name": "cache / PartOpt / cpu / Primal",
            "value": 0.000008999999999999999,
            "unit": "s"
          },
          {
            "name": "cache / IPartOpt / cpu / Primal",
            "value": 0.000008,
            "unit": "s"
          },
          {
            "name": "cache / DefOpt / cpu / Primal",
            "value": 0.000008999999999999999,
            "unit": "s"
          },
          {
            "name": "cache / IDefOpt / cpu / Primal",
            "value": 0.000008,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / Forward",
            "value": 0.00001,
            "unit": "s"
          },
          {
            "name": "cache / Jax / cpu / Forward",
            "value": 0.00001,
            "unit": "s"
          },
          {
            "name": "cache / HLOOpt / cpu / Forward",
            "value": 0.00001,
            "unit": "s"
          },
          {
            "name": "cache / PartOpt / cpu / Forward",
            "value": 0.000011,
            "unit": "s"
          },
          {
            "name": "cache / IPartOpt / cpu / Forward",
            "value": 0.00001,
            "unit": "s"
          },
          {
            "name": "cache / DefOpt / cpu / Forward",
            "value": 0.000011,
            "unit": "s"
          },
          {
            "name": "cache / IDefOpt / cpu / Forward",
            "value": 0.000011,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / PreRev",
            "value": 0.000011,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / PostRev",
            "value": 0.000011,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / BothRev",
            "value": 0.000011,
            "unit": "s"
          },
          {
            "name": "cache / Jax / cpu / BothRev",
            "value": 0.000011,
            "unit": "s"
          },
          {
            "name": "cache / HLOOpt / cpu / PreRev",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "cache / HLOOpt / cpu / PostRev",
            "value": 0.000011,
            "unit": "s"
          },
          {
            "name": "cache / HLOOpt / cpu / BothRev",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "cache / PartOpt / cpu / PreRev",
            "value": 0.000011,
            "unit": "s"
          },
          {
            "name": "cache / PartOpt / cpu / PostRev",
            "value": 0.000034,
            "unit": "s"
          },
          {
            "name": "cache / PartOpt / cpu / BothRev",
            "value": 0.00001,
            "unit": "s"
          },
          {
            "name": "cache / IPartOpt / cpu / PreRev",
            "value": 0.00001,
            "unit": "s"
          },
          {
            "name": "cache / IPartOpt / cpu / PostRev",
            "value": 0.000011,
            "unit": "s"
          },
          {
            "name": "cache / IPartOpt / cpu / BothRev",
            "value": 0.000035000000000000004,
            "unit": "s"
          },
          {
            "name": "cache / DefOpt / cpu / PreRev",
            "value": 0.000011,
            "unit": "s"
          },
          {
            "name": "cache / DefOpt / cpu / PostRev",
            "value": 0.000011,
            "unit": "s"
          },
          {
            "name": "cache / DefOpt / cpu / BothRev",
            "value": 0.00001,
            "unit": "s"
          },
          {
            "name": "cache / IDefOpt / cpu / PreRev",
            "value": 0.000011,
            "unit": "s"
          },
          {
            "name": "cache / IDefOpt / cpu / PostRev",
            "value": 0.000011,
            "unit": "s"
          },
          {
            "name": "cache / IDefOpt / cpu / BothRev",
            "value": 0.000011,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / Primal",
            "value": 0.00000680945997373783,
            "unit": "s"
          },
          {
            "name": "Concat / Jax / cpu / Primal",
            "value": 0.00000685994000377832,
            "unit": "s"
          },
          {
            "name": "Concat / HLOOpt / cpu / Primal",
            "value": 0.000007261000018843333,
            "unit": "s"
          },
          {
            "name": "Concat / PartOpt / cpu / Primal",
            "value": 0.000006560999991052086,
            "unit": "s"
          },
          {
            "name": "Concat / IPartOpt / cpu / Primal",
            "value": 0.00000664078000227164,
            "unit": "s"
          },
          {
            "name": "Concat / DefOpt / cpu / Primal",
            "value": 0.000006791279938624939,
            "unit": "s"
          },
          {
            "name": "Concat / IDefOpt / cpu / Primal",
            "value": 0.000006834340010755113,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / Forward",
            "value": 0.000010419480004202342,
            "unit": "s"
          },
          {
            "name": "Concat / Jax / cpu / Forward",
            "value": 0.000009747220028657466,
            "unit": "s"
          },
          {
            "name": "Concat / HLOOpt / cpu / Forward",
            "value": 0.000010163860006286995,
            "unit": "s"
          },
          {
            "name": "Concat / PartOpt / cpu / Forward",
            "value": 0.000010052399966298254,
            "unit": "s"
          },
          {
            "name": "Concat / IPartOpt / cpu / Forward",
            "value": 0.000010310499992556289,
            "unit": "s"
          },
          {
            "name": "Concat / DefOpt / cpu / Forward",
            "value": 0.000010081060054289992,
            "unit": "s"
          },
          {
            "name": "Concat / IDefOpt / cpu / Forward",
            "value": 0.000009757579982760945,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / PreRev",
            "value": 0.000011575919988899842,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / PostRev",
            "value": 0.000011765639956138327,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / BothRev",
            "value": 0.000011440880025475052,
            "unit": "s"
          },
          {
            "name": "Concat / Jax / cpu / BothRev",
            "value": 0.000011740379995899275,
            "unit": "s"
          },
          {
            "name": "Concat / HLOOpt / cpu / PreRev",
            "value": 0.000012096380041839438,
            "unit": "s"
          },
          {
            "name": "Concat / HLOOpt / cpu / PostRev",
            "value": 0.000013298899993969826,
            "unit": "s"
          },
          {
            "name": "Concat / HLOOpt / cpu / BothRev",
            "value": 0.000011380740015738411,
            "unit": "s"
          },
          {
            "name": "Concat / PartOpt / cpu / PreRev",
            "value": 0.000011581600019781036,
            "unit": "s"
          },
          {
            "name": "Concat / PartOpt / cpu / PostRev",
            "value": 0.000011830000012196251,
            "unit": "s"
          },
          {
            "name": "Concat / PartOpt / cpu / BothRev",
            "value": 0.00001163582000117458,
            "unit": "s"
          },
          {
            "name": "Concat / IPartOpt / cpu / PreRev",
            "value": 0.000011728999961633237,
            "unit": "s"
          },
          {
            "name": "Concat / IPartOpt / cpu / PostRev",
            "value": 0.000011372859962648362,
            "unit": "s"
          },
          {
            "name": "Concat / IPartOpt / cpu / BothRev",
            "value": 0.00001142194001658936,
            "unit": "s"
          },
          {
            "name": "Concat / DefOpt / cpu / PreRev",
            "value": 0.000011049559971070266,
            "unit": "s"
          },
          {
            "name": "Concat / DefOpt / cpu / PostRev",
            "value": 0.000011627620033323182,
            "unit": "s"
          },
          {
            "name": "Concat / DefOpt / cpu / BothRev",
            "value": 0.00001125859998865053,
            "unit": "s"
          },
          {
            "name": "Concat / IDefOpt / cpu / PreRev",
            "value": 0.00001125570005569898,
            "unit": "s"
          },
          {
            "name": "Concat / IDefOpt / cpu / PostRev",
            "value": 0.000011331400010021751,
            "unit": "s"
          },
          {
            "name": "Concat / IDefOpt / cpu / BothRev",
            "value": 0.000011248080018049222,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cuda / Primal",
            "value": 0.0000019200000000000003,
            "unit": "s"
          },
          {
            "name": "Concat / Jax / cuda / Primal",
            "value": 0.0000019200000000000003,
            "unit": "s"
          },
          {
            "name": "Concat / HLOOpt / cuda / Primal",
            "value": 0.000001919,
            "unit": "s"
          },
          {
            "name": "Concat / PartOpt / cuda / Primal",
            "value": 0.000001888,
            "unit": "s"
          },
          {
            "name": "Concat / IPartOpt / cuda / Primal",
            "value": 0.0000019200000000000003,
            "unit": "s"
          },
          {
            "name": "Concat / DefOpt / cuda / Primal",
            "value": 0.0000019200000000000003,
            "unit": "s"
          },
          {
            "name": "Concat / IDefOpt / cuda / Primal",
            "value": 0.0000019200000000000003,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cuda / Forward",
            "value": 0.00001008,
            "unit": "s"
          },
          {
            "name": "Concat / Jax / cuda / Forward",
            "value": 0.000009984,
            "unit": "s"
          },
          {
            "name": "Concat / HLOOpt / cuda / Forward",
            "value": 0.000010048,
            "unit": "s"
          },
          {
            "name": "Concat / PartOpt / cuda / Forward",
            "value": 0.000010145,
            "unit": "s"
          },
          {
            "name": "Concat / IPartOpt / cuda / Forward",
            "value": 0.000009984,
            "unit": "s"
          },
          {
            "name": "Concat / DefOpt / cuda / Forward",
            "value": 0.000009984,
            "unit": "s"
          },
          {
            "name": "Concat / IDefOpt / cuda / Forward",
            "value": 0.000009985,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cuda / PreRev",
            "value": 0.000016607,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cuda / PostRev",
            "value": 0.000015552,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cuda / BothRev",
            "value": 0.00001648,
            "unit": "s"
          },
          {
            "name": "Concat / Jax / cuda / BothRev",
            "value": 0.000016448000000000002,
            "unit": "s"
          },
          {
            "name": "Concat / HLOOpt / cuda / PreRev",
            "value": 0.000016255999999999998,
            "unit": "s"
          },
          {
            "name": "Concat / HLOOpt / cuda / PostRev",
            "value": 0.000016063999999999997,
            "unit": "s"
          },
          {
            "name": "Concat / HLOOpt / cuda / BothRev",
            "value": 0.000016224,
            "unit": "s"
          },
          {
            "name": "Concat / PartOpt / cuda / PreRev",
            "value": 0.000016832,
            "unit": "s"
          },
          {
            "name": "Concat / PartOpt / cuda / PostRev",
            "value": 0.000016192,
            "unit": "s"
          },
          {
            "name": "Concat / PartOpt / cuda / BothRev",
            "value": 0.00001648,
            "unit": "s"
          },
          {
            "name": "Concat / IPartOpt / cuda / PreRev",
            "value": 0.000016576000000000002,
            "unit": "s"
          },
          {
            "name": "Concat / IPartOpt / cuda / PostRev",
            "value": 0.00001664,
            "unit": "s"
          },
          {
            "name": "Concat / IPartOpt / cuda / BothRev",
            "value": 0.000016383999999999998,
            "unit": "s"
          },
          {
            "name": "Concat / DefOpt / cuda / PreRev",
            "value": 0.00001696,
            "unit": "s"
          },
          {
            "name": "Concat / DefOpt / cuda / PostRev",
            "value": 0.000016319,
            "unit": "s"
          },
          {
            "name": "Concat / DefOpt / cuda / BothRev",
            "value": 0.00001664,
            "unit": "s"
          },
          {
            "name": "Concat / IDefOpt / cuda / PreRev",
            "value": 0.00001648,
            "unit": "s"
          },
          {
            "name": "Concat / IDefOpt / cuda / PostRev",
            "value": 0.00001552,
            "unit": "s"
          },
          {
            "name": "Concat / IDefOpt / cuda / BothRev",
            "value": 0.00001616,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / tpu / Primal",
            "value": 0.000001527525,
            "unit": "s"
          },
          {
            "name": "Concat / Jax / tpu / Primal",
            "value": 0.000001528825,
            "unit": "s"
          },
          {
            "name": "Concat / HLOOpt / tpu / Primal",
            "value": 0.00000152625,
            "unit": "s"
          },
          {
            "name": "Concat / PartOpt / tpu / Primal",
            "value": 0.0000015351,
            "unit": "s"
          },
          {
            "name": "Concat / IPartOpt / tpu / Primal",
            "value": 0.000001526125,
            "unit": "s"
          },
          {
            "name": "Concat / DefOpt / tpu / Primal",
            "value": 0.0000015325,
            "unit": "s"
          },
          {
            "name": "Concat / IDefOpt / tpu / Primal",
            "value": 0.000001542,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / tpu / Forward",
            "value": 0.0000015702250000000005,
            "unit": "s"
          },
          {
            "name": "Concat / Jax / tpu / Forward",
            "value": 0.000001549,
            "unit": "s"
          },
          {
            "name": "Concat / HLOOpt / tpu / Forward",
            "value": 0.00000158455,
            "unit": "s"
          },
          {
            "name": "Concat / PartOpt / tpu / Forward",
            "value": 0.00000155495,
            "unit": "s"
          },
          {
            "name": "Concat / IPartOpt / tpu / Forward",
            "value": 0.000001570625,
            "unit": "s"
          },
          {
            "name": "Concat / DefOpt / tpu / Forward",
            "value": 0.0000015566,
            "unit": "s"
          },
          {
            "name": "Concat / IDefOpt / tpu / Forward",
            "value": 0.0000015707750000000002,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / tpu / PreRev",
            "value": 0.000001993525,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / tpu / PostRev",
            "value": 0.00000208665,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / tpu / BothRev",
            "value": 0.0000019998250000000004,
            "unit": "s"
          },
          {
            "name": "Concat / Jax / tpu / BothRev",
            "value": 0.000002072025,
            "unit": "s"
          },
          {
            "name": "Concat / HLOOpt / tpu / PreRev",
            "value": 0.0000019958000000000004,
            "unit": "s"
          },
          {
            "name": "Concat / HLOOpt / tpu / PostRev",
            "value": 0.00000206755,
            "unit": "s"
          },
          {
            "name": "Concat / HLOOpt / tpu / BothRev",
            "value": 0.0000020007,
            "unit": "s"
          },
          {
            "name": "Concat / PartOpt / tpu / PreRev",
            "value": 0.0000020732,
            "unit": "s"
          },
          {
            "name": "Concat / PartOpt / tpu / PostRev",
            "value": 0.0000019972,
            "unit": "s"
          },
          {
            "name": "Concat / PartOpt / tpu / BothRev",
            "value": 0.000002075475,
            "unit": "s"
          },
          {
            "name": "Concat / IPartOpt / tpu / PreRev",
            "value": 0.00000199595,
            "unit": "s"
          },
          {
            "name": "Concat / IPartOpt / tpu / PostRev",
            "value": 0.00000207615,
            "unit": "s"
          },
          {
            "name": "Concat / IPartOpt / tpu / BothRev",
            "value": 0.0000020014,
            "unit": "s"
          },
          {
            "name": "Concat / DefOpt / tpu / PreRev",
            "value": 0.000002076375,
            "unit": "s"
          },
          {
            "name": "Concat / DefOpt / tpu / PostRev",
            "value": 0.000001996825,
            "unit": "s"
          },
          {
            "name": "Concat / DefOpt / tpu / BothRev",
            "value": 0.0000020678,
            "unit": "s"
          },
          {
            "name": "Concat / IDefOpt / tpu / PreRev",
            "value": 0.000002002125,
            "unit": "s"
          },
          {
            "name": "Concat / IDefOpt / tpu / PostRev",
            "value": 0.000002064275,
            "unit": "s"
          },
          {
            "name": "Concat / IDefOpt / tpu / BothRev",
            "value": 0.000001992175,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / Primal",
            "value": 0.000015841,
            "unit": "s"
          },
          {
            "name": "Concat / Jax / cpu / Primal",
            "value": 0.000025944,
            "unit": "s"
          },
          {
            "name": "Concat / HLOOpt / cpu / Primal",
            "value": 0.000015737000000000002,
            "unit": "s"
          },
          {
            "name": "Concat / PartOpt / cpu / Primal",
            "value": 0.000015774,
            "unit": "s"
          },
          {
            "name": "Concat / IPartOpt / cpu / Primal",
            "value": 0.000015619,
            "unit": "s"
          },
          {
            "name": "Concat / DefOpt / cpu / Primal",
            "value": 0.000015637000000000003,
            "unit": "s"
          },
          {
            "name": "Concat / IDefOpt / cpu / Primal",
            "value": 0.000020772,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / Forward",
            "value": 0.000021618,
            "unit": "s"
          },
          {
            "name": "Concat / Jax / cpu / Forward",
            "value": 0.000021132,
            "unit": "s"
          },
          {
            "name": "Concat / HLOOpt / cpu / Forward",
            "value": 0.000021248,
            "unit": "s"
          },
          {
            "name": "Concat / PartOpt / cpu / Forward",
            "value": 0.000021485,
            "unit": "s"
          },
          {
            "name": "Concat / IPartOpt / cpu / Forward",
            "value": 0.00002146,
            "unit": "s"
          },
          {
            "name": "Concat / DefOpt / cpu / Forward",
            "value": 0.000021291,
            "unit": "s"
          },
          {
            "name": "Concat / IDefOpt / cpu / Forward",
            "value": 0.000021604,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / PreRev",
            "value": 0.000024668,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / PostRev",
            "value": 0.000024498,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / BothRev",
            "value": 0.000024195,
            "unit": "s"
          },
          {
            "name": "Concat / Jax / cpu / BothRev",
            "value": 0.000023989,
            "unit": "s"
          },
          {
            "name": "Concat / HLOOpt / cpu / PreRev",
            "value": 0.000024239,
            "unit": "s"
          },
          {
            "name": "Concat / HLOOpt / cpu / PostRev",
            "value": 0.000024045,
            "unit": "s"
          },
          {
            "name": "Concat / HLOOpt / cpu / BothRev",
            "value": 0.000024081,
            "unit": "s"
          },
          {
            "name": "Concat / PartOpt / cpu / PreRev",
            "value": 0.000024068,
            "unit": "s"
          },
          {
            "name": "Concat / PartOpt / cpu / PostRev",
            "value": 0.000024036,
            "unit": "s"
          },
          {
            "name": "Concat / PartOpt / cpu / BothRev",
            "value": 0.000024122,
            "unit": "s"
          },
          {
            "name": "Concat / IPartOpt / cpu / PreRev",
            "value": 0.000024008,
            "unit": "s"
          },
          {
            "name": "Concat / IPartOpt / cpu / PostRev",
            "value": 0.000024387,
            "unit": "s"
          },
          {
            "name": "Concat / IPartOpt / cpu / BothRev",
            "value": 0.000024259,
            "unit": "s"
          },
          {
            "name": "Concat / DefOpt / cpu / PreRev",
            "value": 0.000024422,
            "unit": "s"
          },
          {
            "name": "Concat / DefOpt / cpu / PostRev",
            "value": 0.000024386,
            "unit": "s"
          },
          {
            "name": "Concat / DefOpt / cpu / BothRev",
            "value": 0.000023915,
            "unit": "s"
          },
          {
            "name": "Concat / IDefOpt / cpu / PreRev",
            "value": 0.000024321,
            "unit": "s"
          },
          {
            "name": "Concat / IDefOpt / cpu / PostRev",
            "value": 0.000024196,
            "unit": "s"
          },
          {
            "name": "Concat / IDefOpt / cpu / BothRev",
            "value": 0.000024062,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / Primal",
            "value": 0.000008999999999999999,
            "unit": "s"
          },
          {
            "name": "Concat / Jax / cpu / Primal",
            "value": 0.000008999999999999999,
            "unit": "s"
          },
          {
            "name": "Concat / HLOOpt / cpu / Primal",
            "value": 0.000008999999999999999,
            "unit": "s"
          },
          {
            "name": "Concat / PartOpt / cpu / Primal",
            "value": 0.000008999999999999999,
            "unit": "s"
          },
          {
            "name": "Concat / IPartOpt / cpu / Primal",
            "value": 0.000008999999999999999,
            "unit": "s"
          },
          {
            "name": "Concat / DefOpt / cpu / Primal",
            "value": 0.000008,
            "unit": "s"
          },
          {
            "name": "Concat / IDefOpt / cpu / Primal",
            "value": 0.000008,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / Forward",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "Concat / Jax / cpu / Forward",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "Concat / HLOOpt / cpu / Forward",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "Concat / PartOpt / cpu / Forward",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "Concat / IPartOpt / cpu / Forward",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "Concat / DefOpt / cpu / Forward",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "Concat / IDefOpt / cpu / Forward",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / PreRev",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / PostRev",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / BothRev",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "Concat / Jax / cpu / BothRev",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "Concat / HLOOpt / cpu / PreRev",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "Concat / HLOOpt / cpu / PostRev",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "Concat / HLOOpt / cpu / BothRev",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "Concat / PartOpt / cpu / PreRev",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "Concat / PartOpt / cpu / PostRev",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "Concat / PartOpt / cpu / BothRev",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "Concat / IPartOpt / cpu / PreRev",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "Concat / IPartOpt / cpu / PostRev",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "Concat / IPartOpt / cpu / BothRev",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "Concat / DefOpt / cpu / PreRev",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "Concat / DefOpt / cpu / PostRev",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "Concat / DefOpt / cpu / BothRev",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "Concat / IDefOpt / cpu / PreRev",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "Concat / IDefOpt / cpu / PostRev",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "Concat / IDefOpt / cpu / BothRev",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / cpu / Primal",
            "value": 0.000006517819956570747,
            "unit": "s"
          },
          {
            "name": "const_scatter / Jax / cpu / Primal",
            "value": 0.000006735540018780739,
            "unit": "s"
          },
          {
            "name": "const_scatter / HLOOpt / cpu / Primal",
            "value": 0.000007361260004472569,
            "unit": "s"
          },
          {
            "name": "const_scatter / PartOpt / cpu / Primal",
            "value": 0.000006297840009210632,
            "unit": "s"
          },
          {
            "name": "const_scatter / IPartOpt / cpu / Primal",
            "value": 0.0000062833599986333865,
            "unit": "s"
          },
          {
            "name": "const_scatter / DefOpt / cpu / Primal",
            "value": 0.0000069139400511630815,
            "unit": "s"
          },
          {
            "name": "const_scatter / IDefOpt / cpu / Primal",
            "value": 0.0000069470600101340094,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / cpu / Forward",
            "value": 0.000010408939997432754,
            "unit": "s"
          },
          {
            "name": "const_scatter / Jax / cpu / Forward",
            "value": 0.000009712399996715247,
            "unit": "s"
          },
          {
            "name": "const_scatter / HLOOpt / cpu / Forward",
            "value": 0.00001093629999559198,
            "unit": "s"
          },
          {
            "name": "const_scatter / PartOpt / cpu / Forward",
            "value": 0.00001031351998790342,
            "unit": "s"
          },
          {
            "name": "const_scatter / IPartOpt / cpu / Forward",
            "value": 0.0000109735800378985,
            "unit": "s"
          },
          {
            "name": "const_scatter / DefOpt / cpu / Forward",
            "value": 0.000010581439992165544,
            "unit": "s"
          },
          {
            "name": "const_scatter / IDefOpt / cpu / Forward",
            "value": 0.000010888860024351744,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / cpu / PreRev",
            "value": 0.0002880199600076,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / cpu / PostRev",
            "value": 0.0002795422999861,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / cpu / BothRev",
            "value": 0.0002823235399318,
            "unit": "s"
          },
          {
            "name": "const_scatter / Jax / cpu / BothRev",
            "value": 0.0002785030800077,
            "unit": "s"
          },
          {
            "name": "const_scatter / HLOOpt / cpu / PreRev",
            "value": 0.0002912244199433,
            "unit": "s"
          },
          {
            "name": "const_scatter / HLOOpt / cpu / PostRev",
            "value": 0.0002848686400193,
            "unit": "s"
          },
          {
            "name": "const_scatter / HLOOpt / cpu / BothRev",
            "value": 0.0002826136400199,
            "unit": "s"
          },
          {
            "name": "const_scatter / PartOpt / cpu / PreRev",
            "value": 0.0002812760599772,
            "unit": "s"
          },
          {
            "name": "const_scatter / PartOpt / cpu / PostRev",
            "value": 0.0002809123799943,
            "unit": "s"
          },
          {
            "name": "const_scatter / PartOpt / cpu / BothRev",
            "value": 0.0002803485599906,
            "unit": "s"
          },
          {
            "name": "const_scatter / IPartOpt / cpu / PreRev",
            "value": 0.0002837234400067,
            "unit": "s"
          },
          {
            "name": "const_scatter / IPartOpt / cpu / PostRev",
            "value": 0.0002823269600412,
            "unit": "s"
          },
          {
            "name": "const_scatter / IPartOpt / cpu / BothRev",
            "value": 0.0002820849200179,
            "unit": "s"
          },
          {
            "name": "const_scatter / DefOpt / cpu / PreRev",
            "value": 0.0002825540000412,
            "unit": "s"
          },
          {
            "name": "const_scatter / DefOpt / cpu / PostRev",
            "value": 0.000283008159995,
            "unit": "s"
          },
          {
            "name": "const_scatter / DefOpt / cpu / BothRev",
            "value": 0.0002790038200146,
            "unit": "s"
          },
          {
            "name": "const_scatter / IDefOpt / cpu / PreRev",
            "value": 0.0002833644399834,
            "unit": "s"
          },
          {
            "name": "const_scatter / IDefOpt / cpu / PostRev",
            "value": 0.000283859480005,
            "unit": "s"
          },
          {
            "name": "const_scatter / IDefOpt / cpu / BothRev",
            "value": 0.0002820218600299,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / cuda / Primal",
            "value": 0.000001887,
            "unit": "s"
          },
          {
            "name": "const_scatter / Jax / cuda / Primal",
            "value": 0.000001887,
            "unit": "s"
          },
          {
            "name": "const_scatter / HLOOpt / cuda / Primal",
            "value": 0.000001887,
            "unit": "s"
          },
          {
            "name": "const_scatter / PartOpt / cuda / Primal",
            "value": 0.000001887,
            "unit": "s"
          },
          {
            "name": "const_scatter / IPartOpt / cuda / Primal",
            "value": 0.000001887,
            "unit": "s"
          },
          {
            "name": "const_scatter / DefOpt / cuda / Primal",
            "value": 0.000001887,
            "unit": "s"
          },
          {
            "name": "const_scatter / IDefOpt / cuda / Primal",
            "value": 0.000001887,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / cuda / Forward",
            "value": 0.000010176,
            "unit": "s"
          },
          {
            "name": "const_scatter / Jax / cuda / Forward",
            "value": 0.000009759,
            "unit": "s"
          },
          {
            "name": "const_scatter / HLOOpt / cuda / Forward",
            "value": 0.000010048,
            "unit": "s"
          },
          {
            "name": "const_scatter / PartOpt / cuda / Forward",
            "value": 0.000010048,
            "unit": "s"
          },
          {
            "name": "const_scatter / IPartOpt / cuda / Forward",
            "value": 0.000009856,
            "unit": "s"
          },
          {
            "name": "const_scatter / DefOpt / cuda / Forward",
            "value": 0.000009953,
            "unit": "s"
          },
          {
            "name": "const_scatter / IDefOpt / cuda / Forward",
            "value": 0.000010016,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / cuda / PreRev",
            "value": 0.000016255999999999998,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / cuda / PostRev",
            "value": 0.000016255999999999998,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / cuda / BothRev",
            "value": 0.000016063000000000002,
            "unit": "s"
          },
          {
            "name": "const_scatter / Jax / cuda / BothRev",
            "value": 0.000016192,
            "unit": "s"
          },
          {
            "name": "const_scatter / HLOOpt / cuda / PreRev",
            "value": 0.000016608,
            "unit": "s"
          },
          {
            "name": "const_scatter / HLOOpt / cuda / PostRev",
            "value": 0.000016255999999999998,
            "unit": "s"
          },
          {
            "name": "const_scatter / HLOOpt / cuda / BothRev",
            "value": 0.000015648,
            "unit": "s"
          },
          {
            "name": "const_scatter / PartOpt / cuda / PreRev",
            "value": 0.000016255999999999998,
            "unit": "s"
          },
          {
            "name": "const_scatter / PartOpt / cuda / PostRev",
            "value": 0.000016927999999999998,
            "unit": "s"
          },
          {
            "name": "const_scatter / PartOpt / cuda / BothRev",
            "value": 0.000015712,
            "unit": "s"
          },
          {
            "name": "const_scatter / IPartOpt / cuda / PreRev",
            "value": 0.00001664,
            "unit": "s"
          },
          {
            "name": "const_scatter / IPartOpt / cuda / PostRev",
            "value": 0.000016255999999999998,
            "unit": "s"
          },
          {
            "name": "const_scatter / IPartOpt / cuda / BothRev",
            "value": 0.000016192,
            "unit": "s"
          },
          {
            "name": "const_scatter / DefOpt / cuda / PreRev",
            "value": 0.00001632,
            "unit": "s"
          },
          {
            "name": "const_scatter / DefOpt / cuda / PostRev",
            "value": 0.000016479,
            "unit": "s"
          },
          {
            "name": "const_scatter / DefOpt / cuda / BothRev",
            "value": 0.000016128,
            "unit": "s"
          },
          {
            "name": "const_scatter / IDefOpt / cuda / PreRev",
            "value": 0.000016832,
            "unit": "s"
          },
          {
            "name": "const_scatter / IDefOpt / cuda / PostRev",
            "value": 0.000016192,
            "unit": "s"
          },
          {
            "name": "const_scatter / IDefOpt / cuda / BothRev",
            "value": 0.000016416,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / tpu / Primal",
            "value": 0.000003786025,
            "unit": "s"
          },
          {
            "name": "const_scatter / Jax / tpu / Primal",
            "value": 0.000003809375,
            "unit": "s"
          },
          {
            "name": "const_scatter / HLOOpt / tpu / Primal",
            "value": 0.0000037999,
            "unit": "s"
          },
          {
            "name": "const_scatter / PartOpt / tpu / Primal",
            "value": 0.000003804,
            "unit": "s"
          },
          {
            "name": "const_scatter / IPartOpt / tpu / Primal",
            "value": 0.000003802075,
            "unit": "s"
          },
          {
            "name": "const_scatter / DefOpt / tpu / Primal",
            "value": 0.000003828525,
            "unit": "s"
          },
          {
            "name": "const_scatter / IDefOpt / tpu / Primal",
            "value": 0.00000379765,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / tpu / Forward",
            "value": 0.000006461425,
            "unit": "s"
          },
          {
            "name": "const_scatter / Jax / tpu / Forward",
            "value": 0.000006504925,
            "unit": "s"
          },
          {
            "name": "const_scatter / HLOOpt / tpu / Forward",
            "value": 0.000006455725,
            "unit": "s"
          },
          {
            "name": "const_scatter / PartOpt / tpu / Forward",
            "value": 0.000006500375,
            "unit": "s"
          },
          {
            "name": "const_scatter / IPartOpt / tpu / Forward",
            "value": 0.000006466675,
            "unit": "s"
          },
          {
            "name": "const_scatter / DefOpt / tpu / Forward",
            "value": 0.00000647215,
            "unit": "s"
          },
          {
            "name": "const_scatter / IDefOpt / tpu / Forward",
            "value": 0.000006480924999999999,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / tpu / PreRev",
            "value": 0.000006688075,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / tpu / PostRev",
            "value": 0.000006672975,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / tpu / BothRev",
            "value": 0.000006678150000000001,
            "unit": "s"
          },
          {
            "name": "const_scatter / Jax / tpu / BothRev",
            "value": 0.000006678525,
            "unit": "s"
          },
          {
            "name": "const_scatter / HLOOpt / tpu / PreRev",
            "value": 0.0000066666,
            "unit": "s"
          },
          {
            "name": "const_scatter / HLOOpt / tpu / PostRev",
            "value": 0.000006659000000000001,
            "unit": "s"
          },
          {
            "name": "const_scatter / HLOOpt / tpu / BothRev",
            "value": 0.000006699475,
            "unit": "s"
          },
          {
            "name": "const_scatter / PartOpt / tpu / PreRev",
            "value": 0.000006648425000000001,
            "unit": "s"
          },
          {
            "name": "const_scatter / PartOpt / tpu / PostRev",
            "value": 0.000006682625,
            "unit": "s"
          },
          {
            "name": "const_scatter / PartOpt / tpu / BothRev",
            "value": 0.00000666005,
            "unit": "s"
          },
          {
            "name": "const_scatter / IPartOpt / tpu / PreRev",
            "value": 0.000006676075,
            "unit": "s"
          },
          {
            "name": "const_scatter / IPartOpt / tpu / PostRev",
            "value": 0.00000664445,
            "unit": "s"
          },
          {
            "name": "const_scatter / IPartOpt / tpu / BothRev",
            "value": 0.000006668,
            "unit": "s"
          },
          {
            "name": "const_scatter / DefOpt / tpu / PreRev",
            "value": 0.0000066681,
            "unit": "s"
          },
          {
            "name": "const_scatter / DefOpt / tpu / PostRev",
            "value": 0.000006650825,
            "unit": "s"
          },
          {
            "name": "const_scatter / DefOpt / tpu / BothRev",
            "value": 0.000006653649999999999,
            "unit": "s"
          },
          {
            "name": "const_scatter / IDefOpt / tpu / PreRev",
            "value": 0.0000066552,
            "unit": "s"
          },
          {
            "name": "const_scatter / IDefOpt / tpu / PostRev",
            "value": 0.000006674099999999999,
            "unit": "s"
          },
          {
            "name": "const_scatter / IDefOpt / tpu / BothRev",
            "value": 0.0000066716,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / cpu / Primal",
            "value": 0.000016178,
            "unit": "s"
          },
          {
            "name": "const_scatter / Jax / cpu / Primal",
            "value": 0.000015453,
            "unit": "s"
          },
          {
            "name": "const_scatter / HLOOpt / cpu / Primal",
            "value": 0.000016632999999999998,
            "unit": "s"
          },
          {
            "name": "const_scatter / PartOpt / cpu / Primal",
            "value": 0.000015666,
            "unit": "s"
          },
          {
            "name": "const_scatter / IPartOpt / cpu / Primal",
            "value": 0.000015814,
            "unit": "s"
          },
          {
            "name": "const_scatter / DefOpt / cpu / Primal",
            "value": 0.000016284,
            "unit": "s"
          },
          {
            "name": "const_scatter / IDefOpt / cpu / Primal",
            "value": 0.000016255,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / cpu / Forward",
            "value": 0.00002216,
            "unit": "s"
          },
          {
            "name": "const_scatter / Jax / cpu / Forward",
            "value": 0.000020498,
            "unit": "s"
          },
          {
            "name": "const_scatter / HLOOpt / cpu / Forward",
            "value": 0.000027958,
            "unit": "s"
          },
          {
            "name": "const_scatter / PartOpt / cpu / Forward",
            "value": 0.000022059,
            "unit": "s"
          },
          {
            "name": "const_scatter / IPartOpt / cpu / Forward",
            "value": 0.000021859,
            "unit": "s"
          },
          {
            "name": "const_scatter / DefOpt / cpu / Forward",
            "value": 0.000022267,
            "unit": "s"
          },
          {
            "name": "const_scatter / IDefOpt / cpu / Forward",
            "value": 0.000022002,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / cpu / PreRev",
            "value": 0.000526559,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / cpu / PostRev",
            "value": 0.000521515,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / cpu / BothRev",
            "value": 0.000535829,
            "unit": "s"
          },
          {
            "name": "const_scatter / Jax / cpu / BothRev",
            "value": 0.000532086,
            "unit": "s"
          },
          {
            "name": "const_scatter / HLOOpt / cpu / PreRev",
            "value": 0.00056487,
            "unit": "s"
          },
          {
            "name": "const_scatter / HLOOpt / cpu / PostRev",
            "value": 0.000578136,
            "unit": "s"
          },
          {
            "name": "const_scatter / HLOOpt / cpu / BothRev",
            "value": 0.000545473,
            "unit": "s"
          },
          {
            "name": "const_scatter / PartOpt / cpu / PreRev",
            "value": 0.0005306509999999,
            "unit": "s"
          },
          {
            "name": "const_scatter / PartOpt / cpu / PostRev",
            "value": 0.0005200949999999,
            "unit": "s"
          },
          {
            "name": "const_scatter / PartOpt / cpu / BothRev",
            "value": 0.000533829,
            "unit": "s"
          },
          {
            "name": "const_scatter / IPartOpt / cpu / PreRev",
            "value": 0.0005377779999999,
            "unit": "s"
          },
          {
            "name": "const_scatter / IPartOpt / cpu / PostRev",
            "value": 0.000528133,
            "unit": "s"
          },
          {
            "name": "const_scatter / IPartOpt / cpu / BothRev",
            "value": 0.000534892,
            "unit": "s"
          },
          {
            "name": "const_scatter / DefOpt / cpu / PreRev",
            "value": 0.000532682,
            "unit": "s"
          },
          {
            "name": "const_scatter / DefOpt / cpu / PostRev",
            "value": 0.000529136,
            "unit": "s"
          },
          {
            "name": "const_scatter / DefOpt / cpu / BothRev",
            "value": 0.000522411,
            "unit": "s"
          },
          {
            "name": "const_scatter / IDefOpt / cpu / PreRev",
            "value": 0.000542308,
            "unit": "s"
          },
          {
            "name": "const_scatter / IDefOpt / cpu / PostRev",
            "value": 0.000522331,
            "unit": "s"
          },
          {
            "name": "const_scatter / IDefOpt / cpu / BothRev",
            "value": 0.000551683,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / cpu / Primal",
            "value": 0.000008999999999999999,
            "unit": "s"
          },
          {
            "name": "const_scatter / Jax / cpu / Primal",
            "value": 0.000008,
            "unit": "s"
          },
          {
            "name": "const_scatter / HLOOpt / cpu / Primal",
            "value": 0.000008999999999999999,
            "unit": "s"
          },
          {
            "name": "const_scatter / PartOpt / cpu / Primal",
            "value": 0.000008,
            "unit": "s"
          },
          {
            "name": "const_scatter / IPartOpt / cpu / Primal",
            "value": 0.000008,
            "unit": "s"
          },
          {
            "name": "const_scatter / DefOpt / cpu / Primal",
            "value": 0.000008999999999999999,
            "unit": "s"
          },
          {
            "name": "const_scatter / IDefOpt / cpu / Primal",
            "value": 0.000008,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / cpu / Forward",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "const_scatter / Jax / cpu / Forward",
            "value": 0.000011,
            "unit": "s"
          },
          {
            "name": "const_scatter / HLOOpt / cpu / Forward",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "const_scatter / PartOpt / cpu / Forward",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "const_scatter / IPartOpt / cpu / Forward",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "const_scatter / DefOpt / cpu / Forward",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "const_scatter / IDefOpt / cpu / Forward",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / cpu / PreRev",
            "value": 0.0003599999999999,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / cpu / PostRev",
            "value": 0.000343,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / cpu / BothRev",
            "value": 0.000375,
            "unit": "s"
          },
          {
            "name": "const_scatter / Jax / cpu / BothRev",
            "value": 0.000356,
            "unit": "s"
          },
          {
            "name": "const_scatter / HLOOpt / cpu / PreRev",
            "value": 0.000376,
            "unit": "s"
          },
          {
            "name": "const_scatter / HLOOpt / cpu / PostRev",
            "value": 0.000357,
            "unit": "s"
          },
          {
            "name": "const_scatter / HLOOpt / cpu / BothRev",
            "value": 0.000321,
            "unit": "s"
          },
          {
            "name": "const_scatter / PartOpt / cpu / PreRev",
            "value": 0.00035,
            "unit": "s"
          },
          {
            "name": "const_scatter / PartOpt / cpu / PostRev",
            "value": 0.000341,
            "unit": "s"
          },
          {
            "name": "const_scatter / PartOpt / cpu / BothRev",
            "value": 0.0003459999999999,
            "unit": "s"
          },
          {
            "name": "const_scatter / IPartOpt / cpu / PreRev",
            "value": 0.000345,
            "unit": "s"
          },
          {
            "name": "const_scatter / IPartOpt / cpu / PostRev",
            "value": 0.000365,
            "unit": "s"
          },
          {
            "name": "const_scatter / IPartOpt / cpu / BothRev",
            "value": 0.000367,
            "unit": "s"
          },
          {
            "name": "const_scatter / DefOpt / cpu / PreRev",
            "value": 0.0003689999999999,
            "unit": "s"
          },
          {
            "name": "const_scatter / DefOpt / cpu / PostRev",
            "value": 0.000338,
            "unit": "s"
          },
          {
            "name": "const_scatter / DefOpt / cpu / BothRev",
            "value": 0.000366,
            "unit": "s"
          },
          {
            "name": "const_scatter / IDefOpt / cpu / PreRev",
            "value": 0.0003689999999999,
            "unit": "s"
          },
          {
            "name": "const_scatter / IDefOpt / cpu / PostRev",
            "value": 0.000381,
            "unit": "s"
          },
          {
            "name": "const_scatter / IDefOpt / cpu / BothRev",
            "value": 0.000363,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / Primal",
            "value": 0.000006996919955781777,
            "unit": "s"
          },
          {
            "name": "GenDot / Jax / cpu / Primal",
            "value": 0.000007466960050805938,
            "unit": "s"
          },
          {
            "name": "GenDot / HLOOpt / cpu / Primal",
            "value": 0.000007573020011477638,
            "unit": "s"
          },
          {
            "name": "GenDot / PartOpt / cpu / Primal",
            "value": 0.00000727000000551925,
            "unit": "s"
          },
          {
            "name": "GenDot / IPartOpt / cpu / Primal",
            "value": 0.000007262939989232109,
            "unit": "s"
          },
          {
            "name": "GenDot / DefOpt / cpu / Primal",
            "value": 0.000007189280004240572,
            "unit": "s"
          },
          {
            "name": "GenDot / IDefOpt / cpu / Primal",
            "value": 0.000006937339976502699,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / Forward",
            "value": 0.00001109208003981621,
            "unit": "s"
          },
          {
            "name": "GenDot / Jax / cpu / Forward",
            "value": 0.000010694279944800656,
            "unit": "s"
          },
          {
            "name": "GenDot / HLOOpt / cpu / Forward",
            "value": 0.000011425719985709292,
            "unit": "s"
          },
          {
            "name": "GenDot / PartOpt / cpu / Forward",
            "value": 0.000011010379994331744,
            "unit": "s"
          },
          {
            "name": "GenDot / IPartOpt / cpu / Forward",
            "value": 0.000011673500002871151,
            "unit": "s"
          },
          {
            "name": "GenDot / DefOpt / cpu / Forward",
            "value": 0.000011217820037927595,
            "unit": "s"
          },
          {
            "name": "GenDot / IDefOpt / cpu / Forward",
            "value": 0.000011160699987158296,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / PreRev",
            "value": 0.000011052339996240334,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / PostRev",
            "value": 0.000010355519953009206,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / BothRev",
            "value": 0.000011711480019584996,
            "unit": "s"
          },
          {
            "name": "GenDot / Jax / cpu / BothRev",
            "value": 0.000010143320014321944,
            "unit": "s"
          },
          {
            "name": "GenDot / HLOOpt / cpu / PreRev",
            "value": 0.00001173204003862338,
            "unit": "s"
          },
          {
            "name": "GenDot / HLOOpt / cpu / PostRev",
            "value": 0.000013266199948702706,
            "unit": "s"
          },
          {
            "name": "GenDot / HLOOpt / cpu / BothRev",
            "value": 0.00001131287996940955,
            "unit": "s"
          },
          {
            "name": "GenDot / PartOpt / cpu / PreRev",
            "value": 0.00001143761996900139,
            "unit": "s"
          },
          {
            "name": "GenDot / PartOpt / cpu / PostRev",
            "value": 0.000010051760045826088,
            "unit": "s"
          },
          {
            "name": "GenDot / PartOpt / cpu / BothRev",
            "value": 0.00001174368006104487,
            "unit": "s"
          },
          {
            "name": "GenDot / IPartOpt / cpu / PreRev",
            "value": 0.000011260359970037826,
            "unit": "s"
          },
          {
            "name": "GenDot / IPartOpt / cpu / PostRev",
            "value": 0.000009902019992296118,
            "unit": "s"
          },
          {
            "name": "GenDot / IPartOpt / cpu / BothRev",
            "value": 0.000011261819963692687,
            "unit": "s"
          },
          {
            "name": "GenDot / DefOpt / cpu / PreRev",
            "value": 0.000011598100036280811,
            "unit": "s"
          },
          {
            "name": "GenDot / DefOpt / cpu / PostRev",
            "value": 0.00001124961999266816,
            "unit": "s"
          },
          {
            "name": "GenDot / DefOpt / cpu / BothRev",
            "value": 0.000011092819977420732,
            "unit": "s"
          },
          {
            "name": "GenDot / IDefOpt / cpu / PreRev",
            "value": 0.000011347700010446716,
            "unit": "s"
          },
          {
            "name": "GenDot / IDefOpt / cpu / PostRev",
            "value": 0.000012072240006091306,
            "unit": "s"
          },
          {
            "name": "GenDot / IDefOpt / cpu / BothRev",
            "value": 0.00001116171999456128,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cuda / Primal",
            "value": 0.000002015,
            "unit": "s"
          },
          {
            "name": "GenDot / Jax / cuda / Primal",
            "value": 0.000002015,
            "unit": "s"
          },
          {
            "name": "GenDot / HLOOpt / cuda / Primal",
            "value": 0.000001984,
            "unit": "s"
          },
          {
            "name": "GenDot / PartOpt / cuda / Primal",
            "value": 0.000002015,
            "unit": "s"
          },
          {
            "name": "GenDot / IPartOpt / cuda / Primal",
            "value": 0.000002016,
            "unit": "s"
          },
          {
            "name": "GenDot / DefOpt / cuda / Primal",
            "value": 0.000001984,
            "unit": "s"
          },
          {
            "name": "GenDot / IDefOpt / cuda / Primal",
            "value": 0.000001984,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cuda / Forward",
            "value": 0.000010111,
            "unit": "s"
          },
          {
            "name": "GenDot / Jax / cuda / Forward",
            "value": 0.000010048,
            "unit": "s"
          },
          {
            "name": "GenDot / HLOOpt / cuda / Forward",
            "value": 0.000010912,
            "unit": "s"
          },
          {
            "name": "GenDot / PartOpt / cuda / Forward",
            "value": 0.000010176,
            "unit": "s"
          },
          {
            "name": "GenDot / IPartOpt / cuda / Forward",
            "value": 0.000009984,
            "unit": "s"
          },
          {
            "name": "GenDot / DefOpt / cuda / Forward",
            "value": 0.000010144,
            "unit": "s"
          },
          {
            "name": "GenDot / IDefOpt / cuda / Forward",
            "value": 0.000009792,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cuda / PreRev",
            "value": 0.000010144,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cuda / PostRev",
            "value": 0.000010624,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cuda / BothRev",
            "value": 0.000009824,
            "unit": "s"
          },
          {
            "name": "GenDot / Jax / cuda / BothRev",
            "value": 0.00001008,
            "unit": "s"
          },
          {
            "name": "GenDot / HLOOpt / cuda / PreRev",
            "value": 0.000010144,
            "unit": "s"
          },
          {
            "name": "GenDot / HLOOpt / cuda / PostRev",
            "value": 0.000010048,
            "unit": "s"
          },
          {
            "name": "GenDot / HLOOpt / cuda / BothRev",
            "value": 0.000010016,
            "unit": "s"
          },
          {
            "name": "GenDot / PartOpt / cuda / PreRev",
            "value": 0.000010368,
            "unit": "s"
          },
          {
            "name": "GenDot / PartOpt / cuda / PostRev",
            "value": 0.00000976,
            "unit": "s"
          },
          {
            "name": "GenDot / PartOpt / cuda / BothRev",
            "value": 0.000014848,
            "unit": "s"
          },
          {
            "name": "GenDot / IPartOpt / cuda / PreRev",
            "value": 0.000009984,
            "unit": "s"
          },
          {
            "name": "GenDot / IPartOpt / cuda / PostRev",
            "value": 0.000010176,
            "unit": "s"
          },
          {
            "name": "GenDot / IPartOpt / cuda / BothRev",
            "value": 0.000009568,
            "unit": "s"
          },
          {
            "name": "GenDot / DefOpt / cuda / PreRev",
            "value": 0.000010751,
            "unit": "s"
          },
          {
            "name": "GenDot / DefOpt / cuda / PostRev",
            "value": 0.000010016,
            "unit": "s"
          },
          {
            "name": "GenDot / DefOpt / cuda / BothRev",
            "value": 0.000010336,
            "unit": "s"
          },
          {
            "name": "GenDot / IDefOpt / cuda / PreRev",
            "value": 0.000010368,
            "unit": "s"
          },
          {
            "name": "GenDot / IDefOpt / cuda / PostRev",
            "value": 0.000009984,
            "unit": "s"
          },
          {
            "name": "GenDot / IDefOpt / cuda / BothRev",
            "value": 0.000009888,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / tpu / Primal",
            "value": 9.29575e-7,
            "unit": "s"
          },
          {
            "name": "GenDot / Jax / tpu / Primal",
            "value": 9.25225e-7,
            "unit": "s"
          },
          {
            "name": "GenDot / HLOOpt / tpu / Primal",
            "value": 0.000001577925,
            "unit": "s"
          },
          {
            "name": "GenDot / PartOpt / tpu / Primal",
            "value": 9.25675e-7,
            "unit": "s"
          },
          {
            "name": "GenDot / IPartOpt / tpu / Primal",
            "value": 9.30175e-7,
            "unit": "s"
          },
          {
            "name": "GenDot / DefOpt / tpu / Primal",
            "value": 0.00000149485,
            "unit": "s"
          },
          {
            "name": "GenDot / IDefOpt / tpu / Primal",
            "value": 0.000001579125,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / tpu / Forward",
            "value": 0.0000031612000000000003,
            "unit": "s"
          },
          {
            "name": "GenDot / Jax / tpu / Forward",
            "value": 0.000002318925,
            "unit": "s"
          },
          {
            "name": "GenDot / HLOOpt / tpu / Forward",
            "value": 0.0000031131,
            "unit": "s"
          },
          {
            "name": "GenDot / PartOpt / tpu / Forward",
            "value": 0.00000322215,
            "unit": "s"
          },
          {
            "name": "GenDot / IPartOpt / tpu / Forward",
            "value": 0.000003119,
            "unit": "s"
          },
          {
            "name": "GenDot / DefOpt / tpu / Forward",
            "value": 0.0000032191750000000004,
            "unit": "s"
          },
          {
            "name": "GenDot / IDefOpt / tpu / Forward",
            "value": 0.0000031159,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / tpu / PreRev",
            "value": 0.000002954675,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / tpu / PostRev",
            "value": 0.00000241495,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / tpu / BothRev",
            "value": 0.000002964825,
            "unit": "s"
          },
          {
            "name": "GenDot / Jax / tpu / BothRev",
            "value": 0.000002414275,
            "unit": "s"
          },
          {
            "name": "GenDot / HLOOpt / tpu / PreRev",
            "value": 0.00000296385,
            "unit": "s"
          },
          {
            "name": "GenDot / HLOOpt / tpu / PostRev",
            "value": 0.00000293875,
            "unit": "s"
          },
          {
            "name": "GenDot / HLOOpt / tpu / BothRev",
            "value": 0.000002956875,
            "unit": "s"
          },
          {
            "name": "GenDot / PartOpt / tpu / PreRev",
            "value": 0.000002932425,
            "unit": "s"
          },
          {
            "name": "GenDot / PartOpt / tpu / PostRev",
            "value": 0.00000239235,
            "unit": "s"
          },
          {
            "name": "GenDot / PartOpt / tpu / BothRev",
            "value": 0.0000029372,
            "unit": "s"
          },
          {
            "name": "GenDot / IPartOpt / tpu / PreRev",
            "value": 0.000002964875,
            "unit": "s"
          },
          {
            "name": "GenDot / IPartOpt / tpu / PostRev",
            "value": 0.000002418575,
            "unit": "s"
          },
          {
            "name": "GenDot / IPartOpt / tpu / BothRev",
            "value": 0.00000296245,
            "unit": "s"
          },
          {
            "name": "GenDot / DefOpt / tpu / PreRev",
            "value": 0.000002938775,
            "unit": "s"
          },
          {
            "name": "GenDot / DefOpt / tpu / PostRev",
            "value": 0.0000029676750000000003,
            "unit": "s"
          },
          {
            "name": "GenDot / DefOpt / tpu / BothRev",
            "value": 0.00000292875,
            "unit": "s"
          },
          {
            "name": "GenDot / IDefOpt / tpu / PreRev",
            "value": 0.000002963075,
            "unit": "s"
          },
          {
            "name": "GenDot / IDefOpt / tpu / PostRev",
            "value": 0.0000029291,
            "unit": "s"
          },
          {
            "name": "GenDot / IDefOpt / tpu / BothRev",
            "value": 0.0000029657499999999995,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / Primal",
            "value": 0.000019882,
            "unit": "s"
          },
          {
            "name": "GenDot / Jax / cpu / Primal",
            "value": 0.000018504,
            "unit": "s"
          },
          {
            "name": "GenDot / HLOOpt / cpu / Primal",
            "value": 0.000017148,
            "unit": "s"
          },
          {
            "name": "GenDot / PartOpt / cpu / Primal",
            "value": 0.000017846999999999997,
            "unit": "s"
          },
          {
            "name": "GenDot / IPartOpt / cpu / Primal",
            "value": 0.000018106,
            "unit": "s"
          },
          {
            "name": "GenDot / DefOpt / cpu / Primal",
            "value": 0.000017426,
            "unit": "s"
          },
          {
            "name": "GenDot / IDefOpt / cpu / Primal",
            "value": 0.000017074999999999998,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / Forward",
            "value": 0.000023774,
            "unit": "s"
          },
          {
            "name": "GenDot / Jax / cpu / Forward",
            "value": 0.000024714,
            "unit": "s"
          },
          {
            "name": "GenDot / HLOOpt / cpu / Forward",
            "value": 0.000023398,
            "unit": "s"
          },
          {
            "name": "GenDot / PartOpt / cpu / Forward",
            "value": 0.000023715,
            "unit": "s"
          },
          {
            "name": "GenDot / IPartOpt / cpu / Forward",
            "value": 0.000023677,
            "unit": "s"
          },
          {
            "name": "GenDot / DefOpt / cpu / Forward",
            "value": 0.000023703,
            "unit": "s"
          },
          {
            "name": "GenDot / IDefOpt / cpu / Forward",
            "value": 0.000023499,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / PreRev",
            "value": 0.000024117,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / PostRev",
            "value": 0.000024758,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / BothRev",
            "value": 0.000029763000000000003,
            "unit": "s"
          },
          {
            "name": "GenDot / Jax / cpu / BothRev",
            "value": 0.00002536,
            "unit": "s"
          },
          {
            "name": "GenDot / HLOOpt / cpu / PreRev",
            "value": 0.000024165,
            "unit": "s"
          },
          {
            "name": "GenDot / HLOOpt / cpu / PostRev",
            "value": 0.000024324,
            "unit": "s"
          },
          {
            "name": "GenDot / HLOOpt / cpu / BothRev",
            "value": 0.000023727,
            "unit": "s"
          },
          {
            "name": "GenDot / PartOpt / cpu / PreRev",
            "value": 0.000023443,
            "unit": "s"
          },
          {
            "name": "GenDot / PartOpt / cpu / PostRev",
            "value": 0.000025018,
            "unit": "s"
          },
          {
            "name": "GenDot / PartOpt / cpu / BothRev",
            "value": 0.000024309,
            "unit": "s"
          },
          {
            "name": "GenDot / IPartOpt / cpu / PreRev",
            "value": 0.00002383,
            "unit": "s"
          },
          {
            "name": "GenDot / IPartOpt / cpu / PostRev",
            "value": 0.000024842,
            "unit": "s"
          },
          {
            "name": "GenDot / IPartOpt / cpu / BothRev",
            "value": 0.000023885,
            "unit": "s"
          },
          {
            "name": "GenDot / DefOpt / cpu / PreRev",
            "value": 0.000024072,
            "unit": "s"
          },
          {
            "name": "GenDot / DefOpt / cpu / PostRev",
            "value": 0.00002435,
            "unit": "s"
          },
          {
            "name": "GenDot / DefOpt / cpu / BothRev",
            "value": 0.00002415,
            "unit": "s"
          },
          {
            "name": "GenDot / IDefOpt / cpu / PreRev",
            "value": 0.00002367,
            "unit": "s"
          },
          {
            "name": "GenDot / IDefOpt / cpu / PostRev",
            "value": 0.000024071,
            "unit": "s"
          },
          {
            "name": "GenDot / IDefOpt / cpu / BothRev",
            "value": 0.000023724,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / Primal",
            "value": 0.00001,
            "unit": "s"
          },
          {
            "name": "GenDot / Jax / cpu / Primal",
            "value": 0.00001,
            "unit": "s"
          },
          {
            "name": "GenDot / HLOOpt / cpu / Primal",
            "value": 0.000008999999999999999,
            "unit": "s"
          },
          {
            "name": "GenDot / PartOpt / cpu / Primal",
            "value": 0.00001,
            "unit": "s"
          },
          {
            "name": "GenDot / IPartOpt / cpu / Primal",
            "value": 0.00001,
            "unit": "s"
          },
          {
            "name": "GenDot / DefOpt / cpu / Primal",
            "value": 0.000008999999999999999,
            "unit": "s"
          },
          {
            "name": "GenDot / IDefOpt / cpu / Primal",
            "value": 0.000008999999999999999,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / Forward",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "GenDot / Jax / cpu / Forward",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "GenDot / HLOOpt / cpu / Forward",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "GenDot / PartOpt / cpu / Forward",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "GenDot / IPartOpt / cpu / Forward",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "GenDot / DefOpt / cpu / Forward",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "GenDot / IDefOpt / cpu / Forward",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / PreRev",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / PostRev",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / BothRev",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "GenDot / Jax / cpu / BothRev",
            "value": 0.000015,
            "unit": "s"
          },
          {
            "name": "GenDot / HLOOpt / cpu / PreRev",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "GenDot / HLOOpt / cpu / PostRev",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "GenDot / HLOOpt / cpu / BothRev",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "GenDot / PartOpt / cpu / PreRev",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "GenDot / PartOpt / cpu / PostRev",
            "value": 0.000015,
            "unit": "s"
          },
          {
            "name": "GenDot / PartOpt / cpu / BothRev",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "GenDot / IPartOpt / cpu / PreRev",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "GenDot / IPartOpt / cpu / PostRev",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "GenDot / IPartOpt / cpu / BothRev",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "GenDot / DefOpt / cpu / PreRev",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "GenDot / DefOpt / cpu / PostRev",
            "value": 0.000015,
            "unit": "s"
          },
          {
            "name": "GenDot / DefOpt / cpu / BothRev",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "GenDot / IDefOpt / cpu / PreRev",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "GenDot / IDefOpt / cpu / PostRev",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "GenDot / IDefOpt / cpu / BothRev",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / Primal",
            "value": 0.000010841679968507378,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / Jax / cpu / Primal",
            "value": 0.000010750079982244642,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / HLOOpt / cpu / Primal",
            "value": 0.000010333600012017995,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / PartOpt / cpu / Primal",
            "value": 0.000010718239973357412,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IPartOpt / cpu / Primal",
            "value": 0.00001055310000992904,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / DefOpt / cpu / Primal",
            "value": 0.00000993140000900894,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IDefOpt / cpu / Primal",
            "value": 0.000010487620002095355,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / Forward",
            "value": 0.000014204600020093496,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / Jax / cpu / Forward",
            "value": 0.0000146346400197217,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / HLOOpt / cpu / Forward",
            "value": 0.000015109380010471795,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / PartOpt / cpu / Forward",
            "value": 0.000014914740049789544,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IPartOpt / cpu / Forward",
            "value": 0.000014542079998136617,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / DefOpt / cpu / Forward",
            "value": 0.000015062679985931026,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IDefOpt / cpu / Forward",
            "value": 0.000014916280015313532,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / PreRev",
            "value": 0.00001550459998725273,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / PostRev",
            "value": 0.000014995539968367666,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / BothRev",
            "value": 0.000014779980065213749,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / Jax / cpu / BothRev",
            "value": 0.0000151172800178756,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / HLOOpt / cpu / PreRev",
            "value": 0.000015309580030589133,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / HLOOpt / cpu / PostRev",
            "value": 0.000016761159986344865,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / HLOOpt / cpu / BothRev",
            "value": 0.000014449539994529914,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / PartOpt / cpu / PreRev",
            "value": 0.000015427959988301153,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / PartOpt / cpu / PostRev",
            "value": 0.000014358119988173713,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / PartOpt / cpu / BothRev",
            "value": 0.000014243580008042045,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IPartOpt / cpu / PreRev",
            "value": 0.000015118039955268616,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IPartOpt / cpu / PostRev",
            "value": 0.000014513700016323128,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IPartOpt / cpu / BothRev",
            "value": 0.000014463360030276817,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / DefOpt / cpu / PreRev",
            "value": 0.000014994799976193462,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / DefOpt / cpu / PostRev",
            "value": 0.000014343659986479906,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / DefOpt / cpu / BothRev",
            "value": 0.000014086640021560015,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IDefOpt / cpu / PreRev",
            "value": 0.000015176240003711427,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IDefOpt / cpu / PostRev",
            "value": 0.000014299419990493336,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IDefOpt / cpu / BothRev",
            "value": 0.000014242560055208742,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cuda / Primal",
            "value": 0.000001983,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / Jax / cuda / Primal",
            "value": 0.000001952,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / HLOOpt / cuda / Primal",
            "value": 0.000001951,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / PartOpt / cuda / Primal",
            "value": 0.000001952,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IPartOpt / cuda / Primal",
            "value": 0.000001984,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / DefOpt / cuda / Primal",
            "value": 0.000001983,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IDefOpt / cuda / Primal",
            "value": 0.000001983,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cuda / Forward",
            "value": 0.000002048,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / Jax / cuda / Forward",
            "value": 0.000002047,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / HLOOpt / cuda / Forward",
            "value": 0.000002047,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / PartOpt / cuda / Forward",
            "value": 0.000002047,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IPartOpt / cuda / Forward",
            "value": 0.000002047,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / DefOpt / cuda / Forward",
            "value": 0.000002048,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IDefOpt / cuda / Forward",
            "value": 0.000002047,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cuda / PreRev",
            "value": 0.000002047,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cuda / PostRev",
            "value": 0.000002047,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cuda / BothRev",
            "value": 0.000002047,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / Jax / cuda / BothRev",
            "value": 0.000002047,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / HLOOpt / cuda / PreRev",
            "value": 0.000002047,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / HLOOpt / cuda / PostRev",
            "value": 0.000002047,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / HLOOpt / cuda / BothRev",
            "value": 0.000002047,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / PartOpt / cuda / PreRev",
            "value": 0.000002047,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / PartOpt / cuda / PostRev",
            "value": 0.000002047,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / PartOpt / cuda / BothRev",
            "value": 0.000002047,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IPartOpt / cuda / PreRev",
            "value": 0.000002047,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IPartOpt / cuda / PostRev",
            "value": 0.000002047,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IPartOpt / cuda / BothRev",
            "value": 0.000002047,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / DefOpt / cuda / PreRev",
            "value": 0.000002047,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / DefOpt / cuda / PostRev",
            "value": 0.000002047,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / DefOpt / cuda / BothRev",
            "value": 0.000002047,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IDefOpt / cuda / PreRev",
            "value": 0.000002047,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IDefOpt / cuda / PostRev",
            "value": 0.000002047,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IDefOpt / cuda / BothRev",
            "value": 0.000002047,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / tpu / Primal",
            "value": 9.225e-7,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / Jax / tpu / Primal",
            "value": 9.495e-7,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / HLOOpt / tpu / Primal",
            "value": 9.03125e-7,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / PartOpt / tpu / Primal",
            "value": 9.57925e-7,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IPartOpt / tpu / Primal",
            "value": 9.03575e-7,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / DefOpt / tpu / Primal",
            "value": 9.537e-7,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IDefOpt / tpu / Primal",
            "value": 8.97925e-7,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / tpu / Forward",
            "value": 9.486e-7,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / Jax / tpu / Forward",
            "value": 9.813e-7,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / HLOOpt / tpu / Forward",
            "value": 9.736e-7,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / PartOpt / tpu / Forward",
            "value": 9.33875e-7,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IPartOpt / tpu / Forward",
            "value": 9.73325e-7,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / DefOpt / tpu / Forward",
            "value": 9.337e-7,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IDefOpt / tpu / Forward",
            "value": 9.74125e-7,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / tpu / PreRev",
            "value": 9.317e-7,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / tpu / PostRev",
            "value": 9.64725e-7,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / tpu / BothRev",
            "value": 9.619e-7,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / Jax / tpu / BothRev",
            "value": 9.6445e-7,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / HLOOpt / tpu / PreRev",
            "value": 9.61625e-7,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / HLOOpt / tpu / PostRev",
            "value": 9.650749999999998e-7,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / HLOOpt / tpu / BothRev",
            "value": 9.615e-7,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / PartOpt / tpu / PreRev",
            "value": 9.6405e-7,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / PartOpt / tpu / PostRev",
            "value": 9.61925e-7,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / PartOpt / tpu / BothRev",
            "value": 9.6445e-7,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IPartOpt / tpu / PreRev",
            "value": 9.61975e-7,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IPartOpt / tpu / PostRev",
            "value": 9.65e-7,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IPartOpt / tpu / BothRev",
            "value": 9.62325e-7,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / DefOpt / tpu / PreRev",
            "value": 9.64275e-7,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / DefOpt / tpu / PostRev",
            "value": 9.62e-7,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / DefOpt / tpu / BothRev",
            "value": 9.63975e-7,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IDefOpt / tpu / PreRev",
            "value": 9.62075e-7,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IDefOpt / tpu / PostRev",
            "value": 9.64625e-7,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IDefOpt / tpu / BothRev",
            "value": 9.61825e-7,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / Primal",
            "value": 0.000027858,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / Jax / cpu / Primal",
            "value": 0.000021723,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / HLOOpt / cpu / Primal",
            "value": 0.000021884,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / PartOpt / cpu / Primal",
            "value": 0.000021634,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IPartOpt / cpu / Primal",
            "value": 0.000021882,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / DefOpt / cpu / Primal",
            "value": 0.000021201000000000003,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IDefOpt / cpu / Primal",
            "value": 0.000021499,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / Forward",
            "value": 0.000029932,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / Jax / cpu / Forward",
            "value": 0.000028999,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / HLOOpt / cpu / Forward",
            "value": 0.000029128,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / PartOpt / cpu / Forward",
            "value": 0.00002905,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IPartOpt / cpu / Forward",
            "value": 0.000029086,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / DefOpt / cpu / Forward",
            "value": 0.000034518,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IDefOpt / cpu / Forward",
            "value": 0.000029319,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / PreRev",
            "value": 0.000035401,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / PostRev",
            "value": 0.000029207,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / BothRev",
            "value": 0.000028619,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / Jax / cpu / BothRev",
            "value": 0.00002865,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / HLOOpt / cpu / PreRev",
            "value": 0.000029373,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / HLOOpt / cpu / PostRev",
            "value": 0.000029489,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / HLOOpt / cpu / BothRev",
            "value": 0.000028897,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / PartOpt / cpu / PreRev",
            "value": 0.000029263,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / PartOpt / cpu / PostRev",
            "value": 0.000035214000000000005,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / PartOpt / cpu / BothRev",
            "value": 0.000029267000000000003,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IPartOpt / cpu / PreRev",
            "value": 0.000029075,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IPartOpt / cpu / PostRev",
            "value": 0.000028441,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IPartOpt / cpu / BothRev",
            "value": 0.000028989,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / DefOpt / cpu / PreRev",
            "value": 0.000029098,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / DefOpt / cpu / PostRev",
            "value": 0.000028989,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / DefOpt / cpu / BothRev",
            "value": 0.000029217,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IDefOpt / cpu / PreRev",
            "value": 0.000028835,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IDefOpt / cpu / PostRev",
            "value": 0.000029096,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IDefOpt / cpu / BothRev",
            "value": 0.000029087,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / Primal",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / Jax / cpu / Primal",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / HLOOpt / cpu / Primal",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / PartOpt / cpu / Primal",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IPartOpt / cpu / Primal",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / DefOpt / cpu / Primal",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IDefOpt / cpu / Primal",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / Forward",
            "value": 0.000017,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / Jax / cpu / Forward",
            "value": 0.000017999999999999997,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / HLOOpt / cpu / Forward",
            "value": 0.000017,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / PartOpt / cpu / Forward",
            "value": 0.000017,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IPartOpt / cpu / Forward",
            "value": 0.000017,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / DefOpt / cpu / Forward",
            "value": 0.000017,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IDefOpt / cpu / Forward",
            "value": 0.000017,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / PreRev",
            "value": 0.000017,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / PostRev",
            "value": 0.000017,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / BothRev",
            "value": 0.000017999999999999997,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / Jax / cpu / BothRev",
            "value": 0.000017,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / HLOOpt / cpu / PreRev",
            "value": 0.000017999999999999997,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / HLOOpt / cpu / PostRev",
            "value": 0.000017,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / HLOOpt / cpu / BothRev",
            "value": 0.000017999999999999997,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / PartOpt / cpu / PreRev",
            "value": 0.000017999999999999997,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / PartOpt / cpu / PostRev",
            "value": 0.000017,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / PartOpt / cpu / BothRev",
            "value": 0.000019,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IPartOpt / cpu / PreRev",
            "value": 0.000017,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IPartOpt / cpu / PostRev",
            "value": 0.000017999999999999997,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IPartOpt / cpu / BothRev",
            "value": 0.000017999999999999997,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / DefOpt / cpu / PreRev",
            "value": 0.000017,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / DefOpt / cpu / PostRev",
            "value": 0.000017999999999999997,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / DefOpt / cpu / BothRev",
            "value": 0.000017,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IDefOpt / cpu / PreRev",
            "value": 0.000017999999999999997,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IDefOpt / cpu / PostRev",
            "value": 0.000017999999999999997,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IDefOpt / cpu / BothRev",
            "value": 0.000017,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / JaXPipe / cpu / Primal",
            "value": 0.0009240977999979,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / Jax / cpu / Primal",
            "value": 0.0009339908000583,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / HLOOpt / cpu / Primal",
            "value": 0.0009847250000348,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / PartOpt / cpu / Primal",
            "value": 0.0009646690000408,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IPartOpt / cpu / Primal",
            "value": 0.0009248692001165,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / DefOpt / cpu / Primal",
            "value": 0.0010179868000705,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IDefOpt / cpu / Primal",
            "value": 0.0010224614000435,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / JaXPipe / cpu / Forward",
            "value": 0.0023158776000855,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / Jax / cpu / Forward",
            "value": 0.0024215126000854,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / HLOOpt / cpu / Forward",
            "value": 0.0024494521998349,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / PartOpt / cpu / Forward",
            "value": 0.0023676605998844,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IPartOpt / cpu / Forward",
            "value": 0.0023951875999046,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / DefOpt / cpu / Forward",
            "value": 0.0023946858000272,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IDefOpt / cpu / Forward",
            "value": 0.0024209647998759,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / JaXPipe / cpu / PreRev",
            "value": 0.006682462200024,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / JaXPipe / cpu / PostRev",
            "value": 0.0063939480000044,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / JaXPipe / cpu / BothRev",
            "value": 0.0062240333999397,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / Jax / cpu / BothRev",
            "value": 0.006168826800058,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / HLOOpt / cpu / PreRev",
            "value": 0.0058642033999603,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / HLOOpt / cpu / PostRev",
            "value": 0.0065649374000713,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / HLOOpt / cpu / BothRev",
            "value": 0.0062589523998212,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / PartOpt / cpu / PreRev",
            "value": 0.0055141665999144,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / PartOpt / cpu / PostRev",
            "value": 0.006519494800159,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / PartOpt / cpu / BothRev",
            "value": 0.0060325897999973,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IPartOpt / cpu / PreRev",
            "value": 0.0058518562001154,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IPartOpt / cpu / PostRev",
            "value": 0.0066064911999092,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IPartOpt / cpu / BothRev",
            "value": 0.0049009999999725,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / DefOpt / cpu / PreRev",
            "value": 0.0068223805998968,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / DefOpt / cpu / PostRev",
            "value": 0.0042839250001634,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / DefOpt / cpu / BothRev",
            "value": 0.0049773534000451,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IDefOpt / cpu / PreRev",
            "value": 0.0050564379999741,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IDefOpt / cpu / PostRev",
            "value": 0.0036666862000856,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IDefOpt / cpu / BothRev",
            "value": 0.0049634575999334,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / JaXPipe / cuda / Primal",
            "value": 0.000282525,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / Jax / cuda / Primal",
            "value": 0.000282142,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / HLOOpt / cuda / Primal",
            "value": 0.000288765,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / PartOpt / cuda / Primal",
            "value": 0.000281629,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IPartOpt / cuda / Primal",
            "value": 0.000282238,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / DefOpt / cuda / Primal",
            "value": 0.000289949,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IDefOpt / cuda / Primal",
            "value": 0.000288957,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / JaXPipe / cuda / Forward",
            "value": 0.000560476,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / Jax / cuda / Forward",
            "value": 0.000540347,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / HLOOpt / cuda / Forward",
            "value": 0.000560635,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / PartOpt / cuda / Forward",
            "value": 0.0005582359999999,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IPartOpt / cuda / Forward",
            "value": 0.0005590349999999,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / DefOpt / cuda / Forward",
            "value": 0.000558843,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IDefOpt / cuda / Forward",
            "value": 0.0005579469999999,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / JaXPipe / cuda / PreRev",
            "value": 0.001036343,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / JaXPipe / cuda / PostRev",
            "value": 0.000989623,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / JaXPipe / cuda / BothRev",
            "value": 0.001025015,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / Jax / cuda / BothRev",
            "value": 0.000996023,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / HLOOpt / cuda / PreRev",
            "value": 0.001019096,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / HLOOpt / cuda / PostRev",
            "value": 0.001040856,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / HLOOpt / cuda / BothRev",
            "value": 0.001012472,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / PartOpt / cuda / PreRev",
            "value": 0.001032887,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / PartOpt / cuda / PostRev",
            "value": 0.000979896,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / PartOpt / cuda / BothRev",
            "value": 0.001029496,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IPartOpt / cuda / PreRev",
            "value": 0.001029495,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IPartOpt / cuda / PostRev",
            "value": 0.000981815,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IPartOpt / cuda / BothRev",
            "value": 0.0010351269999999,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / DefOpt / cuda / PreRev",
            "value": 0.001031063,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / DefOpt / cuda / PostRev",
            "value": 0.0009678,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / DefOpt / cuda / BothRev",
            "value": 0.001030904,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IDefOpt / cuda / PreRev",
            "value": 0.001029335,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IDefOpt / cuda / PostRev",
            "value": 0.001028791,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IDefOpt / cuda / BothRev",
            "value": 0.001029911,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / JaXPipe / tpu / Primal",
            "value": 0.0001236612499999,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / Jax / tpu / Primal",
            "value": 0.00012691525,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / HLOOpt / tpu / Primal",
            "value": 0.00015220075,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / PartOpt / tpu / Primal",
            "value": 0.00013378675,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IPartOpt / tpu / Primal",
            "value": 0.000131553,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / DefOpt / tpu / Primal",
            "value": 0.00014771625,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IDefOpt / tpu / Primal",
            "value": 0.00015105825,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / JaXPipe / tpu / Forward",
            "value": 0.00021229,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / Jax / tpu / Forward",
            "value": 0.00026106525,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / HLOOpt / tpu / Forward",
            "value": 0.0002123467499999,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / PartOpt / tpu / Forward",
            "value": 0.000218429,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IPartOpt / tpu / Forward",
            "value": 0.00021238525,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / DefOpt / tpu / Forward",
            "value": 0.0002185895,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IDefOpt / tpu / Forward",
            "value": 0.00021234975,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / JaXPipe / tpu / PreRev",
            "value": 0.00035685875,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / JaXPipe / tpu / PostRev",
            "value": 0.00025662975,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / JaXPipe / tpu / BothRev",
            "value": 0.00035696275,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / Jax / tpu / BothRev",
            "value": 0.0002577185,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / HLOOpt / tpu / PreRev",
            "value": 0.00035700625,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / HLOOpt / tpu / PostRev",
            "value": 0.0002913595,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / HLOOpt / tpu / BothRev",
            "value": 0.0003571645,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / PartOpt / tpu / PreRev",
            "value": 0.00035785075,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / PartOpt / tpu / PostRev",
            "value": 0.0002721935,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / PartOpt / tpu / BothRev",
            "value": 0.000358094,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IPartOpt / tpu / PreRev",
            "value": 0.0003567829999999,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IPartOpt / tpu / PostRev",
            "value": 0.00027296725,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IPartOpt / tpu / BothRev",
            "value": 0.00035733875,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / DefOpt / tpu / PreRev",
            "value": 0.00035909675,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / DefOpt / tpu / PostRev",
            "value": 0.0002837755,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / DefOpt / tpu / BothRev",
            "value": 0.00035903975,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IDefOpt / tpu / PreRev",
            "value": 0.00035801775,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IDefOpt / tpu / PostRev",
            "value": 0.00030154675,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IDefOpt / tpu / BothRev",
            "value": 0.00035856175,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / JaXPipe / cpu / Primal",
            "value": 0.002494088,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / Jax / cpu / Primal",
            "value": 0.002555224,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / HLOOpt / cpu / Primal",
            "value": 0.002778693,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / PartOpt / cpu / Primal",
            "value": 0.002648513,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IPartOpt / cpu / Primal",
            "value": 0.002397975,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / DefOpt / cpu / Primal",
            "value": 0.0028864999999999,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IDefOpt / cpu / Primal",
            "value": 0.002400317,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / JaXPipe / cpu / Forward",
            "value": 0.00643525,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / Jax / cpu / Forward",
            "value": 0.006575364,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / HLOOpt / cpu / Forward",
            "value": 0.006273713,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / PartOpt / cpu / Forward",
            "value": 0.006634411,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IPartOpt / cpu / Forward",
            "value": 0.00631918,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / DefOpt / cpu / Forward",
            "value": 0.006371108,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IDefOpt / cpu / Forward",
            "value": 0.006614437,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / JaXPipe / cpu / PreRev",
            "value": 0.01163928,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / JaXPipe / cpu / PostRev",
            "value": 0.010968114,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / JaXPipe / cpu / BothRev",
            "value": 0.010684773,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / Jax / cpu / BothRev",
            "value": 0.010048507,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / HLOOpt / cpu / PreRev",
            "value": 0.012065109,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / HLOOpt / cpu / PostRev",
            "value": 0.009729218,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / HLOOpt / cpu / BothRev",
            "value": 0.009564409,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / PartOpt / cpu / PreRev",
            "value": 0.010316226,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / PartOpt / cpu / PostRev",
            "value": 0.00966395,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / PartOpt / cpu / BothRev",
            "value": 0.009304501,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IPartOpt / cpu / PreRev",
            "value": 0.008330733,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IPartOpt / cpu / PostRev",
            "value": 0.008826525,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IPartOpt / cpu / BothRev",
            "value": 0.008742937,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / DefOpt / cpu / PreRev",
            "value": 0.008310676,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / DefOpt / cpu / PostRev",
            "value": 0.007644362,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / DefOpt / cpu / BothRev",
            "value": 0.009518901,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IDefOpt / cpu / PreRev",
            "value": 0.008507503,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IDefOpt / cpu / PostRev",
            "value": 0.00890698,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IDefOpt / cpu / BothRev",
            "value": 0.009049301,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / JaXPipe / cpu / Primal",
            "value": 0.001759,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / Jax / cpu / Primal",
            "value": 0.001752,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / HLOOpt / cpu / Primal",
            "value": 0.001825,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / PartOpt / cpu / Primal",
            "value": 0.0018,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IPartOpt / cpu / Primal",
            "value": 0.001812,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / DefOpt / cpu / Primal",
            "value": 0.001792,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IDefOpt / cpu / Primal",
            "value": 0.002234,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / JaXPipe / cpu / Forward",
            "value": 0.004756,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / Jax / cpu / Forward",
            "value": 0.00452,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / HLOOpt / cpu / Forward",
            "value": 0.004378,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / PartOpt / cpu / Forward",
            "value": 0.0044189999999999,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IPartOpt / cpu / Forward",
            "value": 0.0054329999999999,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / DefOpt / cpu / Forward",
            "value": 0.004582,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IDefOpt / cpu / Forward",
            "value": 0.0045839999999999,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / JaXPipe / cpu / PreRev",
            "value": 0.00773,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / JaXPipe / cpu / PostRev",
            "value": 0.0107669999999999,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / JaXPipe / cpu / BothRev",
            "value": 0.007964,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / Jax / cpu / BothRev",
            "value": 0.011379,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / HLOOpt / cpu / PreRev",
            "value": 0.008589,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / HLOOpt / cpu / PostRev",
            "value": 0.008489,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / HLOOpt / cpu / BothRev",
            "value": 0.007926,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / PartOpt / cpu / PreRev",
            "value": 0.007803,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / PartOpt / cpu / PostRev",
            "value": 0.01113,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / PartOpt / cpu / BothRev",
            "value": 0.008685,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IPartOpt / cpu / PreRev",
            "value": 0.008288,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IPartOpt / cpu / PostRev",
            "value": 0.009526,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IPartOpt / cpu / BothRev",
            "value": 0.007953,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / DefOpt / cpu / PreRev",
            "value": 0.007777,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / DefOpt / cpu / PostRev",
            "value": 0.00751,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / DefOpt / cpu / BothRev",
            "value": 0.008409,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IDefOpt / cpu / PreRev",
            "value": 0.007696,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IDefOpt / cpu / PostRev",
            "value": 0.009614,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IDefOpt / cpu / BothRev",
            "value": 0.008018,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000007927520009616273,
            "unit": "s"
          },
          {
            "name": "scatter_sum / Jax / cpu / Primal",
            "value": 0.00000742767999327043,
            "unit": "s"
          },
          {
            "name": "scatter_sum / HLOOpt / cpu / Primal",
            "value": 0.000008308480018968112,
            "unit": "s"
          },
          {
            "name": "scatter_sum / PartOpt / cpu / Primal",
            "value": 0.000007928900022307061,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IPartOpt / cpu / Primal",
            "value": 0.000007562780001535429,
            "unit": "s"
          },
          {
            "name": "scatter_sum / DefOpt / cpu / Primal",
            "value": 0.00000764010001148563,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IDefOpt / cpu / Primal",
            "value": 0.000007460480064764852,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Forward",
            "value": 0.000011345139937475325,
            "unit": "s"
          },
          {
            "name": "scatter_sum / Jax / cpu / Forward",
            "value": 0.000011204619995623944,
            "unit": "s"
          },
          {
            "name": "scatter_sum / HLOOpt / cpu / Forward",
            "value": 0.00001257889996850281,
            "unit": "s"
          },
          {
            "name": "scatter_sum / PartOpt / cpu / Forward",
            "value": 0.00001152361997810658,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IPartOpt / cpu / Forward",
            "value": 0.000014340680008899652,
            "unit": "s"
          },
          {
            "name": "scatter_sum / DefOpt / cpu / Forward",
            "value": 0.000011413759984861828,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IDefOpt / cpu / Forward",
            "value": 0.000011250819998167572,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / PreRev",
            "value": 0.000011935600005017476,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / PostRev",
            "value": 0.000011985679984718445,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / BothRev",
            "value": 0.000012393960014378536,
            "unit": "s"
          },
          {
            "name": "scatter_sum / Jax / cpu / BothRev",
            "value": 0.000011429759997554356,
            "unit": "s"
          },
          {
            "name": "scatter_sum / HLOOpt / cpu / PreRev",
            "value": 0.000012093140012439108,
            "unit": "s"
          },
          {
            "name": "scatter_sum / HLOOpt / cpu / PostRev",
            "value": 0.000014394499994523355,
            "unit": "s"
          },
          {
            "name": "scatter_sum / HLOOpt / cpu / BothRev",
            "value": 0.000011507600011100294,
            "unit": "s"
          },
          {
            "name": "scatter_sum / PartOpt / cpu / PreRev",
            "value": 0.00001178795998384885,
            "unit": "s"
          },
          {
            "name": "scatter_sum / PartOpt / cpu / PostRev",
            "value": 0.00001198563998514146,
            "unit": "s"
          },
          {
            "name": "scatter_sum / PartOpt / cpu / BothRev",
            "value": 0.000012446580021787668,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IPartOpt / cpu / PreRev",
            "value": 0.000011264420027146117,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IPartOpt / cpu / PostRev",
            "value": 0.000011778660000345554,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IPartOpt / cpu / BothRev",
            "value": 0.000011508220040923334,
            "unit": "s"
          },
          {
            "name": "scatter_sum / DefOpt / cpu / PreRev",
            "value": 0.000011381679996702587,
            "unit": "s"
          },
          {
            "name": "scatter_sum / DefOpt / cpu / PostRev",
            "value": 0.0000118991999715945,
            "unit": "s"
          },
          {
            "name": "scatter_sum / DefOpt / cpu / BothRev",
            "value": 0.00001180374005343765,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IDefOpt / cpu / PreRev",
            "value": 0.00001141565998295846,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IDefOpt / cpu / PostRev",
            "value": 0.000011669380010062014,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IDefOpt / cpu / BothRev",
            "value": 0.000012216939976497088,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cuda / Primal",
            "value": 0.000010496,
            "unit": "s"
          },
          {
            "name": "scatter_sum / Jax / cuda / Primal",
            "value": 0.00000992,
            "unit": "s"
          },
          {
            "name": "scatter_sum / HLOOpt / cuda / Primal",
            "value": 0.000010303,
            "unit": "s"
          },
          {
            "name": "scatter_sum / PartOpt / cuda / Primal",
            "value": 0.000010144,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IPartOpt / cuda / Primal",
            "value": 0.000010304,
            "unit": "s"
          },
          {
            "name": "scatter_sum / DefOpt / cuda / Primal",
            "value": 0.00001008,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IDefOpt / cuda / Primal",
            "value": 0.000010495,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cuda / Forward",
            "value": 0.00001728,
            "unit": "s"
          },
          {
            "name": "scatter_sum / Jax / cuda / Forward",
            "value": 0.000017247999999999998,
            "unit": "s"
          },
          {
            "name": "scatter_sum / HLOOpt / cuda / Forward",
            "value": 0.000016927999999999998,
            "unit": "s"
          },
          {
            "name": "scatter_sum / PartOpt / cuda / Forward",
            "value": 0.000016576000000000002,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IPartOpt / cuda / Forward",
            "value": 0.000016864,
            "unit": "s"
          },
          {
            "name": "scatter_sum / DefOpt / cuda / Forward",
            "value": 0.000017056,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IDefOpt / cuda / Forward",
            "value": 0.00001632,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cuda / PreRev",
            "value": 0.000017024,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cuda / PostRev",
            "value": 0.000018431,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cuda / BothRev",
            "value": 0.000016704,
            "unit": "s"
          },
          {
            "name": "scatter_sum / Jax / cuda / BothRev",
            "value": 0.000017088,
            "unit": "s"
          },
          {
            "name": "scatter_sum / HLOOpt / cuda / PreRev",
            "value": 0.000016672,
            "unit": "s"
          },
          {
            "name": "scatter_sum / HLOOpt / cuda / PostRev",
            "value": 0.000016448000000000002,
            "unit": "s"
          },
          {
            "name": "scatter_sum / HLOOpt / cuda / BothRev",
            "value": 0.0000168,
            "unit": "s"
          },
          {
            "name": "scatter_sum / PartOpt / cuda / PreRev",
            "value": 0.000016927999999999998,
            "unit": "s"
          },
          {
            "name": "scatter_sum / PartOpt / cuda / PostRev",
            "value": 0.000016703,
            "unit": "s"
          },
          {
            "name": "scatter_sum / PartOpt / cuda / BothRev",
            "value": 0.00001728,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IPartOpt / cuda / PreRev",
            "value": 0.0000168,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IPartOpt / cuda / PostRev",
            "value": 0.000017024,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IPartOpt / cuda / BothRev",
            "value": 0.000016736,
            "unit": "s"
          },
          {
            "name": "scatter_sum / DefOpt / cuda / PreRev",
            "value": 0.0000176,
            "unit": "s"
          },
          {
            "name": "scatter_sum / DefOpt / cuda / PostRev",
            "value": 0.000016383999999999998,
            "unit": "s"
          },
          {
            "name": "scatter_sum / DefOpt / cuda / BothRev",
            "value": 0.000016542999999999997,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IDefOpt / cuda / PreRev",
            "value": 0.000016736,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IDefOpt / cuda / PostRev",
            "value": 0.0000168,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IDefOpt / cuda / BothRev",
            "value": 0.00001696,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.000001343475,
            "unit": "s"
          },
          {
            "name": "scatter_sum / Jax / tpu / Primal",
            "value": 0.0000014044499999999995,
            "unit": "s"
          },
          {
            "name": "scatter_sum / HLOOpt / tpu / Primal",
            "value": 0.000001343175,
            "unit": "s"
          },
          {
            "name": "scatter_sum / PartOpt / tpu / Primal",
            "value": 0.000001404525,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IPartOpt / tpu / Primal",
            "value": 0.0000013431,
            "unit": "s"
          },
          {
            "name": "scatter_sum / DefOpt / tpu / Primal",
            "value": 0.0000014045,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IDefOpt / tpu / Primal",
            "value": 0.00000134275,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Forward",
            "value": 0.000002702225,
            "unit": "s"
          },
          {
            "name": "scatter_sum / Jax / tpu / Forward",
            "value": 0.000002723475,
            "unit": "s"
          },
          {
            "name": "scatter_sum / HLOOpt / tpu / Forward",
            "value": 0.00000270605,
            "unit": "s"
          },
          {
            "name": "scatter_sum / PartOpt / tpu / Forward",
            "value": 0.000002685575,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IPartOpt / tpu / Forward",
            "value": 0.000002701075,
            "unit": "s"
          },
          {
            "name": "scatter_sum / DefOpt / tpu / Forward",
            "value": 0.000002689975,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IDefOpt / tpu / Forward",
            "value": 0.000002707075,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / PreRev",
            "value": 0.00000268835,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / PostRev",
            "value": 0.000002688925,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / BothRev",
            "value": 0.00000270215,
            "unit": "s"
          },
          {
            "name": "scatter_sum / Jax / tpu / BothRev",
            "value": 0.0000027393,
            "unit": "s"
          },
          {
            "name": "scatter_sum / HLOOpt / tpu / PreRev",
            "value": 0.000002698875,
            "unit": "s"
          },
          {
            "name": "scatter_sum / HLOOpt / tpu / PostRev",
            "value": 0.000002741625,
            "unit": "s"
          },
          {
            "name": "scatter_sum / HLOOpt / tpu / BothRev",
            "value": 0.000002709775,
            "unit": "s"
          },
          {
            "name": "scatter_sum / PartOpt / tpu / PreRev",
            "value": 0.0000027427000000000004,
            "unit": "s"
          },
          {
            "name": "scatter_sum / PartOpt / tpu / PostRev",
            "value": 0.0000026978249999999994,
            "unit": "s"
          },
          {
            "name": "scatter_sum / PartOpt / tpu / BothRev",
            "value": 0.000002742175,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IPartOpt / tpu / PreRev",
            "value": 0.00000270555,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IPartOpt / tpu / PostRev",
            "value": 0.000002738875,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IPartOpt / tpu / BothRev",
            "value": 0.000002701575,
            "unit": "s"
          },
          {
            "name": "scatter_sum / DefOpt / tpu / PreRev",
            "value": 0.0000027513,
            "unit": "s"
          },
          {
            "name": "scatter_sum / DefOpt / tpu / PostRev",
            "value": 0.000002702125,
            "unit": "s"
          },
          {
            "name": "scatter_sum / DefOpt / tpu / BothRev",
            "value": 0.0000027480500000000003,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IDefOpt / tpu / PreRev",
            "value": 0.00000270145,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IDefOpt / tpu / PostRev",
            "value": 0.00000274525,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IDefOpt / tpu / BothRev",
            "value": 0.000002697975,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000019449,
            "unit": "s"
          },
          {
            "name": "scatter_sum / Jax / cpu / Primal",
            "value": 0.000019068,
            "unit": "s"
          },
          {
            "name": "scatter_sum / HLOOpt / cpu / Primal",
            "value": 0.000024798,
            "unit": "s"
          },
          {
            "name": "scatter_sum / PartOpt / cpu / Primal",
            "value": 0.000019234000000000003,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IPartOpt / cpu / Primal",
            "value": 0.000019028,
            "unit": "s"
          },
          {
            "name": "scatter_sum / DefOpt / cpu / Primal",
            "value": 0.000019384,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IDefOpt / cpu / Primal",
            "value": 0.000018862,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Forward",
            "value": 0.000027536,
            "unit": "s"
          },
          {
            "name": "scatter_sum / Jax / cpu / Forward",
            "value": 0.000027023,
            "unit": "s"
          },
          {
            "name": "scatter_sum / HLOOpt / cpu / Forward",
            "value": 0.000027564,
            "unit": "s"
          },
          {
            "name": "scatter_sum / PartOpt / cpu / Forward",
            "value": 0.000027243,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IPartOpt / cpu / Forward",
            "value": 0.000027791,
            "unit": "s"
          },
          {
            "name": "scatter_sum / DefOpt / cpu / Forward",
            "value": 0.000027244,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IDefOpt / cpu / Forward",
            "value": 0.000027455,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / PreRev",
            "value": 0.000028039,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / PostRev",
            "value": 0.000027622,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / BothRev",
            "value": 0.000027648,
            "unit": "s"
          },
          {
            "name": "scatter_sum / Jax / cpu / BothRev",
            "value": 0.000027919,
            "unit": "s"
          },
          {
            "name": "scatter_sum / HLOOpt / cpu / PreRev",
            "value": 0.000027534,
            "unit": "s"
          },
          {
            "name": "scatter_sum / HLOOpt / cpu / PostRev",
            "value": 0.000027186,
            "unit": "s"
          },
          {
            "name": "scatter_sum / HLOOpt / cpu / BothRev",
            "value": 0.000027105,
            "unit": "s"
          },
          {
            "name": "scatter_sum / PartOpt / cpu / PreRev",
            "value": 0.000027778,
            "unit": "s"
          },
          {
            "name": "scatter_sum / PartOpt / cpu / PostRev",
            "value": 0.000027594,
            "unit": "s"
          },
          {
            "name": "scatter_sum / PartOpt / cpu / BothRev",
            "value": 0.000027698,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IPartOpt / cpu / PreRev",
            "value": 0.000034469,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IPartOpt / cpu / PostRev",
            "value": 0.000033648,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IPartOpt / cpu / BothRev",
            "value": 0.000028078,
            "unit": "s"
          },
          {
            "name": "scatter_sum / DefOpt / cpu / PreRev",
            "value": 0.00002794,
            "unit": "s"
          },
          {
            "name": "scatter_sum / DefOpt / cpu / PostRev",
            "value": 0.000027693,
            "unit": "s"
          },
          {
            "name": "scatter_sum / DefOpt / cpu / BothRev",
            "value": 0.000027712,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IDefOpt / cpu / PreRev",
            "value": 0.000027531,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IDefOpt / cpu / PostRev",
            "value": 0.000027136,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IDefOpt / cpu / BothRev",
            "value": 0.000027693,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000011,
            "unit": "s"
          },
          {
            "name": "scatter_sum / Jax / cpu / Primal",
            "value": 0.00001,
            "unit": "s"
          },
          {
            "name": "scatter_sum / HLOOpt / cpu / Primal",
            "value": 0.000011,
            "unit": "s"
          },
          {
            "name": "scatter_sum / PartOpt / cpu / Primal",
            "value": 0.000011,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IPartOpt / cpu / Primal",
            "value": 0.000011,
            "unit": "s"
          },
          {
            "name": "scatter_sum / DefOpt / cpu / Primal",
            "value": 0.000011,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IDefOpt / cpu / Primal",
            "value": 0.000011,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Forward",
            "value": 0.000016,
            "unit": "s"
          },
          {
            "name": "scatter_sum / Jax / cpu / Forward",
            "value": 0.000016,
            "unit": "s"
          },
          {
            "name": "scatter_sum / HLOOpt / cpu / Forward",
            "value": 0.000016,
            "unit": "s"
          },
          {
            "name": "scatter_sum / PartOpt / cpu / Forward",
            "value": 0.000017,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IPartOpt / cpu / Forward",
            "value": 0.000016,
            "unit": "s"
          },
          {
            "name": "scatter_sum / DefOpt / cpu / Forward",
            "value": 0.000016,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IDefOpt / cpu / Forward",
            "value": 0.000016,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / PreRev",
            "value": 0.000016,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / PostRev",
            "value": 0.000017999999999999997,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / BothRev",
            "value": 0.000017,
            "unit": "s"
          },
          {
            "name": "scatter_sum / Jax / cpu / BothRev",
            "value": 0.000016,
            "unit": "s"
          },
          {
            "name": "scatter_sum / HLOOpt / cpu / PreRev",
            "value": 0.000016,
            "unit": "s"
          },
          {
            "name": "scatter_sum / HLOOpt / cpu / PostRev",
            "value": 0.000016,
            "unit": "s"
          },
          {
            "name": "scatter_sum / HLOOpt / cpu / BothRev",
            "value": 0.000017,
            "unit": "s"
          },
          {
            "name": "scatter_sum / PartOpt / cpu / PreRev",
            "value": 0.000017,
            "unit": "s"
          },
          {
            "name": "scatter_sum / PartOpt / cpu / PostRev",
            "value": 0.000016,
            "unit": "s"
          },
          {
            "name": "scatter_sum / PartOpt / cpu / BothRev",
            "value": 0.000016,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IPartOpt / cpu / PreRev",
            "value": 0.000016,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IPartOpt / cpu / PostRev",
            "value": 0.000016,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IPartOpt / cpu / BothRev",
            "value": 0.000017999999999999997,
            "unit": "s"
          },
          {
            "name": "scatter_sum / DefOpt / cpu / PreRev",
            "value": 0.000016,
            "unit": "s"
          },
          {
            "name": "scatter_sum / DefOpt / cpu / PostRev",
            "value": 0.000016,
            "unit": "s"
          },
          {
            "name": "scatter_sum / DefOpt / cpu / BothRev",
            "value": 0.000016,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IDefOpt / cpu / PreRev",
            "value": 0.000016,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IDefOpt / cpu / PostRev",
            "value": 0.000016,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IDefOpt / cpu / BothRev",
            "value": 0.000016,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / Primal",
            "value": 0.0000065718999940145295,
            "unit": "s"
          },
          {
            "name": "slicing / Jax / cpu / Primal",
            "value": 0.000006047020051482832,
            "unit": "s"
          },
          {
            "name": "slicing / HLOOpt / cpu / Primal",
            "value": 0.000006182960005389759,
            "unit": "s"
          },
          {
            "name": "slicing / PartOpt / cpu / Primal",
            "value": 0.000006284980026975973,
            "unit": "s"
          },
          {
            "name": "slicing / IPartOpt / cpu / Primal",
            "value": 0.0000066760800291376655,
            "unit": "s"
          },
          {
            "name": "slicing / DefOpt / cpu / Primal",
            "value": 0.000006011119994582259,
            "unit": "s"
          },
          {
            "name": "slicing / IDefOpt / cpu / Primal",
            "value": 0.00000617611998677603,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / Forward",
            "value": 0.00000992725998003152,
            "unit": "s"
          },
          {
            "name": "slicing / Jax / cpu / Forward",
            "value": 0.000009392720012328936,
            "unit": "s"
          },
          {
            "name": "slicing / HLOOpt / cpu / Forward",
            "value": 0.000010251100020468584,
            "unit": "s"
          },
          {
            "name": "slicing / PartOpt / cpu / Forward",
            "value": 0.000009866499995041525,
            "unit": "s"
          },
          {
            "name": "slicing / IPartOpt / cpu / Forward",
            "value": 0.000010019239998655394,
            "unit": "s"
          },
          {
            "name": "slicing / DefOpt / cpu / Forward",
            "value": 0.000009773899992069344,
            "unit": "s"
          },
          {
            "name": "slicing / IDefOpt / cpu / Forward",
            "value": 0.000009806559974094852,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / PreRev",
            "value": 0.000010159940056837514,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / PostRev",
            "value": 0.000010119239987034234,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / BothRev",
            "value": 0.000010195520026172744,
            "unit": "s"
          },
          {
            "name": "slicing / Jax / cpu / BothRev",
            "value": 0.000010721000016928885,
            "unit": "s"
          },
          {
            "name": "slicing / HLOOpt / cpu / PreRev",
            "value": 0.000010868420013139255,
            "unit": "s"
          },
          {
            "name": "slicing / HLOOpt / cpu / PostRev",
            "value": 0.000012423019970810856,
            "unit": "s"
          },
          {
            "name": "slicing / HLOOpt / cpu / BothRev",
            "value": 0.000009850219994405052,
            "unit": "s"
          },
          {
            "name": "slicing / PartOpt / cpu / PreRev",
            "value": 0.00000993288001154724,
            "unit": "s"
          },
          {
            "name": "slicing / PartOpt / cpu / PostRev",
            "value": 0.000010278080017087632,
            "unit": "s"
          },
          {
            "name": "slicing / PartOpt / cpu / BothRev",
            "value": 0.00001018304002172954,
            "unit": "s"
          },
          {
            "name": "slicing / IPartOpt / cpu / PreRev",
            "value": 0.000010028739980043611,
            "unit": "s"
          },
          {
            "name": "slicing / IPartOpt / cpu / PostRev",
            "value": 0.00001013477996821166,
            "unit": "s"
          },
          {
            "name": "slicing / IPartOpt / cpu / BothRev",
            "value": 0.000009986599998228483,
            "unit": "s"
          },
          {
            "name": "slicing / DefOpt / cpu / PreRev",
            "value": 0.000010073839976030283,
            "unit": "s"
          },
          {
            "name": "slicing / DefOpt / cpu / PostRev",
            "value": 0.000010306819967809134,
            "unit": "s"
          },
          {
            "name": "slicing / DefOpt / cpu / BothRev",
            "value": 0.000009928880026564,
            "unit": "s"
          },
          {
            "name": "slicing / IDefOpt / cpu / PreRev",
            "value": 0.000010205140033576756,
            "unit": "s"
          },
          {
            "name": "slicing / IDefOpt / cpu / PostRev",
            "value": 0.00001020074000734894,
            "unit": "s"
          },
          {
            "name": "slicing / IDefOpt / cpu / BothRev",
            "value": 0.000010028259966929908,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cuda / Primal",
            "value": 0.000001887,
            "unit": "s"
          },
          {
            "name": "slicing / Jax / cuda / Primal",
            "value": 0.000001888,
            "unit": "s"
          },
          {
            "name": "slicing / HLOOpt / cuda / Primal",
            "value": 0.000001887,
            "unit": "s"
          },
          {
            "name": "slicing / PartOpt / cuda / Primal",
            "value": 0.000001888,
            "unit": "s"
          },
          {
            "name": "slicing / IPartOpt / cuda / Primal",
            "value": 0.000001888,
            "unit": "s"
          },
          {
            "name": "slicing / DefOpt / cuda / Primal",
            "value": 0.000001887,
            "unit": "s"
          },
          {
            "name": "slicing / IDefOpt / cuda / Primal",
            "value": 0.000001887,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cuda / Forward",
            "value": 0.000010208,
            "unit": "s"
          },
          {
            "name": "slicing / Jax / cuda / Forward",
            "value": 0.00001008,
            "unit": "s"
          },
          {
            "name": "slicing / HLOOpt / cuda / Forward",
            "value": 0.000010015,
            "unit": "s"
          },
          {
            "name": "slicing / PartOpt / cuda / Forward",
            "value": 0.000010176,
            "unit": "s"
          },
          {
            "name": "slicing / IPartOpt / cuda / Forward",
            "value": 0.000009664,
            "unit": "s"
          },
          {
            "name": "slicing / DefOpt / cuda / Forward",
            "value": 0.000010048,
            "unit": "s"
          },
          {
            "name": "slicing / IDefOpt / cuda / Forward",
            "value": 0.000009984,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cuda / PreRev",
            "value": 0.000009695,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cuda / PostRev",
            "value": 0.000009696,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cuda / BothRev",
            "value": 0.000009791,
            "unit": "s"
          },
          {
            "name": "slicing / Jax / cuda / BothRev",
            "value": 0.000009536,
            "unit": "s"
          },
          {
            "name": "slicing / HLOOpt / cuda / PreRev",
            "value": 0.000009792,
            "unit": "s"
          },
          {
            "name": "slicing / HLOOpt / cuda / PostRev",
            "value": 0.000009311,
            "unit": "s"
          },
          {
            "name": "slicing / HLOOpt / cuda / BothRev",
            "value": 0.000009824,
            "unit": "s"
          },
          {
            "name": "slicing / PartOpt / cuda / PreRev",
            "value": 0.000010208,
            "unit": "s"
          },
          {
            "name": "slicing / PartOpt / cuda / PostRev",
            "value": 0.00001008,
            "unit": "s"
          },
          {
            "name": "slicing / PartOpt / cuda / BothRev",
            "value": 0.00001008,
            "unit": "s"
          },
          {
            "name": "slicing / IPartOpt / cuda / PreRev",
            "value": 0.000010049,
            "unit": "s"
          },
          {
            "name": "slicing / IPartOpt / cuda / PostRev",
            "value": 0.00001072,
            "unit": "s"
          },
          {
            "name": "slicing / IPartOpt / cuda / BothRev",
            "value": 0.000009505,
            "unit": "s"
          },
          {
            "name": "slicing / DefOpt / cuda / PreRev",
            "value": 0.000010016,
            "unit": "s"
          },
          {
            "name": "slicing / DefOpt / cuda / PostRev",
            "value": 0.000010208,
            "unit": "s"
          },
          {
            "name": "slicing / DefOpt / cuda / BothRev",
            "value": 0.00001024,
            "unit": "s"
          },
          {
            "name": "slicing / IDefOpt / cuda / PreRev",
            "value": 0.000010144,
            "unit": "s"
          },
          {
            "name": "slicing / IDefOpt / cuda / PostRev",
            "value": 0.00000976,
            "unit": "s"
          },
          {
            "name": "slicing / IDefOpt / cuda / BothRev",
            "value": 0.000010111,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / tpu / Primal",
            "value": 0.000001018175,
            "unit": "s"
          },
          {
            "name": "slicing / Jax / tpu / Primal",
            "value": 9.61625e-7,
            "unit": "s"
          },
          {
            "name": "slicing / HLOOpt / tpu / Primal",
            "value": 0.0000010201000000000002,
            "unit": "s"
          },
          {
            "name": "slicing / PartOpt / tpu / Primal",
            "value": 9.68875e-7,
            "unit": "s"
          },
          {
            "name": "slicing / IPartOpt / tpu / Primal",
            "value": 0.0000010229,
            "unit": "s"
          },
          {
            "name": "slicing / DefOpt / tpu / Primal",
            "value": 9.7125e-7,
            "unit": "s"
          },
          {
            "name": "slicing / IDefOpt / tpu / Primal",
            "value": 0.0000010199999999999998,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / tpu / Forward",
            "value": 0.0000014013999999999998,
            "unit": "s"
          },
          {
            "name": "slicing / Jax / tpu / Forward",
            "value": 0.0000014771,
            "unit": "s"
          },
          {
            "name": "slicing / HLOOpt / tpu / Forward",
            "value": 0.000001513875,
            "unit": "s"
          },
          {
            "name": "slicing / PartOpt / tpu / Forward",
            "value": 0.000001488725,
            "unit": "s"
          },
          {
            "name": "slicing / IPartOpt / tpu / Forward",
            "value": 0.000001513225,
            "unit": "s"
          },
          {
            "name": "slicing / DefOpt / tpu / Forward",
            "value": 0.000001490975,
            "unit": "s"
          },
          {
            "name": "slicing / IDefOpt / tpu / Forward",
            "value": 0.000001517,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / tpu / PreRev",
            "value": 0.000002567925,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / tpu / PostRev",
            "value": 0.000002539775,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / tpu / BothRev",
            "value": 0.00000257955,
            "unit": "s"
          },
          {
            "name": "slicing / Jax / tpu / BothRev",
            "value": 0.0000025363,
            "unit": "s"
          },
          {
            "name": "slicing / HLOOpt / tpu / PreRev",
            "value": 0.00000258515,
            "unit": "s"
          },
          {
            "name": "slicing / HLOOpt / tpu / PostRev",
            "value": 0.0000025454500000000003,
            "unit": "s"
          },
          {
            "name": "slicing / HLOOpt / tpu / BothRev",
            "value": 0.000002587675,
            "unit": "s"
          },
          {
            "name": "slicing / PartOpt / tpu / PreRev",
            "value": 0.000002533125,
            "unit": "s"
          },
          {
            "name": "slicing / PartOpt / tpu / PostRev",
            "value": 0.00000257785,
            "unit": "s"
          },
          {
            "name": "slicing / PartOpt / tpu / BothRev",
            "value": 0.00000254905,
            "unit": "s"
          },
          {
            "name": "slicing / IPartOpt / tpu / PreRev",
            "value": 0.000002587675,
            "unit": "s"
          },
          {
            "name": "slicing / IPartOpt / tpu / PostRev",
            "value": 0.000002545775,
            "unit": "s"
          },
          {
            "name": "slicing / IPartOpt / tpu / BothRev",
            "value": 0.000002578575,
            "unit": "s"
          },
          {
            "name": "slicing / DefOpt / tpu / PreRev",
            "value": 0.000002541625,
            "unit": "s"
          },
          {
            "name": "slicing / DefOpt / tpu / PostRev",
            "value": 0.00000258455,
            "unit": "s"
          },
          {
            "name": "slicing / DefOpt / tpu / BothRev",
            "value": 0.0000025394250000000003,
            "unit": "s"
          },
          {
            "name": "slicing / IDefOpt / tpu / PreRev",
            "value": 0.00000257635,
            "unit": "s"
          },
          {
            "name": "slicing / IDefOpt / tpu / PostRev",
            "value": 0.0000025470500000000005,
            "unit": "s"
          },
          {
            "name": "slicing / IDefOpt / tpu / BothRev",
            "value": 0.000002589975,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / Primal",
            "value": 0.000015691,
            "unit": "s"
          },
          {
            "name": "slicing / Jax / cpu / Primal",
            "value": 0.000015425,
            "unit": "s"
          },
          {
            "name": "slicing / HLOOpt / cpu / Primal",
            "value": 0.000015515,
            "unit": "s"
          },
          {
            "name": "slicing / PartOpt / cpu / Primal",
            "value": 0.00001568,
            "unit": "s"
          },
          {
            "name": "slicing / IPartOpt / cpu / Primal",
            "value": 0.000015647,
            "unit": "s"
          },
          {
            "name": "slicing / DefOpt / cpu / Primal",
            "value": 0.000015692999999999997,
            "unit": "s"
          },
          {
            "name": "slicing / IDefOpt / cpu / Primal",
            "value": 0.00001514,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / Forward",
            "value": 0.000020794,
            "unit": "s"
          },
          {
            "name": "slicing / Jax / cpu / Forward",
            "value": 0.000020506,
            "unit": "s"
          },
          {
            "name": "slicing / HLOOpt / cpu / Forward",
            "value": 0.000020367,
            "unit": "s"
          },
          {
            "name": "slicing / PartOpt / cpu / Forward",
            "value": 0.000020598,
            "unit": "s"
          },
          {
            "name": "slicing / IPartOpt / cpu / Forward",
            "value": 0.000020516,
            "unit": "s"
          },
          {
            "name": "slicing / DefOpt / cpu / Forward",
            "value": 0.000020365,
            "unit": "s"
          },
          {
            "name": "slicing / IDefOpt / cpu / Forward",
            "value": 0.000026157,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / PreRev",
            "value": 0.000021229000000000003,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / PostRev",
            "value": 0.000021228,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / BothRev",
            "value": 0.000020781,
            "unit": "s"
          },
          {
            "name": "slicing / Jax / cpu / BothRev",
            "value": 0.000027089,
            "unit": "s"
          },
          {
            "name": "slicing / HLOOpt / cpu / PreRev",
            "value": 0.000021435,
            "unit": "s"
          },
          {
            "name": "slicing / HLOOpt / cpu / PostRev",
            "value": 0.000021052,
            "unit": "s"
          },
          {
            "name": "slicing / HLOOpt / cpu / BothRev",
            "value": 0.000020912,
            "unit": "s"
          },
          {
            "name": "slicing / PartOpt / cpu / PreRev",
            "value": 0.000021117,
            "unit": "s"
          },
          {
            "name": "slicing / PartOpt / cpu / PostRev",
            "value": 0.000021037,
            "unit": "s"
          },
          {
            "name": "slicing / PartOpt / cpu / BothRev",
            "value": 0.000021472,
            "unit": "s"
          },
          {
            "name": "slicing / IPartOpt / cpu / PreRev",
            "value": 0.000021522,
            "unit": "s"
          },
          {
            "name": "slicing / IPartOpt / cpu / PostRev",
            "value": 0.000020948,
            "unit": "s"
          },
          {
            "name": "slicing / IPartOpt / cpu / BothRev",
            "value": 0.000021484,
            "unit": "s"
          },
          {
            "name": "slicing / DefOpt / cpu / PreRev",
            "value": 0.000021243000000000003,
            "unit": "s"
          },
          {
            "name": "slicing / DefOpt / cpu / PostRev",
            "value": 0.000020996,
            "unit": "s"
          },
          {
            "name": "slicing / DefOpt / cpu / BothRev",
            "value": 0.000021108,
            "unit": "s"
          },
          {
            "name": "slicing / IDefOpt / cpu / PreRev",
            "value": 0.000021233,
            "unit": "s"
          },
          {
            "name": "slicing / IDefOpt / cpu / PostRev",
            "value": 0.000020987,
            "unit": "s"
          },
          {
            "name": "slicing / IDefOpt / cpu / BothRev",
            "value": 0.00002109,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / Primal",
            "value": 0.000008,
            "unit": "s"
          },
          {
            "name": "slicing / Jax / cpu / Primal",
            "value": 0.000008,
            "unit": "s"
          },
          {
            "name": "slicing / HLOOpt / cpu / Primal",
            "value": 0.000008999999999999999,
            "unit": "s"
          },
          {
            "name": "slicing / PartOpt / cpu / Primal",
            "value": 0.000008999999999999999,
            "unit": "s"
          },
          {
            "name": "slicing / IPartOpt / cpu / Primal",
            "value": 0.000008,
            "unit": "s"
          },
          {
            "name": "slicing / DefOpt / cpu / Primal",
            "value": 0.000008999999999999999,
            "unit": "s"
          },
          {
            "name": "slicing / IDefOpt / cpu / Primal",
            "value": 0.000008999999999999999,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / Forward",
            "value": 0.000011,
            "unit": "s"
          },
          {
            "name": "slicing / Jax / cpu / Forward",
            "value": 0.000011,
            "unit": "s"
          },
          {
            "name": "slicing / HLOOpt / cpu / Forward",
            "value": 0.000011,
            "unit": "s"
          },
          {
            "name": "slicing / PartOpt / cpu / Forward",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "slicing / IPartOpt / cpu / Forward",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "slicing / DefOpt / cpu / Forward",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "slicing / IDefOpt / cpu / Forward",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / PreRev",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / PostRev",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / BothRev",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "slicing / Jax / cpu / BothRev",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "slicing / HLOOpt / cpu / PreRev",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "slicing / HLOOpt / cpu / PostRev",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "slicing / HLOOpt / cpu / BothRev",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "slicing / PartOpt / cpu / PreRev",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "slicing / PartOpt / cpu / PostRev",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "slicing / PartOpt / cpu / BothRev",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "slicing / IPartOpt / cpu / PreRev",
            "value": 0.000011,
            "unit": "s"
          },
          {
            "name": "slicing / IPartOpt / cpu / PostRev",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "slicing / IPartOpt / cpu / BothRev",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "slicing / DefOpt / cpu / PreRev",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "slicing / DefOpt / cpu / PostRev",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "slicing / DefOpt / cpu / BothRev",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "slicing / IDefOpt / cpu / PreRev",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "slicing / IDefOpt / cpu / PostRev",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "slicing / IDefOpt / cpu / BothRev",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / Primal",
            "value": 0.00000816919998214871,
            "unit": "s"
          },
          {
            "name": "sum    / Jax / cpu / Primal",
            "value": 0.000007457820038325735,
            "unit": "s"
          },
          {
            "name": "sum    / HLOOpt / cpu / Primal",
            "value": 0.00000763924002967542,
            "unit": "s"
          },
          {
            "name": "sum    / PartOpt / cpu / Primal",
            "value": 0.000007604439997521695,
            "unit": "s"
          },
          {
            "name": "sum    / IPartOpt / cpu / Primal",
            "value": 0.000008017280024432693,
            "unit": "s"
          },
          {
            "name": "sum    / DefOpt / cpu / Primal",
            "value": 0.000007534759988629957,
            "unit": "s"
          },
          {
            "name": "sum    / IDefOpt / cpu / Primal",
            "value": 0.000007843719968150254,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / Forward",
            "value": 0.0000113872000201809,
            "unit": "s"
          },
          {
            "name": "sum    / Jax / cpu / Forward",
            "value": 0.000011374180012353465,
            "unit": "s"
          },
          {
            "name": "sum    / HLOOpt / cpu / Forward",
            "value": 0.000011870960024680244,
            "unit": "s"
          },
          {
            "name": "sum    / PartOpt / cpu / Forward",
            "value": 0.000011473179984022864,
            "unit": "s"
          },
          {
            "name": "sum    / IPartOpt / cpu / Forward",
            "value": 0.000011139440020997428,
            "unit": "s"
          },
          {
            "name": "sum    / DefOpt / cpu / Forward",
            "value": 0.000011762800049837096,
            "unit": "s"
          },
          {
            "name": "sum    / IDefOpt / cpu / Forward",
            "value": 0.000011532799962878926,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / PreRev",
            "value": 0.000011403500038795756,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / PostRev",
            "value": 0.00001117342003453814,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / BothRev",
            "value": 0.000011444460023994906,
            "unit": "s"
          },
          {
            "name": "sum    / Jax / cpu / BothRev",
            "value": 0.00001101707997804624,
            "unit": "s"
          },
          {
            "name": "sum    / HLOOpt / cpu / PreRev",
            "value": 0.000011423160003687371,
            "unit": "s"
          },
          {
            "name": "sum    / HLOOpt / cpu / PostRev",
            "value": 0.000013029080000706018,
            "unit": "s"
          },
          {
            "name": "sum    / HLOOpt / cpu / BothRev",
            "value": 0.000010510639995118254,
            "unit": "s"
          },
          {
            "name": "sum    / PartOpt / cpu / PreRev",
            "value": 0.000011088860055679106,
            "unit": "s"
          },
          {
            "name": "sum    / PartOpt / cpu / PostRev",
            "value": 0.000011051279980165418,
            "unit": "s"
          },
          {
            "name": "sum    / PartOpt / cpu / BothRev",
            "value": 0.000010995619968525716,
            "unit": "s"
          },
          {
            "name": "sum    / IPartOpt / cpu / PreRev",
            "value": 0.000011126600020361364,
            "unit": "s"
          },
          {
            "name": "sum    / IPartOpt / cpu / PostRev",
            "value": 0.000011172479962624494,
            "unit": "s"
          },
          {
            "name": "sum    / IPartOpt / cpu / BothRev",
            "value": 0.000011434539965193835,
            "unit": "s"
          },
          {
            "name": "sum    / DefOpt / cpu / PreRev",
            "value": 0.00001090151998141664,
            "unit": "s"
          },
          {
            "name": "sum    / DefOpt / cpu / PostRev",
            "value": 0.000010847459998331031,
            "unit": "s"
          },
          {
            "name": "sum    / DefOpt / cpu / BothRev",
            "value": 0.00001036037999256223,
            "unit": "s"
          },
          {
            "name": "sum    / IDefOpt / cpu / PreRev",
            "value": 0.000010492980036360678,
            "unit": "s"
          },
          {
            "name": "sum    / IDefOpt / cpu / PostRev",
            "value": 0.000010710420028772205,
            "unit": "s"
          },
          {
            "name": "sum    / IDefOpt / cpu / BothRev",
            "value": 0.00001095347998671059,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cuda / Primal",
            "value": 0.000002047,
            "unit": "s"
          },
          {
            "name": "sum    / Jax / cuda / Primal",
            "value": 0.000002047,
            "unit": "s"
          },
          {
            "name": "sum    / HLOOpt / cuda / Primal",
            "value": 0.000002047,
            "unit": "s"
          },
          {
            "name": "sum    / PartOpt / cuda / Primal",
            "value": 0.000002047,
            "unit": "s"
          },
          {
            "name": "sum    / IPartOpt / cuda / Primal",
            "value": 0.000002047,
            "unit": "s"
          },
          {
            "name": "sum    / DefOpt / cuda / Primal",
            "value": 0.000002047,
            "unit": "s"
          },
          {
            "name": "sum    / IDefOpt / cuda / Primal",
            "value": 0.000002047,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cuda / Forward",
            "value": 0.000010336,
            "unit": "s"
          },
          {
            "name": "sum    / Jax / cuda / Forward",
            "value": 0.000010432,
            "unit": "s"
          },
          {
            "name": "sum    / HLOOpt / cuda / Forward",
            "value": 0.000009696,
            "unit": "s"
          },
          {
            "name": "sum    / PartOpt / cuda / Forward",
            "value": 0.00001024,
            "unit": "s"
          },
          {
            "name": "sum    / IPartOpt / cuda / Forward",
            "value": 0.0000096,
            "unit": "s"
          },
          {
            "name": "sum    / DefOpt / cuda / Forward",
            "value": 0.000010048,
            "unit": "s"
          },
          {
            "name": "sum    / IDefOpt / cuda / Forward",
            "value": 0.000010176,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cuda / PreRev",
            "value": 0.000009791,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cuda / PostRev",
            "value": 0.000009664,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cuda / BothRev",
            "value": 0.000009823,
            "unit": "s"
          },
          {
            "name": "sum    / Jax / cuda / BothRev",
            "value": 0.000008928,
            "unit": "s"
          },
          {
            "name": "sum    / HLOOpt / cuda / PreRev",
            "value": 0.00000944,
            "unit": "s"
          },
          {
            "name": "sum    / HLOOpt / cuda / PostRev",
            "value": 0.000009728,
            "unit": "s"
          },
          {
            "name": "sum    / HLOOpt / cuda / BothRev",
            "value": 0.000009855,
            "unit": "s"
          },
          {
            "name": "sum    / PartOpt / cuda / PreRev",
            "value": 0.000010016,
            "unit": "s"
          },
          {
            "name": "sum    / PartOpt / cuda / PostRev",
            "value": 0.000009504,
            "unit": "s"
          },
          {
            "name": "sum    / PartOpt / cuda / BothRev",
            "value": 0.000009792,
            "unit": "s"
          },
          {
            "name": "sum    / IPartOpt / cuda / PreRev",
            "value": 0.00000976,
            "unit": "s"
          },
          {
            "name": "sum    / IPartOpt / cuda / PostRev",
            "value": 0.000010016,
            "unit": "s"
          },
          {
            "name": "sum    / IPartOpt / cuda / BothRev",
            "value": 0.000009824,
            "unit": "s"
          },
          {
            "name": "sum    / DefOpt / cuda / PreRev",
            "value": 0.000009824,
            "unit": "s"
          },
          {
            "name": "sum    / DefOpt / cuda / PostRev",
            "value": 0.000009568,
            "unit": "s"
          },
          {
            "name": "sum    / DefOpt / cuda / BothRev",
            "value": 0.000009472,
            "unit": "s"
          },
          {
            "name": "sum    / IDefOpt / cuda / PreRev",
            "value": 0.000009952,
            "unit": "s"
          },
          {
            "name": "sum    / IDefOpt / cuda / PostRev",
            "value": 0.000009728,
            "unit": "s"
          },
          {
            "name": "sum    / IDefOpt / cuda / BothRev",
            "value": 0.000009727,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / tpu / Primal",
            "value": 5.102750000000001e-7,
            "unit": "s"
          },
          {
            "name": "sum    / Jax / tpu / Primal",
            "value": 5.47075e-7,
            "unit": "s"
          },
          {
            "name": "sum    / HLOOpt / tpu / Primal",
            "value": 5.106750000000001e-7,
            "unit": "s"
          },
          {
            "name": "sum    / PartOpt / tpu / Primal",
            "value": 5.473e-7,
            "unit": "s"
          },
          {
            "name": "sum    / IPartOpt / tpu / Primal",
            "value": 5.10375e-7,
            "unit": "s"
          },
          {
            "name": "sum    / DefOpt / tpu / Primal",
            "value": 5.467249999999999e-7,
            "unit": "s"
          },
          {
            "name": "sum    / IDefOpt / tpu / Primal",
            "value": 5.104e-7,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / tpu / Forward",
            "value": 0.0000015484,
            "unit": "s"
          },
          {
            "name": "sum    / Jax / tpu / Forward",
            "value": 0.0000015049000000000002,
            "unit": "s"
          },
          {
            "name": "sum    / HLOOpt / tpu / Forward",
            "value": 0.000001533725,
            "unit": "s"
          },
          {
            "name": "sum    / PartOpt / tpu / Forward",
            "value": 0.0000014942,
            "unit": "s"
          },
          {
            "name": "sum    / IPartOpt / tpu / Forward",
            "value": 0.0000015378249999999998,
            "unit": "s"
          },
          {
            "name": "sum    / DefOpt / tpu / Forward",
            "value": 0.0000014951,
            "unit": "s"
          },
          {
            "name": "sum    / IDefOpt / tpu / Forward",
            "value": 0.000001531225,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / tpu / PreRev",
            "value": 0.0000010535,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / tpu / PostRev",
            "value": 0.00000109215,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / tpu / BothRev",
            "value": 0.000001053,
            "unit": "s"
          },
          {
            "name": "sum    / Jax / tpu / BothRev",
            "value": 0.000001083075,
            "unit": "s"
          },
          {
            "name": "sum    / HLOOpt / tpu / PreRev",
            "value": 0.000001052425,
            "unit": "s"
          },
          {
            "name": "sum    / HLOOpt / tpu / PostRev",
            "value": 0.00000109205,
            "unit": "s"
          },
          {
            "name": "sum    / HLOOpt / tpu / BothRev",
            "value": 0.000001050175,
            "unit": "s"
          },
          {
            "name": "sum    / PartOpt / tpu / PreRev",
            "value": 0.0000010928000000000002,
            "unit": "s"
          },
          {
            "name": "sum    / PartOpt / tpu / PostRev",
            "value": 0.00000105045,
            "unit": "s"
          },
          {
            "name": "sum    / PartOpt / tpu / BothRev",
            "value": 0.000001091575,
            "unit": "s"
          },
          {
            "name": "sum    / IPartOpt / tpu / PreRev",
            "value": 0.00000105165,
            "unit": "s"
          },
          {
            "name": "sum    / IPartOpt / tpu / PostRev",
            "value": 0.00000109235,
            "unit": "s"
          },
          {
            "name": "sum    / IPartOpt / tpu / BothRev",
            "value": 0.000001055825,
            "unit": "s"
          },
          {
            "name": "sum    / DefOpt / tpu / PreRev",
            "value": 0.0000010853,
            "unit": "s"
          },
          {
            "name": "sum    / DefOpt / tpu / PostRev",
            "value": 0.00000104685,
            "unit": "s"
          },
          {
            "name": "sum    / DefOpt / tpu / BothRev",
            "value": 0.000001083325,
            "unit": "s"
          },
          {
            "name": "sum    / IDefOpt / tpu / PreRev",
            "value": 0.000001047075,
            "unit": "s"
          },
          {
            "name": "sum    / IDefOpt / tpu / PostRev",
            "value": 0.000001092225,
            "unit": "s"
          },
          {
            "name": "sum    / IDefOpt / tpu / BothRev",
            "value": 0.000001053925,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / Primal",
            "value": 0.000017923,
            "unit": "s"
          },
          {
            "name": "sum    / Jax / cpu / Primal",
            "value": 0.000017807,
            "unit": "s"
          },
          {
            "name": "sum    / HLOOpt / cpu / Primal",
            "value": 0.000017851,
            "unit": "s"
          },
          {
            "name": "sum    / PartOpt / cpu / Primal",
            "value": 0.000017639,
            "unit": "s"
          },
          {
            "name": "sum    / IPartOpt / cpu / Primal",
            "value": 0.000017523,
            "unit": "s"
          },
          {
            "name": "sum    / DefOpt / cpu / Primal",
            "value": 0.000017735000000000002,
            "unit": "s"
          },
          {
            "name": "sum    / IDefOpt / cpu / Primal",
            "value": 0.000017645,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / Forward",
            "value": 0.000024507,
            "unit": "s"
          },
          {
            "name": "sum    / Jax / cpu / Forward",
            "value": 0.000023999,
            "unit": "s"
          },
          {
            "name": "sum    / HLOOpt / cpu / Forward",
            "value": 0.000024283,
            "unit": "s"
          },
          {
            "name": "sum    / PartOpt / cpu / Forward",
            "value": 0.000024219,
            "unit": "s"
          },
          {
            "name": "sum    / IPartOpt / cpu / Forward",
            "value": 0.000025032,
            "unit": "s"
          },
          {
            "name": "sum    / DefOpt / cpu / Forward",
            "value": 0.000024424,
            "unit": "s"
          },
          {
            "name": "sum    / IDefOpt / cpu / Forward",
            "value": 0.000024392,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / PreRev",
            "value": 0.000023016,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / PostRev",
            "value": 0.000023033,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / BothRev",
            "value": 0.000023151,
            "unit": "s"
          },
          {
            "name": "sum    / Jax / cpu / BothRev",
            "value": 0.000022878,
            "unit": "s"
          },
          {
            "name": "sum    / HLOOpt / cpu / PreRev",
            "value": 0.000022986,
            "unit": "s"
          },
          {
            "name": "sum    / HLOOpt / cpu / PostRev",
            "value": 0.000023171,
            "unit": "s"
          },
          {
            "name": "sum    / HLOOpt / cpu / BothRev",
            "value": 0.000023042,
            "unit": "s"
          },
          {
            "name": "sum    / PartOpt / cpu / PreRev",
            "value": 0.000022991,
            "unit": "s"
          },
          {
            "name": "sum    / PartOpt / cpu / PostRev",
            "value": 0.000023028,
            "unit": "s"
          },
          {
            "name": "sum    / PartOpt / cpu / BothRev",
            "value": 0.000022731,
            "unit": "s"
          },
          {
            "name": "sum    / IPartOpt / cpu / PreRev",
            "value": 0.00002314,
            "unit": "s"
          },
          {
            "name": "sum    / IPartOpt / cpu / PostRev",
            "value": 0.000023065,
            "unit": "s"
          },
          {
            "name": "sum    / IPartOpt / cpu / BothRev",
            "value": 0.000023228,
            "unit": "s"
          },
          {
            "name": "sum    / DefOpt / cpu / PreRev",
            "value": 0.00002938,
            "unit": "s"
          },
          {
            "name": "sum    / DefOpt / cpu / PostRev",
            "value": 0.00002311,
            "unit": "s"
          },
          {
            "name": "sum    / DefOpt / cpu / BothRev",
            "value": 0.000023154,
            "unit": "s"
          },
          {
            "name": "sum    / IDefOpt / cpu / PreRev",
            "value": 0.000023278,
            "unit": "s"
          },
          {
            "name": "sum    / IDefOpt / cpu / PostRev",
            "value": 0.00002924,
            "unit": "s"
          },
          {
            "name": "sum    / IDefOpt / cpu / BothRev",
            "value": 0.000022679,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / Primal",
            "value": 0.00001,
            "unit": "s"
          },
          {
            "name": "sum    / Jax / cpu / Primal",
            "value": 0.00001,
            "unit": "s"
          },
          {
            "name": "sum    / HLOOpt / cpu / Primal",
            "value": 0.00001,
            "unit": "s"
          },
          {
            "name": "sum    / PartOpt / cpu / Primal",
            "value": 0.00001,
            "unit": "s"
          },
          {
            "name": "sum    / IPartOpt / cpu / Primal",
            "value": 0.00001,
            "unit": "s"
          },
          {
            "name": "sum    / DefOpt / cpu / Primal",
            "value": 0.00001,
            "unit": "s"
          },
          {
            "name": "sum    / IDefOpt / cpu / Primal",
            "value": 0.00001,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / Forward",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "sum    / Jax / cpu / Forward",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "sum    / HLOOpt / cpu / Forward",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "sum    / PartOpt / cpu / Forward",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "sum    / IPartOpt / cpu / Forward",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "sum    / DefOpt / cpu / Forward",
            "value": 0.000015,
            "unit": "s"
          },
          {
            "name": "sum    / IDefOpt / cpu / Forward",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / PreRev",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / PostRev",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / BothRev",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "sum    / Jax / cpu / BothRev",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "sum    / HLOOpt / cpu / PreRev",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "sum    / HLOOpt / cpu / PostRev",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "sum    / HLOOpt / cpu / BothRev",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "sum    / PartOpt / cpu / PreRev",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "sum    / PartOpt / cpu / PostRev",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "sum    / PartOpt / cpu / BothRev",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "sum    / IPartOpt / cpu / PreRev",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "sum    / IPartOpt / cpu / PostRev",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "sum    / IPartOpt / cpu / BothRev",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "sum    / DefOpt / cpu / PreRev",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "sum    / DefOpt / cpu / PostRev",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "sum    / DefOpt / cpu / BothRev",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "sum    / IDefOpt / cpu / PreRev",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "sum    / IDefOpt / cpu / PostRev",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "sum    / IDefOpt / cpu / BothRev",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "value_and_grad / JaXPipe / cpu / Primal",
            "value": 0.000014257219991122838,
            "unit": "s"
          },
          {
            "name": "value_and_grad / Jax / cpu / Primal",
            "value": 0.000014079739994485862,
            "unit": "s"
          },
          {
            "name": "value_and_grad / HLOOpt / cpu / Primal",
            "value": 0.0000138554400200519,
            "unit": "s"
          },
          {
            "name": "value_and_grad / PartOpt / cpu / Primal",
            "value": 0.000013732859997617196,
            "unit": "s"
          },
          {
            "name": "value_and_grad / IPartOpt / cpu / Primal",
            "value": 0.000013589980017059134,
            "unit": "s"
          },
          {
            "name": "value_and_grad / DefOpt / cpu / Primal",
            "value": 0.000013590080016001592,
            "unit": "s"
          },
          {
            "name": "value_and_grad / IDefOpt / cpu / Primal",
            "value": 0.000013416999981927802,
            "unit": "s"
          },
          {
            "name": "value_and_grad / JaXPipe / cuda / Primal",
            "value": 0.000033056,
            "unit": "s"
          },
          {
            "name": "value_and_grad / Jax / cuda / Primal",
            "value": 0.000032705,
            "unit": "s"
          },
          {
            "name": "value_and_grad / HLOOpt / cuda / Primal",
            "value": 0.000033119999999999995,
            "unit": "s"
          },
          {
            "name": "value_and_grad / PartOpt / cuda / Primal",
            "value": 0.000033024,
            "unit": "s"
          },
          {
            "name": "value_and_grad / IPartOpt / cuda / Primal",
            "value": 0.000032608,
            "unit": "s"
          },
          {
            "name": "value_and_grad / DefOpt / cuda / Primal",
            "value": 0.000032288,
            "unit": "s"
          },
          {
            "name": "value_and_grad / IDefOpt / cuda / Primal",
            "value": 0.000032864,
            "unit": "s"
          },
          {
            "name": "value_and_grad / JaXPipe / tpu / Primal",
            "value": 0,
            "unit": "s"
          },
          {
            "name": "value_and_grad / Jax / tpu / Primal",
            "value": 0,
            "unit": "s"
          },
          {
            "name": "value_and_grad / HLOOpt / tpu / Primal",
            "value": 0,
            "unit": "s"
          },
          {
            "name": "value_and_grad / PartOpt / tpu / Primal",
            "value": 0,
            "unit": "s"
          },
          {
            "name": "value_and_grad / IPartOpt / tpu / Primal",
            "value": 0,
            "unit": "s"
          },
          {
            "name": "value_and_grad / DefOpt / tpu / Primal",
            "value": 0,
            "unit": "s"
          },
          {
            "name": "value_and_grad / IDefOpt / tpu / Primal",
            "value": 0,
            "unit": "s"
          },
          {
            "name": "value_and_grad / JaXPipe / cpu / Primal",
            "value": 0.000027664,
            "unit": "s"
          },
          {
            "name": "value_and_grad / Jax / cpu / Primal",
            "value": 0.000027243,
            "unit": "s"
          },
          {
            "name": "value_and_grad / HLOOpt / cpu / Primal",
            "value": 0.000027408,
            "unit": "s"
          },
          {
            "name": "value_and_grad / PartOpt / cpu / Primal",
            "value": 0.000027667,
            "unit": "s"
          },
          {
            "name": "value_and_grad / IPartOpt / cpu / Primal",
            "value": 0.000028165,
            "unit": "s"
          },
          {
            "name": "value_and_grad / DefOpt / cpu / Primal",
            "value": 0.000027674,
            "unit": "s"
          },
          {
            "name": "value_and_grad / IDefOpt / cpu / Primal",
            "value": 0.000027838,
            "unit": "s"
          },
          {
            "name": "value_and_grad / JaXPipe / cpu / Primal",
            "value": 0.000016,
            "unit": "s"
          },
          {
            "name": "value_and_grad / Jax / cpu / Primal",
            "value": 0.000016,
            "unit": "s"
          },
          {
            "name": "value_and_grad / HLOOpt / cpu / Primal",
            "value": 0.000016,
            "unit": "s"
          },
          {
            "name": "value_and_grad / PartOpt / cpu / Primal",
            "value": 0.000016,
            "unit": "s"
          },
          {
            "name": "value_and_grad / IPartOpt / cpu / Primal",
            "value": 0.000016,
            "unit": "s"
          },
          {
            "name": "value_and_grad / DefOpt / cpu / Primal",
            "value": 0.000016,
            "unit": "s"
          },
          {
            "name": "value_and_grad / IDefOpt / cpu / Primal",
            "value": 0.000016,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / JaXPipe / cuda / Primal",
            "value": 0.001457397,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / Jax / cuda / Primal",
            "value": 0.001451123,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / HLOOpt / cuda / Primal",
            "value": 0.001325206,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / PartOpt / cuda / Primal",
            "value": 0.00132207,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / IPartOpt / cuda / Primal",
            "value": 0.001354484,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / DefOpt / cuda / Primal",
            "value": 0.000915832,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / IDefOpt / cuda / Primal",
            "value": 0.000946264,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / JaXPipe / cuda / Forward",
            "value": 0.001565299,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / Jax / cuda / Forward",
            "value": 0.00177445,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / HLOOpt / cuda / Forward",
            "value": 0.0016276029999999,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / PartOpt / cuda / Forward",
            "value": 0.001644435,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / IPartOpt / cuda / Forward",
            "value": 0.001633073,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / DefOpt / cuda / Forward",
            "value": 0.001641395,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / IDefOpt / cuda / Forward",
            "value": 0.001626067,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / JaXPipe / cuda / PreRev",
            "value": 0.002650986,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / JaXPipe / cuda / PostRev",
            "value": 0.005307668,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / JaXPipe / cuda / BothRev",
            "value": 0.002674538,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / Jax / cuda / BothRev",
            "value": 0.005279286,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / HLOOpt / cuda / PreRev",
            "value": 0.002728745,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / HLOOpt / cuda / PostRev",
            "value": 0.00529202,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / HLOOpt / cuda / BothRev",
            "value": 0.002721225,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / PartOpt / cuda / PreRev",
            "value": 0.002842761,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / PartOpt / cuda / PostRev",
            "value": 0.005400819,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / PartOpt / cuda / BothRev",
            "value": 0.002791464,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / IPartOpt / cuda / PreRev",
            "value": 0.002799721,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / IPartOpt / cuda / PostRev",
            "value": 0.005345107,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / IPartOpt / cuda / BothRev",
            "value": 0.0027779289999999,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / DefOpt / cuda / PreRev",
            "value": 0.0028215439999999,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / DefOpt / cuda / PostRev",
            "value": 0.002743369,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / DefOpt / cuda / BothRev",
            "value": 0.002758602,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / IDefOpt / cuda / PreRev",
            "value": 0.002804522,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / IDefOpt / cuda / PostRev",
            "value": 0.00230478,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / IDefOpt / cuda / BothRev",
            "value": 0.002734473,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / JaXPipe / tpu / Primal",
            "value": 0.009273954375,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / Jax / tpu / Primal",
            "value": 0.00926473125,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / HLOOpt / tpu / Primal",
            "value": 0.009170430625,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / PartOpt / tpu / Primal",
            "value": 0.00919640125,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / IPartOpt / tpu / Primal",
            "value": 0.0092012087499999,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / DefOpt / tpu / Primal",
            "value": 0.0087924475,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / IDefOpt / tpu / Primal",
            "value": 0.00869956375,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / JaXPipe / tpu / Forward",
            "value": 0.0174177849999999,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / Jax / tpu / Forward",
            "value": 0.018727536875,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / HLOOpt / tpu / Forward",
            "value": 0.0173932675,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / PartOpt / tpu / Forward",
            "value": 0.017409841875,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / IPartOpt / tpu / Forward",
            "value": 0.01741184875,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / DefOpt / tpu / Forward",
            "value": 0.017417063125,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / IDefOpt / tpu / Forward",
            "value": 0.017414838125,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / JaXPipe / tpu / PreRev",
            "value": 0.025457535,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / JaXPipe / tpu / PostRev",
            "value": 0.0218942125,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / JaXPipe / tpu / BothRev",
            "value": 0.02547417375,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / Jax / tpu / BothRev",
            "value": 0.021891351875,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / HLOOpt / tpu / PreRev",
            "value": 0.0255871593749999,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / HLOOpt / tpu / PostRev",
            "value": 0.02083006625,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / HLOOpt / tpu / BothRev",
            "value": 0.025685535,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / PartOpt / tpu / PreRev",
            "value": 0.02550769125,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / PartOpt / tpu / PostRev",
            "value": 0.02150850125,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / PartOpt / tpu / BothRev",
            "value": 0.025594076875,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / IPartOpt / tpu / PreRev",
            "value": 0.025477834375,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / IPartOpt / tpu / PostRev",
            "value": 0.021536304375,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / IPartOpt / tpu / BothRev",
            "value": 0.025555453125,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / DefOpt / tpu / PreRev",
            "value": 0.025506061875,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / DefOpt / tpu / PostRev",
            "value": 0.01880455625,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / DefOpt / tpu / BothRev",
            "value": 0.02559681875,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / IDefOpt / tpu / PreRev",
            "value": 0.025477278125,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / IDefOpt / tpu / PostRev",
            "value": 0.01834256875,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / IDefOpt / tpu / BothRev",
            "value": 0.025552339375,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / JaXPipe / cpu / Primal",
            "value": 0.080197468,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / Jax / cpu / Primal",
            "value": 0.083655234,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / HLOOpt / cpu / Primal",
            "value": 0.096154745,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / PartOpt / cpu / Primal",
            "value": 0.083441797,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / IPartOpt / cpu / Primal",
            "value": 0.07348125,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / DefOpt / cpu / Primal",
            "value": 0.1080883209999999,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / IDefOpt / cpu / Primal",
            "value": 0.102355934,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / JaXPipe / cpu / Forward",
            "value": 0.194925377,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / Jax / cpu / Forward",
            "value": 0.102705963,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / HLOOpt / cpu / Forward",
            "value": 0.1943103319999999,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / PartOpt / cpu / Forward",
            "value": 0.192890015,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / IPartOpt / cpu / Forward",
            "value": 0.1889543649999999,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / DefOpt / cpu / Forward",
            "value": 0.187598506,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / IDefOpt / cpu / Forward",
            "value": 0.1914111,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / JaXPipe / cpu / PreRev",
            "value": 0.2570904029999999,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / JaXPipe / cpu / PostRev",
            "value": 0.164598411,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / JaXPipe / cpu / BothRev",
            "value": 0.243595934,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / Jax / cpu / BothRev",
            "value": 0.167637033,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / HLOOpt / cpu / PreRev",
            "value": 0.247185092,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / HLOOpt / cpu / PostRev",
            "value": 0.214137107,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / HLOOpt / cpu / BothRev",
            "value": 0.28978097,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / PartOpt / cpu / PreRev",
            "value": 0.245707196,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / PartOpt / cpu / PostRev",
            "value": 0.148089955,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / PartOpt / cpu / BothRev",
            "value": 0.289875612,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / IPartOpt / cpu / PreRev",
            "value": 0.250292784,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / IPartOpt / cpu / PostRev",
            "value": 0.155665586,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / IPartOpt / cpu / BothRev",
            "value": 0.280760357,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / DefOpt / cpu / PreRev",
            "value": 0.246668072,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / DefOpt / cpu / PostRev",
            "value": 0.208282383,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / DefOpt / cpu / BothRev",
            "value": 0.286270934,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / IDefOpt / cpu / PreRev",
            "value": 0.243850494,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / IDefOpt / cpu / PostRev",
            "value": 0.2124084,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / IDefOpt / cpu / BothRev",
            "value": 0.2716668399999999,
            "unit": "s"
          },
          {
            "name": "v1/deterministic_2_8_deg_inner_steps_24_outer_steps_4 / JaXPipe / cuda / Primal",
            "value": 1.7028338120000002,
            "unit": "s"
          },
          {
            "name": "v1/deterministic_2_8_deg_inner_steps_24_outer_steps_4 / Jax / cuda / Primal",
            "value": 1.704407339,
            "unit": "s"
          },
          {
            "name": "v1/deterministic_2_8_deg_inner_steps_24_outer_steps_4 / HLOOpt / cuda / Primal",
            "value": 1.715157461,
            "unit": "s"
          },
          {
            "name": "v1/deterministic_2_8_deg_inner_steps_24_outer_steps_4 / PartOpt / cuda / Primal",
            "value": 1.695699103,
            "unit": "s"
          },
          {
            "name": "v1/deterministic_2_8_deg_inner_steps_24_outer_steps_4 / IPartOpt / cuda / Primal",
            "value": 1.6934279449999998,
            "unit": "s"
          },
          {
            "name": "v1/deterministic_2_8_deg_inner_steps_24_outer_steps_4 / DefOpt / cuda / Primal",
            "value": 1.664433072,
            "unit": "s"
          },
          {
            "name": "v1/deterministic_2_8_deg_inner_steps_24_outer_steps_4 / IDefOpt / cuda / Primal",
            "value": 1.921661585,
            "unit": "s"
          },
          {
            "name": "v1/deterministic_2_8_deg_inner_steps_24_outer_steps_4 / JaXPipe / tpu / Primal",
            "value": 3.038416416875,
            "unit": "s"
          },
          {
            "name": "v1/deterministic_2_8_deg_inner_steps_24_outer_steps_4 / Jax / tpu / Primal",
            "value": 3.038972848125,
            "unit": "s"
          },
          {
            "name": "v1/deterministic_2_8_deg_inner_steps_24_outer_steps_4 / HLOOpt / tpu / Primal",
            "value": 3.12138719375,
            "unit": "s"
          },
          {
            "name": "v1/deterministic_2_8_deg_inner_steps_24_outer_steps_4 / PartOpt / tpu / Primal",
            "value": 3.059688449375,
            "unit": "s"
          },
          {
            "name": "v1/deterministic_2_8_deg_inner_steps_24_outer_steps_4 / IPartOpt / tpu / Primal",
            "value": 3.05992763375,
            "unit": "s"
          },
          {
            "name": "v1/deterministic_2_8_deg_inner_steps_24_outer_steps_4 / DefOpt / tpu / Primal",
            "value": 2.102344878125,
            "unit": "s"
          },
          {
            "name": "v1/deterministic_2_8_deg_inner_steps_24_outer_steps_4 / IDefOpt / tpu / Primal",
            "value": 2.944463474375,
            "unit": "s"
          },
          {
            "name": "v1/deterministic_2_8_deg_inner_steps_2_outer_steps_2 / JaXPipe / cpu / Primal",
            "value": 6.8007833060000005,
            "unit": "s"
          },
          {
            "name": "v1/deterministic_2_8_deg_inner_steps_2_outer_steps_2 / Jax / cpu / Primal",
            "value": 6.818972752,
            "unit": "s"
          },
          {
            "name": "v1/deterministic_2_8_deg_inner_steps_2_outer_steps_2 / HLOOpt / cpu / Primal",
            "value": 6.647598244,
            "unit": "s"
          },
          {
            "name": "v1/deterministic_2_8_deg_inner_steps_2_outer_steps_2 / PartOpt / cpu / Primal",
            "value": 6.899143466000001,
            "unit": "s"
          },
          {
            "name": "v1/deterministic_2_8_deg_inner_steps_2_outer_steps_2 / IPartOpt / cpu / Primal",
            "value": 6.961332216,
            "unit": "s"
          },
          {
            "name": "v1/deterministic_2_8_deg_inner_steps_2_outer_steps_2 / DefOpt / cpu / Primal",
            "value": 2.786819339,
            "unit": "s"
          },
          {
            "name": "v1/deterministic_2_8_deg_inner_steps_2_outer_steps_2 / IDefOpt / cpu / Primal",
            "value": 7.423087524,
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
          "id": "30eb4e716fb12e657a3cd05d9bc00aa5d686feed",
          "message": "Update EnzymeAD/Enzyme to commit ee83e6901c5032086801e91be9a86f8195883f0d (#1846)\n\nDiff: https://github.com/EnzymeAD/Enzyme/compare/9aa1bec873e957120c5544e033227c47963be4f5...ee83e6901c5032086801e91be9a86f8195883f0d\n\nCo-authored-by: enzymead-bot[bot] <238314553+enzymead-bot[bot]@users.noreply.github.com>",
          "timestamp": "2025-12-25T16:21:16-05:00",
          "tree_id": "d43f818c7e2fdb5936cb4cbdbc6dce023fd71d51",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/30eb4e716fb12e657a3cd05d9bc00aa5d686feed"
        },
        "date": 1766704919473,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "actmtch / JaXPipe / cpu / Primal",
            "value": 0.000007178099976954399,
            "unit": "s"
          },
          {
            "name": "actmtch / Jax / cpu / Primal",
            "value": 0.00000742670001272927,
            "unit": "s"
          },
          {
            "name": "actmtch / HLOOpt / cpu / Primal",
            "value": 0.00000942116002079274,
            "unit": "s"
          },
          {
            "name": "actmtch / PartOpt / cpu / Primal",
            "value": 0.000008231920019170502,
            "unit": "s"
          },
          {
            "name": "actmtch / IPartOpt / cpu / Primal",
            "value": 0.000009108859985644812,
            "unit": "s"
          },
          {
            "name": "actmtch / DefOpt / cpu / Primal",
            "value": 0.000009177840011034278,
            "unit": "s"
          },
          {
            "name": "actmtch / IDefOpt / cpu / Primal",
            "value": 0.000009169459972326876,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / Forward",
            "value": 0.00001240557998244185,
            "unit": "s"
          },
          {
            "name": "actmtch / Jax / cpu / Forward",
            "value": 0.000011727619985322234,
            "unit": "s"
          },
          {
            "name": "actmtch / HLOOpt / cpu / Forward",
            "value": 0.000013135259996488456,
            "unit": "s"
          },
          {
            "name": "actmtch / PartOpt / cpu / Forward",
            "value": 0.000012186899984953924,
            "unit": "s"
          },
          {
            "name": "actmtch / IPartOpt / cpu / Forward",
            "value": 0.00001277817995287478,
            "unit": "s"
          },
          {
            "name": "actmtch / DefOpt / cpu / Forward",
            "value": 0.000012611259999175672,
            "unit": "s"
          },
          {
            "name": "actmtch / IDefOpt / cpu / Forward",
            "value": 0.000013058779995844815,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / PreRev",
            "value": 0.000012500760039984015,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / PostRev",
            "value": 0.000011682659987855004,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / BothRev",
            "value": 0.000013351459965633694,
            "unit": "s"
          },
          {
            "name": "actmtch / Jax / cpu / BothRev",
            "value": 0.0000101748600081919,
            "unit": "s"
          },
          {
            "name": "actmtch / HLOOpt / cpu / PreRev",
            "value": 0.000012809399968318758,
            "unit": "s"
          },
          {
            "name": "actmtch / HLOOpt / cpu / PostRev",
            "value": 0.000014683079989481484,
            "unit": "s"
          },
          {
            "name": "actmtch / HLOOpt / cpu / BothRev",
            "value": 0.000012941140003022156,
            "unit": "s"
          },
          {
            "name": "actmtch / PartOpt / cpu / PreRev",
            "value": 0.00001237502001458779,
            "unit": "s"
          },
          {
            "name": "actmtch / PartOpt / cpu / PostRev",
            "value": 0.000010807039989231271,
            "unit": "s"
          },
          {
            "name": "actmtch / PartOpt / cpu / BothRev",
            "value": 0.000013357679927139543,
            "unit": "s"
          },
          {
            "name": "actmtch / IPartOpt / cpu / PreRev",
            "value": 0.000011974959961662535,
            "unit": "s"
          },
          {
            "name": "actmtch / IPartOpt / cpu / PostRev",
            "value": 0.000011647979999906963,
            "unit": "s"
          },
          {
            "name": "actmtch / IPartOpt / cpu / BothRev",
            "value": 0.000012888160035799956,
            "unit": "s"
          },
          {
            "name": "actmtch / DefOpt / cpu / PreRev",
            "value": 0.000012172299993835623,
            "unit": "s"
          },
          {
            "name": "actmtch / DefOpt / cpu / PostRev",
            "value": 0.0000126511599864898,
            "unit": "s"
          },
          {
            "name": "actmtch / DefOpt / cpu / BothRev",
            "value": 0.000012905660032629384,
            "unit": "s"
          },
          {
            "name": "actmtch / IDefOpt / cpu / PreRev",
            "value": 0.00001217844000166224,
            "unit": "s"
          },
          {
            "name": "actmtch / IDefOpt / cpu / PostRev",
            "value": 0.000012553719934658148,
            "unit": "s"
          },
          {
            "name": "actmtch / IDefOpt / cpu / BothRev",
            "value": 0.000012284119984542483,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cuda / Primal",
            "value": 0.000002015,
            "unit": "s"
          },
          {
            "name": "actmtch / Jax / cuda / Primal",
            "value": 0.000002015,
            "unit": "s"
          },
          {
            "name": "actmtch / HLOOpt / cuda / Primal",
            "value": 0.000002015,
            "unit": "s"
          },
          {
            "name": "actmtch / PartOpt / cuda / Primal",
            "value": 0.000002015,
            "unit": "s"
          },
          {
            "name": "actmtch / IPartOpt / cuda / Primal",
            "value": 0.000002015,
            "unit": "s"
          },
          {
            "name": "actmtch / DefOpt / cuda / Primal",
            "value": 0.000002015,
            "unit": "s"
          },
          {
            "name": "actmtch / IDefOpt / cuda / Primal",
            "value": 0.000002016,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cuda / Forward",
            "value": 0.000009536,
            "unit": "s"
          },
          {
            "name": "actmtch / Jax / cuda / Forward",
            "value": 0.000009504,
            "unit": "s"
          },
          {
            "name": "actmtch / HLOOpt / cuda / Forward",
            "value": 0.000010016,
            "unit": "s"
          },
          {
            "name": "actmtch / PartOpt / cuda / Forward",
            "value": 0.000009664,
            "unit": "s"
          },
          {
            "name": "actmtch / IPartOpt / cuda / Forward",
            "value": 0.000009568,
            "unit": "s"
          },
          {
            "name": "actmtch / DefOpt / cuda / Forward",
            "value": 0.000009504,
            "unit": "s"
          },
          {
            "name": "actmtch / IDefOpt / cuda / Forward",
            "value": 0.000010016,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cuda / PreRev",
            "value": 0.00000992,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cuda / PostRev",
            "value": 0.000009696,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cuda / BothRev",
            "value": 0.000009984,
            "unit": "s"
          },
          {
            "name": "actmtch / Jax / cuda / BothRev",
            "value": 0.000009632,
            "unit": "s"
          },
          {
            "name": "actmtch / HLOOpt / cuda / PreRev",
            "value": 0.000010688,
            "unit": "s"
          },
          {
            "name": "actmtch / HLOOpt / cuda / PostRev",
            "value": 0.000010528,
            "unit": "s"
          },
          {
            "name": "actmtch / HLOOpt / cuda / BothRev",
            "value": 0.000009727,
            "unit": "s"
          },
          {
            "name": "actmtch / PartOpt / cuda / PreRev",
            "value": 0.000010112,
            "unit": "s"
          },
          {
            "name": "actmtch / PartOpt / cuda / PostRev",
            "value": 0.000009952,
            "unit": "s"
          },
          {
            "name": "actmtch / PartOpt / cuda / BothRev",
            "value": 0.00001024,
            "unit": "s"
          },
          {
            "name": "actmtch / IPartOpt / cuda / PreRev",
            "value": 0.000010112,
            "unit": "s"
          },
          {
            "name": "actmtch / IPartOpt / cuda / PostRev",
            "value": 0.000009727,
            "unit": "s"
          },
          {
            "name": "actmtch / IPartOpt / cuda / BothRev",
            "value": 0.0000096,
            "unit": "s"
          },
          {
            "name": "actmtch / DefOpt / cuda / PreRev",
            "value": 0.000010049,
            "unit": "s"
          },
          {
            "name": "actmtch / DefOpt / cuda / PostRev",
            "value": 0.000009696,
            "unit": "s"
          },
          {
            "name": "actmtch / DefOpt / cuda / BothRev",
            "value": 0.00001008,
            "unit": "s"
          },
          {
            "name": "actmtch / IDefOpt / cuda / PreRev",
            "value": 0.000010176,
            "unit": "s"
          },
          {
            "name": "actmtch / IDefOpt / cuda / PostRev",
            "value": 0.00001008,
            "unit": "s"
          },
          {
            "name": "actmtch / IDefOpt / cuda / BothRev",
            "value": 0.000010047,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / tpu / Primal",
            "value": 5.63475e-7,
            "unit": "s"
          },
          {
            "name": "actmtch / Jax / tpu / Primal",
            "value": 5.965e-7,
            "unit": "s"
          },
          {
            "name": "actmtch / HLOOpt / tpu / Primal",
            "value": 0.000002103725,
            "unit": "s"
          },
          {
            "name": "actmtch / PartOpt / tpu / Primal",
            "value": 5.968500000000001e-7,
            "unit": "s"
          },
          {
            "name": "actmtch / IPartOpt / tpu / Primal",
            "value": 5.5225e-7,
            "unit": "s"
          },
          {
            "name": "actmtch / DefOpt / tpu / Primal",
            "value": 0.000002154625,
            "unit": "s"
          },
          {
            "name": "actmtch / IDefOpt / tpu / Primal",
            "value": 0.0000021011750000000003,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / tpu / Forward",
            "value": 0.000003833475000000001,
            "unit": "s"
          },
          {
            "name": "actmtch / Jax / tpu / Forward",
            "value": 0.000001207225,
            "unit": "s"
          },
          {
            "name": "actmtch / HLOOpt / tpu / Forward",
            "value": 0.000003939525,
            "unit": "s"
          },
          {
            "name": "actmtch / PartOpt / tpu / Forward",
            "value": 0.0000039168,
            "unit": "s"
          },
          {
            "name": "actmtch / IPartOpt / tpu / Forward",
            "value": 0.000003937025000000001,
            "unit": "s"
          },
          {
            "name": "actmtch / DefOpt / tpu / Forward",
            "value": 0.000003927975,
            "unit": "s"
          },
          {
            "name": "actmtch / IDefOpt / tpu / Forward",
            "value": 0.000003936375,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / tpu / PreRev",
            "value": 0.000003475,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / tpu / PostRev",
            "value": 0.00000163465,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / tpu / BothRev",
            "value": 0.00000349725,
            "unit": "s"
          },
          {
            "name": "actmtch / Jax / tpu / BothRev",
            "value": 0.0000016403750000000002,
            "unit": "s"
          },
          {
            "name": "actmtch / HLOOpt / tpu / PreRev",
            "value": 0.000003481025,
            "unit": "s"
          },
          {
            "name": "actmtch / HLOOpt / tpu / PostRev",
            "value": 0.00000342085,
            "unit": "s"
          },
          {
            "name": "actmtch / HLOOpt / tpu / BothRev",
            "value": 0.000003482925,
            "unit": "s"
          },
          {
            "name": "actmtch / PartOpt / tpu / PreRev",
            "value": 0.0000034052,
            "unit": "s"
          },
          {
            "name": "actmtch / PartOpt / tpu / PostRev",
            "value": 0.0000016022,
            "unit": "s"
          },
          {
            "name": "actmtch / PartOpt / tpu / BothRev",
            "value": 0.0000034209750000000003,
            "unit": "s"
          },
          {
            "name": "actmtch / IPartOpt / tpu / PreRev",
            "value": 0.0000034736,
            "unit": "s"
          },
          {
            "name": "actmtch / IPartOpt / tpu / PostRev",
            "value": 0.000001655825,
            "unit": "s"
          },
          {
            "name": "actmtch / IPartOpt / tpu / BothRev",
            "value": 0.0000034814,
            "unit": "s"
          },
          {
            "name": "actmtch / DefOpt / tpu / PreRev",
            "value": 0.00000340395,
            "unit": "s"
          },
          {
            "name": "actmtch / DefOpt / tpu / PostRev",
            "value": 0.000003402575,
            "unit": "s"
          },
          {
            "name": "actmtch / DefOpt / tpu / BothRev",
            "value": 0.0000034213,
            "unit": "s"
          },
          {
            "name": "actmtch / IDefOpt / tpu / PreRev",
            "value": 0.00000348275,
            "unit": "s"
          },
          {
            "name": "actmtch / IDefOpt / tpu / PostRev",
            "value": 0.0000034287000000000003,
            "unit": "s"
          },
          {
            "name": "actmtch / IDefOpt / tpu / BothRev",
            "value": 0.0000034788,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / Primal",
            "value": 0.000013103,
            "unit": "s"
          },
          {
            "name": "actmtch / Jax / cpu / Primal",
            "value": 0.000013236,
            "unit": "s"
          },
          {
            "name": "actmtch / HLOOpt / cpu / Primal",
            "value": 0.000014039,
            "unit": "s"
          },
          {
            "name": "actmtch / PartOpt / cpu / Primal",
            "value": 0.000013145,
            "unit": "s"
          },
          {
            "name": "actmtch / IPartOpt / cpu / Primal",
            "value": 0.000013216,
            "unit": "s"
          },
          {
            "name": "actmtch / DefOpt / cpu / Primal",
            "value": 0.000014225,
            "unit": "s"
          },
          {
            "name": "actmtch / IDefOpt / cpu / Primal",
            "value": 0.000013908,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / Forward",
            "value": 0.000019751,
            "unit": "s"
          },
          {
            "name": "actmtch / Jax / cpu / Forward",
            "value": 0.000018275,
            "unit": "s"
          },
          {
            "name": "actmtch / HLOOpt / cpu / Forward",
            "value": 0.000020098,
            "unit": "s"
          },
          {
            "name": "actmtch / PartOpt / cpu / Forward",
            "value": 0.000019389,
            "unit": "s"
          },
          {
            "name": "actmtch / IPartOpt / cpu / Forward",
            "value": 0.000019552,
            "unit": "s"
          },
          {
            "name": "actmtch / DefOpt / cpu / Forward",
            "value": 0.000019369,
            "unit": "s"
          },
          {
            "name": "actmtch / IDefOpt / cpu / Forward",
            "value": 0.000019336,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / PreRev",
            "value": 0.000020174,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / PostRev",
            "value": 0.00001763,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / BothRev",
            "value": 0.000019452,
            "unit": "s"
          },
          {
            "name": "actmtch / Jax / cpu / BothRev",
            "value": 0.000017676999999999997,
            "unit": "s"
          },
          {
            "name": "actmtch / HLOOpt / cpu / PreRev",
            "value": 0.000020278,
            "unit": "s"
          },
          {
            "name": "actmtch / HLOOpt / cpu / PostRev",
            "value": 0.000019829,
            "unit": "s"
          },
          {
            "name": "actmtch / HLOOpt / cpu / BothRev",
            "value": 0.000019138,
            "unit": "s"
          },
          {
            "name": "actmtch / PartOpt / cpu / PreRev",
            "value": 0.000019489,
            "unit": "s"
          },
          {
            "name": "actmtch / PartOpt / cpu / PostRev",
            "value": 0.000017760999999999998,
            "unit": "s"
          },
          {
            "name": "actmtch / PartOpt / cpu / BothRev",
            "value": 0.00001934,
            "unit": "s"
          },
          {
            "name": "actmtch / IPartOpt / cpu / PreRev",
            "value": 0.000019098,
            "unit": "s"
          },
          {
            "name": "actmtch / IPartOpt / cpu / PostRev",
            "value": 0.000017818,
            "unit": "s"
          },
          {
            "name": "actmtch / IPartOpt / cpu / BothRev",
            "value": 0.000019289,
            "unit": "s"
          },
          {
            "name": "actmtch / DefOpt / cpu / PreRev",
            "value": 0.000018905000000000003,
            "unit": "s"
          },
          {
            "name": "actmtch / DefOpt / cpu / PostRev",
            "value": 0.000019313,
            "unit": "s"
          },
          {
            "name": "actmtch / DefOpt / cpu / BothRev",
            "value": 0.000018973,
            "unit": "s"
          },
          {
            "name": "actmtch / IDefOpt / cpu / PreRev",
            "value": 0.000019312,
            "unit": "s"
          },
          {
            "name": "actmtch / IDefOpt / cpu / PostRev",
            "value": 0.000019426,
            "unit": "s"
          },
          {
            "name": "actmtch / IDefOpt / cpu / BothRev",
            "value": 0.000019297,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / Primal",
            "value": 0.000008999999999999999,
            "unit": "s"
          },
          {
            "name": "actmtch / Jax / cpu / Primal",
            "value": 0.000008999999999999999,
            "unit": "s"
          },
          {
            "name": "actmtch / HLOOpt / cpu / Primal",
            "value": 0.00001,
            "unit": "s"
          },
          {
            "name": "actmtch / PartOpt / cpu / Primal",
            "value": 0.000008999999999999999,
            "unit": "s"
          },
          {
            "name": "actmtch / IPartOpt / cpu / Primal",
            "value": 0.00001,
            "unit": "s"
          },
          {
            "name": "actmtch / DefOpt / cpu / Primal",
            "value": 0.00001,
            "unit": "s"
          },
          {
            "name": "actmtch / IDefOpt / cpu / Primal",
            "value": 0.00001,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / Forward",
            "value": 0.000015,
            "unit": "s"
          },
          {
            "name": "actmtch / Jax / cpu / Forward",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "actmtch / HLOOpt / cpu / Forward",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "actmtch / PartOpt / cpu / Forward",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "actmtch / IPartOpt / cpu / Forward",
            "value": 0.000015,
            "unit": "s"
          },
          {
            "name": "actmtch / DefOpt / cpu / Forward",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "actmtch / IDefOpt / cpu / Forward",
            "value": 0.000015,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / PreRev",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / PostRev",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / BothRev",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "actmtch / Jax / cpu / BothRev",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "actmtch / HLOOpt / cpu / PreRev",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "actmtch / HLOOpt / cpu / PostRev",
            "value": 0.000016,
            "unit": "s"
          },
          {
            "name": "actmtch / HLOOpt / cpu / BothRev",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "actmtch / PartOpt / cpu / PreRev",
            "value": 0.000015,
            "unit": "s"
          },
          {
            "name": "actmtch / PartOpt / cpu / PostRev",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "actmtch / PartOpt / cpu / BothRev",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "actmtch / IPartOpt / cpu / PreRev",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "actmtch / IPartOpt / cpu / PostRev",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "actmtch / IPartOpt / cpu / BothRev",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "actmtch / DefOpt / cpu / PreRev",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "actmtch / DefOpt / cpu / PostRev",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "actmtch / DefOpt / cpu / BothRev",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "actmtch / IDefOpt / cpu / PreRev",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "actmtch / IDefOpt / cpu / PostRev",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "actmtch / IDefOpt / cpu / BothRev",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / Primal",
            "value": 0.000007177039997259271,
            "unit": "s"
          },
          {
            "name": "add_one / Jax / cpu / Primal",
            "value": 0.000007734440014246502,
            "unit": "s"
          },
          {
            "name": "add_one / HLOOpt / cpu / Primal",
            "value": 0.00000762009998652502,
            "unit": "s"
          },
          {
            "name": "add_one / PartOpt / cpu / Primal",
            "value": 0.000007244479984365171,
            "unit": "s"
          },
          {
            "name": "add_one / IPartOpt / cpu / Primal",
            "value": 0.000007930979982120335,
            "unit": "s"
          },
          {
            "name": "add_one / DefOpt / cpu / Primal",
            "value": 0.000007669920023545274,
            "unit": "s"
          },
          {
            "name": "add_one / IDefOpt / cpu / Primal",
            "value": 0.000007121160024325945,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / Forward",
            "value": 0.00001077120000445575,
            "unit": "s"
          },
          {
            "name": "add_one / Jax / cpu / Forward",
            "value": 0.000011104940058430656,
            "unit": "s"
          },
          {
            "name": "add_one / HLOOpt / cpu / Forward",
            "value": 0.000010922199990091033,
            "unit": "s"
          },
          {
            "name": "add_one / PartOpt / cpu / Forward",
            "value": 0.000010968379992846168,
            "unit": "s"
          },
          {
            "name": "add_one / IPartOpt / cpu / Forward",
            "value": 0.000011612179969233694,
            "unit": "s"
          },
          {
            "name": "add_one / DefOpt / cpu / Forward",
            "value": 0.000011284179972790298,
            "unit": "s"
          },
          {
            "name": "add_one / IDefOpt / cpu / Forward",
            "value": 0.000010938719979094458,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / PreRev",
            "value": 0.000012953139976161765,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / PostRev",
            "value": 0.000013293079982759082,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / BothRev",
            "value": 0.000014024859983692296,
            "unit": "s"
          },
          {
            "name": "add_one / Jax / cpu / BothRev",
            "value": 0.000012893459916085705,
            "unit": "s"
          },
          {
            "name": "add_one / HLOOpt / cpu / PreRev",
            "value": 0.00001295961996220285,
            "unit": "s"
          },
          {
            "name": "add_one / HLOOpt / cpu / PostRev",
            "value": 0.000015428020005856525,
            "unit": "s"
          },
          {
            "name": "add_one / HLOOpt / cpu / BothRev",
            "value": 0.00001350904002720199,
            "unit": "s"
          },
          {
            "name": "add_one / PartOpt / cpu / PreRev",
            "value": 0.00001265406001039082,
            "unit": "s"
          },
          {
            "name": "add_one / PartOpt / cpu / PostRev",
            "value": 0.000013283080015753512,
            "unit": "s"
          },
          {
            "name": "add_one / PartOpt / cpu / BothRev",
            "value": 0.00001378497997393424,
            "unit": "s"
          },
          {
            "name": "add_one / IPartOpt / cpu / PreRev",
            "value": 0.000013247099986983811,
            "unit": "s"
          },
          {
            "name": "add_one / IPartOpt / cpu / PostRev",
            "value": 0.00001345441994999419,
            "unit": "s"
          },
          {
            "name": "add_one / IPartOpt / cpu / BothRev",
            "value": 0.000013223599971752264,
            "unit": "s"
          },
          {
            "name": "add_one / DefOpt / cpu / PreRev",
            "value": 0.000012625260042113953,
            "unit": "s"
          },
          {
            "name": "add_one / DefOpt / cpu / PostRev",
            "value": 0.000013961339964225773,
            "unit": "s"
          },
          {
            "name": "add_one / DefOpt / cpu / BothRev",
            "value": 0.000013434940001388895,
            "unit": "s"
          },
          {
            "name": "add_one / IDefOpt / cpu / PreRev",
            "value": 0.00001282082000216178,
            "unit": "s"
          },
          {
            "name": "add_one / IDefOpt / cpu / PostRev",
            "value": 0.000013155759997971472,
            "unit": "s"
          },
          {
            "name": "add_one / IDefOpt / cpu / BothRev",
            "value": 0.00001394135998452839,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cuda / Primal",
            "value": 0.000001919,
            "unit": "s"
          },
          {
            "name": "add_one / Jax / cuda / Primal",
            "value": 0.000001919,
            "unit": "s"
          },
          {
            "name": "add_one / HLOOpt / cuda / Primal",
            "value": 0.000001919,
            "unit": "s"
          },
          {
            "name": "add_one / PartOpt / cuda / Primal",
            "value": 0.0000019200000000000003,
            "unit": "s"
          },
          {
            "name": "add_one / IPartOpt / cuda / Primal",
            "value": 0.000001919,
            "unit": "s"
          },
          {
            "name": "add_one / DefOpt / cuda / Primal",
            "value": 0.000001919,
            "unit": "s"
          },
          {
            "name": "add_one / IDefOpt / cuda / Primal",
            "value": 0.0000019200000000000003,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cuda / Forward",
            "value": 0.000009408,
            "unit": "s"
          },
          {
            "name": "add_one / Jax / cuda / Forward",
            "value": 0.000014623,
            "unit": "s"
          },
          {
            "name": "add_one / HLOOpt / cuda / Forward",
            "value": 0.00000976,
            "unit": "s"
          },
          {
            "name": "add_one / PartOpt / cuda / Forward",
            "value": 0.000010144,
            "unit": "s"
          },
          {
            "name": "add_one / IPartOpt / cuda / Forward",
            "value": 0.000010176,
            "unit": "s"
          },
          {
            "name": "add_one / DefOpt / cuda / Forward",
            "value": 0.000009696,
            "unit": "s"
          },
          {
            "name": "add_one / IDefOpt / cuda / Forward",
            "value": 0.000010144,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cuda / PreRev",
            "value": 0.000026847,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cuda / PostRev",
            "value": 0.00002496,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cuda / BothRev",
            "value": 0.000024639,
            "unit": "s"
          },
          {
            "name": "add_one / Jax / cuda / BothRev",
            "value": 0.000024895,
            "unit": "s"
          },
          {
            "name": "add_one / HLOOpt / cuda / PreRev",
            "value": 0.000038112,
            "unit": "s"
          },
          {
            "name": "add_one / HLOOpt / cuda / PostRev",
            "value": 0.000028256,
            "unit": "s"
          },
          {
            "name": "add_one / HLOOpt / cuda / BothRev",
            "value": 0.00002816,
            "unit": "s"
          },
          {
            "name": "add_one / PartOpt / cuda / PreRev",
            "value": 0.000024576,
            "unit": "s"
          },
          {
            "name": "add_one / PartOpt / cuda / PostRev",
            "value": 0.00002448,
            "unit": "s"
          },
          {
            "name": "add_one / PartOpt / cuda / BothRev",
            "value": 0.000024768,
            "unit": "s"
          },
          {
            "name": "add_one / IPartOpt / cuda / PreRev",
            "value": 0.000024896,
            "unit": "s"
          },
          {
            "name": "add_one / IPartOpt / cuda / PostRev",
            "value": 0.000025024,
            "unit": "s"
          },
          {
            "name": "add_one / IPartOpt / cuda / BothRev",
            "value": 0.000025024,
            "unit": "s"
          },
          {
            "name": "add_one / DefOpt / cuda / PreRev",
            "value": 0.000024991,
            "unit": "s"
          },
          {
            "name": "add_one / DefOpt / cuda / PostRev",
            "value": 0.000024448,
            "unit": "s"
          },
          {
            "name": "add_one / DefOpt / cuda / BothRev",
            "value": 0.000024576,
            "unit": "s"
          },
          {
            "name": "add_one / IDefOpt / cuda / PreRev",
            "value": 0.000025055,
            "unit": "s"
          },
          {
            "name": "add_one / IDefOpt / cuda / PostRev",
            "value": 0.000024735,
            "unit": "s"
          },
          {
            "name": "add_one / IDefOpt / cuda / BothRev",
            "value": 0.000032608,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / tpu / Primal",
            "value": 0.0000014201,
            "unit": "s"
          },
          {
            "name": "add_one / Jax / tpu / Primal",
            "value": 0.0000014049,
            "unit": "s"
          },
          {
            "name": "add_one / HLOOpt / tpu / Primal",
            "value": 0.000001435675,
            "unit": "s"
          },
          {
            "name": "add_one / PartOpt / tpu / Primal",
            "value": 0.0000014039999999999998,
            "unit": "s"
          },
          {
            "name": "add_one / IPartOpt / tpu / Primal",
            "value": 0.0000014377750000000002,
            "unit": "s"
          },
          {
            "name": "add_one / DefOpt / tpu / Primal",
            "value": 0.0000014016500000000002,
            "unit": "s"
          },
          {
            "name": "add_one / IDefOpt / tpu / Primal",
            "value": 0.00000144195,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / tpu / Forward",
            "value": 0.000001850925,
            "unit": "s"
          },
          {
            "name": "add_one / Jax / tpu / Forward",
            "value": 0.000001836275,
            "unit": "s"
          },
          {
            "name": "add_one / HLOOpt / tpu / Forward",
            "value": 0.0000018559,
            "unit": "s"
          },
          {
            "name": "add_one / PartOpt / tpu / Forward",
            "value": 0.00000184635,
            "unit": "s"
          },
          {
            "name": "add_one / IPartOpt / tpu / Forward",
            "value": 0.000001853875,
            "unit": "s"
          },
          {
            "name": "add_one / DefOpt / tpu / Forward",
            "value": 0.000001841125,
            "unit": "s"
          },
          {
            "name": "add_one / IDefOpt / tpu / Forward",
            "value": 0.0000018604,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / tpu / PreRev",
            "value": 0.0000022337250000000004,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / tpu / PostRev",
            "value": 0.000002251875,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / tpu / BothRev",
            "value": 0.000002245775,
            "unit": "s"
          },
          {
            "name": "add_one / Jax / tpu / BothRev",
            "value": 0.000002232875,
            "unit": "s"
          },
          {
            "name": "add_one / HLOOpt / tpu / PreRev",
            "value": 0.0000022363,
            "unit": "s"
          },
          {
            "name": "add_one / HLOOpt / tpu / PostRev",
            "value": 0.00000224635,
            "unit": "s"
          },
          {
            "name": "add_one / HLOOpt / tpu / BothRev",
            "value": 0.00000224005,
            "unit": "s"
          },
          {
            "name": "add_one / PartOpt / tpu / PreRev",
            "value": 0.000002236475,
            "unit": "s"
          },
          {
            "name": "add_one / PartOpt / tpu / PostRev",
            "value": 0.000002253325,
            "unit": "s"
          },
          {
            "name": "add_one / PartOpt / tpu / BothRev",
            "value": 0.0000022447,
            "unit": "s"
          },
          {
            "name": "add_one / IPartOpt / tpu / PreRev",
            "value": 0.0000022506750000000003,
            "unit": "s"
          },
          {
            "name": "add_one / IPartOpt / tpu / PostRev",
            "value": 0.00000225235,
            "unit": "s"
          },
          {
            "name": "add_one / IPartOpt / tpu / BothRev",
            "value": 0.0000022373,
            "unit": "s"
          },
          {
            "name": "add_one / DefOpt / tpu / PreRev",
            "value": 0.000002243875,
            "unit": "s"
          },
          {
            "name": "add_one / DefOpt / tpu / PostRev",
            "value": 0.000002236525,
            "unit": "s"
          },
          {
            "name": "add_one / DefOpt / tpu / BothRev",
            "value": 0.000002253425,
            "unit": "s"
          },
          {
            "name": "add_one / IDefOpt / tpu / PreRev",
            "value": 0.0000022355,
            "unit": "s"
          },
          {
            "name": "add_one / IDefOpt / tpu / PostRev",
            "value": 0.00000224675,
            "unit": "s"
          },
          {
            "name": "add_one / IDefOpt / tpu / BothRev",
            "value": 0.00000224285,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / Primal",
            "value": 0.000012864,
            "unit": "s"
          },
          {
            "name": "add_one / Jax / cpu / Primal",
            "value": 0.000013288,
            "unit": "s"
          },
          {
            "name": "add_one / HLOOpt / cpu / Primal",
            "value": 0.000013049,
            "unit": "s"
          },
          {
            "name": "add_one / PartOpt / cpu / Primal",
            "value": 0.000012775,
            "unit": "s"
          },
          {
            "name": "add_one / IPartOpt / cpu / Primal",
            "value": 0.0000127,
            "unit": "s"
          },
          {
            "name": "add_one / DefOpt / cpu / Primal",
            "value": 0.000012703,
            "unit": "s"
          },
          {
            "name": "add_one / IDefOpt / cpu / Primal",
            "value": 0.000012665,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / Forward",
            "value": 0.000017271,
            "unit": "s"
          },
          {
            "name": "add_one / Jax / cpu / Forward",
            "value": 0.000017496,
            "unit": "s"
          },
          {
            "name": "add_one / HLOOpt / cpu / Forward",
            "value": 0.000017581,
            "unit": "s"
          },
          {
            "name": "add_one / PartOpt / cpu / Forward",
            "value": 0.000017392000000000002,
            "unit": "s"
          },
          {
            "name": "add_one / IPartOpt / cpu / Forward",
            "value": 0.000017373,
            "unit": "s"
          },
          {
            "name": "add_one / DefOpt / cpu / Forward",
            "value": 0.000017515,
            "unit": "s"
          },
          {
            "name": "add_one / IDefOpt / cpu / Forward",
            "value": 0.000017218,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / PreRev",
            "value": 0.000019649,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / PostRev",
            "value": 0.000019505,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / BothRev",
            "value": 0.000019546,
            "unit": "s"
          },
          {
            "name": "add_one / Jax / cpu / BothRev",
            "value": 0.000019668,
            "unit": "s"
          },
          {
            "name": "add_one / HLOOpt / cpu / PreRev",
            "value": 0.0000195,
            "unit": "s"
          },
          {
            "name": "add_one / HLOOpt / cpu / PostRev",
            "value": 0.000019822,
            "unit": "s"
          },
          {
            "name": "add_one / HLOOpt / cpu / BothRev",
            "value": 0.000019745,
            "unit": "s"
          },
          {
            "name": "add_one / PartOpt / cpu / PreRev",
            "value": 0.000019759,
            "unit": "s"
          },
          {
            "name": "add_one / PartOpt / cpu / PostRev",
            "value": 0.000020369,
            "unit": "s"
          },
          {
            "name": "add_one / PartOpt / cpu / BothRev",
            "value": 0.000020007,
            "unit": "s"
          },
          {
            "name": "add_one / IPartOpt / cpu / PreRev",
            "value": 0.000019648,
            "unit": "s"
          },
          {
            "name": "add_one / IPartOpt / cpu / PostRev",
            "value": 0.000019504,
            "unit": "s"
          },
          {
            "name": "add_one / IPartOpt / cpu / BothRev",
            "value": 0.000019657,
            "unit": "s"
          },
          {
            "name": "add_one / DefOpt / cpu / PreRev",
            "value": 0.000019613,
            "unit": "s"
          },
          {
            "name": "add_one / DefOpt / cpu / PostRev",
            "value": 0.000019378,
            "unit": "s"
          },
          {
            "name": "add_one / DefOpt / cpu / BothRev",
            "value": 0.000019937,
            "unit": "s"
          },
          {
            "name": "add_one / IDefOpt / cpu / PreRev",
            "value": 0.000020077,
            "unit": "s"
          },
          {
            "name": "add_one / IDefOpt / cpu / PostRev",
            "value": 0.000019725,
            "unit": "s"
          },
          {
            "name": "add_one / IDefOpt / cpu / BothRev",
            "value": 0.000019783,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / Primal",
            "value": 0.00001,
            "unit": "s"
          },
          {
            "name": "add_one / Jax / cpu / Primal",
            "value": 0.000008999999999999999,
            "unit": "s"
          },
          {
            "name": "add_one / HLOOpt / cpu / Primal",
            "value": 0.000008999999999999999,
            "unit": "s"
          },
          {
            "name": "add_one / PartOpt / cpu / Primal",
            "value": 0.000008999999999999999,
            "unit": "s"
          },
          {
            "name": "add_one / IPartOpt / cpu / Primal",
            "value": 0.000008999999999999999,
            "unit": "s"
          },
          {
            "name": "add_one / DefOpt / cpu / Primal",
            "value": 0.000008,
            "unit": "s"
          },
          {
            "name": "add_one / IDefOpt / cpu / Primal",
            "value": 0.000008999999999999999,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / Forward",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "add_one / Jax / cpu / Forward",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "add_one / HLOOpt / cpu / Forward",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "add_one / PartOpt / cpu / Forward",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "add_one / IPartOpt / cpu / Forward",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "add_one / DefOpt / cpu / Forward",
            "value": 0.000011,
            "unit": "s"
          },
          {
            "name": "add_one / IDefOpt / cpu / Forward",
            "value": 0.000011,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / PreRev",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / PostRev",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / BothRev",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "add_one / Jax / cpu / BothRev",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "add_one / HLOOpt / cpu / PreRev",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "add_one / HLOOpt / cpu / PostRev",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "add_one / HLOOpt / cpu / BothRev",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "add_one / PartOpt / cpu / PreRev",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "add_one / PartOpt / cpu / PostRev",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "add_one / PartOpt / cpu / BothRev",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "add_one / IPartOpt / cpu / PreRev",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "add_one / IPartOpt / cpu / PostRev",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "add_one / IPartOpt / cpu / BothRev",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "add_one / DefOpt / cpu / PreRev",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "add_one / DefOpt / cpu / PostRev",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "add_one / DefOpt / cpu / BothRev",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "add_one / IDefOpt / cpu / PreRev",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "add_one / IDefOpt / cpu / PostRev",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "add_one / IDefOpt / cpu / BothRev",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / Primal",
            "value": 0.000007506659994760412,
            "unit": "s"
          },
          {
            "name": "add_two / Jax / cpu / Primal",
            "value": 0.000007353320033871569,
            "unit": "s"
          },
          {
            "name": "add_two / HLOOpt / cpu / Primal",
            "value": 0.000007573599987154012,
            "unit": "s"
          },
          {
            "name": "add_two / PartOpt / cpu / Primal",
            "value": 0.000007316919982258696,
            "unit": "s"
          },
          {
            "name": "add_two / IPartOpt / cpu / Primal",
            "value": 0.000007347219980147201,
            "unit": "s"
          },
          {
            "name": "add_two / DefOpt / cpu / Primal",
            "value": 0.0000074251600290153874,
            "unit": "s"
          },
          {
            "name": "add_two / IDefOpt / cpu / Primal",
            "value": 0.000006994880013735383,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / Forward",
            "value": 0.000010793679984999472,
            "unit": "s"
          },
          {
            "name": "add_two / Jax / cpu / Forward",
            "value": 0.000011223360015719663,
            "unit": "s"
          },
          {
            "name": "add_two / HLOOpt / cpu / Forward",
            "value": 0.00001163689995337336,
            "unit": "s"
          },
          {
            "name": "add_two / PartOpt / cpu / Forward",
            "value": 0.000011552759970072657,
            "unit": "s"
          },
          {
            "name": "add_two / IPartOpt / cpu / Forward",
            "value": 0.000011149400006615904,
            "unit": "s"
          },
          {
            "name": "add_two / DefOpt / cpu / Forward",
            "value": 0.000010866600005101646,
            "unit": "s"
          },
          {
            "name": "add_two / IDefOpt / cpu / Forward",
            "value": 0.00001085120003153861,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / PreRev",
            "value": 0.00001607411998520547,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / PostRev",
            "value": 0.000015767160030009108,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / BothRev",
            "value": 0.00001591828001437534,
            "unit": "s"
          },
          {
            "name": "add_two / Jax / cpu / BothRev",
            "value": 0.000016008080037863693,
            "unit": "s"
          },
          {
            "name": "add_two / HLOOpt / cpu / PreRev",
            "value": 0.000016023040034269796,
            "unit": "s"
          },
          {
            "name": "add_two / HLOOpt / cpu / PostRev",
            "value": 0.00001724719999401714,
            "unit": "s"
          },
          {
            "name": "add_two / HLOOpt / cpu / BothRev",
            "value": 0.000015348320039265674,
            "unit": "s"
          },
          {
            "name": "add_two / PartOpt / cpu / PreRev",
            "value": 0.000016130079948197818,
            "unit": "s"
          },
          {
            "name": "add_two / PartOpt / cpu / PostRev",
            "value": 0.00001545496003018343,
            "unit": "s"
          },
          {
            "name": "add_two / PartOpt / cpu / BothRev",
            "value": 0.000015658540050935698,
            "unit": "s"
          },
          {
            "name": "add_two / IPartOpt / cpu / PreRev",
            "value": 0.000015700720014137913,
            "unit": "s"
          },
          {
            "name": "add_two / IPartOpt / cpu / PostRev",
            "value": 0.000015940499997668668,
            "unit": "s"
          },
          {
            "name": "add_two / IPartOpt / cpu / BothRev",
            "value": 0.000016176559984160122,
            "unit": "s"
          },
          {
            "name": "add_two / DefOpt / cpu / PreRev",
            "value": 0.00001615454000784666,
            "unit": "s"
          },
          {
            "name": "add_two / DefOpt / cpu / PostRev",
            "value": 0.000016489659992657833,
            "unit": "s"
          },
          {
            "name": "add_two / DefOpt / cpu / BothRev",
            "value": 0.00001642914002331963,
            "unit": "s"
          },
          {
            "name": "add_two / IDefOpt / cpu / PreRev",
            "value": 0.00001574142003846646,
            "unit": "s"
          },
          {
            "name": "add_two / IDefOpt / cpu / PostRev",
            "value": 0.000016233939986705083,
            "unit": "s"
          },
          {
            "name": "add_two / IDefOpt / cpu / BothRev",
            "value": 0.000016458300024169148,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cuda / Primal",
            "value": 0.000001887,
            "unit": "s"
          },
          {
            "name": "add_two / Jax / cuda / Primal",
            "value": 0.000001887,
            "unit": "s"
          },
          {
            "name": "add_two / HLOOpt / cuda / Primal",
            "value": 0.000001887,
            "unit": "s"
          },
          {
            "name": "add_two / PartOpt / cuda / Primal",
            "value": 0.000001888,
            "unit": "s"
          },
          {
            "name": "add_two / IPartOpt / cuda / Primal",
            "value": 0.000001887,
            "unit": "s"
          },
          {
            "name": "add_two / DefOpt / cuda / Primal",
            "value": 0.000001888,
            "unit": "s"
          },
          {
            "name": "add_two / IDefOpt / cuda / Primal",
            "value": 0.000001887,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cuda / Forward",
            "value": 0.000009632,
            "unit": "s"
          },
          {
            "name": "add_two / Jax / cuda / Forward",
            "value": 0.000009792,
            "unit": "s"
          },
          {
            "name": "add_two / HLOOpt / cuda / Forward",
            "value": 0.000009472,
            "unit": "s"
          },
          {
            "name": "add_two / PartOpt / cuda / Forward",
            "value": 0.00000992,
            "unit": "s"
          },
          {
            "name": "add_two / IPartOpt / cuda / Forward",
            "value": 0.000009952,
            "unit": "s"
          },
          {
            "name": "add_two / DefOpt / cuda / Forward",
            "value": 0.000009408,
            "unit": "s"
          },
          {
            "name": "add_two / IDefOpt / cuda / Forward",
            "value": 0.000009824,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cuda / PreRev",
            "value": 0.000031199000000000004,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cuda / PostRev",
            "value": 0.000031744,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cuda / BothRev",
            "value": 0.000031711,
            "unit": "s"
          },
          {
            "name": "add_two / Jax / cuda / BothRev",
            "value": 0.000032319,
            "unit": "s"
          },
          {
            "name": "add_two / HLOOpt / cuda / PreRev",
            "value": 0.000032287000000000004,
            "unit": "s"
          },
          {
            "name": "add_two / HLOOpt / cuda / PostRev",
            "value": 0.000032064,
            "unit": "s"
          },
          {
            "name": "add_two / HLOOpt / cuda / BothRev",
            "value": 0.000032032,
            "unit": "s"
          },
          {
            "name": "add_two / PartOpt / cuda / PreRev",
            "value": 0.000031808000000000004,
            "unit": "s"
          },
          {
            "name": "add_two / PartOpt / cuda / PostRev",
            "value": 0.000031584,
            "unit": "s"
          },
          {
            "name": "add_two / PartOpt / cuda / BothRev",
            "value": 0.000031585,
            "unit": "s"
          },
          {
            "name": "add_two / IPartOpt / cuda / PreRev",
            "value": 0.000031455,
            "unit": "s"
          },
          {
            "name": "add_two / IPartOpt / cuda / PostRev",
            "value": 0.000031808000000000004,
            "unit": "s"
          },
          {
            "name": "add_two / IPartOpt / cuda / BothRev",
            "value": 0.000031872,
            "unit": "s"
          },
          {
            "name": "add_two / DefOpt / cuda / PreRev",
            "value": 0.000031808000000000004,
            "unit": "s"
          },
          {
            "name": "add_two / DefOpt / cuda / PostRev",
            "value": 0.00003232,
            "unit": "s"
          },
          {
            "name": "add_two / DefOpt / cuda / BothRev",
            "value": 0.000031968,
            "unit": "s"
          },
          {
            "name": "add_two / IDefOpt / cuda / PreRev",
            "value": 0.000032672,
            "unit": "s"
          },
          {
            "name": "add_two / IDefOpt / cuda / PostRev",
            "value": 0.000032352,
            "unit": "s"
          },
          {
            "name": "add_two / IDefOpt / cuda / BothRev",
            "value": 0.000032736,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / tpu / Primal",
            "value": 0.000001433525,
            "unit": "s"
          },
          {
            "name": "add_two / Jax / tpu / Primal",
            "value": 0.0000014727,
            "unit": "s"
          },
          {
            "name": "add_two / HLOOpt / tpu / Primal",
            "value": 0.00000142615,
            "unit": "s"
          },
          {
            "name": "add_two / PartOpt / tpu / Primal",
            "value": 0.0000014754249999999995,
            "unit": "s"
          },
          {
            "name": "add_two / IPartOpt / tpu / Primal",
            "value": 0.0000014310999999999995,
            "unit": "s"
          },
          {
            "name": "add_two / DefOpt / tpu / Primal",
            "value": 0.000001473875,
            "unit": "s"
          },
          {
            "name": "add_two / IDefOpt / tpu / Primal",
            "value": 0.0000014331749999999998,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / tpu / Forward",
            "value": 0.000001824325,
            "unit": "s"
          },
          {
            "name": "add_two / Jax / tpu / Forward",
            "value": 0.000001836725,
            "unit": "s"
          },
          {
            "name": "add_two / HLOOpt / tpu / Forward",
            "value": 0.000001826375,
            "unit": "s"
          },
          {
            "name": "add_two / PartOpt / tpu / Forward",
            "value": 0.000001831475,
            "unit": "s"
          },
          {
            "name": "add_two / IPartOpt / tpu / Forward",
            "value": 0.00000183205,
            "unit": "s"
          },
          {
            "name": "add_two / DefOpt / tpu / Forward",
            "value": 0.00000183725,
            "unit": "s"
          },
          {
            "name": "add_two / IDefOpt / tpu / Forward",
            "value": 0.000001828525,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / tpu / PreRev",
            "value": 0.000002843875,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / tpu / PostRev",
            "value": 0.00000276105,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / tpu / BothRev",
            "value": 0.0000028427,
            "unit": "s"
          },
          {
            "name": "add_two / Jax / tpu / BothRev",
            "value": 0.0000027576,
            "unit": "s"
          },
          {
            "name": "add_two / HLOOpt / tpu / PreRev",
            "value": 0.000002844375,
            "unit": "s"
          },
          {
            "name": "add_two / HLOOpt / tpu / PostRev",
            "value": 0.000002748775,
            "unit": "s"
          },
          {
            "name": "add_two / HLOOpt / tpu / BothRev",
            "value": 0.0000028381250000000005,
            "unit": "s"
          },
          {
            "name": "add_two / PartOpt / tpu / PreRev",
            "value": 0.0000027584750000000003,
            "unit": "s"
          },
          {
            "name": "add_two / PartOpt / tpu / PostRev",
            "value": 0.0000028425,
            "unit": "s"
          },
          {
            "name": "add_two / PartOpt / tpu / BothRev",
            "value": 0.0000027695,
            "unit": "s"
          },
          {
            "name": "add_two / IPartOpt / tpu / PreRev",
            "value": 0.00000285875,
            "unit": "s"
          },
          {
            "name": "add_two / IPartOpt / tpu / PostRev",
            "value": 0.00000276865,
            "unit": "s"
          },
          {
            "name": "add_two / IPartOpt / tpu / BothRev",
            "value": 0.000002844875,
            "unit": "s"
          },
          {
            "name": "add_two / DefOpt / tpu / PreRev",
            "value": 0.0000027608,
            "unit": "s"
          },
          {
            "name": "add_two / DefOpt / tpu / PostRev",
            "value": 0.00000283915,
            "unit": "s"
          },
          {
            "name": "add_two / DefOpt / tpu / BothRev",
            "value": 0.0000027521250000000004,
            "unit": "s"
          },
          {
            "name": "add_two / IDefOpt / tpu / PreRev",
            "value": 0.0000028433000000000004,
            "unit": "s"
          },
          {
            "name": "add_two / IDefOpt / tpu / PostRev",
            "value": 0.00000276145,
            "unit": "s"
          },
          {
            "name": "add_two / IDefOpt / tpu / BothRev",
            "value": 0.0000028359250000000003,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / Primal",
            "value": 0.000013414,
            "unit": "s"
          },
          {
            "name": "add_two / Jax / cpu / Primal",
            "value": 0.000013551,
            "unit": "s"
          },
          {
            "name": "add_two / HLOOpt / cpu / Primal",
            "value": 0.00001359,
            "unit": "s"
          },
          {
            "name": "add_two / PartOpt / cpu / Primal",
            "value": 0.000013364,
            "unit": "s"
          },
          {
            "name": "add_two / IPartOpt / cpu / Primal",
            "value": 0.000013675,
            "unit": "s"
          },
          {
            "name": "add_two / DefOpt / cpu / Primal",
            "value": 0.000013555,
            "unit": "s"
          },
          {
            "name": "add_two / IDefOpt / cpu / Primal",
            "value": 0.000013094,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / Forward",
            "value": 0.000018254,
            "unit": "s"
          },
          {
            "name": "add_two / Jax / cpu / Forward",
            "value": 0.00001828,
            "unit": "s"
          },
          {
            "name": "add_two / HLOOpt / cpu / Forward",
            "value": 0.000018404,
            "unit": "s"
          },
          {
            "name": "add_two / PartOpt / cpu / Forward",
            "value": 0.000018438,
            "unit": "s"
          },
          {
            "name": "add_two / IPartOpt / cpu / Forward",
            "value": 0.000018124,
            "unit": "s"
          },
          {
            "name": "add_two / DefOpt / cpu / Forward",
            "value": 0.000018270000000000003,
            "unit": "s"
          },
          {
            "name": "add_two / IDefOpt / cpu / Forward",
            "value": 0.00001795,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / PreRev",
            "value": 0.000024418,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / PostRev",
            "value": 0.000023532,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / BothRev",
            "value": 0.000023139,
            "unit": "s"
          },
          {
            "name": "add_two / Jax / cpu / BothRev",
            "value": 0.000023074,
            "unit": "s"
          },
          {
            "name": "add_two / HLOOpt / cpu / PreRev",
            "value": 0.000022497,
            "unit": "s"
          },
          {
            "name": "add_two / HLOOpt / cpu / PostRev",
            "value": 0.000024122,
            "unit": "s"
          },
          {
            "name": "add_two / HLOOpt / cpu / BothRev",
            "value": 0.000023232,
            "unit": "s"
          },
          {
            "name": "add_two / PartOpt / cpu / PreRev",
            "value": 0.000023275,
            "unit": "s"
          },
          {
            "name": "add_two / PartOpt / cpu / PostRev",
            "value": 0.000023217,
            "unit": "s"
          },
          {
            "name": "add_two / PartOpt / cpu / BothRev",
            "value": 0.000023254,
            "unit": "s"
          },
          {
            "name": "add_two / IPartOpt / cpu / PreRev",
            "value": 0.000023631,
            "unit": "s"
          },
          {
            "name": "add_two / IPartOpt / cpu / PostRev",
            "value": 0.00002333,
            "unit": "s"
          },
          {
            "name": "add_two / IPartOpt / cpu / BothRev",
            "value": 0.000023309,
            "unit": "s"
          },
          {
            "name": "add_two / DefOpt / cpu / PreRev",
            "value": 0.000023562,
            "unit": "s"
          },
          {
            "name": "add_two / DefOpt / cpu / PostRev",
            "value": 0.000023083,
            "unit": "s"
          },
          {
            "name": "add_two / DefOpt / cpu / BothRev",
            "value": 0.000023365,
            "unit": "s"
          },
          {
            "name": "add_two / IDefOpt / cpu / PreRev",
            "value": 0.000023837,
            "unit": "s"
          },
          {
            "name": "add_two / IDefOpt / cpu / PostRev",
            "value": 0.000023509,
            "unit": "s"
          },
          {
            "name": "add_two / IDefOpt / cpu / BothRev",
            "value": 0.000023551,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / Primal",
            "value": 0.00001,
            "unit": "s"
          },
          {
            "name": "add_two / Jax / cpu / Primal",
            "value": 0.000008999999999999999,
            "unit": "s"
          },
          {
            "name": "add_two / HLOOpt / cpu / Primal",
            "value": 0.000008999999999999999,
            "unit": "s"
          },
          {
            "name": "add_two / PartOpt / cpu / Primal",
            "value": 0.000008999999999999999,
            "unit": "s"
          },
          {
            "name": "add_two / IPartOpt / cpu / Primal",
            "value": 0.000008999999999999999,
            "unit": "s"
          },
          {
            "name": "add_two / DefOpt / cpu / Primal",
            "value": 0.000008999999999999999,
            "unit": "s"
          },
          {
            "name": "add_two / IDefOpt / cpu / Primal",
            "value": 0.000008999999999999999,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / Forward",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "add_two / Jax / cpu / Forward",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "add_two / HLOOpt / cpu / Forward",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "add_two / PartOpt / cpu / Forward",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "add_two / IPartOpt / cpu / Forward",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "add_two / DefOpt / cpu / Forward",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "add_two / IDefOpt / cpu / Forward",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / PreRev",
            "value": 0.000017,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / PostRev",
            "value": 0.000017,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / BothRev",
            "value": 0.000017,
            "unit": "s"
          },
          {
            "name": "add_two / Jax / cpu / BothRev",
            "value": 0.000016,
            "unit": "s"
          },
          {
            "name": "add_two / HLOOpt / cpu / PreRev",
            "value": 0.000017,
            "unit": "s"
          },
          {
            "name": "add_two / HLOOpt / cpu / PostRev",
            "value": 0.000017999999999999997,
            "unit": "s"
          },
          {
            "name": "add_two / HLOOpt / cpu / BothRev",
            "value": 0.000017,
            "unit": "s"
          },
          {
            "name": "add_two / PartOpt / cpu / PreRev",
            "value": 0.000016,
            "unit": "s"
          },
          {
            "name": "add_two / PartOpt / cpu / PostRev",
            "value": 0.000017,
            "unit": "s"
          },
          {
            "name": "add_two / PartOpt / cpu / BothRev",
            "value": 0.000017,
            "unit": "s"
          },
          {
            "name": "add_two / IPartOpt / cpu / PreRev",
            "value": 0.000017,
            "unit": "s"
          },
          {
            "name": "add_two / IPartOpt / cpu / PostRev",
            "value": 0.000017,
            "unit": "s"
          },
          {
            "name": "add_two / IPartOpt / cpu / BothRev",
            "value": 0.000016,
            "unit": "s"
          },
          {
            "name": "add_two / DefOpt / cpu / PreRev",
            "value": 0.000016,
            "unit": "s"
          },
          {
            "name": "add_two / DefOpt / cpu / PostRev",
            "value": 0.000017,
            "unit": "s"
          },
          {
            "name": "add_two / DefOpt / cpu / BothRev",
            "value": 0.000017,
            "unit": "s"
          },
          {
            "name": "add_two / IDefOpt / cpu / PreRev",
            "value": 0.000017,
            "unit": "s"
          },
          {
            "name": "add_two / IDefOpt / cpu / PostRev",
            "value": 0.000017,
            "unit": "s"
          },
          {
            "name": "add_two / IDefOpt / cpu / BothRev",
            "value": 0.000016,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / Primal",
            "value": 0.0000072663399805605875,
            "unit": "s"
          },
          {
            "name": "cache / Jax / cpu / Primal",
            "value": 0.000007059020026645157,
            "unit": "s"
          },
          {
            "name": "cache / HLOOpt / cpu / Primal",
            "value": 0.00000714279999556311,
            "unit": "s"
          },
          {
            "name": "cache / PartOpt / cpu / Primal",
            "value": 0.000007266960028573521,
            "unit": "s"
          },
          {
            "name": "cache / IPartOpt / cpu / Primal",
            "value": 0.000006997760056037805,
            "unit": "s"
          },
          {
            "name": "cache / DefOpt / cpu / Primal",
            "value": 0.000007151300005716621,
            "unit": "s"
          },
          {
            "name": "cache / IDefOpt / cpu / Primal",
            "value": 0.00000695411999913631,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / Forward",
            "value": 0.000015701859983892063,
            "unit": "s"
          },
          {
            "name": "cache / Jax / cpu / Forward",
            "value": 0.00001483424003708933,
            "unit": "s"
          },
          {
            "name": "cache / HLOOpt / cpu / Forward",
            "value": 0.000016832900009831064,
            "unit": "s"
          },
          {
            "name": "cache / PartOpt / cpu / Forward",
            "value": 0.00001612193999790179,
            "unit": "s"
          },
          {
            "name": "cache / IPartOpt / cpu / Forward",
            "value": 0.00001666036001552129,
            "unit": "s"
          },
          {
            "name": "cache / DefOpt / cpu / Forward",
            "value": 0.00001574375998643518,
            "unit": "s"
          },
          {
            "name": "cache / IDefOpt / cpu / Forward",
            "value": 0.00001478907997807255,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / PreRev",
            "value": 0.00001646296000217262,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / PostRev",
            "value": 0.00002128442002685915,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / BothRev",
            "value": 0.000016913399995246437,
            "unit": "s"
          },
          {
            "name": "cache / Jax / cpu / BothRev",
            "value": 0.000021109800045451265,
            "unit": "s"
          },
          {
            "name": "cache / HLOOpt / cpu / PreRev",
            "value": 0.00001809326004149625,
            "unit": "s"
          },
          {
            "name": "cache / HLOOpt / cpu / PostRev",
            "value": 0.00001996429998143867,
            "unit": "s"
          },
          {
            "name": "cache / HLOOpt / cpu / BothRev",
            "value": 0.000018035499979305315,
            "unit": "s"
          },
          {
            "name": "cache / PartOpt / cpu / PreRev",
            "value": 0.00001996600002712512,
            "unit": "s"
          },
          {
            "name": "cache / PartOpt / cpu / PostRev",
            "value": 0.000022108299999672452,
            "unit": "s"
          },
          {
            "name": "cache / PartOpt / cpu / BothRev",
            "value": 0.000016975299977275427,
            "unit": "s"
          },
          {
            "name": "cache / IPartOpt / cpu / PreRev",
            "value": 0.000017175520024466096,
            "unit": "s"
          },
          {
            "name": "cache / IPartOpt / cpu / PostRev",
            "value": 0.00002078946001347504,
            "unit": "s"
          },
          {
            "name": "cache / IPartOpt / cpu / BothRev",
            "value": 0.000017755019971446018,
            "unit": "s"
          },
          {
            "name": "cache / DefOpt / cpu / PreRev",
            "value": 0.00001808053998502146,
            "unit": "s"
          },
          {
            "name": "cache / DefOpt / cpu / PostRev",
            "value": 0.000018206260010629192,
            "unit": "s"
          },
          {
            "name": "cache / DefOpt / cpu / BothRev",
            "value": 0.00001633273996958451,
            "unit": "s"
          },
          {
            "name": "cache / IDefOpt / cpu / PreRev",
            "value": 0.000017065619977074677,
            "unit": "s"
          },
          {
            "name": "cache / IDefOpt / cpu / PostRev",
            "value": 0.000017194100000779144,
            "unit": "s"
          },
          {
            "name": "cache / IDefOpt / cpu / BothRev",
            "value": 0.00003606621998187621,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cuda / Primal",
            "value": 0.000002336,
            "unit": "s"
          },
          {
            "name": "cache / Jax / cuda / Primal",
            "value": 0.000002336,
            "unit": "s"
          },
          {
            "name": "cache / HLOOpt / cuda / Primal",
            "value": 0.000002272,
            "unit": "s"
          },
          {
            "name": "cache / PartOpt / cuda / Primal",
            "value": 0.000002304,
            "unit": "s"
          },
          {
            "name": "cache / IPartOpt / cuda / Primal",
            "value": 0.000002335,
            "unit": "s"
          },
          {
            "name": "cache / DefOpt / cuda / Primal",
            "value": 0.000002304,
            "unit": "s"
          },
          {
            "name": "cache / IDefOpt / cuda / Primal",
            "value": 0.000002272,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cuda / Forward",
            "value": 0.0000023670000000000004,
            "unit": "s"
          },
          {
            "name": "cache / Jax / cuda / Forward",
            "value": 0.0000023670000000000004,
            "unit": "s"
          },
          {
            "name": "cache / HLOOpt / cuda / Forward",
            "value": 0.0000023670000000000004,
            "unit": "s"
          },
          {
            "name": "cache / PartOpt / cuda / Forward",
            "value": 0.0000023670000000000004,
            "unit": "s"
          },
          {
            "name": "cache / IPartOpt / cuda / Forward",
            "value": 0.0000023670000000000004,
            "unit": "s"
          },
          {
            "name": "cache / DefOpt / cuda / Forward",
            "value": 0.000002304,
            "unit": "s"
          },
          {
            "name": "cache / IDefOpt / cuda / Forward",
            "value": 0.0000023670000000000004,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cuda / PreRev",
            "value": 0.00001024,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cuda / PostRev",
            "value": 0.000010496,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cuda / BothRev",
            "value": 0.000010433,
            "unit": "s"
          },
          {
            "name": "cache / Jax / cuda / BothRev",
            "value": 0.000011008,
            "unit": "s"
          },
          {
            "name": "cache / HLOOpt / cuda / PreRev",
            "value": 0.000013087,
            "unit": "s"
          },
          {
            "name": "cache / HLOOpt / cuda / PostRev",
            "value": 0.000013055,
            "unit": "s"
          },
          {
            "name": "cache / HLOOpt / cuda / BothRev",
            "value": 0.000013087,
            "unit": "s"
          },
          {
            "name": "cache / PartOpt / cuda / PreRev",
            "value": 0.00001088,
            "unit": "s"
          },
          {
            "name": "cache / PartOpt / cuda / PostRev",
            "value": 0.000010816,
            "unit": "s"
          },
          {
            "name": "cache / PartOpt / cuda / BothRev",
            "value": 0.000010913,
            "unit": "s"
          },
          {
            "name": "cache / IPartOpt / cuda / PreRev",
            "value": 0.000011103,
            "unit": "s"
          },
          {
            "name": "cache / IPartOpt / cuda / PostRev",
            "value": 0.000010912,
            "unit": "s"
          },
          {
            "name": "cache / IPartOpt / cuda / BothRev",
            "value": 0.00001056,
            "unit": "s"
          },
          {
            "name": "cache / DefOpt / cuda / PreRev",
            "value": 0.000011712,
            "unit": "s"
          },
          {
            "name": "cache / DefOpt / cuda / PostRev",
            "value": 0.000010943,
            "unit": "s"
          },
          {
            "name": "cache / DefOpt / cuda / BothRev",
            "value": 0.000010496,
            "unit": "s"
          },
          {
            "name": "cache / IDefOpt / cuda / PreRev",
            "value": 0.000010848,
            "unit": "s"
          },
          {
            "name": "cache / IDefOpt / cuda / PostRev",
            "value": 0.000010816,
            "unit": "s"
          },
          {
            "name": "cache / IDefOpt / cuda / BothRev",
            "value": 0.000010753,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / tpu / Primal",
            "value": 0.000002475475,
            "unit": "s"
          },
          {
            "name": "cache / Jax / tpu / Primal",
            "value": 0.000002462875,
            "unit": "s"
          },
          {
            "name": "cache / HLOOpt / tpu / Primal",
            "value": 0.000002463575,
            "unit": "s"
          },
          {
            "name": "cache / PartOpt / tpu / Primal",
            "value": 0.0000024692,
            "unit": "s"
          },
          {
            "name": "cache / IPartOpt / tpu / Primal",
            "value": 0.000002475325,
            "unit": "s"
          },
          {
            "name": "cache / DefOpt / tpu / Primal",
            "value": 0.000002473,
            "unit": "s"
          },
          {
            "name": "cache / IDefOpt / tpu / Primal",
            "value": 0.00000245865,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / tpu / Forward",
            "value": 0.000003546375,
            "unit": "s"
          },
          {
            "name": "cache / Jax / tpu / Forward",
            "value": 0.000003542725,
            "unit": "s"
          },
          {
            "name": "cache / HLOOpt / tpu / Forward",
            "value": 0.000003557725,
            "unit": "s"
          },
          {
            "name": "cache / PartOpt / tpu / Forward",
            "value": 0.000003531725,
            "unit": "s"
          },
          {
            "name": "cache / IPartOpt / tpu / Forward",
            "value": 0.000003584575,
            "unit": "s"
          },
          {
            "name": "cache / DefOpt / tpu / Forward",
            "value": 0.000003544125000000001,
            "unit": "s"
          },
          {
            "name": "cache / IDefOpt / tpu / Forward",
            "value": 0.00000355125,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / tpu / PreRev",
            "value": 0.000004941975,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / tpu / PostRev",
            "value": 0.000004950099999999999,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / tpu / BothRev",
            "value": 0.0000049529250000000005,
            "unit": "s"
          },
          {
            "name": "cache / Jax / tpu / BothRev",
            "value": 0.000004971324999999999,
            "unit": "s"
          },
          {
            "name": "cache / HLOOpt / tpu / PreRev",
            "value": 0.0000039344,
            "unit": "s"
          },
          {
            "name": "cache / HLOOpt / tpu / PostRev",
            "value": 0.000004119175,
            "unit": "s"
          },
          {
            "name": "cache / HLOOpt / tpu / BothRev",
            "value": 0.000003933925,
            "unit": "s"
          },
          {
            "name": "cache / PartOpt / tpu / PreRev",
            "value": 0.000004969925,
            "unit": "s"
          },
          {
            "name": "cache / PartOpt / tpu / PostRev",
            "value": 0.0000049710500000000005,
            "unit": "s"
          },
          {
            "name": "cache / PartOpt / tpu / BothRev",
            "value": 0.0000049886,
            "unit": "s"
          },
          {
            "name": "cache / IPartOpt / tpu / PreRev",
            "value": 0.000004944075,
            "unit": "s"
          },
          {
            "name": "cache / IPartOpt / tpu / PostRev",
            "value": 0.000004964275000000001,
            "unit": "s"
          },
          {
            "name": "cache / IPartOpt / tpu / BothRev",
            "value": 0.000004961025,
            "unit": "s"
          },
          {
            "name": "cache / DefOpt / tpu / PreRev",
            "value": 0.00000497215,
            "unit": "s"
          },
          {
            "name": "cache / DefOpt / tpu / PostRev",
            "value": 0.000004977124999999999,
            "unit": "s"
          },
          {
            "name": "cache / DefOpt / tpu / BothRev",
            "value": 0.000004970425,
            "unit": "s"
          },
          {
            "name": "cache / IDefOpt / tpu / PreRev",
            "value": 0.0000049747,
            "unit": "s"
          },
          {
            "name": "cache / IDefOpt / tpu / PostRev",
            "value": 0.0000049627,
            "unit": "s"
          },
          {
            "name": "cache / IDefOpt / tpu / BothRev",
            "value": 0.00000497935,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / Primal",
            "value": 0.000013055,
            "unit": "s"
          },
          {
            "name": "cache / Jax / cpu / Primal",
            "value": 0.000012511,
            "unit": "s"
          },
          {
            "name": "cache / HLOOpt / cpu / Primal",
            "value": 0.000012688,
            "unit": "s"
          },
          {
            "name": "cache / PartOpt / cpu / Primal",
            "value": 0.000012418,
            "unit": "s"
          },
          {
            "name": "cache / IPartOpt / cpu / Primal",
            "value": 0.000012681,
            "unit": "s"
          },
          {
            "name": "cache / DefOpt / cpu / Primal",
            "value": 0.000012733,
            "unit": "s"
          },
          {
            "name": "cache / IDefOpt / cpu / Primal",
            "value": 0.000012802,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / Forward",
            "value": 0.00002314,
            "unit": "s"
          },
          {
            "name": "cache / Jax / cpu / Forward",
            "value": 0.000021556,
            "unit": "s"
          },
          {
            "name": "cache / HLOOpt / cpu / Forward",
            "value": 0.000018048,
            "unit": "s"
          },
          {
            "name": "cache / PartOpt / cpu / Forward",
            "value": 0.000024327000000000003,
            "unit": "s"
          },
          {
            "name": "cache / IPartOpt / cpu / Forward",
            "value": 0.000017034,
            "unit": "s"
          },
          {
            "name": "cache / DefOpt / cpu / Forward",
            "value": 0.00001753,
            "unit": "s"
          },
          {
            "name": "cache / IDefOpt / cpu / Forward",
            "value": 0.000017007,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / PreRev",
            "value": 0.000018243,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / PostRev",
            "value": 0.000022367,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / BothRev",
            "value": 0.00002626,
            "unit": "s"
          },
          {
            "name": "cache / Jax / cpu / BothRev",
            "value": 0.00002034,
            "unit": "s"
          },
          {
            "name": "cache / HLOOpt / cpu / PreRev",
            "value": 0.000017923,
            "unit": "s"
          },
          {
            "name": "cache / HLOOpt / cpu / PostRev",
            "value": 0.000017517999999999997,
            "unit": "s"
          },
          {
            "name": "cache / HLOOpt / cpu / BothRev",
            "value": 0.000017517999999999997,
            "unit": "s"
          },
          {
            "name": "cache / PartOpt / cpu / PreRev",
            "value": 0.000017572,
            "unit": "s"
          },
          {
            "name": "cache / PartOpt / cpu / PostRev",
            "value": 0.000019504,
            "unit": "s"
          },
          {
            "name": "cache / PartOpt / cpu / BothRev",
            "value": 0.000017661,
            "unit": "s"
          },
          {
            "name": "cache / IPartOpt / cpu / PreRev",
            "value": 0.000019325,
            "unit": "s"
          },
          {
            "name": "cache / IPartOpt / cpu / PostRev",
            "value": 0.000019476,
            "unit": "s"
          },
          {
            "name": "cache / IPartOpt / cpu / BothRev",
            "value": 0.00001799,
            "unit": "s"
          },
          {
            "name": "cache / DefOpt / cpu / PreRev",
            "value": 0.000018415,
            "unit": "s"
          },
          {
            "name": "cache / DefOpt / cpu / PostRev",
            "value": 0.000017612,
            "unit": "s"
          },
          {
            "name": "cache / DefOpt / cpu / BothRev",
            "value": 0.000017617,
            "unit": "s"
          },
          {
            "name": "cache / IDefOpt / cpu / PreRev",
            "value": 0.000018403,
            "unit": "s"
          },
          {
            "name": "cache / IDefOpt / cpu / PostRev",
            "value": 0.000017168,
            "unit": "s"
          },
          {
            "name": "cache / IDefOpt / cpu / BothRev",
            "value": 0.000018012,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / Primal",
            "value": 0.000008,
            "unit": "s"
          },
          {
            "name": "cache / Jax / cpu / Primal",
            "value": 0.000008999999999999999,
            "unit": "s"
          },
          {
            "name": "cache / HLOOpt / cpu / Primal",
            "value": 0.000008999999999999999,
            "unit": "s"
          },
          {
            "name": "cache / PartOpt / cpu / Primal",
            "value": 0.000008999999999999999,
            "unit": "s"
          },
          {
            "name": "cache / IPartOpt / cpu / Primal",
            "value": 0.000008,
            "unit": "s"
          },
          {
            "name": "cache / DefOpt / cpu / Primal",
            "value": 0.000008,
            "unit": "s"
          },
          {
            "name": "cache / IDefOpt / cpu / Primal",
            "value": 0.000008,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / Forward",
            "value": 0.000011,
            "unit": "s"
          },
          {
            "name": "cache / Jax / cpu / Forward",
            "value": 0.000011,
            "unit": "s"
          },
          {
            "name": "cache / HLOOpt / cpu / Forward",
            "value": 0.000011,
            "unit": "s"
          },
          {
            "name": "cache / PartOpt / cpu / Forward",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "cache / IPartOpt / cpu / Forward",
            "value": 0.000011,
            "unit": "s"
          },
          {
            "name": "cache / DefOpt / cpu / Forward",
            "value": 0.000011,
            "unit": "s"
          },
          {
            "name": "cache / IDefOpt / cpu / Forward",
            "value": 0.000011,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / PreRev",
            "value": 0.000011,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / PostRev",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / BothRev",
            "value": 0.000011,
            "unit": "s"
          },
          {
            "name": "cache / Jax / cpu / BothRev",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "cache / HLOOpt / cpu / PreRev",
            "value": 0.000011,
            "unit": "s"
          },
          {
            "name": "cache / HLOOpt / cpu / PostRev",
            "value": 0.000035000000000000004,
            "unit": "s"
          },
          {
            "name": "cache / HLOOpt / cpu / BothRev",
            "value": 0.000011,
            "unit": "s"
          },
          {
            "name": "cache / PartOpt / cpu / PreRev",
            "value": 0.000011,
            "unit": "s"
          },
          {
            "name": "cache / PartOpt / cpu / PostRev",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "cache / PartOpt / cpu / BothRev",
            "value": 0.00001,
            "unit": "s"
          },
          {
            "name": "cache / IPartOpt / cpu / PreRev",
            "value": 0.000035999999999999994,
            "unit": "s"
          },
          {
            "name": "cache / IPartOpt / cpu / PostRev",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "cache / IPartOpt / cpu / BothRev",
            "value": 0.000011,
            "unit": "s"
          },
          {
            "name": "cache / DefOpt / cpu / PreRev",
            "value": 0.000037,
            "unit": "s"
          },
          {
            "name": "cache / DefOpt / cpu / PostRev",
            "value": 0.000011,
            "unit": "s"
          },
          {
            "name": "cache / DefOpt / cpu / BothRev",
            "value": 0.000011,
            "unit": "s"
          },
          {
            "name": "cache / IDefOpt / cpu / PreRev",
            "value": 0.000011,
            "unit": "s"
          },
          {
            "name": "cache / IDefOpt / cpu / PostRev",
            "value": 0.000043,
            "unit": "s"
          },
          {
            "name": "cache / IDefOpt / cpu / BothRev",
            "value": 0.000035999999999999994,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / Primal",
            "value": 0.000007722380005361628,
            "unit": "s"
          },
          {
            "name": "Concat / Jax / cpu / Primal",
            "value": 0.000007469840011253837,
            "unit": "s"
          },
          {
            "name": "Concat / HLOOpt / cpu / Primal",
            "value": 0.000007509980005124816,
            "unit": "s"
          },
          {
            "name": "Concat / PartOpt / cpu / Primal",
            "value": 0.00000716315999852668,
            "unit": "s"
          },
          {
            "name": "Concat / IPartOpt / cpu / Primal",
            "value": 0.0000073569000232964755,
            "unit": "s"
          },
          {
            "name": "Concat / DefOpt / cpu / Primal",
            "value": 0.000006935940000403207,
            "unit": "s"
          },
          {
            "name": "Concat / IDefOpt / cpu / Primal",
            "value": 0.000007187700048234547,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / Forward",
            "value": 0.00001155280003331427,
            "unit": "s"
          },
          {
            "name": "Concat / Jax / cpu / Forward",
            "value": 0.000011150919972351405,
            "unit": "s"
          },
          {
            "name": "Concat / HLOOpt / cpu / Forward",
            "value": 0.000010956260020975605,
            "unit": "s"
          },
          {
            "name": "Concat / PartOpt / cpu / Forward",
            "value": 0.00001142896002420457,
            "unit": "s"
          },
          {
            "name": "Concat / IPartOpt / cpu / Forward",
            "value": 0.000011185860003024571,
            "unit": "s"
          },
          {
            "name": "Concat / DefOpt / cpu / Forward",
            "value": 0.000011048579999624053,
            "unit": "s"
          },
          {
            "name": "Concat / IDefOpt / cpu / Forward",
            "value": 0.000011640999982773792,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / PreRev",
            "value": 0.000013337879954633535,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / PostRev",
            "value": 0.00001346205998743244,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / BothRev",
            "value": 0.000012496040026235278,
            "unit": "s"
          },
          {
            "name": "Concat / Jax / cpu / BothRev",
            "value": 0.00001323239996963821,
            "unit": "s"
          },
          {
            "name": "Concat / HLOOpt / cpu / PreRev",
            "value": 0.000013045039950156931,
            "unit": "s"
          },
          {
            "name": "Concat / HLOOpt / cpu / PostRev",
            "value": 0.000015257439999913913,
            "unit": "s"
          },
          {
            "name": "Concat / HLOOpt / cpu / BothRev",
            "value": 0.00001330468005107832,
            "unit": "s"
          },
          {
            "name": "Concat / PartOpt / cpu / PreRev",
            "value": 0.000012892439972347347,
            "unit": "s"
          },
          {
            "name": "Concat / PartOpt / cpu / PostRev",
            "value": 0.000012447819972294383,
            "unit": "s"
          },
          {
            "name": "Concat / PartOpt / cpu / BothRev",
            "value": 0.000014018979973116077,
            "unit": "s"
          },
          {
            "name": "Concat / IPartOpt / cpu / PreRev",
            "value": 0.000013137420000930434,
            "unit": "s"
          },
          {
            "name": "Concat / IPartOpt / cpu / PostRev",
            "value": 0.00001297132000217971,
            "unit": "s"
          },
          {
            "name": "Concat / IPartOpt / cpu / BothRev",
            "value": 0.000013204779988882364,
            "unit": "s"
          },
          {
            "name": "Concat / DefOpt / cpu / PreRev",
            "value": 0.0000141949400131125,
            "unit": "s"
          },
          {
            "name": "Concat / DefOpt / cpu / PostRev",
            "value": 0.00001276810002309503,
            "unit": "s"
          },
          {
            "name": "Concat / DefOpt / cpu / BothRev",
            "value": 0.000013300199989316751,
            "unit": "s"
          },
          {
            "name": "Concat / IDefOpt / cpu / PreRev",
            "value": 0.000013066080000498916,
            "unit": "s"
          },
          {
            "name": "Concat / IDefOpt / cpu / PostRev",
            "value": 0.000013508679958249558,
            "unit": "s"
          },
          {
            "name": "Concat / IDefOpt / cpu / BothRev",
            "value": 0.000012933540010635624,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cuda / Primal",
            "value": 0.000001888,
            "unit": "s"
          },
          {
            "name": "Concat / Jax / cuda / Primal",
            "value": 0.0000019200000000000003,
            "unit": "s"
          },
          {
            "name": "Concat / HLOOpt / cuda / Primal",
            "value": 0.0000019200000000000003,
            "unit": "s"
          },
          {
            "name": "Concat / PartOpt / cuda / Primal",
            "value": 0.000001919,
            "unit": "s"
          },
          {
            "name": "Concat / IPartOpt / cuda / Primal",
            "value": 0.0000019200000000000003,
            "unit": "s"
          },
          {
            "name": "Concat / DefOpt / cuda / Primal",
            "value": 0.0000019200000000000003,
            "unit": "s"
          },
          {
            "name": "Concat / IDefOpt / cuda / Primal",
            "value": 0.0000019200000000000003,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cuda / Forward",
            "value": 0.000009408,
            "unit": "s"
          },
          {
            "name": "Concat / Jax / cuda / Forward",
            "value": 0.0000096,
            "unit": "s"
          },
          {
            "name": "Concat / HLOOpt / cuda / Forward",
            "value": 0.000009472,
            "unit": "s"
          },
          {
            "name": "Concat / PartOpt / cuda / Forward",
            "value": 0.000009728,
            "unit": "s"
          },
          {
            "name": "Concat / IPartOpt / cuda / Forward",
            "value": 0.000009824,
            "unit": "s"
          },
          {
            "name": "Concat / DefOpt / cuda / Forward",
            "value": 0.000009824,
            "unit": "s"
          },
          {
            "name": "Concat / IDefOpt / cuda / Forward",
            "value": 0.000009792,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cuda / PreRev",
            "value": 0.000015743,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cuda / PostRev",
            "value": 0.000015999,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cuda / BothRev",
            "value": 0.000025088,
            "unit": "s"
          },
          {
            "name": "Concat / Jax / cuda / BothRev",
            "value": 0.000016448000000000002,
            "unit": "s"
          },
          {
            "name": "Concat / HLOOpt / cuda / PreRev",
            "value": 0.00001632,
            "unit": "s"
          },
          {
            "name": "Concat / HLOOpt / cuda / PostRev",
            "value": 0.000015999,
            "unit": "s"
          },
          {
            "name": "Concat / HLOOpt / cuda / BothRev",
            "value": 0.000015966999999999998,
            "unit": "s"
          },
          {
            "name": "Concat / PartOpt / cuda / PreRev",
            "value": 0.000016,
            "unit": "s"
          },
          {
            "name": "Concat / PartOpt / cuda / PostRev",
            "value": 0.000016352,
            "unit": "s"
          },
          {
            "name": "Concat / PartOpt / cuda / BothRev",
            "value": 0.000016063999999999997,
            "unit": "s"
          },
          {
            "name": "Concat / IPartOpt / cuda / PreRev",
            "value": 0.00001632,
            "unit": "s"
          },
          {
            "name": "Concat / IPartOpt / cuda / PostRev",
            "value": 0.000016224,
            "unit": "s"
          },
          {
            "name": "Concat / IPartOpt / cuda / BothRev",
            "value": 0.000016288,
            "unit": "s"
          },
          {
            "name": "Concat / DefOpt / cuda / PreRev",
            "value": 0.000016032,
            "unit": "s"
          },
          {
            "name": "Concat / DefOpt / cuda / PostRev",
            "value": 0.000015872,
            "unit": "s"
          },
          {
            "name": "Concat / DefOpt / cuda / BothRev",
            "value": 0.000016224,
            "unit": "s"
          },
          {
            "name": "Concat / IDefOpt / cuda / PreRev",
            "value": 0.000016576000000000002,
            "unit": "s"
          },
          {
            "name": "Concat / IDefOpt / cuda / PostRev",
            "value": 0.000015999,
            "unit": "s"
          },
          {
            "name": "Concat / IDefOpt / cuda / BothRev",
            "value": 0.00001632,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / tpu / Primal",
            "value": 0.00000153335,
            "unit": "s"
          },
          {
            "name": "Concat / Jax / tpu / Primal",
            "value": 0.000001526325,
            "unit": "s"
          },
          {
            "name": "Concat / HLOOpt / tpu / Primal",
            "value": 0.0000015363,
            "unit": "s"
          },
          {
            "name": "Concat / PartOpt / tpu / Primal",
            "value": 0.00000153295,
            "unit": "s"
          },
          {
            "name": "Concat / IPartOpt / tpu / Primal",
            "value": 0.000001527325,
            "unit": "s"
          },
          {
            "name": "Concat / DefOpt / tpu / Primal",
            "value": 0.000001520875,
            "unit": "s"
          },
          {
            "name": "Concat / IDefOpt / tpu / Primal",
            "value": 0.0000015308250000000005,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / tpu / Forward",
            "value": 0.000001571175,
            "unit": "s"
          },
          {
            "name": "Concat / Jax / tpu / Forward",
            "value": 0.00000154745,
            "unit": "s"
          },
          {
            "name": "Concat / HLOOpt / tpu / Forward",
            "value": 0.0000015902,
            "unit": "s"
          },
          {
            "name": "Concat / PartOpt / tpu / Forward",
            "value": 0.000001550125,
            "unit": "s"
          },
          {
            "name": "Concat / IPartOpt / tpu / Forward",
            "value": 0.0000015801,
            "unit": "s"
          },
          {
            "name": "Concat / DefOpt / tpu / Forward",
            "value": 0.00000155265,
            "unit": "s"
          },
          {
            "name": "Concat / IDefOpt / tpu / Forward",
            "value": 0.0000015643999999999998,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / tpu / PreRev",
            "value": 0.0000020065500000000003,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / tpu / PostRev",
            "value": 0.000002083225,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / tpu / BothRev",
            "value": 0.000002003425,
            "unit": "s"
          },
          {
            "name": "Concat / Jax / tpu / BothRev",
            "value": 0.000002071975,
            "unit": "s"
          },
          {
            "name": "Concat / HLOOpt / tpu / PreRev",
            "value": 0.0000020164000000000003,
            "unit": "s"
          },
          {
            "name": "Concat / HLOOpt / tpu / PostRev",
            "value": 0.00000206995,
            "unit": "s"
          },
          {
            "name": "Concat / HLOOpt / tpu / BothRev",
            "value": 0.0000020086,
            "unit": "s"
          },
          {
            "name": "Concat / PartOpt / tpu / PreRev",
            "value": 0.0000020772,
            "unit": "s"
          },
          {
            "name": "Concat / PartOpt / tpu / PostRev",
            "value": 0.000002007975,
            "unit": "s"
          },
          {
            "name": "Concat / PartOpt / tpu / BothRev",
            "value": 0.0000020912,
            "unit": "s"
          },
          {
            "name": "Concat / IPartOpt / tpu / PreRev",
            "value": 0.00000200175,
            "unit": "s"
          },
          {
            "name": "Concat / IPartOpt / tpu / PostRev",
            "value": 0.000002068875,
            "unit": "s"
          },
          {
            "name": "Concat / IPartOpt / tpu / BothRev",
            "value": 0.000002016125,
            "unit": "s"
          },
          {
            "name": "Concat / DefOpt / tpu / PreRev",
            "value": 0.00000207255,
            "unit": "s"
          },
          {
            "name": "Concat / DefOpt / tpu / PostRev",
            "value": 0.0000020205,
            "unit": "s"
          },
          {
            "name": "Concat / DefOpt / tpu / BothRev",
            "value": 0.000002074075,
            "unit": "s"
          },
          {
            "name": "Concat / IDefOpt / tpu / PreRev",
            "value": 0.0000020079,
            "unit": "s"
          },
          {
            "name": "Concat / IDefOpt / tpu / PostRev",
            "value": 0.000002076925,
            "unit": "s"
          },
          {
            "name": "Concat / IDefOpt / tpu / BothRev",
            "value": 0.000002008125,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / Primal",
            "value": 0.000012762,
            "unit": "s"
          },
          {
            "name": "Concat / Jax / cpu / Primal",
            "value": 0.000013045,
            "unit": "s"
          },
          {
            "name": "Concat / HLOOpt / cpu / Primal",
            "value": 0.000012759,
            "unit": "s"
          },
          {
            "name": "Concat / PartOpt / cpu / Primal",
            "value": 0.000012827,
            "unit": "s"
          },
          {
            "name": "Concat / IPartOpt / cpu / Primal",
            "value": 0.00001274,
            "unit": "s"
          },
          {
            "name": "Concat / DefOpt / cpu / Primal",
            "value": 0.000012548,
            "unit": "s"
          },
          {
            "name": "Concat / IDefOpt / cpu / Primal",
            "value": 0.000013034,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / Forward",
            "value": 0.000017412000000000002,
            "unit": "s"
          },
          {
            "name": "Concat / Jax / cpu / Forward",
            "value": 0.000017175,
            "unit": "s"
          },
          {
            "name": "Concat / HLOOpt / cpu / Forward",
            "value": 0.000017403,
            "unit": "s"
          },
          {
            "name": "Concat / PartOpt / cpu / Forward",
            "value": 0.000017962,
            "unit": "s"
          },
          {
            "name": "Concat / IPartOpt / cpu / Forward",
            "value": 0.000017406000000000002,
            "unit": "s"
          },
          {
            "name": "Concat / DefOpt / cpu / Forward",
            "value": 0.000017703,
            "unit": "s"
          },
          {
            "name": "Concat / IDefOpt / cpu / Forward",
            "value": 0.000017697,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / PreRev",
            "value": 0.000020833,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / PostRev",
            "value": 0.000020884,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / BothRev",
            "value": 0.000019884,
            "unit": "s"
          },
          {
            "name": "Concat / Jax / cpu / BothRev",
            "value": 0.000019806,
            "unit": "s"
          },
          {
            "name": "Concat / HLOOpt / cpu / PreRev",
            "value": 0.000019707,
            "unit": "s"
          },
          {
            "name": "Concat / HLOOpt / cpu / PostRev",
            "value": 0.000019712,
            "unit": "s"
          },
          {
            "name": "Concat / HLOOpt / cpu / BothRev",
            "value": 0.000019952,
            "unit": "s"
          },
          {
            "name": "Concat / PartOpt / cpu / PreRev",
            "value": 0.000019766,
            "unit": "s"
          },
          {
            "name": "Concat / PartOpt / cpu / PostRev",
            "value": 0.000019485,
            "unit": "s"
          },
          {
            "name": "Concat / PartOpt / cpu / BothRev",
            "value": 0.00001985,
            "unit": "s"
          },
          {
            "name": "Concat / IPartOpt / cpu / PreRev",
            "value": 0.000019848,
            "unit": "s"
          },
          {
            "name": "Concat / IPartOpt / cpu / PostRev",
            "value": 0.000020322,
            "unit": "s"
          },
          {
            "name": "Concat / IPartOpt / cpu / BothRev",
            "value": 0.000019428,
            "unit": "s"
          },
          {
            "name": "Concat / DefOpt / cpu / PreRev",
            "value": 0.00002051,
            "unit": "s"
          },
          {
            "name": "Concat / DefOpt / cpu / PostRev",
            "value": 0.000020146,
            "unit": "s"
          },
          {
            "name": "Concat / DefOpt / cpu / BothRev",
            "value": 0.000020151,
            "unit": "s"
          },
          {
            "name": "Concat / IDefOpt / cpu / PreRev",
            "value": 0.000019927,
            "unit": "s"
          },
          {
            "name": "Concat / IDefOpt / cpu / PostRev",
            "value": 0.000019716,
            "unit": "s"
          },
          {
            "name": "Concat / IDefOpt / cpu / BothRev",
            "value": 0.000019877,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / Primal",
            "value": 0.000008,
            "unit": "s"
          },
          {
            "name": "Concat / Jax / cpu / Primal",
            "value": 0.000008999999999999999,
            "unit": "s"
          },
          {
            "name": "Concat / HLOOpt / cpu / Primal",
            "value": 0.000008999999999999999,
            "unit": "s"
          },
          {
            "name": "Concat / PartOpt / cpu / Primal",
            "value": 0.000008,
            "unit": "s"
          },
          {
            "name": "Concat / IPartOpt / cpu / Primal",
            "value": 0.000008,
            "unit": "s"
          },
          {
            "name": "Concat / DefOpt / cpu / Primal",
            "value": 0.000008,
            "unit": "s"
          },
          {
            "name": "Concat / IDefOpt / cpu / Primal",
            "value": 0.000008,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / Forward",
            "value": 0.000011,
            "unit": "s"
          },
          {
            "name": "Concat / Jax / cpu / Forward",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "Concat / HLOOpt / cpu / Forward",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "Concat / PartOpt / cpu / Forward",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "Concat / IPartOpt / cpu / Forward",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "Concat / DefOpt / cpu / Forward",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "Concat / IDefOpt / cpu / Forward",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / PreRev",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / PostRev",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / BothRev",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "Concat / Jax / cpu / BothRev",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "Concat / HLOOpt / cpu / PreRev",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "Concat / HLOOpt / cpu / PostRev",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "Concat / HLOOpt / cpu / BothRev",
            "value": 0.000015,
            "unit": "s"
          },
          {
            "name": "Concat / PartOpt / cpu / PreRev",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "Concat / PartOpt / cpu / PostRev",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "Concat / PartOpt / cpu / BothRev",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "Concat / IPartOpt / cpu / PreRev",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "Concat / IPartOpt / cpu / PostRev",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "Concat / IPartOpt / cpu / BothRev",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "Concat / DefOpt / cpu / PreRev",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "Concat / DefOpt / cpu / PostRev",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "Concat / DefOpt / cpu / BothRev",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "Concat / IDefOpt / cpu / PreRev",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "Concat / IDefOpt / cpu / PostRev",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "Concat / IDefOpt / cpu / BothRev",
            "value": 0.000015,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / cpu / Primal",
            "value": 0.000006842320008217939,
            "unit": "s"
          },
          {
            "name": "const_scatter / Jax / cpu / Primal",
            "value": 0.000007548859975941014,
            "unit": "s"
          },
          {
            "name": "const_scatter / HLOOpt / cpu / Primal",
            "value": 0.000008170320033968893,
            "unit": "s"
          },
          {
            "name": "const_scatter / PartOpt / cpu / Primal",
            "value": 0.000006784740025977953,
            "unit": "s"
          },
          {
            "name": "const_scatter / IPartOpt / cpu / Primal",
            "value": 0.000007173680005507777,
            "unit": "s"
          },
          {
            "name": "const_scatter / DefOpt / cpu / Primal",
            "value": 0.000007593259997520363,
            "unit": "s"
          },
          {
            "name": "const_scatter / IDefOpt / cpu / Primal",
            "value": 0.000007684459988013259,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / cpu / Forward",
            "value": 0.000011905840028703095,
            "unit": "s"
          },
          {
            "name": "const_scatter / Jax / cpu / Forward",
            "value": 0.000011410720007916098,
            "unit": "s"
          },
          {
            "name": "const_scatter / HLOOpt / cpu / Forward",
            "value": 0.000012375200003589271,
            "unit": "s"
          },
          {
            "name": "const_scatter / PartOpt / cpu / Forward",
            "value": 0.0000117505800062645,
            "unit": "s"
          },
          {
            "name": "const_scatter / IPartOpt / cpu / Forward",
            "value": 0.0000122887799898308,
            "unit": "s"
          },
          {
            "name": "const_scatter / DefOpt / cpu / Forward",
            "value": 0.0000120592799885344,
            "unit": "s"
          },
          {
            "name": "const_scatter / IDefOpt / cpu / Forward",
            "value": 0.000011614740005825297,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / cpu / PreRev",
            "value": 0.0002893409799889,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / cpu / PostRev",
            "value": 0.0002851822799402,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / cpu / BothRev",
            "value": 0.0002877293199799,
            "unit": "s"
          },
          {
            "name": "const_scatter / Jax / cpu / BothRev",
            "value": 0.0002843011199638,
            "unit": "s"
          },
          {
            "name": "const_scatter / HLOOpt / cpu / PreRev",
            "value": 0.0002843069400023,
            "unit": "s"
          },
          {
            "name": "const_scatter / HLOOpt / cpu / PostRev",
            "value": 0.000288153240026,
            "unit": "s"
          },
          {
            "name": "const_scatter / HLOOpt / cpu / BothRev",
            "value": 0.0002838666400111,
            "unit": "s"
          },
          {
            "name": "const_scatter / PartOpt / cpu / PreRev",
            "value": 0.0002882542400038,
            "unit": "s"
          },
          {
            "name": "const_scatter / PartOpt / cpu / PostRev",
            "value": 0.0002828643999964,
            "unit": "s"
          },
          {
            "name": "const_scatter / PartOpt / cpu / BothRev",
            "value": 0.0002836606200071,
            "unit": "s"
          },
          {
            "name": "const_scatter / IPartOpt / cpu / PreRev",
            "value": 0.0002861782600211,
            "unit": "s"
          },
          {
            "name": "const_scatter / IPartOpt / cpu / PostRev",
            "value": 0.0002866531799645,
            "unit": "s"
          },
          {
            "name": "const_scatter / IPartOpt / cpu / BothRev",
            "value": 0.0002857870600018,
            "unit": "s"
          },
          {
            "name": "const_scatter / DefOpt / cpu / PreRev",
            "value": 0.0002845654799966,
            "unit": "s"
          },
          {
            "name": "const_scatter / DefOpt / cpu / PostRev",
            "value": 0.0002846889000102,
            "unit": "s"
          },
          {
            "name": "const_scatter / DefOpt / cpu / BothRev",
            "value": 0.0002862369200192,
            "unit": "s"
          },
          {
            "name": "const_scatter / IDefOpt / cpu / PreRev",
            "value": 0.000284361999993,
            "unit": "s"
          },
          {
            "name": "const_scatter / IDefOpt / cpu / PostRev",
            "value": 0.0002850745399882,
            "unit": "s"
          },
          {
            "name": "const_scatter / IDefOpt / cpu / BothRev",
            "value": 0.0002843939999911,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / cuda / Primal",
            "value": 0.000001887,
            "unit": "s"
          },
          {
            "name": "const_scatter / Jax / cuda / Primal",
            "value": 0.000001887,
            "unit": "s"
          },
          {
            "name": "const_scatter / HLOOpt / cuda / Primal",
            "value": 0.000001887,
            "unit": "s"
          },
          {
            "name": "const_scatter / PartOpt / cuda / Primal",
            "value": 0.000001887,
            "unit": "s"
          },
          {
            "name": "const_scatter / IPartOpt / cuda / Primal",
            "value": 0.000001887,
            "unit": "s"
          },
          {
            "name": "const_scatter / DefOpt / cuda / Primal",
            "value": 0.000001887,
            "unit": "s"
          },
          {
            "name": "const_scatter / IDefOpt / cuda / Primal",
            "value": 0.000001887,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / cuda / Forward",
            "value": 0.000009472,
            "unit": "s"
          },
          {
            "name": "const_scatter / Jax / cuda / Forward",
            "value": 0.000010047,
            "unit": "s"
          },
          {
            "name": "const_scatter / HLOOpt / cuda / Forward",
            "value": 0.000008736,
            "unit": "s"
          },
          {
            "name": "const_scatter / PartOpt / cuda / Forward",
            "value": 0.000009919,
            "unit": "s"
          },
          {
            "name": "const_scatter / IPartOpt / cuda / Forward",
            "value": 0.000009599,
            "unit": "s"
          },
          {
            "name": "const_scatter / DefOpt / cuda / Forward",
            "value": 0.000009472,
            "unit": "s"
          },
          {
            "name": "const_scatter / IDefOpt / cuda / Forward",
            "value": 0.000009952,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / cuda / PreRev",
            "value": 0.00001632,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / cuda / PostRev",
            "value": 0.000015904000000000002,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / cuda / BothRev",
            "value": 0.000015872,
            "unit": "s"
          },
          {
            "name": "const_scatter / Jax / cuda / BothRev",
            "value": 0.000016096,
            "unit": "s"
          },
          {
            "name": "const_scatter / HLOOpt / cuda / PreRev",
            "value": 0.000015968,
            "unit": "s"
          },
          {
            "name": "const_scatter / HLOOpt / cuda / PostRev",
            "value": 0.000016,
            "unit": "s"
          },
          {
            "name": "const_scatter / HLOOpt / cuda / BothRev",
            "value": 0.000015776,
            "unit": "s"
          },
          {
            "name": "const_scatter / PartOpt / cuda / PreRev",
            "value": 0.000016192,
            "unit": "s"
          },
          {
            "name": "const_scatter / PartOpt / cuda / PostRev",
            "value": 0.000015743,
            "unit": "s"
          },
          {
            "name": "const_scatter / PartOpt / cuda / BothRev",
            "value": 0.000016,
            "unit": "s"
          },
          {
            "name": "const_scatter / IPartOpt / cuda / PreRev",
            "value": 0.000016128,
            "unit": "s"
          },
          {
            "name": "const_scatter / IPartOpt / cuda / PostRev",
            "value": 0.000015968,
            "unit": "s"
          },
          {
            "name": "const_scatter / IPartOpt / cuda / BothRev",
            "value": 0.000015776,
            "unit": "s"
          },
          {
            "name": "const_scatter / DefOpt / cuda / PreRev",
            "value": 0.000016,
            "unit": "s"
          },
          {
            "name": "const_scatter / DefOpt / cuda / PostRev",
            "value": 0.000015424,
            "unit": "s"
          },
          {
            "name": "const_scatter / DefOpt / cuda / BothRev",
            "value": 0.000015904000000000002,
            "unit": "s"
          },
          {
            "name": "const_scatter / IDefOpt / cuda / PreRev",
            "value": 0.0000152,
            "unit": "s"
          },
          {
            "name": "const_scatter / IDefOpt / cuda / PostRev",
            "value": 0.000014848,
            "unit": "s"
          },
          {
            "name": "const_scatter / IDefOpt / cuda / BothRev",
            "value": 0.000015648,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / tpu / Primal",
            "value": 0.000003797925,
            "unit": "s"
          },
          {
            "name": "const_scatter / Jax / tpu / Primal",
            "value": 0.000003836575,
            "unit": "s"
          },
          {
            "name": "const_scatter / HLOOpt / tpu / Primal",
            "value": 0.0000037955,
            "unit": "s"
          },
          {
            "name": "const_scatter / PartOpt / tpu / Primal",
            "value": 0.000003830325,
            "unit": "s"
          },
          {
            "name": "const_scatter / IPartOpt / tpu / Primal",
            "value": 0.0000037967,
            "unit": "s"
          },
          {
            "name": "const_scatter / DefOpt / tpu / Primal",
            "value": 0.0000038172000000000005,
            "unit": "s"
          },
          {
            "name": "const_scatter / IDefOpt / tpu / Primal",
            "value": 0.000003794975,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / tpu / Forward",
            "value": 0.000006446174999999999,
            "unit": "s"
          },
          {
            "name": "const_scatter / Jax / tpu / Forward",
            "value": 0.0000064799250000000005,
            "unit": "s"
          },
          {
            "name": "const_scatter / HLOOpt / tpu / Forward",
            "value": 0.00000645555,
            "unit": "s"
          },
          {
            "name": "const_scatter / PartOpt / tpu / Forward",
            "value": 0.000006470350000000001,
            "unit": "s"
          },
          {
            "name": "const_scatter / IPartOpt / tpu / Forward",
            "value": 0.000006477775,
            "unit": "s"
          },
          {
            "name": "const_scatter / DefOpt / tpu / Forward",
            "value": 0.000006482825,
            "unit": "s"
          },
          {
            "name": "const_scatter / IDefOpt / tpu / Forward",
            "value": 0.00000645875,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / tpu / PreRev",
            "value": 0.000006622525000000001,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / tpu / PostRev",
            "value": 0.000006631975,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / tpu / BothRev",
            "value": 0.000006599350000000001,
            "unit": "s"
          },
          {
            "name": "const_scatter / Jax / tpu / BothRev",
            "value": 0.0000066286750000000005,
            "unit": "s"
          },
          {
            "name": "const_scatter / HLOOpt / tpu / PreRev",
            "value": 0.00000662385,
            "unit": "s"
          },
          {
            "name": "const_scatter / HLOOpt / tpu / PostRev",
            "value": 0.000006609475,
            "unit": "s"
          },
          {
            "name": "const_scatter / HLOOpt / tpu / BothRev",
            "value": 0.000006623925,
            "unit": "s"
          },
          {
            "name": "const_scatter / PartOpt / tpu / PreRev",
            "value": 0.000006621200000000001,
            "unit": "s"
          },
          {
            "name": "const_scatter / PartOpt / tpu / PostRev",
            "value": 0.00000661575,
            "unit": "s"
          },
          {
            "name": "const_scatter / PartOpt / tpu / BothRev",
            "value": 0.00000662395,
            "unit": "s"
          },
          {
            "name": "const_scatter / IPartOpt / tpu / PreRev",
            "value": 0.000006612,
            "unit": "s"
          },
          {
            "name": "const_scatter / IPartOpt / tpu / PostRev",
            "value": 0.0000066036,
            "unit": "s"
          },
          {
            "name": "const_scatter / IPartOpt / tpu / BothRev",
            "value": 0.0000066035750000000006,
            "unit": "s"
          },
          {
            "name": "const_scatter / DefOpt / tpu / PreRev",
            "value": 0.000006590049999999999,
            "unit": "s"
          },
          {
            "name": "const_scatter / DefOpt / tpu / PostRev",
            "value": 0.0000065854,
            "unit": "s"
          },
          {
            "name": "const_scatter / DefOpt / tpu / BothRev",
            "value": 0.0000066295,
            "unit": "s"
          },
          {
            "name": "const_scatter / IDefOpt / tpu / PreRev",
            "value": 0.00000660685,
            "unit": "s"
          },
          {
            "name": "const_scatter / IDefOpt / tpu / PostRev",
            "value": 0.00000661475,
            "unit": "s"
          },
          {
            "name": "const_scatter / IDefOpt / tpu / BothRev",
            "value": 0.000006601825,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / cpu / Primal",
            "value": 0.000012794,
            "unit": "s"
          },
          {
            "name": "const_scatter / Jax / cpu / Primal",
            "value": 0.000012857,
            "unit": "s"
          },
          {
            "name": "const_scatter / HLOOpt / cpu / Primal",
            "value": 0.000013306,
            "unit": "s"
          },
          {
            "name": "const_scatter / PartOpt / cpu / Primal",
            "value": 0.00001279,
            "unit": "s"
          },
          {
            "name": "const_scatter / IPartOpt / cpu / Primal",
            "value": 0.00001258,
            "unit": "s"
          },
          {
            "name": "const_scatter / DefOpt / cpu / Primal",
            "value": 0.000013652,
            "unit": "s"
          },
          {
            "name": "const_scatter / IDefOpt / cpu / Primal",
            "value": 0.000013738,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / cpu / Forward",
            "value": 0.000018742,
            "unit": "s"
          },
          {
            "name": "const_scatter / Jax / cpu / Forward",
            "value": 0.000016802999999999998,
            "unit": "s"
          },
          {
            "name": "const_scatter / HLOOpt / cpu / Forward",
            "value": 0.000018442,
            "unit": "s"
          },
          {
            "name": "const_scatter / PartOpt / cpu / Forward",
            "value": 0.000018102,
            "unit": "s"
          },
          {
            "name": "const_scatter / IPartOpt / cpu / Forward",
            "value": 0.000018083,
            "unit": "s"
          },
          {
            "name": "const_scatter / DefOpt / cpu / Forward",
            "value": 0.000017436,
            "unit": "s"
          },
          {
            "name": "const_scatter / IDefOpt / cpu / Forward",
            "value": 0.00001789,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / cpu / PreRev",
            "value": 0.0005180909999999,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / cpu / PostRev",
            "value": 0.000515389,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / cpu / BothRev",
            "value": 0.00051198,
            "unit": "s"
          },
          {
            "name": "const_scatter / Jax / cpu / BothRev",
            "value": 0.0005073839999999,
            "unit": "s"
          },
          {
            "name": "const_scatter / HLOOpt / cpu / PreRev",
            "value": 0.000528244,
            "unit": "s"
          },
          {
            "name": "const_scatter / HLOOpt / cpu / PostRev",
            "value": 0.000522279,
            "unit": "s"
          },
          {
            "name": "const_scatter / HLOOpt / cpu / BothRev",
            "value": 0.000544146,
            "unit": "s"
          },
          {
            "name": "const_scatter / PartOpt / cpu / PreRev",
            "value": 0.000527919,
            "unit": "s"
          },
          {
            "name": "const_scatter / PartOpt / cpu / PostRev",
            "value": 0.000525195,
            "unit": "s"
          },
          {
            "name": "const_scatter / PartOpt / cpu / BothRev",
            "value": 0.00052177,
            "unit": "s"
          },
          {
            "name": "const_scatter / IPartOpt / cpu / PreRev",
            "value": 0.0005198,
            "unit": "s"
          },
          {
            "name": "const_scatter / IPartOpt / cpu / PostRev",
            "value": 0.000518424,
            "unit": "s"
          },
          {
            "name": "const_scatter / IPartOpt / cpu / BothRev",
            "value": 0.000519773,
            "unit": "s"
          },
          {
            "name": "const_scatter / DefOpt / cpu / PreRev",
            "value": 0.000529995,
            "unit": "s"
          },
          {
            "name": "const_scatter / DefOpt / cpu / PostRev",
            "value": 0.0005208669999999,
            "unit": "s"
          },
          {
            "name": "const_scatter / DefOpt / cpu / BothRev",
            "value": 0.000506916,
            "unit": "s"
          },
          {
            "name": "const_scatter / IDefOpt / cpu / PreRev",
            "value": 0.000515161,
            "unit": "s"
          },
          {
            "name": "const_scatter / IDefOpt / cpu / PostRev",
            "value": 0.000512016,
            "unit": "s"
          },
          {
            "name": "const_scatter / IDefOpt / cpu / BothRev",
            "value": 0.000529545,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / cpu / Primal",
            "value": 0.000008999999999999999,
            "unit": "s"
          },
          {
            "name": "const_scatter / Jax / cpu / Primal",
            "value": 0.000008,
            "unit": "s"
          },
          {
            "name": "const_scatter / HLOOpt / cpu / Primal",
            "value": 0.000008999999999999999,
            "unit": "s"
          },
          {
            "name": "const_scatter / PartOpt / cpu / Primal",
            "value": 0.000008999999999999999,
            "unit": "s"
          },
          {
            "name": "const_scatter / IPartOpt / cpu / Primal",
            "value": 0.000008999999999999999,
            "unit": "s"
          },
          {
            "name": "const_scatter / DefOpt / cpu / Primal",
            "value": 0.000008999999999999999,
            "unit": "s"
          },
          {
            "name": "const_scatter / IDefOpt / cpu / Primal",
            "value": 0.000008999999999999999,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / cpu / Forward",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "const_scatter / Jax / cpu / Forward",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "const_scatter / HLOOpt / cpu / Forward",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "const_scatter / PartOpt / cpu / Forward",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "const_scatter / IPartOpt / cpu / Forward",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "const_scatter / DefOpt / cpu / Forward",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "const_scatter / IDefOpt / cpu / Forward",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / cpu / PreRev",
            "value": 0.0003619999999999,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / cpu / PostRev",
            "value": 0.000333,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / cpu / BothRev",
            "value": 0.000328,
            "unit": "s"
          },
          {
            "name": "const_scatter / Jax / cpu / BothRev",
            "value": 0.000325,
            "unit": "s"
          },
          {
            "name": "const_scatter / HLOOpt / cpu / PreRev",
            "value": 0.000326,
            "unit": "s"
          },
          {
            "name": "const_scatter / HLOOpt / cpu / PostRev",
            "value": 0.000325,
            "unit": "s"
          },
          {
            "name": "const_scatter / HLOOpt / cpu / BothRev",
            "value": 0.00033,
            "unit": "s"
          },
          {
            "name": "const_scatter / PartOpt / cpu / PreRev",
            "value": 0.000324,
            "unit": "s"
          },
          {
            "name": "const_scatter / PartOpt / cpu / PostRev",
            "value": 0.000371,
            "unit": "s"
          },
          {
            "name": "const_scatter / PartOpt / cpu / BothRev",
            "value": 0.000333,
            "unit": "s"
          },
          {
            "name": "const_scatter / IPartOpt / cpu / PreRev",
            "value": 0.00033,
            "unit": "s"
          },
          {
            "name": "const_scatter / IPartOpt / cpu / PostRev",
            "value": 0.0003459999999999,
            "unit": "s"
          },
          {
            "name": "const_scatter / IPartOpt / cpu / BothRev",
            "value": 0.000361,
            "unit": "s"
          },
          {
            "name": "const_scatter / DefOpt / cpu / PreRev",
            "value": 0.000368,
            "unit": "s"
          },
          {
            "name": "const_scatter / DefOpt / cpu / PostRev",
            "value": 0.000329,
            "unit": "s"
          },
          {
            "name": "const_scatter / DefOpt / cpu / BothRev",
            "value": 0.00034,
            "unit": "s"
          },
          {
            "name": "const_scatter / IDefOpt / cpu / PreRev",
            "value": 0.000313,
            "unit": "s"
          },
          {
            "name": "const_scatter / IDefOpt / cpu / PostRev",
            "value": 0.000305,
            "unit": "s"
          },
          {
            "name": "const_scatter / IDefOpt / cpu / BothRev",
            "value": 0.000313,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / Primal",
            "value": 0.000009136419994320022,
            "unit": "s"
          },
          {
            "name": "GenDot / Jax / cpu / Primal",
            "value": 0.000008939100025600055,
            "unit": "s"
          },
          {
            "name": "GenDot / HLOOpt / cpu / Primal",
            "value": 0.00000937842002713296,
            "unit": "s"
          },
          {
            "name": "GenDot / PartOpt / cpu / Primal",
            "value": 0.00000840610002342146,
            "unit": "s"
          },
          {
            "name": "GenDot / IPartOpt / cpu / Primal",
            "value": 0.000008393619991693412,
            "unit": "s"
          },
          {
            "name": "GenDot / DefOpt / cpu / Primal",
            "value": 0.000008787400020082714,
            "unit": "s"
          },
          {
            "name": "GenDot / IDefOpt / cpu / Primal",
            "value": 0.000008077780030362192,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / Forward",
            "value": 0.00001278371995795169,
            "unit": "s"
          },
          {
            "name": "GenDot / Jax / cpu / Forward",
            "value": 0.00001215408001371543,
            "unit": "s"
          },
          {
            "name": "GenDot / HLOOpt / cpu / Forward",
            "value": 0.000013325959980647894,
            "unit": "s"
          },
          {
            "name": "GenDot / PartOpt / cpu / Forward",
            "value": 0.000013168640016374411,
            "unit": "s"
          },
          {
            "name": "GenDot / IPartOpt / cpu / Forward",
            "value": 0.000012963319986738498,
            "unit": "s"
          },
          {
            "name": "GenDot / DefOpt / cpu / Forward",
            "value": 0.00001290447997234878,
            "unit": "s"
          },
          {
            "name": "GenDot / IDefOpt / cpu / Forward",
            "value": 0.000012347279953246471,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / PreRev",
            "value": 0.000012486799978432827,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / PostRev",
            "value": 0.000010974139977406594,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / BothRev",
            "value": 0.000013244699966890038,
            "unit": "s"
          },
          {
            "name": "GenDot / Jax / cpu / BothRev",
            "value": 0.000011685120052788988,
            "unit": "s"
          },
          {
            "name": "GenDot / HLOOpt / cpu / PreRev",
            "value": 0.00001274009998269321,
            "unit": "s"
          },
          {
            "name": "GenDot / HLOOpt / cpu / PostRev",
            "value": 0.000015062699985719518,
            "unit": "s"
          },
          {
            "name": "GenDot / HLOOpt / cpu / BothRev",
            "value": 0.000012676980022661156,
            "unit": "s"
          },
          {
            "name": "GenDot / PartOpt / cpu / PreRev",
            "value": 0.000012681280022661667,
            "unit": "s"
          },
          {
            "name": "GenDot / PartOpt / cpu / PostRev",
            "value": 0.000011119320033685652,
            "unit": "s"
          },
          {
            "name": "GenDot / PartOpt / cpu / BothRev",
            "value": 0.0000128565000341041,
            "unit": "s"
          },
          {
            "name": "GenDot / IPartOpt / cpu / PreRev",
            "value": 0.000012805879996449222,
            "unit": "s"
          },
          {
            "name": "GenDot / IPartOpt / cpu / PostRev",
            "value": 0.000011597299971981556,
            "unit": "s"
          },
          {
            "name": "GenDot / IPartOpt / cpu / BothRev",
            "value": 0.000013051699979769185,
            "unit": "s"
          },
          {
            "name": "GenDot / DefOpt / cpu / PreRev",
            "value": 0.000012490140024965512,
            "unit": "s"
          },
          {
            "name": "GenDot / DefOpt / cpu / PostRev",
            "value": 0.000012669499974435891,
            "unit": "s"
          },
          {
            "name": "GenDot / DefOpt / cpu / BothRev",
            "value": 0.000012742800045089098,
            "unit": "s"
          },
          {
            "name": "GenDot / IDefOpt / cpu / PreRev",
            "value": 0.000012604300027305724,
            "unit": "s"
          },
          {
            "name": "GenDot / IDefOpt / cpu / PostRev",
            "value": 0.000012143400017521344,
            "unit": "s"
          },
          {
            "name": "GenDot / IDefOpt / cpu / BothRev",
            "value": 0.000012777719985024304,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cuda / Primal",
            "value": 0.000002016,
            "unit": "s"
          },
          {
            "name": "GenDot / Jax / cuda / Primal",
            "value": 0.000002016,
            "unit": "s"
          },
          {
            "name": "GenDot / HLOOpt / cuda / Primal",
            "value": 0.000001984,
            "unit": "s"
          },
          {
            "name": "GenDot / PartOpt / cuda / Primal",
            "value": 0.000002015,
            "unit": "s"
          },
          {
            "name": "GenDot / IPartOpt / cuda / Primal",
            "value": 0.000002015,
            "unit": "s"
          },
          {
            "name": "GenDot / DefOpt / cuda / Primal",
            "value": 0.000002016,
            "unit": "s"
          },
          {
            "name": "GenDot / IDefOpt / cuda / Primal",
            "value": 0.000002015,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cuda / Forward",
            "value": 0.00001008,
            "unit": "s"
          },
          {
            "name": "GenDot / Jax / cuda / Forward",
            "value": 0.000009728,
            "unit": "s"
          },
          {
            "name": "GenDot / HLOOpt / cuda / Forward",
            "value": 0.000009856,
            "unit": "s"
          },
          {
            "name": "GenDot / PartOpt / cuda / Forward",
            "value": 0.000009791,
            "unit": "s"
          },
          {
            "name": "GenDot / IPartOpt / cuda / Forward",
            "value": 0.000009696,
            "unit": "s"
          },
          {
            "name": "GenDot / DefOpt / cuda / Forward",
            "value": 0.00000944,
            "unit": "s"
          },
          {
            "name": "GenDot / IDefOpt / cuda / Forward",
            "value": 0.00000928,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cuda / PreRev",
            "value": 0.000009568,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cuda / PostRev",
            "value": 0.000009952,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cuda / BothRev",
            "value": 0.000009888,
            "unit": "s"
          },
          {
            "name": "GenDot / Jax / cuda / BothRev",
            "value": 0.000009856,
            "unit": "s"
          },
          {
            "name": "GenDot / HLOOpt / cuda / PreRev",
            "value": 0.000009856,
            "unit": "s"
          },
          {
            "name": "GenDot / HLOOpt / cuda / PostRev",
            "value": 0.000009376,
            "unit": "s"
          },
          {
            "name": "GenDot / HLOOpt / cuda / BothRev",
            "value": 0.000010048,
            "unit": "s"
          },
          {
            "name": "GenDot / PartOpt / cuda / PreRev",
            "value": 0.000009728,
            "unit": "s"
          },
          {
            "name": "GenDot / PartOpt / cuda / PostRev",
            "value": 0.000009664,
            "unit": "s"
          },
          {
            "name": "GenDot / PartOpt / cuda / BothRev",
            "value": 0.000009632,
            "unit": "s"
          },
          {
            "name": "GenDot / IPartOpt / cuda / PreRev",
            "value": 0.000009632,
            "unit": "s"
          },
          {
            "name": "GenDot / IPartOpt / cuda / PostRev",
            "value": 0.000009504,
            "unit": "s"
          },
          {
            "name": "GenDot / IPartOpt / cuda / BothRev",
            "value": 0.000009536,
            "unit": "s"
          },
          {
            "name": "GenDot / DefOpt / cuda / PreRev",
            "value": 0.000009536,
            "unit": "s"
          },
          {
            "name": "GenDot / DefOpt / cuda / PostRev",
            "value": 0.000009472,
            "unit": "s"
          },
          {
            "name": "GenDot / DefOpt / cuda / BothRev",
            "value": 0.000009793,
            "unit": "s"
          },
          {
            "name": "GenDot / IDefOpt / cuda / PreRev",
            "value": 0.00000928,
            "unit": "s"
          },
          {
            "name": "GenDot / IDefOpt / cuda / PostRev",
            "value": 0.000009759,
            "unit": "s"
          },
          {
            "name": "GenDot / IDefOpt / cuda / BothRev",
            "value": 0.000009888,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / tpu / Primal",
            "value": 9.3015e-7,
            "unit": "s"
          },
          {
            "name": "GenDot / Jax / tpu / Primal",
            "value": 9.2575e-7,
            "unit": "s"
          },
          {
            "name": "GenDot / HLOOpt / tpu / Primal",
            "value": 0.000001571925,
            "unit": "s"
          },
          {
            "name": "GenDot / PartOpt / tpu / Primal",
            "value": 9.25575e-7,
            "unit": "s"
          },
          {
            "name": "GenDot / IPartOpt / tpu / Primal",
            "value": 9.30375e-7,
            "unit": "s"
          },
          {
            "name": "GenDot / DefOpt / tpu / Primal",
            "value": 0.000001487025,
            "unit": "s"
          },
          {
            "name": "GenDot / IDefOpt / tpu / Primal",
            "value": 0.000001566975,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / tpu / Forward",
            "value": 0.000003157,
            "unit": "s"
          },
          {
            "name": "GenDot / Jax / tpu / Forward",
            "value": 0.000002316875,
            "unit": "s"
          },
          {
            "name": "GenDot / HLOOpt / tpu / Forward",
            "value": 0.0000031079000000000003,
            "unit": "s"
          },
          {
            "name": "GenDot / PartOpt / tpu / Forward",
            "value": 0.0000032048500000000003,
            "unit": "s"
          },
          {
            "name": "GenDot / IPartOpt / tpu / Forward",
            "value": 0.0000031134250000000006,
            "unit": "s"
          },
          {
            "name": "GenDot / DefOpt / tpu / Forward",
            "value": 0.00000321465,
            "unit": "s"
          },
          {
            "name": "GenDot / IDefOpt / tpu / Forward",
            "value": 0.00000311865,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / tpu / PreRev",
            "value": 0.0000029508,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / tpu / PostRev",
            "value": 0.0000023988,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / tpu / BothRev",
            "value": 0.00000295575,
            "unit": "s"
          },
          {
            "name": "GenDot / Jax / tpu / BothRev",
            "value": 0.00000240435,
            "unit": "s"
          },
          {
            "name": "GenDot / HLOOpt / tpu / PreRev",
            "value": 0.000002953375,
            "unit": "s"
          },
          {
            "name": "GenDot / HLOOpt / tpu / PostRev",
            "value": 0.000002926775,
            "unit": "s"
          },
          {
            "name": "GenDot / HLOOpt / tpu / BothRev",
            "value": 0.0000029604,
            "unit": "s"
          },
          {
            "name": "GenDot / PartOpt / tpu / PreRev",
            "value": 0.000002933975,
            "unit": "s"
          },
          {
            "name": "GenDot / PartOpt / tpu / PostRev",
            "value": 0.000002391025,
            "unit": "s"
          },
          {
            "name": "GenDot / PartOpt / tpu / BothRev",
            "value": 0.0000029292000000000003,
            "unit": "s"
          },
          {
            "name": "GenDot / IPartOpt / tpu / PreRev",
            "value": 0.0000029615000000000004,
            "unit": "s"
          },
          {
            "name": "GenDot / IPartOpt / tpu / PostRev",
            "value": 0.000002405875,
            "unit": "s"
          },
          {
            "name": "GenDot / IPartOpt / tpu / BothRev",
            "value": 0.000002958875,
            "unit": "s"
          },
          {
            "name": "GenDot / DefOpt / tpu / PreRev",
            "value": 0.00000293265,
            "unit": "s"
          },
          {
            "name": "GenDot / DefOpt / tpu / PostRev",
            "value": 0.0000029616749999999995,
            "unit": "s"
          },
          {
            "name": "GenDot / DefOpt / tpu / BothRev",
            "value": 0.000002943025,
            "unit": "s"
          },
          {
            "name": "GenDot / IDefOpt / tpu / PreRev",
            "value": 0.00000296195,
            "unit": "s"
          },
          {
            "name": "GenDot / IDefOpt / tpu / PostRev",
            "value": 0.000002926125,
            "unit": "s"
          },
          {
            "name": "GenDot / IDefOpt / tpu / BothRev",
            "value": 0.000002955175,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / Primal",
            "value": 0.000015037,
            "unit": "s"
          },
          {
            "name": "GenDot / Jax / cpu / Primal",
            "value": 0.000014573,
            "unit": "s"
          },
          {
            "name": "GenDot / HLOOpt / cpu / Primal",
            "value": 0.000013947,
            "unit": "s"
          },
          {
            "name": "GenDot / PartOpt / cpu / Primal",
            "value": 0.000014888,
            "unit": "s"
          },
          {
            "name": "GenDot / IPartOpt / cpu / Primal",
            "value": 0.000014724,
            "unit": "s"
          },
          {
            "name": "GenDot / DefOpt / cpu / Primal",
            "value": 0.000014155,
            "unit": "s"
          },
          {
            "name": "GenDot / IDefOpt / cpu / Primal",
            "value": 0.000013929,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / Forward",
            "value": 0.000019464,
            "unit": "s"
          },
          {
            "name": "GenDot / Jax / cpu / Forward",
            "value": 0.000020421000000000003,
            "unit": "s"
          },
          {
            "name": "GenDot / HLOOpt / cpu / Forward",
            "value": 0.000019308,
            "unit": "s"
          },
          {
            "name": "GenDot / PartOpt / cpu / Forward",
            "value": 0.000018865,
            "unit": "s"
          },
          {
            "name": "GenDot / IPartOpt / cpu / Forward",
            "value": 0.000018651,
            "unit": "s"
          },
          {
            "name": "GenDot / DefOpt / cpu / Forward",
            "value": 0.00001896,
            "unit": "s"
          },
          {
            "name": "GenDot / IDefOpt / cpu / Forward",
            "value": 0.000018818,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / PreRev",
            "value": 0.00001992,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / PostRev",
            "value": 0.000019927,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / BothRev",
            "value": 0.000019305,
            "unit": "s"
          },
          {
            "name": "GenDot / Jax / cpu / BothRev",
            "value": 0.000020323,
            "unit": "s"
          },
          {
            "name": "GenDot / HLOOpt / cpu / PreRev",
            "value": 0.000019171,
            "unit": "s"
          },
          {
            "name": "GenDot / HLOOpt / cpu / PostRev",
            "value": 0.000020311,
            "unit": "s"
          },
          {
            "name": "GenDot / HLOOpt / cpu / BothRev",
            "value": 0.000020294,
            "unit": "s"
          },
          {
            "name": "GenDot / PartOpt / cpu / PreRev",
            "value": 0.000019615,
            "unit": "s"
          },
          {
            "name": "GenDot / PartOpt / cpu / PostRev",
            "value": 0.000020374,
            "unit": "s"
          },
          {
            "name": "GenDot / PartOpt / cpu / BothRev",
            "value": 0.000019275,
            "unit": "s"
          },
          {
            "name": "GenDot / IPartOpt / cpu / PreRev",
            "value": 0.000018759,
            "unit": "s"
          },
          {
            "name": "GenDot / IPartOpt / cpu / PostRev",
            "value": 0.000019703,
            "unit": "s"
          },
          {
            "name": "GenDot / IPartOpt / cpu / BothRev",
            "value": 0.000019213,
            "unit": "s"
          },
          {
            "name": "GenDot / DefOpt / cpu / PreRev",
            "value": 0.000019475,
            "unit": "s"
          },
          {
            "name": "GenDot / DefOpt / cpu / PostRev",
            "value": 0.00001932,
            "unit": "s"
          },
          {
            "name": "GenDot / DefOpt / cpu / BothRev",
            "value": 0.000019717,
            "unit": "s"
          },
          {
            "name": "GenDot / IDefOpt / cpu / PreRev",
            "value": 0.000019333,
            "unit": "s"
          },
          {
            "name": "GenDot / IDefOpt / cpu / PostRev",
            "value": 0.000019813,
            "unit": "s"
          },
          {
            "name": "GenDot / IDefOpt / cpu / BothRev",
            "value": 0.000019686,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / Primal",
            "value": 0.00001,
            "unit": "s"
          },
          {
            "name": "GenDot / Jax / cpu / Primal",
            "value": 0.000011,
            "unit": "s"
          },
          {
            "name": "GenDot / HLOOpt / cpu / Primal",
            "value": 0.000008999999999999999,
            "unit": "s"
          },
          {
            "name": "GenDot / PartOpt / cpu / Primal",
            "value": 0.00001,
            "unit": "s"
          },
          {
            "name": "GenDot / IPartOpt / cpu / Primal",
            "value": 0.00001,
            "unit": "s"
          },
          {
            "name": "GenDot / DefOpt / cpu / Primal",
            "value": 0.000008999999999999999,
            "unit": "s"
          },
          {
            "name": "GenDot / IDefOpt / cpu / Primal",
            "value": 0.000008999999999999999,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / Forward",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "GenDot / Jax / cpu / Forward",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "GenDot / HLOOpt / cpu / Forward",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "GenDot / PartOpt / cpu / Forward",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "GenDot / IPartOpt / cpu / Forward",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "GenDot / DefOpt / cpu / Forward",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "GenDot / IDefOpt / cpu / Forward",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / PreRev",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / PostRev",
            "value": 0.000015,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / BothRev",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "GenDot / Jax / cpu / BothRev",
            "value": 0.000015,
            "unit": "s"
          },
          {
            "name": "GenDot / HLOOpt / cpu / PreRev",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "GenDot / HLOOpt / cpu / PostRev",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "GenDot / HLOOpt / cpu / BothRev",
            "value": 0.000015,
            "unit": "s"
          },
          {
            "name": "GenDot / PartOpt / cpu / PreRev",
            "value": 0.000015,
            "unit": "s"
          },
          {
            "name": "GenDot / PartOpt / cpu / PostRev",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "GenDot / PartOpt / cpu / BothRev",
            "value": 0.000015,
            "unit": "s"
          },
          {
            "name": "GenDot / IPartOpt / cpu / PreRev",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "GenDot / IPartOpt / cpu / PostRev",
            "value": 0.000015,
            "unit": "s"
          },
          {
            "name": "GenDot / IPartOpt / cpu / BothRev",
            "value": 0.000015,
            "unit": "s"
          },
          {
            "name": "GenDot / DefOpt / cpu / PreRev",
            "value": 0.000015,
            "unit": "s"
          },
          {
            "name": "GenDot / DefOpt / cpu / PostRev",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "GenDot / DefOpt / cpu / BothRev",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "GenDot / IDefOpt / cpu / PreRev",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "GenDot / IDefOpt / cpu / PostRev",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "GenDot / IDefOpt / cpu / BothRev",
            "value": 0.000015,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / Primal",
            "value": 0.000011769219991037972,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / Jax / cpu / Primal",
            "value": 0.000011390040008336656,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / HLOOpt / cpu / Primal",
            "value": 0.00001107128002331592,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / PartOpt / cpu / Primal",
            "value": 0.000010834639997483464,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IPartOpt / cpu / Primal",
            "value": 0.000011298039999019238,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / DefOpt / cpu / Primal",
            "value": 0.000010621599985825014,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IDefOpt / cpu / Primal",
            "value": 0.000010755800030892714,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / Forward",
            "value": 0.000016757680014052313,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / Jax / cpu / Forward",
            "value": 0.000016056699996624958,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / HLOOpt / cpu / Forward",
            "value": 0.000016304439968735094,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / PartOpt / cpu / Forward",
            "value": 0.000015897820067038992,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IPartOpt / cpu / Forward",
            "value": 0.000016493580014866895,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / DefOpt / cpu / Forward",
            "value": 0.0000166749799700483,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IDefOpt / cpu / Forward",
            "value": 0.00001626992002456973,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / PreRev",
            "value": 0.000016673420032020658,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / PostRev",
            "value": 0.000014827939985480045,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / BothRev",
            "value": 0.00001554261997625872,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / Jax / cpu / BothRev",
            "value": 0.000016992420014503296,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / HLOOpt / cpu / PreRev",
            "value": 0.000016757339963078267,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / HLOOpt / cpu / PostRev",
            "value": 0.00001766755997778091,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / HLOOpt / cpu / BothRev",
            "value": 0.000015197859993349991,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / PartOpt / cpu / PreRev",
            "value": 0.000016054180032369914,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / PartOpt / cpu / PostRev",
            "value": 0.000019289579959149703,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / PartOpt / cpu / BothRev",
            "value": 0.00001536798000415729,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IPartOpt / cpu / PreRev",
            "value": 0.00001672369996413181,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IPartOpt / cpu / PostRev",
            "value": 0.0000154835399644071,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IPartOpt / cpu / BothRev",
            "value": 0.000016247380035565584,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / DefOpt / cpu / PreRev",
            "value": 0.00001578282000082254,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / DefOpt / cpu / PostRev",
            "value": 0.000015375199964182685,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / DefOpt / cpu / BothRev",
            "value": 0.000015243000007103548,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IDefOpt / cpu / PreRev",
            "value": 0.000016734960063331527,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IDefOpt / cpu / PostRev",
            "value": 0.00001536071995360544,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IDefOpt / cpu / BothRev",
            "value": 0.000015463039990208926,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cuda / Primal",
            "value": 0.000001952,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / Jax / cuda / Primal",
            "value": 0.000001952,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / HLOOpt / cuda / Primal",
            "value": 0.000001984,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / PartOpt / cuda / Primal",
            "value": 0.000001983,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IPartOpt / cuda / Primal",
            "value": 0.000001983,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / DefOpt / cuda / Primal",
            "value": 0.000001952,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IDefOpt / cuda / Primal",
            "value": 0.000001952,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cuda / Forward",
            "value": 0.000002047,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / Jax / cuda / Forward",
            "value": 0.000002047,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / HLOOpt / cuda / Forward",
            "value": 0.000002048,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / PartOpt / cuda / Forward",
            "value": 0.000002047,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IPartOpt / cuda / Forward",
            "value": 0.000002047,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / DefOpt / cuda / Forward",
            "value": 0.000002047,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IDefOpt / cuda / Forward",
            "value": 0.000002047,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cuda / PreRev",
            "value": 0.000002047,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cuda / PostRev",
            "value": 0.000002047,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cuda / BothRev",
            "value": 0.000002047,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / Jax / cuda / BothRev",
            "value": 0.000002047,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / HLOOpt / cuda / PreRev",
            "value": 0.000002047,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / HLOOpt / cuda / PostRev",
            "value": 0.000002047,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / HLOOpt / cuda / BothRev",
            "value": 0.000002047,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / PartOpt / cuda / PreRev",
            "value": 0.000002047,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / PartOpt / cuda / PostRev",
            "value": 0.000002047,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / PartOpt / cuda / BothRev",
            "value": 0.000002047,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IPartOpt / cuda / PreRev",
            "value": 0.000002047,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IPartOpt / cuda / PostRev",
            "value": 0.000002047,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IPartOpt / cuda / BothRev",
            "value": 0.000002047,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / DefOpt / cuda / PreRev",
            "value": 0.000002047,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / DefOpt / cuda / PostRev",
            "value": 0.000002047,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / DefOpt / cuda / BothRev",
            "value": 0.000002047,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IDefOpt / cuda / PreRev",
            "value": 0.000002047,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IDefOpt / cuda / PostRev",
            "value": 0.000002047,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IDefOpt / cuda / BothRev",
            "value": 0.000002047,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / tpu / Primal",
            "value": 9.29025e-7,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / Jax / tpu / Primal",
            "value": 9.5445e-7,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / HLOOpt / tpu / Primal",
            "value": 9.07425e-7,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / PartOpt / tpu / Primal",
            "value": 9.53725e-7,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IPartOpt / tpu / Primal",
            "value": 9.1555e-7,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / DefOpt / tpu / Primal",
            "value": 9.55225e-7,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IDefOpt / tpu / Primal",
            "value": 9.09475e-7,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / tpu / Forward",
            "value": 9.492e-7,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / Jax / tpu / Forward",
            "value": 9.81125e-7,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / HLOOpt / tpu / Forward",
            "value": 9.7465e-7,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / PartOpt / tpu / Forward",
            "value": 9.34175e-7,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IPartOpt / tpu / Forward",
            "value": 9.74175e-7,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / DefOpt / tpu / Forward",
            "value": 9.33975e-7,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IDefOpt / tpu / Forward",
            "value": 9.74625e-7,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / tpu / PreRev",
            "value": 9.3795e-7,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / tpu / PostRev",
            "value": 9.657499999999998e-7,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / tpu / BothRev",
            "value": 9.62875e-7,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / Jax / tpu / BothRev",
            "value": 9.65e-7,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / HLOOpt / tpu / PreRev",
            "value": 9.625250000000002e-7,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / HLOOpt / tpu / PostRev",
            "value": 9.647e-7,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / HLOOpt / tpu / BothRev",
            "value": 9.618e-7,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / PartOpt / tpu / PreRev",
            "value": 9.6495e-7,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / PartOpt / tpu / PostRev",
            "value": 9.625749999999998e-7,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / PartOpt / tpu / BothRev",
            "value": 9.65425e-7,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IPartOpt / tpu / PreRev",
            "value": 9.6275e-7,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IPartOpt / tpu / PostRev",
            "value": 9.650749999999998e-7,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IPartOpt / tpu / BothRev",
            "value": 9.62725e-7,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / DefOpt / tpu / PreRev",
            "value": 9.65325e-7,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / DefOpt / tpu / PostRev",
            "value": 9.6215e-7,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / DefOpt / tpu / BothRev",
            "value": 9.652e-7,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IDefOpt / tpu / PreRev",
            "value": 9.624e-7,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IDefOpt / tpu / PostRev",
            "value": 9.654749999999998e-7,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IDefOpt / tpu / BothRev",
            "value": 9.625e-7,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / Primal",
            "value": 0.000017603,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / Jax / cpu / Primal",
            "value": 0.000017128999999999998,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / HLOOpt / cpu / Primal",
            "value": 0.000017082,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / PartOpt / cpu / Primal",
            "value": 0.000017468,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IPartOpt / cpu / Primal",
            "value": 0.000017463,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / DefOpt / cpu / Primal",
            "value": 0.000017175,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IDefOpt / cpu / Primal",
            "value": 0.000017624,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / Forward",
            "value": 0.000024418,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / Jax / cpu / Forward",
            "value": 0.000023137,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / HLOOpt / cpu / Forward",
            "value": 0.00002393,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / PartOpt / cpu / Forward",
            "value": 0.000024218,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IPartOpt / cpu / Forward",
            "value": 0.000023728,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / DefOpt / cpu / Forward",
            "value": 0.000023965,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IDefOpt / cpu / Forward",
            "value": 0.000024078,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / PreRev",
            "value": 0.00002452,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / PostRev",
            "value": 0.000023793,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / BothRev",
            "value": 0.000023555,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / Jax / cpu / BothRev",
            "value": 0.000023691,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / HLOOpt / cpu / PreRev",
            "value": 0.000023938,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / HLOOpt / cpu / PostRev",
            "value": 0.000023471,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / HLOOpt / cpu / BothRev",
            "value": 0.000023371,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / PartOpt / cpu / PreRev",
            "value": 0.000023696,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / PartOpt / cpu / PostRev",
            "value": 0.000023721,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / PartOpt / cpu / BothRev",
            "value": 0.000023895,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IPartOpt / cpu / PreRev",
            "value": 0.000023468,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IPartOpt / cpu / PostRev",
            "value": 0.000023477,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IPartOpt / cpu / BothRev",
            "value": 0.000023546,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / DefOpt / cpu / PreRev",
            "value": 0.000024394,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / DefOpt / cpu / PostRev",
            "value": 0.000023511,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / DefOpt / cpu / BothRev",
            "value": 0.000024214,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IDefOpt / cpu / PreRev",
            "value": 0.000024124,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IDefOpt / cpu / PostRev",
            "value": 0.000023767,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IDefOpt / cpu / BothRev",
            "value": 0.000025183,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / Primal",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / Jax / cpu / Primal",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / HLOOpt / cpu / Primal",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / PartOpt / cpu / Primal",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IPartOpt / cpu / Primal",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / DefOpt / cpu / Primal",
            "value": 0.000011,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IDefOpt / cpu / Primal",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / Forward",
            "value": 0.000017,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / Jax / cpu / Forward",
            "value": 0.000017,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / HLOOpt / cpu / Forward",
            "value": 0.000017,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / PartOpt / cpu / Forward",
            "value": 0.000017,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IPartOpt / cpu / Forward",
            "value": 0.000017999999999999997,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / DefOpt / cpu / Forward",
            "value": 0.000017,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IDefOpt / cpu / Forward",
            "value": 0.000017,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / PreRev",
            "value": 0.000016,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / PostRev",
            "value": 0.000017,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / BothRev",
            "value": 0.000016,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / Jax / cpu / BothRev",
            "value": 0.000017,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / HLOOpt / cpu / PreRev",
            "value": 0.000017,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / HLOOpt / cpu / PostRev",
            "value": 0.000017,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / HLOOpt / cpu / BothRev",
            "value": 0.000017999999999999997,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / PartOpt / cpu / PreRev",
            "value": 0.000017,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / PartOpt / cpu / PostRev",
            "value": 0.000017999999999999997,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / PartOpt / cpu / BothRev",
            "value": 0.000017,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IPartOpt / cpu / PreRev",
            "value": 0.000017,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IPartOpt / cpu / PostRev",
            "value": 0.000017,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IPartOpt / cpu / BothRev",
            "value": 0.000017,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / DefOpt / cpu / PreRev",
            "value": 0.000016,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / DefOpt / cpu / PostRev",
            "value": 0.000017,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / DefOpt / cpu / BothRev",
            "value": 0.000017,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IDefOpt / cpu / PreRev",
            "value": 0.000017,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IDefOpt / cpu / PostRev",
            "value": 0.000016,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IDefOpt / cpu / BothRev",
            "value": 0.000016,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / JaXPipe / cpu / Primal",
            "value": 0.0011267505998148,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / Jax / cpu / Primal",
            "value": 0.0009418846000698,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / HLOOpt / cpu / Primal",
            "value": 0.0010541087999627,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / PartOpt / cpu / Primal",
            "value": 0.0011027717998331,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IPartOpt / cpu / Primal",
            "value": 0.0011475706000965,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / DefOpt / cpu / Primal",
            "value": 0.0011051118000068,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IDefOpt / cpu / Primal",
            "value": 0.0011550227999578,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / JaXPipe / cpu / Forward",
            "value": 0.0026810323999598,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / Jax / cpu / Forward",
            "value": 0.0029001091998907,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / HLOOpt / cpu / Forward",
            "value": 0.0026212021999526,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / PartOpt / cpu / Forward",
            "value": 0.0024966237999251,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IPartOpt / cpu / Forward",
            "value": 0.0025083886000174,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / DefOpt / cpu / Forward",
            "value": 0.0025393892000465,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IDefOpt / cpu / Forward",
            "value": 0.0027567053999518,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / JaXPipe / cpu / PreRev",
            "value": 0.0072784822001267,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / JaXPipe / cpu / PostRev",
            "value": 0.0069289246000153,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / JaXPipe / cpu / BothRev",
            "value": 0.0065746002000196,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / Jax / cpu / BothRev",
            "value": 0.0065450790000795,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / HLOOpt / cpu / PreRev",
            "value": 0.0067159168001126,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / HLOOpt / cpu / PostRev",
            "value": 0.0037599254001179,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / HLOOpt / cpu / BothRev",
            "value": 0.005835021599978,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / PartOpt / cpu / PreRev",
            "value": 0.0061067225999067,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / PartOpt / cpu / PostRev",
            "value": 0.006158757600042,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / PartOpt / cpu / BothRev",
            "value": 0.0063330867998956,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IPartOpt / cpu / PreRev",
            "value": 0.0058289966001211,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IPartOpt / cpu / PostRev",
            "value": 0.0063528508000672,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IPartOpt / cpu / BothRev",
            "value": 0.0056682509999518,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / DefOpt / cpu / PreRev",
            "value": 0.0062407683998571,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / DefOpt / cpu / PostRev",
            "value": 0.005692110000109,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / DefOpt / cpu / BothRev",
            "value": 0.0059548058000473,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IDefOpt / cpu / PreRev",
            "value": 0.005899918400064,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IDefOpt / cpu / PostRev",
            "value": 0.0052036910001334,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IDefOpt / cpu / BothRev",
            "value": 0.0063195202000315,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / JaXPipe / cuda / Primal",
            "value": 0.000284158,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / Jax / cuda / Primal",
            "value": 0.000284094,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / HLOOpt / cuda / Primal",
            "value": 0.000290462,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / PartOpt / cuda / Primal",
            "value": 0.000282494,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IPartOpt / cuda / Primal",
            "value": 0.000283006,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / DefOpt / cuda / Primal",
            "value": 0.000289405,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IDefOpt / cuda / Primal",
            "value": 0.00029075,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / JaXPipe / cuda / Forward",
            "value": 0.000560156,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / Jax / cuda / Forward",
            "value": 0.0005427149999999,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / HLOOpt / cuda / Forward",
            "value": 0.000560763,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / PartOpt / cuda / Forward",
            "value": 0.000560283,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IPartOpt / cuda / Forward",
            "value": 0.000560795,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / DefOpt / cuda / Forward",
            "value": 0.000560348,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IDefOpt / cuda / Forward",
            "value": 0.000561148,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / JaXPipe / cuda / PreRev",
            "value": 0.001034327,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / JaXPipe / cuda / PostRev",
            "value": 0.000992599,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / JaXPipe / cuda / BothRev",
            "value": 0.001026936,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / Jax / cuda / BothRev",
            "value": 0.000997399,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / HLOOpt / cuda / PreRev",
            "value": 0.001018808,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / HLOOpt / cuda / PostRev",
            "value": 0.001043767,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / HLOOpt / cuda / BothRev",
            "value": 0.001016216,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / PartOpt / cuda / PreRev",
            "value": 0.001033015,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / PartOpt / cuda / PostRev",
            "value": 0.000983607,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / PartOpt / cuda / BothRev",
            "value": 0.00103452,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IPartOpt / cuda / PreRev",
            "value": 0.0010326,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IPartOpt / cuda / PostRev",
            "value": 0.000980727,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IPartOpt / cuda / BothRev",
            "value": 0.001032823,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / DefOpt / cuda / PreRev",
            "value": 0.001029592,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / DefOpt / cuda / PostRev",
            "value": 0.0009675759999999,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / DefOpt / cuda / BothRev",
            "value": 0.001031735,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IDefOpt / cuda / PreRev",
            "value": 0.00102364,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IDefOpt / cuda / PostRev",
            "value": 0.00102812,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IDefOpt / cuda / BothRev",
            "value": 0.0010308399999999,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / JaXPipe / tpu / Primal",
            "value": 0.0001236115,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / Jax / tpu / Primal",
            "value": 0.00012627875,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / HLOOpt / tpu / Primal",
            "value": 0.0001527022499999,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / PartOpt / tpu / Primal",
            "value": 0.00013420725,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IPartOpt / tpu / Primal",
            "value": 0.0001311715,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / DefOpt / tpu / Primal",
            "value": 0.00014800375,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IDefOpt / tpu / Primal",
            "value": 0.000150843,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / JaXPipe / tpu / Forward",
            "value": 0.00021210625,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / Jax / tpu / Forward",
            "value": 0.000261026,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / HLOOpt / tpu / Forward",
            "value": 0.0002117882499999,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / PartOpt / tpu / Forward",
            "value": 0.0002183129999999,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IPartOpt / tpu / Forward",
            "value": 0.00021214525,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / DefOpt / tpu / Forward",
            "value": 0.00021827375,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IDefOpt / tpu / Forward",
            "value": 0.00021204925,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / JaXPipe / tpu / PreRev",
            "value": 0.0003565605,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / JaXPipe / tpu / PostRev",
            "value": 0.0002593045,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / JaXPipe / tpu / BothRev",
            "value": 0.00035667,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / Jax / tpu / BothRev",
            "value": 0.00025923775,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / HLOOpt / tpu / PreRev",
            "value": 0.00035666175,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / HLOOpt / tpu / PostRev",
            "value": 0.0002918622499999,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / HLOOpt / tpu / BothRev",
            "value": 0.00035662875,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / PartOpt / tpu / PreRev",
            "value": 0.00035846425,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / PartOpt / tpu / PostRev",
            "value": 0.00027207375,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / PartOpt / tpu / BothRev",
            "value": 0.0003585587499999,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IPartOpt / tpu / PreRev",
            "value": 0.0003570305,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IPartOpt / tpu / PostRev",
            "value": 0.00027476175,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IPartOpt / tpu / BothRev",
            "value": 0.00035703525,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / DefOpt / tpu / PreRev",
            "value": 0.0003595432499999,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / DefOpt / tpu / PostRev",
            "value": 0.00028369525,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / DefOpt / tpu / BothRev",
            "value": 0.00035994525,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IDefOpt / tpu / PreRev",
            "value": 0.00035769625,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IDefOpt / tpu / PostRev",
            "value": 0.00030205625,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IDefOpt / tpu / BothRev",
            "value": 0.00035778875,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / JaXPipe / cpu / Primal",
            "value": 0.002194247,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / Jax / cpu / Primal",
            "value": 0.002406188,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / HLOOpt / cpu / Primal",
            "value": 0.002428191,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / PartOpt / cpu / Primal",
            "value": 0.002458628,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IPartOpt / cpu / Primal",
            "value": 0.002710221,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / DefOpt / cpu / Primal",
            "value": 0.002524433,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IDefOpt / cpu / Primal",
            "value": 0.002399772,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / JaXPipe / cpu / Forward",
            "value": 0.0060692,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / Jax / cpu / Forward",
            "value": 0.005858512,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / HLOOpt / cpu / Forward",
            "value": 0.005889533,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / PartOpt / cpu / Forward",
            "value": 0.005961251,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IPartOpt / cpu / Forward",
            "value": 0.005544403,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / DefOpt / cpu / Forward",
            "value": 0.005591603,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IDefOpt / cpu / Forward",
            "value": 0.006025609,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / JaXPipe / cpu / PreRev",
            "value": 0.009420141,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / JaXPipe / cpu / PostRev",
            "value": 0.008702639,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / JaXPipe / cpu / BothRev",
            "value": 0.0095133399999999,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / Jax / cpu / BothRev",
            "value": 0.0103057529999999,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / HLOOpt / cpu / PreRev",
            "value": 0.008790603,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / HLOOpt / cpu / PostRev",
            "value": 0.008169906,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / HLOOpt / cpu / BothRev",
            "value": 0.008740592,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / PartOpt / cpu / PreRev",
            "value": 0.008992118,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / PartOpt / cpu / PostRev",
            "value": 0.0107465729999999,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / PartOpt / cpu / BothRev",
            "value": 0.008688985,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IPartOpt / cpu / PreRev",
            "value": 0.008629312,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IPartOpt / cpu / PostRev",
            "value": 0.009531245,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IPartOpt / cpu / BothRev",
            "value": 0.00904836,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / DefOpt / cpu / PreRev",
            "value": 0.008163044,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / DefOpt / cpu / PostRev",
            "value": 0.008415091,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / DefOpt / cpu / BothRev",
            "value": 0.009260837,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IDefOpt / cpu / PreRev",
            "value": 0.0088217909999999,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IDefOpt / cpu / PostRev",
            "value": 0.008369044,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IDefOpt / cpu / BothRev",
            "value": 0.009465614,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / JaXPipe / cpu / Primal",
            "value": 0.001787,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / Jax / cpu / Primal",
            "value": 0.001538,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / HLOOpt / cpu / Primal",
            "value": 0.001591,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / PartOpt / cpu / Primal",
            "value": 0.00159,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IPartOpt / cpu / Primal",
            "value": 0.001477,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / DefOpt / cpu / Primal",
            "value": 0.0016519999999999,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IDefOpt / cpu / Primal",
            "value": 0.001637,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / JaXPipe / cpu / Forward",
            "value": 0.004413,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / Jax / cpu / Forward",
            "value": 0.004339,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / HLOOpt / cpu / Forward",
            "value": 0.004409,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / PartOpt / cpu / Forward",
            "value": 0.0040469999999999,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IPartOpt / cpu / Forward",
            "value": 0.0042309999999999,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / DefOpt / cpu / Forward",
            "value": 0.004378,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IDefOpt / cpu / Forward",
            "value": 0.004368,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / JaXPipe / cpu / PreRev",
            "value": 0.007244,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / JaXPipe / cpu / PostRev",
            "value": 0.0093059999999999,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / JaXPipe / cpu / BothRev",
            "value": 0.007644,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / Jax / cpu / BothRev",
            "value": 0.010618,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / HLOOpt / cpu / PreRev",
            "value": 0.0080139999999999,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / HLOOpt / cpu / PostRev",
            "value": 0.008681,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / HLOOpt / cpu / BothRev",
            "value": 0.007975,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / PartOpt / cpu / PreRev",
            "value": 0.007456,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / PartOpt / cpu / PostRev",
            "value": 0.01115,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / PartOpt / cpu / BothRev",
            "value": 0.007591,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IPartOpt / cpu / PreRev",
            "value": 0.007799,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IPartOpt / cpu / PostRev",
            "value": 0.0093859999999999,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IPartOpt / cpu / BothRev",
            "value": 0.008108,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / DefOpt / cpu / PreRev",
            "value": 0.007859,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / DefOpt / cpu / PostRev",
            "value": 0.00717,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / DefOpt / cpu / BothRev",
            "value": 0.007515,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IDefOpt / cpu / PreRev",
            "value": 0.00811,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IDefOpt / cpu / PostRev",
            "value": 0.008754,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IDefOpt / cpu / BothRev",
            "value": 0.007264,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000009541880017422954,
            "unit": "s"
          },
          {
            "name": "scatter_sum / Jax / cpu / Primal",
            "value": 0.000009051719980561756,
            "unit": "s"
          },
          {
            "name": "scatter_sum / HLOOpt / cpu / Primal",
            "value": 0.000009232299989889725,
            "unit": "s"
          },
          {
            "name": "scatter_sum / PartOpt / cpu / Primal",
            "value": 0.000008413739942625398,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IPartOpt / cpu / Primal",
            "value": 0.000008346000031451695,
            "unit": "s"
          },
          {
            "name": "scatter_sum / DefOpt / cpu / Primal",
            "value": 0.000008606619967395091,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IDefOpt / cpu / Primal",
            "value": 0.000008915020007407293,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Forward",
            "value": 0.000014127079957688693,
            "unit": "s"
          },
          {
            "name": "scatter_sum / Jax / cpu / Forward",
            "value": 0.000013720980014113591,
            "unit": "s"
          },
          {
            "name": "scatter_sum / HLOOpt / cpu / Forward",
            "value": 0.000014498360051220517,
            "unit": "s"
          },
          {
            "name": "scatter_sum / PartOpt / cpu / Forward",
            "value": 0.00001362183998026012,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IPartOpt / cpu / Forward",
            "value": 0.00001518636006039742,
            "unit": "s"
          },
          {
            "name": "scatter_sum / DefOpt / cpu / Forward",
            "value": 0.000013790399952995358,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IDefOpt / cpu / Forward",
            "value": 0.000013915299978179974,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / PreRev",
            "value": 0.000013283560037962162,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / PostRev",
            "value": 0.000013325800000529852,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / BothRev",
            "value": 0.00001360463998025807,
            "unit": "s"
          },
          {
            "name": "scatter_sum / Jax / cpu / BothRev",
            "value": 0.000012925799992444807,
            "unit": "s"
          },
          {
            "name": "scatter_sum / HLOOpt / cpu / PreRev",
            "value": 0.000014037100054338226,
            "unit": "s"
          },
          {
            "name": "scatter_sum / HLOOpt / cpu / PostRev",
            "value": 0.000015473340035896398,
            "unit": "s"
          },
          {
            "name": "scatter_sum / HLOOpt / cpu / BothRev",
            "value": 0.00001335435997134482,
            "unit": "s"
          },
          {
            "name": "scatter_sum / PartOpt / cpu / PreRev",
            "value": 0.000013849779970769304,
            "unit": "s"
          },
          {
            "name": "scatter_sum / PartOpt / cpu / PostRev",
            "value": 0.000013569119992098424,
            "unit": "s"
          },
          {
            "name": "scatter_sum / PartOpt / cpu / BothRev",
            "value": 0.000013870199991288246,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IPartOpt / cpu / PreRev",
            "value": 0.000013387979970502784,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IPartOpt / cpu / PostRev",
            "value": 0.000013002060004509986,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IPartOpt / cpu / BothRev",
            "value": 0.000012920400004077236,
            "unit": "s"
          },
          {
            "name": "scatter_sum / DefOpt / cpu / PreRev",
            "value": 0.000013036360005571623,
            "unit": "s"
          },
          {
            "name": "scatter_sum / DefOpt / cpu / PostRev",
            "value": 0.000013914739993197143,
            "unit": "s"
          },
          {
            "name": "scatter_sum / DefOpt / cpu / BothRev",
            "value": 0.000014243039995562869,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IDefOpt / cpu / PreRev",
            "value": 0.00001324805996773648,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IDefOpt / cpu / PostRev",
            "value": 0.000013406620037130778,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IDefOpt / cpu / BothRev",
            "value": 0.000013913320008214216,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cuda / Primal",
            "value": 0.000009664,
            "unit": "s"
          },
          {
            "name": "scatter_sum / Jax / cuda / Primal",
            "value": 0.000009664,
            "unit": "s"
          },
          {
            "name": "scatter_sum / HLOOpt / cuda / Primal",
            "value": 0.000009888,
            "unit": "s"
          },
          {
            "name": "scatter_sum / PartOpt / cuda / Primal",
            "value": 0.000009952,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IPartOpt / cuda / Primal",
            "value": 0.000009889,
            "unit": "s"
          },
          {
            "name": "scatter_sum / DefOpt / cuda / Primal",
            "value": 0.00000976,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IDefOpt / cuda / Primal",
            "value": 0.000009184,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cuda / Forward",
            "value": 0.000016479,
            "unit": "s"
          },
          {
            "name": "scatter_sum / Jax / cuda / Forward",
            "value": 0.000016767,
            "unit": "s"
          },
          {
            "name": "scatter_sum / HLOOpt / cuda / Forward",
            "value": 0.000016544,
            "unit": "s"
          },
          {
            "name": "scatter_sum / PartOpt / cuda / Forward",
            "value": 0.000016352,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IPartOpt / cuda / Forward",
            "value": 0.000016704,
            "unit": "s"
          },
          {
            "name": "scatter_sum / DefOpt / cuda / Forward",
            "value": 0.000018496,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IDefOpt / cuda / Forward",
            "value": 0.000017919999999999998,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cuda / PreRev",
            "value": 0.000016864,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cuda / PostRev",
            "value": 0.000016352,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cuda / BothRev",
            "value": 0.000016512,
            "unit": "s"
          },
          {
            "name": "scatter_sum / Jax / cuda / BothRev",
            "value": 0.000016512,
            "unit": "s"
          },
          {
            "name": "scatter_sum / HLOOpt / cuda / PreRev",
            "value": 0.000016672,
            "unit": "s"
          },
          {
            "name": "scatter_sum / HLOOpt / cuda / PostRev",
            "value": 0.0000168,
            "unit": "s"
          },
          {
            "name": "scatter_sum / HLOOpt / cuda / BothRev",
            "value": 0.000016544,
            "unit": "s"
          },
          {
            "name": "scatter_sum / PartOpt / cuda / PreRev",
            "value": 0.00001712,
            "unit": "s"
          },
          {
            "name": "scatter_sum / PartOpt / cuda / PostRev",
            "value": 0.000016192,
            "unit": "s"
          },
          {
            "name": "scatter_sum / PartOpt / cuda / BothRev",
            "value": 0.00001664,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IPartOpt / cuda / PreRev",
            "value": 0.000016703,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IPartOpt / cuda / PostRev",
            "value": 0.000016192,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IPartOpt / cuda / BothRev",
            "value": 0.000016832,
            "unit": "s"
          },
          {
            "name": "scatter_sum / DefOpt / cuda / PreRev",
            "value": 0.000016832,
            "unit": "s"
          },
          {
            "name": "scatter_sum / DefOpt / cuda / PostRev",
            "value": 0.000015872,
            "unit": "s"
          },
          {
            "name": "scatter_sum / DefOpt / cuda / BothRev",
            "value": 0.000017056,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IDefOpt / cuda / PreRev",
            "value": 0.00001696,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IDefOpt / cuda / PostRev",
            "value": 0.000017024,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IDefOpt / cuda / BothRev",
            "value": 0.000016832,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.000001344,
            "unit": "s"
          },
          {
            "name": "scatter_sum / Jax / tpu / Primal",
            "value": 0.0000014048,
            "unit": "s"
          },
          {
            "name": "scatter_sum / HLOOpt / tpu / Primal",
            "value": 0.000001343875,
            "unit": "s"
          },
          {
            "name": "scatter_sum / PartOpt / tpu / Primal",
            "value": 0.0000014045750000000002,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IPartOpt / tpu / Primal",
            "value": 0.000001343725,
            "unit": "s"
          },
          {
            "name": "scatter_sum / DefOpt / tpu / Primal",
            "value": 0.000001404925,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IDefOpt / tpu / Primal",
            "value": 0.0000013436,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Forward",
            "value": 0.00000270495,
            "unit": "s"
          },
          {
            "name": "scatter_sum / Jax / tpu / Forward",
            "value": 0.000002722675,
            "unit": "s"
          },
          {
            "name": "scatter_sum / HLOOpt / tpu / Forward",
            "value": 0.0000027063,
            "unit": "s"
          },
          {
            "name": "scatter_sum / PartOpt / tpu / Forward",
            "value": 0.00000268545,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IPartOpt / tpu / Forward",
            "value": 0.00000270835,
            "unit": "s"
          },
          {
            "name": "scatter_sum / DefOpt / tpu / Forward",
            "value": 0.000002685175,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IDefOpt / tpu / Forward",
            "value": 0.00000270065,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / PreRev",
            "value": 0.000002686075,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / PostRev",
            "value": 0.00000269025,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / BothRev",
            "value": 0.000002700875,
            "unit": "s"
          },
          {
            "name": "scatter_sum / Jax / tpu / BothRev",
            "value": 0.0000027426,
            "unit": "s"
          },
          {
            "name": "scatter_sum / HLOOpt / tpu / PreRev",
            "value": 0.000002695875,
            "unit": "s"
          },
          {
            "name": "scatter_sum / HLOOpt / tpu / PostRev",
            "value": 0.000002746125,
            "unit": "s"
          },
          {
            "name": "scatter_sum / HLOOpt / tpu / BothRev",
            "value": 0.000002700975,
            "unit": "s"
          },
          {
            "name": "scatter_sum / PartOpt / tpu / PreRev",
            "value": 0.0000027438000000000003,
            "unit": "s"
          },
          {
            "name": "scatter_sum / PartOpt / tpu / PostRev",
            "value": 0.0000026997,
            "unit": "s"
          },
          {
            "name": "scatter_sum / PartOpt / tpu / BothRev",
            "value": 0.0000027439,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IPartOpt / tpu / PreRev",
            "value": 0.0000027068,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IPartOpt / tpu / PostRev",
            "value": 0.00000274555,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IPartOpt / tpu / BothRev",
            "value": 0.0000027067249999999995,
            "unit": "s"
          },
          {
            "name": "scatter_sum / DefOpt / tpu / PreRev",
            "value": 0.000002747,
            "unit": "s"
          },
          {
            "name": "scatter_sum / DefOpt / tpu / PostRev",
            "value": 0.00000269455,
            "unit": "s"
          },
          {
            "name": "scatter_sum / DefOpt / tpu / BothRev",
            "value": 0.000002745425,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IDefOpt / tpu / PreRev",
            "value": 0.0000027003,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IDefOpt / tpu / PostRev",
            "value": 0.000002745825,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IDefOpt / tpu / BothRev",
            "value": 0.000002696975,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000015980999999999998,
            "unit": "s"
          },
          {
            "name": "scatter_sum / Jax / cpu / Primal",
            "value": 0.000016052,
            "unit": "s"
          },
          {
            "name": "scatter_sum / HLOOpt / cpu / Primal",
            "value": 0.000015499000000000002,
            "unit": "s"
          },
          {
            "name": "scatter_sum / PartOpt / cpu / Primal",
            "value": 0.000015386,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IPartOpt / cpu / Primal",
            "value": 0.000015197,
            "unit": "s"
          },
          {
            "name": "scatter_sum / DefOpt / cpu / Primal",
            "value": 0.00001575,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IDefOpt / cpu / Primal",
            "value": 0.00001555,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Forward",
            "value": 0.000023196000000000003,
            "unit": "s"
          },
          {
            "name": "scatter_sum / Jax / cpu / Forward",
            "value": 0.000022207000000000003,
            "unit": "s"
          },
          {
            "name": "scatter_sum / HLOOpt / cpu / Forward",
            "value": 0.000022482,
            "unit": "s"
          },
          {
            "name": "scatter_sum / PartOpt / cpu / Forward",
            "value": 0.000022309,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IPartOpt / cpu / Forward",
            "value": 0.000022729,
            "unit": "s"
          },
          {
            "name": "scatter_sum / DefOpt / cpu / Forward",
            "value": 0.000023037,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IDefOpt / cpu / Forward",
            "value": 0.000022594,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / PreRev",
            "value": 0.000023253,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / PostRev",
            "value": 0.000022779,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / BothRev",
            "value": 0.000021727,
            "unit": "s"
          },
          {
            "name": "scatter_sum / Jax / cpu / BothRev",
            "value": 0.000022891,
            "unit": "s"
          },
          {
            "name": "scatter_sum / HLOOpt / cpu / PreRev",
            "value": 0.0000229,
            "unit": "s"
          },
          {
            "name": "scatter_sum / HLOOpt / cpu / PostRev",
            "value": 0.000022461,
            "unit": "s"
          },
          {
            "name": "scatter_sum / HLOOpt / cpu / BothRev",
            "value": 0.00002306,
            "unit": "s"
          },
          {
            "name": "scatter_sum / PartOpt / cpu / PreRev",
            "value": 0.0000225,
            "unit": "s"
          },
          {
            "name": "scatter_sum / PartOpt / cpu / PostRev",
            "value": 0.000022766,
            "unit": "s"
          },
          {
            "name": "scatter_sum / PartOpt / cpu / BothRev",
            "value": 0.000022211,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IPartOpt / cpu / PreRev",
            "value": 0.000023384,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IPartOpt / cpu / PostRev",
            "value": 0.000023929,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IPartOpt / cpu / BothRev",
            "value": 0.000022452,
            "unit": "s"
          },
          {
            "name": "scatter_sum / DefOpt / cpu / PreRev",
            "value": 0.000023302,
            "unit": "s"
          },
          {
            "name": "scatter_sum / DefOpt / cpu / PostRev",
            "value": 0.000023276,
            "unit": "s"
          },
          {
            "name": "scatter_sum / DefOpt / cpu / BothRev",
            "value": 0.00002355,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IDefOpt / cpu / PreRev",
            "value": 0.000022436,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IDefOpt / cpu / PostRev",
            "value": 0.000024456,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IDefOpt / cpu / BothRev",
            "value": 0.000022754,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000011,
            "unit": "s"
          },
          {
            "name": "scatter_sum / Jax / cpu / Primal",
            "value": 0.000011,
            "unit": "s"
          },
          {
            "name": "scatter_sum / HLOOpt / cpu / Primal",
            "value": 0.000011,
            "unit": "s"
          },
          {
            "name": "scatter_sum / PartOpt / cpu / Primal",
            "value": 0.00001,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IPartOpt / cpu / Primal",
            "value": 0.00001,
            "unit": "s"
          },
          {
            "name": "scatter_sum / DefOpt / cpu / Primal",
            "value": 0.00001,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IDefOpt / cpu / Primal",
            "value": 0.000011,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Forward",
            "value": 0.000016,
            "unit": "s"
          },
          {
            "name": "scatter_sum / Jax / cpu / Forward",
            "value": 0.000017,
            "unit": "s"
          },
          {
            "name": "scatter_sum / HLOOpt / cpu / Forward",
            "value": 0.000016,
            "unit": "s"
          },
          {
            "name": "scatter_sum / PartOpt / cpu / Forward",
            "value": 0.000016,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IPartOpt / cpu / Forward",
            "value": 0.000016,
            "unit": "s"
          },
          {
            "name": "scatter_sum / DefOpt / cpu / Forward",
            "value": 0.000017,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IDefOpt / cpu / Forward",
            "value": 0.000017,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / PreRev",
            "value": 0.000017,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / PostRev",
            "value": 0.000017,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / BothRev",
            "value": 0.000017999999999999997,
            "unit": "s"
          },
          {
            "name": "scatter_sum / Jax / cpu / BothRev",
            "value": 0.000017,
            "unit": "s"
          },
          {
            "name": "scatter_sum / HLOOpt / cpu / PreRev",
            "value": 0.000016,
            "unit": "s"
          },
          {
            "name": "scatter_sum / HLOOpt / cpu / PostRev",
            "value": 0.000016,
            "unit": "s"
          },
          {
            "name": "scatter_sum / HLOOpt / cpu / BothRev",
            "value": 0.000017,
            "unit": "s"
          },
          {
            "name": "scatter_sum / PartOpt / cpu / PreRev",
            "value": 0.000017,
            "unit": "s"
          },
          {
            "name": "scatter_sum / PartOpt / cpu / PostRev",
            "value": 0.000017999999999999997,
            "unit": "s"
          },
          {
            "name": "scatter_sum / PartOpt / cpu / BothRev",
            "value": 0.000017999999999999997,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IPartOpt / cpu / PreRev",
            "value": 0.000017,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IPartOpt / cpu / PostRev",
            "value": 0.000016,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IPartOpt / cpu / BothRev",
            "value": 0.000017,
            "unit": "s"
          },
          {
            "name": "scatter_sum / DefOpt / cpu / PreRev",
            "value": 0.000016,
            "unit": "s"
          },
          {
            "name": "scatter_sum / DefOpt / cpu / PostRev",
            "value": 0.000016,
            "unit": "s"
          },
          {
            "name": "scatter_sum / DefOpt / cpu / BothRev",
            "value": 0.000016,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IDefOpt / cpu / PreRev",
            "value": 0.000015,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IDefOpt / cpu / PostRev",
            "value": 0.000015,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IDefOpt / cpu / BothRev",
            "value": 0.000016,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / Primal",
            "value": 0.000007281100033651456,
            "unit": "s"
          },
          {
            "name": "slicing / Jax / cpu / Primal",
            "value": 0.000007417620008709491,
            "unit": "s"
          },
          {
            "name": "slicing / HLOOpt / cpu / Primal",
            "value": 0.000006967280014578136,
            "unit": "s"
          },
          {
            "name": "slicing / PartOpt / cpu / Primal",
            "value": 0.00000741012001526542,
            "unit": "s"
          },
          {
            "name": "slicing / IPartOpt / cpu / Primal",
            "value": 0.000006921139993210091,
            "unit": "s"
          },
          {
            "name": "slicing / DefOpt / cpu / Primal",
            "value": 0.000007031080021988601,
            "unit": "s"
          },
          {
            "name": "slicing / IDefOpt / cpu / Primal",
            "value": 0.000006688900002700393,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / Forward",
            "value": 0.00001028114003020164,
            "unit": "s"
          },
          {
            "name": "slicing / Jax / cpu / Forward",
            "value": 0.000011345779976181802,
            "unit": "s"
          },
          {
            "name": "slicing / HLOOpt / cpu / Forward",
            "value": 0.000011428660027377193,
            "unit": "s"
          },
          {
            "name": "slicing / PartOpt / cpu / Forward",
            "value": 0.00001100200001928897,
            "unit": "s"
          },
          {
            "name": "slicing / IPartOpt / cpu / Forward",
            "value": 0.000011208740024812867,
            "unit": "s"
          },
          {
            "name": "slicing / DefOpt / cpu / Forward",
            "value": 0.000010430400025143173,
            "unit": "s"
          },
          {
            "name": "slicing / IDefOpt / cpu / Forward",
            "value": 0.000010656319973350036,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / PreRev",
            "value": 0.000011447760016380926,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / PostRev",
            "value": 0.000011277679996055666,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / BothRev",
            "value": 0.000011822359983852948,
            "unit": "s"
          },
          {
            "name": "slicing / Jax / cpu / BothRev",
            "value": 0.00001135638000050676,
            "unit": "s"
          },
          {
            "name": "slicing / HLOOpt / cpu / PreRev",
            "value": 0.000011924040018129743,
            "unit": "s"
          },
          {
            "name": "slicing / HLOOpt / cpu / PostRev",
            "value": 0.000013196680001783532,
            "unit": "s"
          },
          {
            "name": "slicing / HLOOpt / cpu / BothRev",
            "value": 0.00001109399996494176,
            "unit": "s"
          },
          {
            "name": "slicing / PartOpt / cpu / PreRev",
            "value": 0.000011041359948649188,
            "unit": "s"
          },
          {
            "name": "slicing / PartOpt / cpu / PostRev",
            "value": 0.000011046320032619406,
            "unit": "s"
          },
          {
            "name": "slicing / PartOpt / cpu / BothRev",
            "value": 0.00001108768004087324,
            "unit": "s"
          },
          {
            "name": "slicing / IPartOpt / cpu / PreRev",
            "value": 0.000011140319975311284,
            "unit": "s"
          },
          {
            "name": "slicing / IPartOpt / cpu / PostRev",
            "value": 0.000011385600000721752,
            "unit": "s"
          },
          {
            "name": "slicing / IPartOpt / cpu / BothRev",
            "value": 0.00001075728005162091,
            "unit": "s"
          },
          {
            "name": "slicing / DefOpt / cpu / PreRev",
            "value": 0.00001135841998802789,
            "unit": "s"
          },
          {
            "name": "slicing / DefOpt / cpu / PostRev",
            "value": 0.000011674980032694294,
            "unit": "s"
          },
          {
            "name": "slicing / DefOpt / cpu / BothRev",
            "value": 0.00001120979999541305,
            "unit": "s"
          },
          {
            "name": "slicing / IDefOpt / cpu / PreRev",
            "value": 0.000011482079980851268,
            "unit": "s"
          },
          {
            "name": "slicing / IDefOpt / cpu / PostRev",
            "value": 0.000011328559985486208,
            "unit": "s"
          },
          {
            "name": "slicing / IDefOpt / cpu / BothRev",
            "value": 0.000011219779962630129,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cuda / Primal",
            "value": 0.000001887,
            "unit": "s"
          },
          {
            "name": "slicing / Jax / cuda / Primal",
            "value": 0.000001887,
            "unit": "s"
          },
          {
            "name": "slicing / HLOOpt / cuda / Primal",
            "value": 0.000001888,
            "unit": "s"
          },
          {
            "name": "slicing / PartOpt / cuda / Primal",
            "value": 0.000001887,
            "unit": "s"
          },
          {
            "name": "slicing / IPartOpt / cuda / Primal",
            "value": 0.000001887,
            "unit": "s"
          },
          {
            "name": "slicing / DefOpt / cuda / Primal",
            "value": 0.000001887,
            "unit": "s"
          },
          {
            "name": "slicing / IDefOpt / cuda / Primal",
            "value": 0.000001887,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cuda / Forward",
            "value": 0.000009632,
            "unit": "s"
          },
          {
            "name": "slicing / Jax / cuda / Forward",
            "value": 0.000009824,
            "unit": "s"
          },
          {
            "name": "slicing / HLOOpt / cuda / Forward",
            "value": 0.00000944,
            "unit": "s"
          },
          {
            "name": "slicing / PartOpt / cuda / Forward",
            "value": 0.000009792,
            "unit": "s"
          },
          {
            "name": "slicing / IPartOpt / cuda / Forward",
            "value": 0.00000976,
            "unit": "s"
          },
          {
            "name": "slicing / DefOpt / cuda / Forward",
            "value": 0.00000976,
            "unit": "s"
          },
          {
            "name": "slicing / IDefOpt / cuda / Forward",
            "value": 0.000009951,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cuda / PreRev",
            "value": 0.000009344,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cuda / PostRev",
            "value": 0.000010016,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cuda / BothRev",
            "value": 0.000009504,
            "unit": "s"
          },
          {
            "name": "slicing / Jax / cuda / BothRev",
            "value": 0.000009824,
            "unit": "s"
          },
          {
            "name": "slicing / HLOOpt / cuda / PreRev",
            "value": 0.000009536,
            "unit": "s"
          },
          {
            "name": "slicing / HLOOpt / cuda / PostRev",
            "value": 0.000009504,
            "unit": "s"
          },
          {
            "name": "slicing / HLOOpt / cuda / BothRev",
            "value": 0.000009471,
            "unit": "s"
          },
          {
            "name": "slicing / PartOpt / cuda / PreRev",
            "value": 0.000009824,
            "unit": "s"
          },
          {
            "name": "slicing / PartOpt / cuda / PostRev",
            "value": 0.000009664,
            "unit": "s"
          },
          {
            "name": "slicing / PartOpt / cuda / BothRev",
            "value": 0.000009248,
            "unit": "s"
          },
          {
            "name": "slicing / IPartOpt / cuda / PreRev",
            "value": 0.000009856,
            "unit": "s"
          },
          {
            "name": "slicing / IPartOpt / cuda / PostRev",
            "value": 0.000009344,
            "unit": "s"
          },
          {
            "name": "slicing / IPartOpt / cuda / BothRev",
            "value": 0.000010368,
            "unit": "s"
          },
          {
            "name": "slicing / DefOpt / cuda / PreRev",
            "value": 0.000009728,
            "unit": "s"
          },
          {
            "name": "slicing / DefOpt / cuda / PostRev",
            "value": 0.0000096,
            "unit": "s"
          },
          {
            "name": "slicing / DefOpt / cuda / BothRev",
            "value": 0.000009696,
            "unit": "s"
          },
          {
            "name": "slicing / IDefOpt / cuda / PreRev",
            "value": 0.00000992,
            "unit": "s"
          },
          {
            "name": "slicing / IDefOpt / cuda / PostRev",
            "value": 0.000009439,
            "unit": "s"
          },
          {
            "name": "slicing / IDefOpt / cuda / BothRev",
            "value": 0.000009888,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / tpu / Primal",
            "value": 0.000001017725,
            "unit": "s"
          },
          {
            "name": "slicing / Jax / tpu / Primal",
            "value": 9.66075e-7,
            "unit": "s"
          },
          {
            "name": "slicing / HLOOpt / tpu / Primal",
            "value": 0.000001019725,
            "unit": "s"
          },
          {
            "name": "slicing / PartOpt / tpu / Primal",
            "value": 9.635e-7,
            "unit": "s"
          },
          {
            "name": "slicing / IPartOpt / tpu / Primal",
            "value": 0.00000102175,
            "unit": "s"
          },
          {
            "name": "slicing / DefOpt / tpu / Primal",
            "value": 9.69275e-7,
            "unit": "s"
          },
          {
            "name": "slicing / IDefOpt / tpu / Primal",
            "value": 0.000001017375,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / tpu / Forward",
            "value": 0.000001402875,
            "unit": "s"
          },
          {
            "name": "slicing / Jax / tpu / Forward",
            "value": 0.0000014684249999999998,
            "unit": "s"
          },
          {
            "name": "slicing / HLOOpt / tpu / Forward",
            "value": 0.000001511325,
            "unit": "s"
          },
          {
            "name": "slicing / PartOpt / tpu / Forward",
            "value": 0.000001492375,
            "unit": "s"
          },
          {
            "name": "slicing / IPartOpt / tpu / Forward",
            "value": 0.0000015116,
            "unit": "s"
          },
          {
            "name": "slicing / DefOpt / tpu / Forward",
            "value": 0.000001490025,
            "unit": "s"
          },
          {
            "name": "slicing / IDefOpt / tpu / Forward",
            "value": 0.000001513325,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / tpu / PreRev",
            "value": 0.000002577125,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / tpu / PostRev",
            "value": 0.0000025213249999999995,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / tpu / BothRev",
            "value": 0.000002582525,
            "unit": "s"
          },
          {
            "name": "slicing / Jax / tpu / BothRev",
            "value": 0.00000253535,
            "unit": "s"
          },
          {
            "name": "slicing / HLOOpt / tpu / PreRev",
            "value": 0.0000025862249999999995,
            "unit": "s"
          },
          {
            "name": "slicing / HLOOpt / tpu / PostRev",
            "value": 0.0000025417,
            "unit": "s"
          },
          {
            "name": "slicing / HLOOpt / tpu / BothRev",
            "value": 0.000002583125,
            "unit": "s"
          },
          {
            "name": "slicing / PartOpt / tpu / PreRev",
            "value": 0.0000025458750000000004,
            "unit": "s"
          },
          {
            "name": "slicing / PartOpt / tpu / PostRev",
            "value": 0.000002588825,
            "unit": "s"
          },
          {
            "name": "slicing / PartOpt / tpu / BothRev",
            "value": 0.00000253965,
            "unit": "s"
          },
          {
            "name": "slicing / IPartOpt / tpu / PreRev",
            "value": 0.0000025968000000000003,
            "unit": "s"
          },
          {
            "name": "slicing / IPartOpt / tpu / PostRev",
            "value": 0.0000025466,
            "unit": "s"
          },
          {
            "name": "slicing / IPartOpt / tpu / BothRev",
            "value": 0.000002589625,
            "unit": "s"
          },
          {
            "name": "slicing / DefOpt / tpu / PreRev",
            "value": 0.00000253795,
            "unit": "s"
          },
          {
            "name": "slicing / DefOpt / tpu / PostRev",
            "value": 0.00000258885,
            "unit": "s"
          },
          {
            "name": "slicing / DefOpt / tpu / BothRev",
            "value": 0.0000025395,
            "unit": "s"
          },
          {
            "name": "slicing / IDefOpt / tpu / PreRev",
            "value": 0.0000025814500000000005,
            "unit": "s"
          },
          {
            "name": "slicing / IDefOpt / tpu / PostRev",
            "value": 0.0000025348,
            "unit": "s"
          },
          {
            "name": "slicing / IDefOpt / tpu / BothRev",
            "value": 0.000002584975,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / Primal",
            "value": 0.000012508,
            "unit": "s"
          },
          {
            "name": "slicing / Jax / cpu / Primal",
            "value": 0.000012227,
            "unit": "s"
          },
          {
            "name": "slicing / HLOOpt / cpu / Primal",
            "value": 0.000012631,
            "unit": "s"
          },
          {
            "name": "slicing / PartOpt / cpu / Primal",
            "value": 0.000012435,
            "unit": "s"
          },
          {
            "name": "slicing / IPartOpt / cpu / Primal",
            "value": 0.00001295,
            "unit": "s"
          },
          {
            "name": "slicing / DefOpt / cpu / Primal",
            "value": 0.00001228,
            "unit": "s"
          },
          {
            "name": "slicing / IDefOpt / cpu / Primal",
            "value": 0.0000128,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / Forward",
            "value": 0.000017092,
            "unit": "s"
          },
          {
            "name": "slicing / Jax / cpu / Forward",
            "value": 0.000017079,
            "unit": "s"
          },
          {
            "name": "slicing / HLOOpt / cpu / Forward",
            "value": 0.000016391,
            "unit": "s"
          },
          {
            "name": "slicing / PartOpt / cpu / Forward",
            "value": 0.000016484,
            "unit": "s"
          },
          {
            "name": "slicing / IPartOpt / cpu / Forward",
            "value": 0.000016498,
            "unit": "s"
          },
          {
            "name": "slicing / DefOpt / cpu / Forward",
            "value": 0.000016513,
            "unit": "s"
          },
          {
            "name": "slicing / IDefOpt / cpu / Forward",
            "value": 0.000016844,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / PreRev",
            "value": 0.000017643,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / PostRev",
            "value": 0.000017447,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / BothRev",
            "value": 0.000017169,
            "unit": "s"
          },
          {
            "name": "slicing / Jax / cpu / BothRev",
            "value": 0.000017169,
            "unit": "s"
          },
          {
            "name": "slicing / HLOOpt / cpu / PreRev",
            "value": 0.000017263,
            "unit": "s"
          },
          {
            "name": "slicing / HLOOpt / cpu / PostRev",
            "value": 0.000017066000000000002,
            "unit": "s"
          },
          {
            "name": "slicing / HLOOpt / cpu / BothRev",
            "value": 0.000017186,
            "unit": "s"
          },
          {
            "name": "slicing / PartOpt / cpu / PreRev",
            "value": 0.000017448000000000003,
            "unit": "s"
          },
          {
            "name": "slicing / PartOpt / cpu / PostRev",
            "value": 0.000017386,
            "unit": "s"
          },
          {
            "name": "slicing / PartOpt / cpu / BothRev",
            "value": 0.000017003,
            "unit": "s"
          },
          {
            "name": "slicing / IPartOpt / cpu / PreRev",
            "value": 0.000017003,
            "unit": "s"
          },
          {
            "name": "slicing / IPartOpt / cpu / PostRev",
            "value": 0.000017423000000000002,
            "unit": "s"
          },
          {
            "name": "slicing / IPartOpt / cpu / BothRev",
            "value": 0.000017207,
            "unit": "s"
          },
          {
            "name": "slicing / DefOpt / cpu / PreRev",
            "value": 0.000017603,
            "unit": "s"
          },
          {
            "name": "slicing / DefOpt / cpu / PostRev",
            "value": 0.000017517000000000002,
            "unit": "s"
          },
          {
            "name": "slicing / DefOpt / cpu / BothRev",
            "value": 0.0000174,
            "unit": "s"
          },
          {
            "name": "slicing / IDefOpt / cpu / PreRev",
            "value": 0.000018039,
            "unit": "s"
          },
          {
            "name": "slicing / IDefOpt / cpu / PostRev",
            "value": 0.000017925,
            "unit": "s"
          },
          {
            "name": "slicing / IDefOpt / cpu / BothRev",
            "value": 0.000017451999999999998,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / Primal",
            "value": 0.000008999999999999999,
            "unit": "s"
          },
          {
            "name": "slicing / Jax / cpu / Primal",
            "value": 0.000008999999999999999,
            "unit": "s"
          },
          {
            "name": "slicing / HLOOpt / cpu / Primal",
            "value": 0.000008999999999999999,
            "unit": "s"
          },
          {
            "name": "slicing / PartOpt / cpu / Primal",
            "value": 0.000008,
            "unit": "s"
          },
          {
            "name": "slicing / IPartOpt / cpu / Primal",
            "value": 0.000008999999999999999,
            "unit": "s"
          },
          {
            "name": "slicing / DefOpt / cpu / Primal",
            "value": 0.000008999999999999999,
            "unit": "s"
          },
          {
            "name": "slicing / IDefOpt / cpu / Primal",
            "value": 0.000008999999999999999,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / Forward",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "slicing / Jax / cpu / Forward",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "slicing / HLOOpt / cpu / Forward",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "slicing / PartOpt / cpu / Forward",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "slicing / IPartOpt / cpu / Forward",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "slicing / DefOpt / cpu / Forward",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "slicing / IDefOpt / cpu / Forward",
            "value": 0.000011,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / PreRev",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / PostRev",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / BothRev",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "slicing / Jax / cpu / BothRev",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "slicing / HLOOpt / cpu / PreRev",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "slicing / HLOOpt / cpu / PostRev",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "slicing / HLOOpt / cpu / BothRev",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "slicing / PartOpt / cpu / PreRev",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "slicing / PartOpt / cpu / PostRev",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "slicing / PartOpt / cpu / BothRev",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "slicing / IPartOpt / cpu / PreRev",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "slicing / IPartOpt / cpu / PostRev",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "slicing / IPartOpt / cpu / BothRev",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "slicing / DefOpt / cpu / PreRev",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "slicing / DefOpt / cpu / PostRev",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "slicing / DefOpt / cpu / BothRev",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "slicing / IDefOpt / cpu / PreRev",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "slicing / IDefOpt / cpu / PostRev",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "slicing / IDefOpt / cpu / BothRev",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / Primal",
            "value": 0.000008694479984114878,
            "unit": "s"
          },
          {
            "name": "sum    / Jax / cpu / Primal",
            "value": 0.000008546899998691515,
            "unit": "s"
          },
          {
            "name": "sum    / HLOOpt / cpu / Primal",
            "value": 0.000008662599948365823,
            "unit": "s"
          },
          {
            "name": "sum    / PartOpt / cpu / Primal",
            "value": 0.000008412260031036568,
            "unit": "s"
          },
          {
            "name": "sum    / IPartOpt / cpu / Primal",
            "value": 0.00000878825994732324,
            "unit": "s"
          },
          {
            "name": "sum    / DefOpt / cpu / Primal",
            "value": 0.000008437960023002233,
            "unit": "s"
          },
          {
            "name": "sum    / IDefOpt / cpu / Primal",
            "value": 0.000008664920005685418,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / Forward",
            "value": 0.0000127289999727509,
            "unit": "s"
          },
          {
            "name": "sum    / Jax / cpu / Forward",
            "value": 0.00001288352003939508,
            "unit": "s"
          },
          {
            "name": "sum    / HLOOpt / cpu / Forward",
            "value": 0.000012899100038339384,
            "unit": "s"
          },
          {
            "name": "sum    / PartOpt / cpu / Forward",
            "value": 0.00001233078003679111,
            "unit": "s"
          },
          {
            "name": "sum    / IPartOpt / cpu / Forward",
            "value": 0.000012780420038325246,
            "unit": "s"
          },
          {
            "name": "sum    / DefOpt / cpu / Forward",
            "value": 0.00001248081999619899,
            "unit": "s"
          },
          {
            "name": "sum    / IDefOpt / cpu / Forward",
            "value": 0.000012676520000241,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / PreRev",
            "value": 0.000012363699952402384,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / PostRev",
            "value": 0.00001218069995957194,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / BothRev",
            "value": 0.000012381719943732606,
            "unit": "s"
          },
          {
            "name": "sum    / Jax / cpu / BothRev",
            "value": 0.00001194789999317436,
            "unit": "s"
          },
          {
            "name": "sum    / HLOOpt / cpu / PreRev",
            "value": 0.000012409460023263818,
            "unit": "s"
          },
          {
            "name": "sum    / HLOOpt / cpu / PostRev",
            "value": 0.000014344759974846966,
            "unit": "s"
          },
          {
            "name": "sum    / HLOOpt / cpu / BothRev",
            "value": 0.000012370700023893732,
            "unit": "s"
          },
          {
            "name": "sum    / PartOpt / cpu / PreRev",
            "value": 0.00001217521999933524,
            "unit": "s"
          },
          {
            "name": "sum    / PartOpt / cpu / PostRev",
            "value": 0.000012258639990250232,
            "unit": "s"
          },
          {
            "name": "sum    / PartOpt / cpu / BothRev",
            "value": 0.00001247643996066472,
            "unit": "s"
          },
          {
            "name": "sum    / IPartOpt / cpu / PreRev",
            "value": 0.000012240759997439454,
            "unit": "s"
          },
          {
            "name": "sum    / IPartOpt / cpu / PostRev",
            "value": 0.000011836020021291916,
            "unit": "s"
          },
          {
            "name": "sum    / IPartOpt / cpu / BothRev",
            "value": 0.000011904080001841069,
            "unit": "s"
          },
          {
            "name": "sum    / DefOpt / cpu / PreRev",
            "value": 0.000011943660019824163,
            "unit": "s"
          },
          {
            "name": "sum    / DefOpt / cpu / PostRev",
            "value": 0.000012389420035106014,
            "unit": "s"
          },
          {
            "name": "sum    / DefOpt / cpu / BothRev",
            "value": 0.000011991000019406784,
            "unit": "s"
          },
          {
            "name": "sum    / IDefOpt / cpu / PreRev",
            "value": 0.000011959200001001592,
            "unit": "s"
          },
          {
            "name": "sum    / IDefOpt / cpu / PostRev",
            "value": 0.000011911360052181409,
            "unit": "s"
          },
          {
            "name": "sum    / IDefOpt / cpu / BothRev",
            "value": 0.000012511519989857336,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cuda / Primal",
            "value": 0.000002047,
            "unit": "s"
          },
          {
            "name": "sum    / Jax / cuda / Primal",
            "value": 0.000002047,
            "unit": "s"
          },
          {
            "name": "sum    / HLOOpt / cuda / Primal",
            "value": 0.000002047,
            "unit": "s"
          },
          {
            "name": "sum    / PartOpt / cuda / Primal",
            "value": 0.000002047,
            "unit": "s"
          },
          {
            "name": "sum    / IPartOpt / cuda / Primal",
            "value": 0.000002047,
            "unit": "s"
          },
          {
            "name": "sum    / DefOpt / cuda / Primal",
            "value": 0.000002047,
            "unit": "s"
          },
          {
            "name": "sum    / IDefOpt / cuda / Primal",
            "value": 0.000002047,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cuda / Forward",
            "value": 0.000009952,
            "unit": "s"
          },
          {
            "name": "sum    / Jax / cuda / Forward",
            "value": 0.00001008,
            "unit": "s"
          },
          {
            "name": "sum    / HLOOpt / cuda / Forward",
            "value": 0.000009952,
            "unit": "s"
          },
          {
            "name": "sum    / PartOpt / cuda / Forward",
            "value": 0.000010144,
            "unit": "s"
          },
          {
            "name": "sum    / IPartOpt / cuda / Forward",
            "value": 0.000009952,
            "unit": "s"
          },
          {
            "name": "sum    / DefOpt / cuda / Forward",
            "value": 0.00000992,
            "unit": "s"
          },
          {
            "name": "sum    / IDefOpt / cuda / Forward",
            "value": 0.00001024,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cuda / PreRev",
            "value": 0.000009345,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cuda / PostRev",
            "value": 0.00000928,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cuda / BothRev",
            "value": 0.00000944,
            "unit": "s"
          },
          {
            "name": "sum    / Jax / cuda / BothRev",
            "value": 0.00000992,
            "unit": "s"
          },
          {
            "name": "sum    / HLOOpt / cuda / PreRev",
            "value": 0.000009984,
            "unit": "s"
          },
          {
            "name": "sum    / HLOOpt / cuda / PostRev",
            "value": 0.000009632,
            "unit": "s"
          },
          {
            "name": "sum    / HLOOpt / cuda / BothRev",
            "value": 0.000009472,
            "unit": "s"
          },
          {
            "name": "sum    / PartOpt / cuda / PreRev",
            "value": 0.00000992,
            "unit": "s"
          },
          {
            "name": "sum    / PartOpt / cuda / PostRev",
            "value": 0.000009696,
            "unit": "s"
          },
          {
            "name": "sum    / PartOpt / cuda / BothRev",
            "value": 0.000009536,
            "unit": "s"
          },
          {
            "name": "sum    / IPartOpt / cuda / PreRev",
            "value": 0.000009408,
            "unit": "s"
          },
          {
            "name": "sum    / IPartOpt / cuda / PostRev",
            "value": 0.000009536,
            "unit": "s"
          },
          {
            "name": "sum    / IPartOpt / cuda / BothRev",
            "value": 0.000009568,
            "unit": "s"
          },
          {
            "name": "sum    / DefOpt / cuda / PreRev",
            "value": 0.000009856,
            "unit": "s"
          },
          {
            "name": "sum    / DefOpt / cuda / PostRev",
            "value": 0.000009376,
            "unit": "s"
          },
          {
            "name": "sum    / DefOpt / cuda / BothRev",
            "value": 0.0000096,
            "unit": "s"
          },
          {
            "name": "sum    / IDefOpt / cuda / PreRev",
            "value": 0.00000992,
            "unit": "s"
          },
          {
            "name": "sum    / IDefOpt / cuda / PostRev",
            "value": 0.000009728,
            "unit": "s"
          },
          {
            "name": "sum    / IDefOpt / cuda / BothRev",
            "value": 0.000009856,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / tpu / Primal",
            "value": 5.1075e-7,
            "unit": "s"
          },
          {
            "name": "sum    / Jax / tpu / Primal",
            "value": 5.4685e-7,
            "unit": "s"
          },
          {
            "name": "sum    / HLOOpt / tpu / Primal",
            "value": 5.104499999999999e-7,
            "unit": "s"
          },
          {
            "name": "sum    / PartOpt / tpu / Primal",
            "value": 5.46625e-7,
            "unit": "s"
          },
          {
            "name": "sum    / IPartOpt / tpu / Primal",
            "value": 5.10425e-7,
            "unit": "s"
          },
          {
            "name": "sum    / DefOpt / tpu / Primal",
            "value": 5.473499999999999e-7,
            "unit": "s"
          },
          {
            "name": "sum    / IDefOpt / tpu / Primal",
            "value": 5.1025e-7,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / tpu / Forward",
            "value": 0.000001551575,
            "unit": "s"
          },
          {
            "name": "sum    / Jax / tpu / Forward",
            "value": 0.000001508825,
            "unit": "s"
          },
          {
            "name": "sum    / HLOOpt / tpu / Forward",
            "value": 0.000001528575,
            "unit": "s"
          },
          {
            "name": "sum    / PartOpt / tpu / Forward",
            "value": 0.000001493275,
            "unit": "s"
          },
          {
            "name": "sum    / IPartOpt / tpu / Forward",
            "value": 0.00000152845,
            "unit": "s"
          },
          {
            "name": "sum    / DefOpt / tpu / Forward",
            "value": 0.000001493475,
            "unit": "s"
          },
          {
            "name": "sum    / IDefOpt / tpu / Forward",
            "value": 0.0000015284,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / tpu / PreRev",
            "value": 0.0000010533,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / tpu / PostRev",
            "value": 0.00000108595,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / tpu / BothRev",
            "value": 0.0000010503,
            "unit": "s"
          },
          {
            "name": "sum    / Jax / tpu / BothRev",
            "value": 0.000001090075,
            "unit": "s"
          },
          {
            "name": "sum    / HLOOpt / tpu / PreRev",
            "value": 0.000001049675,
            "unit": "s"
          },
          {
            "name": "sum    / HLOOpt / tpu / PostRev",
            "value": 0.0000010878000000000002,
            "unit": "s"
          },
          {
            "name": "sum    / HLOOpt / tpu / BothRev",
            "value": 0.0000010470500000000002,
            "unit": "s"
          },
          {
            "name": "sum    / PartOpt / tpu / PreRev",
            "value": 0.000001084425,
            "unit": "s"
          },
          {
            "name": "sum    / PartOpt / tpu / PostRev",
            "value": 0.000001054,
            "unit": "s"
          },
          {
            "name": "sum    / PartOpt / tpu / BothRev",
            "value": 0.0000010963000000000002,
            "unit": "s"
          },
          {
            "name": "sum    / IPartOpt / tpu / PreRev",
            "value": 0.00000105545,
            "unit": "s"
          },
          {
            "name": "sum    / IPartOpt / tpu / PostRev",
            "value": 0.000001087275,
            "unit": "s"
          },
          {
            "name": "sum    / IPartOpt / tpu / BothRev",
            "value": 0.00000105555,
            "unit": "s"
          },
          {
            "name": "sum    / DefOpt / tpu / PreRev",
            "value": 0.000001088475,
            "unit": "s"
          },
          {
            "name": "sum    / DefOpt / tpu / PostRev",
            "value": 0.00000105905,
            "unit": "s"
          },
          {
            "name": "sum    / DefOpt / tpu / BothRev",
            "value": 0.000001083975,
            "unit": "s"
          },
          {
            "name": "sum    / IDefOpt / tpu / PreRev",
            "value": 0.0000010464,
            "unit": "s"
          },
          {
            "name": "sum    / IDefOpt / tpu / PostRev",
            "value": 0.000001092975,
            "unit": "s"
          },
          {
            "name": "sum    / IDefOpt / tpu / BothRev",
            "value": 0.0000010563500000000005,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / Primal",
            "value": 0.000014441,
            "unit": "s"
          },
          {
            "name": "sum    / Jax / cpu / Primal",
            "value": 0.000014468,
            "unit": "s"
          },
          {
            "name": "sum    / HLOOpt / cpu / Primal",
            "value": 0.000014656,
            "unit": "s"
          },
          {
            "name": "sum    / PartOpt / cpu / Primal",
            "value": 0.000015005,
            "unit": "s"
          },
          {
            "name": "sum    / IPartOpt / cpu / Primal",
            "value": 0.000015006,
            "unit": "s"
          },
          {
            "name": "sum    / DefOpt / cpu / Primal",
            "value": 0.000014609,
            "unit": "s"
          },
          {
            "name": "sum    / IDefOpt / cpu / Primal",
            "value": 0.000014785,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / Forward",
            "value": 0.000020332,
            "unit": "s"
          },
          {
            "name": "sum    / Jax / cpu / Forward",
            "value": 0.00001993,
            "unit": "s"
          },
          {
            "name": "sum    / HLOOpt / cpu / Forward",
            "value": 0.00002025,
            "unit": "s"
          },
          {
            "name": "sum    / PartOpt / cpu / Forward",
            "value": 0.000020168,
            "unit": "s"
          },
          {
            "name": "sum    / IPartOpt / cpu / Forward",
            "value": 0.000019943,
            "unit": "s"
          },
          {
            "name": "sum    / DefOpt / cpu / Forward",
            "value": 0.000019774,
            "unit": "s"
          },
          {
            "name": "sum    / IDefOpt / cpu / Forward",
            "value": 0.000019657,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / PreRev",
            "value": 0.000018788,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / PostRev",
            "value": 0.000018464,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / BothRev",
            "value": 0.000018367,
            "unit": "s"
          },
          {
            "name": "sum    / Jax / cpu / BothRev",
            "value": 0.000018384,
            "unit": "s"
          },
          {
            "name": "sum    / HLOOpt / cpu / PreRev",
            "value": 0.000018582000000000003,
            "unit": "s"
          },
          {
            "name": "sum    / HLOOpt / cpu / PostRev",
            "value": 0.000019198,
            "unit": "s"
          },
          {
            "name": "sum    / HLOOpt / cpu / BothRev",
            "value": 0.000018508,
            "unit": "s"
          },
          {
            "name": "sum    / PartOpt / cpu / PreRev",
            "value": 0.000018756,
            "unit": "s"
          },
          {
            "name": "sum    / PartOpt / cpu / PostRev",
            "value": 0.000018602,
            "unit": "s"
          },
          {
            "name": "sum    / PartOpt / cpu / BothRev",
            "value": 0.000019516,
            "unit": "s"
          },
          {
            "name": "sum    / IPartOpt / cpu / PreRev",
            "value": 0.000019072,
            "unit": "s"
          },
          {
            "name": "sum    / IPartOpt / cpu / PostRev",
            "value": 0.000018535,
            "unit": "s"
          },
          {
            "name": "sum    / IPartOpt / cpu / BothRev",
            "value": 0.000018693,
            "unit": "s"
          },
          {
            "name": "sum    / DefOpt / cpu / PreRev",
            "value": 0.000018442,
            "unit": "s"
          },
          {
            "name": "sum    / DefOpt / cpu / PostRev",
            "value": 0.000018627,
            "unit": "s"
          },
          {
            "name": "sum    / DefOpt / cpu / BothRev",
            "value": 0.000018893,
            "unit": "s"
          },
          {
            "name": "sum    / IDefOpt / cpu / PreRev",
            "value": 0.000018708,
            "unit": "s"
          },
          {
            "name": "sum    / IDefOpt / cpu / PostRev",
            "value": 0.000018215,
            "unit": "s"
          },
          {
            "name": "sum    / IDefOpt / cpu / BothRev",
            "value": 0.00001898,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / Primal",
            "value": 0.00001,
            "unit": "s"
          },
          {
            "name": "sum    / Jax / cpu / Primal",
            "value": 0.00001,
            "unit": "s"
          },
          {
            "name": "sum    / HLOOpt / cpu / Primal",
            "value": 0.00001,
            "unit": "s"
          },
          {
            "name": "sum    / PartOpt / cpu / Primal",
            "value": 0.00001,
            "unit": "s"
          },
          {
            "name": "sum    / IPartOpt / cpu / Primal",
            "value": 0.00001,
            "unit": "s"
          },
          {
            "name": "sum    / DefOpt / cpu / Primal",
            "value": 0.00001,
            "unit": "s"
          },
          {
            "name": "sum    / IDefOpt / cpu / Primal",
            "value": 0.00001,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / Forward",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "sum    / Jax / cpu / Forward",
            "value": 0.000015,
            "unit": "s"
          },
          {
            "name": "sum    / HLOOpt / cpu / Forward",
            "value": 0.000015,
            "unit": "s"
          },
          {
            "name": "sum    / PartOpt / cpu / Forward",
            "value": 0.000015,
            "unit": "s"
          },
          {
            "name": "sum    / IPartOpt / cpu / Forward",
            "value": 0.000015,
            "unit": "s"
          },
          {
            "name": "sum    / DefOpt / cpu / Forward",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "sum    / IDefOpt / cpu / Forward",
            "value": 0.000015,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / PreRev",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / PostRev",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / BothRev",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "sum    / Jax / cpu / BothRev",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "sum    / HLOOpt / cpu / PreRev",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "sum    / HLOOpt / cpu / PostRev",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "sum    / HLOOpt / cpu / BothRev",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "sum    / PartOpt / cpu / PreRev",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "sum    / PartOpt / cpu / PostRev",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "sum    / PartOpt / cpu / BothRev",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "sum    / IPartOpt / cpu / PreRev",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "sum    / IPartOpt / cpu / PostRev",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "sum    / IPartOpt / cpu / BothRev",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "sum    / DefOpt / cpu / PreRev",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "sum    / DefOpt / cpu / PostRev",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "sum    / DefOpt / cpu / BothRev",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "sum    / IDefOpt / cpu / PreRev",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "sum    / IDefOpt / cpu / PostRev",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "sum    / IDefOpt / cpu / BothRev",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "value_and_grad / JaXPipe / cpu / Primal",
            "value": 0.00001542877999781922,
            "unit": "s"
          },
          {
            "name": "value_and_grad / Jax / cpu / Primal",
            "value": 0.000015932120040815788,
            "unit": "s"
          },
          {
            "name": "value_and_grad / HLOOpt / cpu / Primal",
            "value": 0.00001495937996878638,
            "unit": "s"
          },
          {
            "name": "value_and_grad / PartOpt / cpu / Primal",
            "value": 0.000015459419983017143,
            "unit": "s"
          },
          {
            "name": "value_and_grad / IPartOpt / cpu / Primal",
            "value": 0.000016022300014810755,
            "unit": "s"
          },
          {
            "name": "value_and_grad / DefOpt / cpu / Primal",
            "value": 0.000015086439989318025,
            "unit": "s"
          },
          {
            "name": "value_and_grad / IDefOpt / cpu / Primal",
            "value": 0.000015966919945640257,
            "unit": "s"
          },
          {
            "name": "value_and_grad / JaXPipe / cuda / Primal",
            "value": 0.000032192,
            "unit": "s"
          },
          {
            "name": "value_and_grad / Jax / cuda / Primal",
            "value": 0.000032832,
            "unit": "s"
          },
          {
            "name": "value_and_grad / HLOOpt / cuda / Primal",
            "value": 0.000033119999999999995,
            "unit": "s"
          },
          {
            "name": "value_and_grad / PartOpt / cuda / Primal",
            "value": 0.000032959,
            "unit": "s"
          },
          {
            "name": "value_and_grad / IPartOpt / cuda / Primal",
            "value": 0.000032543,
            "unit": "s"
          },
          {
            "name": "value_and_grad / DefOpt / cuda / Primal",
            "value": 0.000032288,
            "unit": "s"
          },
          {
            "name": "value_and_grad / IDefOpt / cuda / Primal",
            "value": 0.00003184,
            "unit": "s"
          },
          {
            "name": "value_and_grad / JaXPipe / tpu / Primal",
            "value": 0,
            "unit": "s"
          },
          {
            "name": "value_and_grad / Jax / tpu / Primal",
            "value": 0,
            "unit": "s"
          },
          {
            "name": "value_and_grad / HLOOpt / tpu / Primal",
            "value": 0,
            "unit": "s"
          },
          {
            "name": "value_and_grad / PartOpt / tpu / Primal",
            "value": 0,
            "unit": "s"
          },
          {
            "name": "value_and_grad / IPartOpt / tpu / Primal",
            "value": 0,
            "unit": "s"
          },
          {
            "name": "value_and_grad / DefOpt / tpu / Primal",
            "value": 0,
            "unit": "s"
          },
          {
            "name": "value_and_grad / IDefOpt / tpu / Primal",
            "value": 0,
            "unit": "s"
          },
          {
            "name": "value_and_grad / JaXPipe / cpu / Primal",
            "value": 0.000023279,
            "unit": "s"
          },
          {
            "name": "value_and_grad / Jax / cpu / Primal",
            "value": 0.000022178,
            "unit": "s"
          },
          {
            "name": "value_and_grad / HLOOpt / cpu / Primal",
            "value": 0.000022716,
            "unit": "s"
          },
          {
            "name": "value_and_grad / PartOpt / cpu / Primal",
            "value": 0.000022852,
            "unit": "s"
          },
          {
            "name": "value_and_grad / IPartOpt / cpu / Primal",
            "value": 0.000023704,
            "unit": "s"
          },
          {
            "name": "value_and_grad / DefOpt / cpu / Primal",
            "value": 0.000023115,
            "unit": "s"
          },
          {
            "name": "value_and_grad / IDefOpt / cpu / Primal",
            "value": 0.000023488,
            "unit": "s"
          },
          {
            "name": "value_and_grad / JaXPipe / cpu / Primal",
            "value": 0.000015,
            "unit": "s"
          },
          {
            "name": "value_and_grad / Jax / cpu / Primal",
            "value": 0.000016,
            "unit": "s"
          },
          {
            "name": "value_and_grad / HLOOpt / cpu / Primal",
            "value": 0.000017,
            "unit": "s"
          },
          {
            "name": "value_and_grad / PartOpt / cpu / Primal",
            "value": 0.000016,
            "unit": "s"
          },
          {
            "name": "value_and_grad / IPartOpt / cpu / Primal",
            "value": 0.000016,
            "unit": "s"
          },
          {
            "name": "value_and_grad / DefOpt / cpu / Primal",
            "value": 0.000016,
            "unit": "s"
          },
          {
            "name": "value_and_grad / IDefOpt / cpu / Primal",
            "value": 0.000015,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / JaXPipe / cuda / Primal",
            "value": 0.001502068,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / Jax / cuda / Primal",
            "value": 0.001515699,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / HLOOpt / cuda / Primal",
            "value": 0.001316338,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / PartOpt / cuda / Primal",
            "value": 0.001303734,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / IPartOpt / cuda / Primal",
            "value": 0.001324502,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / DefOpt / cuda / Primal",
            "value": 0.000916986,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / IDefOpt / cuda / Primal",
            "value": 0.000950392,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / JaXPipe / cuda / Forward",
            "value": 0.001554066,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / Jax / cuda / Forward",
            "value": 0.00180357,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / HLOOpt / cuda / Forward",
            "value": 0.001628338,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / PartOpt / cuda / Forward",
            "value": 0.001636595,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / IPartOpt / cuda / Forward",
            "value": 0.001628435,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / DefOpt / cuda / Forward",
            "value": 0.001648402,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / IDefOpt / cuda / Forward",
            "value": 0.001610963,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / JaXPipe / cuda / PreRev",
            "value": 0.00267236,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / JaXPipe / cuda / PostRev",
            "value": 0.005362591,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / JaXPipe / cuda / BothRev",
            "value": 0.0027405529999999,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / Jax / cuda / BothRev",
            "value": 0.005374405,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / HLOOpt / cuda / PreRev",
            "value": 0.002730279,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / HLOOpt / cuda / PostRev",
            "value": 0.005283831,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / HLOOpt / cuda / BothRev",
            "value": 0.00273217,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / PartOpt / cuda / PreRev",
            "value": 0.002867273,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / PartOpt / cuda / PostRev",
            "value": 0.005403988,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / PartOpt / cuda / BothRev",
            "value": 0.002758153,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / IPartOpt / cuda / PreRev",
            "value": 0.002793064,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / IPartOpt / cuda / PostRev",
            "value": 0.005419859,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / IPartOpt / cuda / BothRev",
            "value": 0.002777959,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / DefOpt / cuda / PreRev",
            "value": 0.002815432,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / DefOpt / cuda / PostRev",
            "value": 0.002721953,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / DefOpt / cuda / BothRev",
            "value": 0.0027546,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / IDefOpt / cuda / PreRev",
            "value": 0.002798055,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / IDefOpt / cuda / PostRev",
            "value": 0.00230251,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / IDefOpt / cuda / BothRev",
            "value": 0.002761108,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / JaXPipe / tpu / Primal",
            "value": 0.009287865,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / Jax / tpu / Primal",
            "value": 0.009278386875,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / HLOOpt / tpu / Primal",
            "value": 0.00917926125,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / PartOpt / tpu / Primal",
            "value": 0.009197413125,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / IPartOpt / tpu / Primal",
            "value": 0.009201908125,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / DefOpt / tpu / Primal",
            "value": 0.00879857125,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / IDefOpt / tpu / Primal",
            "value": 0.008701575,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / JaXPipe / tpu / Forward",
            "value": 0.017421941875,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / Jax / tpu / Forward",
            "value": 0.018750668125,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / HLOOpt / tpu / Forward",
            "value": 0.01740896375,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / PartOpt / tpu / Forward",
            "value": 0.017423065,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / IPartOpt / tpu / Forward",
            "value": 0.01741626125,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / DefOpt / tpu / Forward",
            "value": 0.0174240481249999,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / IDefOpt / tpu / Forward",
            "value": 0.0174109625,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / JaXPipe / tpu / PreRev",
            "value": 0.0254711174999999,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / JaXPipe / tpu / PostRev",
            "value": 0.02187455375,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / JaXPipe / tpu / BothRev",
            "value": 0.025454668125,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / Jax / tpu / BothRev",
            "value": 0.02187549375,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / HLOOpt / tpu / PreRev",
            "value": 0.025571803125,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / HLOOpt / tpu / PostRev",
            "value": 0.0208195193749999,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / HLOOpt / tpu / BothRev",
            "value": 0.025676069375,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / PartOpt / tpu / PreRev",
            "value": 0.025500503125,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / PartOpt / tpu / PostRev",
            "value": 0.021509539375,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / PartOpt / tpu / BothRev",
            "value": 0.025581229375,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / IPartOpt / tpu / PreRev",
            "value": 0.02546226125,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / IPartOpt / tpu / PostRev",
            "value": 0.021520011875,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / IPartOpt / tpu / BothRev",
            "value": 0.02555030875,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / DefOpt / tpu / PreRev",
            "value": 0.025501114375,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / DefOpt / tpu / PostRev",
            "value": 0.01881101125,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / DefOpt / tpu / BothRev",
            "value": 0.02558739625,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / IDefOpt / tpu / PreRev",
            "value": 0.02546454875,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / IDefOpt / tpu / PostRev",
            "value": 0.0183198049999999,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / IDefOpt / tpu / BothRev",
            "value": 0.025556130625,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / JaXPipe / cpu / Primal",
            "value": 0.0710995,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / Jax / cpu / Primal",
            "value": 0.064726397,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / HLOOpt / cpu / Primal",
            "value": 0.091431759,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / PartOpt / cpu / Primal",
            "value": 0.0735390809999999,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / IPartOpt / cpu / Primal",
            "value": 0.074325466,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / DefOpt / cpu / Primal",
            "value": 0.092193808,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / IDefOpt / cpu / Primal",
            "value": 0.0868588709999999,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / JaXPipe / cpu / Forward",
            "value": 0.169716206,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / Jax / cpu / Forward",
            "value": 0.096498974,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / HLOOpt / cpu / Forward",
            "value": 0.173637867,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / PartOpt / cpu / Forward",
            "value": 0.171666692,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / IPartOpt / cpu / Forward",
            "value": 0.166021188,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / DefOpt / cpu / Forward",
            "value": 0.163862997,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / IDefOpt / cpu / Forward",
            "value": 0.164408476,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / JaXPipe / cpu / PreRev",
            "value": 0.246420341,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / JaXPipe / cpu / PostRev",
            "value": 0.142790002,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / JaXPipe / cpu / BothRev",
            "value": 0.235643853,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / Jax / cpu / BothRev",
            "value": 0.14647163,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / HLOOpt / cpu / PreRev",
            "value": 0.242932352,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / HLOOpt / cpu / PostRev",
            "value": 0.179192015,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / HLOOpt / cpu / BothRev",
            "value": 0.257848239,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / PartOpt / cpu / PreRev",
            "value": 0.238793437,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / PartOpt / cpu / PostRev",
            "value": 0.130844221,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / PartOpt / cpu / BothRev",
            "value": 0.234667297,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / IPartOpt / cpu / PreRev",
            "value": 0.2315206159999999,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / IPartOpt / cpu / PostRev",
            "value": 0.139821482,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / IPartOpt / cpu / BothRev",
            "value": 0.25754629,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / DefOpt / cpu / PreRev",
            "value": 0.229896429,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / DefOpt / cpu / PostRev",
            "value": 0.1780652939999999,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / DefOpt / cpu / BothRev",
            "value": 0.232200429,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / IDefOpt / cpu / PreRev",
            "value": 0.227594538,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / IDefOpt / cpu / PostRev",
            "value": 0.179249319,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / IDefOpt / cpu / BothRev",
            "value": 0.238341839,
            "unit": "s"
          },
          {
            "name": "v1/deterministic_2_8_deg_inner_steps_24_outer_steps_4 / JaXPipe / cuda / Primal",
            "value": 1.702839258,
            "unit": "s"
          },
          {
            "name": "v1/deterministic_2_8_deg_inner_steps_24_outer_steps_4 / Jax / cuda / Primal",
            "value": 1.704278017,
            "unit": "s"
          },
          {
            "name": "v1/deterministic_2_8_deg_inner_steps_24_outer_steps_4 / HLOOpt / cuda / Primal",
            "value": 1.715041343,
            "unit": "s"
          },
          {
            "name": "v1/deterministic_2_8_deg_inner_steps_24_outer_steps_4 / PartOpt / cuda / Primal",
            "value": 1.695830297,
            "unit": "s"
          },
          {
            "name": "v1/deterministic_2_8_deg_inner_steps_24_outer_steps_4 / IPartOpt / cuda / Primal",
            "value": 1.694144053,
            "unit": "s"
          },
          {
            "name": "v1/deterministic_2_8_deg_inner_steps_24_outer_steps_4 / DefOpt / cuda / Primal",
            "value": 1.664635456,
            "unit": "s"
          },
          {
            "name": "v1/deterministic_2_8_deg_inner_steps_24_outer_steps_4 / IDefOpt / cuda / Primal",
            "value": 1.922402025,
            "unit": "s"
          },
          {
            "name": "v1/deterministic_2_8_deg_inner_steps_24_outer_steps_4 / JaXPipe / tpu / Primal",
            "value": 3.03854487875,
            "unit": "s"
          },
          {
            "name": "v1/deterministic_2_8_deg_inner_steps_24_outer_steps_4 / Jax / tpu / Primal",
            "value": 3.039158746875,
            "unit": "s"
          },
          {
            "name": "v1/deterministic_2_8_deg_inner_steps_24_outer_steps_4 / HLOOpt / tpu / Primal",
            "value": 3.12142853,
            "unit": "s"
          },
          {
            "name": "v1/deterministic_2_8_deg_inner_steps_24_outer_steps_4 / PartOpt / tpu / Primal",
            "value": 3.059994203125,
            "unit": "s"
          },
          {
            "name": "v1/deterministic_2_8_deg_inner_steps_24_outer_steps_4 / IPartOpt / tpu / Primal",
            "value": 3.060206984375,
            "unit": "s"
          },
          {
            "name": "v1/deterministic_2_8_deg_inner_steps_24_outer_steps_4 / DefOpt / tpu / Primal",
            "value": 2.102319526875,
            "unit": "s"
          },
          {
            "name": "v1/deterministic_2_8_deg_inner_steps_24_outer_steps_4 / IDefOpt / tpu / Primal",
            "value": 2.944417758125,
            "unit": "s"
          },
          {
            "name": "v1/deterministic_2_8_deg_inner_steps_2_outer_steps_2 / JaXPipe / cpu / Primal",
            "value": 6.302432162,
            "unit": "s"
          },
          {
            "name": "v1/deterministic_2_8_deg_inner_steps_2_outer_steps_2 / Jax / cpu / Primal",
            "value": 6.278327559,
            "unit": "s"
          },
          {
            "name": "v1/deterministic_2_8_deg_inner_steps_2_outer_steps_2 / HLOOpt / cpu / Primal",
            "value": 6.270322101,
            "unit": "s"
          },
          {
            "name": "v1/deterministic_2_8_deg_inner_steps_2_outer_steps_2 / PartOpt / cpu / Primal",
            "value": 6.222798666,
            "unit": "s"
          },
          {
            "name": "v1/deterministic_2_8_deg_inner_steps_2_outer_steps_2 / IPartOpt / cpu / Primal",
            "value": 6.359798141000001,
            "unit": "s"
          },
          {
            "name": "v1/deterministic_2_8_deg_inner_steps_2_outer_steps_2 / DefOpt / cpu / Primal",
            "value": 2.505259261,
            "unit": "s"
          },
          {
            "name": "v1/deterministic_2_8_deg_inner_steps_2_outer_steps_2 / IDefOpt / cpu / Primal",
            "value": 6.657719991,
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
          "id": "cd10cc0b58c6f2c065829870a7a77aa43d60266b",
          "message": "Fix update (#1849)",
          "timestamp": "2025-12-25T22:03:49-05:00",
          "tree_id": "a6428a3ae42c3ea264232ff02c9b68dbc38a78ec",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/cd10cc0b58c6f2c065829870a7a77aa43d60266b"
        },
        "date": 1766728973893,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "actmtch / JaXPipe / cpu / Primal",
            "value": 0.000006677699984720676,
            "unit": "s"
          },
          {
            "name": "actmtch / Jax / cpu / Primal",
            "value": 0.000006327040118776494,
            "unit": "s"
          },
          {
            "name": "actmtch / HLOOpt / cpu / Primal",
            "value": 0.000007292380032595247,
            "unit": "s"
          },
          {
            "name": "actmtch / PartOpt / cpu / Primal",
            "value": 0.000006817760040576104,
            "unit": "s"
          },
          {
            "name": "actmtch / IPartOpt / cpu / Primal",
            "value": 0.000006633379925915506,
            "unit": "s"
          },
          {
            "name": "actmtch / DefOpt / cpu / Primal",
            "value": 0.0000073802199949568605,
            "unit": "s"
          },
          {
            "name": "actmtch / IDefOpt / cpu / Primal",
            "value": 0.00000775096003053477,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / Forward",
            "value": 0.000010984460041072452,
            "unit": "s"
          },
          {
            "name": "actmtch / Jax / cpu / Forward",
            "value": 0.000012903859988000476,
            "unit": "s"
          },
          {
            "name": "actmtch / HLOOpt / cpu / Forward",
            "value": 0.000011052339941670652,
            "unit": "s"
          },
          {
            "name": "actmtch / PartOpt / cpu / Forward",
            "value": 0.000011166740023327291,
            "unit": "s"
          },
          {
            "name": "actmtch / IPartOpt / cpu / Forward",
            "value": 0.000010720759964897298,
            "unit": "s"
          },
          {
            "name": "actmtch / DefOpt / cpu / Forward",
            "value": 0.00001091942000130075,
            "unit": "s"
          },
          {
            "name": "actmtch / IDefOpt / cpu / Forward",
            "value": 0.000010500459993636468,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / PreRev",
            "value": 0.00001086826001483132,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / PostRev",
            "value": 0.000009911479992297243,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / BothRev",
            "value": 0.00002423846004603547,
            "unit": "s"
          },
          {
            "name": "actmtch / Jax / cpu / BothRev",
            "value": 0.000009888759996101726,
            "unit": "s"
          },
          {
            "name": "actmtch / HLOOpt / cpu / PreRev",
            "value": 0.000010959399969578953,
            "unit": "s"
          },
          {
            "name": "actmtch / HLOOpt / cpu / PostRev",
            "value": 0.000013419560073089087,
            "unit": "s"
          },
          {
            "name": "actmtch / HLOOpt / cpu / BothRev",
            "value": 0.000011109420083812438,
            "unit": "s"
          },
          {
            "name": "actmtch / PartOpt / cpu / PreRev",
            "value": 0.00001066435999746318,
            "unit": "s"
          },
          {
            "name": "actmtch / PartOpt / cpu / PostRev",
            "value": 0.000009673620097601087,
            "unit": "s"
          },
          {
            "name": "actmtch / PartOpt / cpu / BothRev",
            "value": 0.000011795880072895673,
            "unit": "s"
          },
          {
            "name": "actmtch / IPartOpt / cpu / PreRev",
            "value": 0.00001131957991674426,
            "unit": "s"
          },
          {
            "name": "actmtch / IPartOpt / cpu / PostRev",
            "value": 0.000009492879999015713,
            "unit": "s"
          },
          {
            "name": "actmtch / IPartOpt / cpu / BothRev",
            "value": 0.000011221099921385756,
            "unit": "s"
          },
          {
            "name": "actmtch / DefOpt / cpu / PreRev",
            "value": 0.000011053079906560016,
            "unit": "s"
          },
          {
            "name": "actmtch / DefOpt / cpu / PostRev",
            "value": 0.000011209419990336756,
            "unit": "s"
          },
          {
            "name": "actmtch / DefOpt / cpu / BothRev",
            "value": 0.00001118627995310817,
            "unit": "s"
          },
          {
            "name": "actmtch / IDefOpt / cpu / PreRev",
            "value": 0.000010687300036806846,
            "unit": "s"
          },
          {
            "name": "actmtch / IDefOpt / cpu / PostRev",
            "value": 0.00001195313991047442,
            "unit": "s"
          },
          {
            "name": "actmtch / IDefOpt / cpu / BothRev",
            "value": 0.000011010879989044042,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cuda / Primal",
            "value": 0.000002015,
            "unit": "s"
          },
          {
            "name": "actmtch / Jax / cuda / Primal",
            "value": 0.000002016,
            "unit": "s"
          },
          {
            "name": "actmtch / HLOOpt / cuda / Primal",
            "value": 0.000002015,
            "unit": "s"
          },
          {
            "name": "actmtch / PartOpt / cuda / Primal",
            "value": 0.000002015,
            "unit": "s"
          },
          {
            "name": "actmtch / IPartOpt / cuda / Primal",
            "value": 0.000002015,
            "unit": "s"
          },
          {
            "name": "actmtch / DefOpt / cuda / Primal",
            "value": 0.000002016,
            "unit": "s"
          },
          {
            "name": "actmtch / IDefOpt / cuda / Primal",
            "value": 0.000002016,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cuda / Forward",
            "value": 0.000010112,
            "unit": "s"
          },
          {
            "name": "actmtch / Jax / cuda / Forward",
            "value": 0.000009951,
            "unit": "s"
          },
          {
            "name": "actmtch / HLOOpt / cuda / Forward",
            "value": 0.000010336,
            "unit": "s"
          },
          {
            "name": "actmtch / PartOpt / cuda / Forward",
            "value": 0.000009984,
            "unit": "s"
          },
          {
            "name": "actmtch / IPartOpt / cuda / Forward",
            "value": 0.00001008,
            "unit": "s"
          },
          {
            "name": "actmtch / DefOpt / cuda / Forward",
            "value": 0.00001024,
            "unit": "s"
          },
          {
            "name": "actmtch / IDefOpt / cuda / Forward",
            "value": 0.000010144,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cuda / PreRev",
            "value": 0.000010112,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cuda / PostRev",
            "value": 0.000010528,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cuda / BothRev",
            "value": 0.000010272,
            "unit": "s"
          },
          {
            "name": "actmtch / Jax / cuda / BothRev",
            "value": 0.00001008,
            "unit": "s"
          },
          {
            "name": "actmtch / HLOOpt / cuda / PreRev",
            "value": 0.000010048,
            "unit": "s"
          },
          {
            "name": "actmtch / HLOOpt / cuda / PostRev",
            "value": 0.00001024,
            "unit": "s"
          },
          {
            "name": "actmtch / HLOOpt / cuda / BothRev",
            "value": 0.000010528,
            "unit": "s"
          },
          {
            "name": "actmtch / PartOpt / cuda / PreRev",
            "value": 0.0000104,
            "unit": "s"
          },
          {
            "name": "actmtch / PartOpt / cuda / PostRev",
            "value": 0.000010336,
            "unit": "s"
          },
          {
            "name": "actmtch / PartOpt / cuda / BothRev",
            "value": 0.000010496,
            "unit": "s"
          },
          {
            "name": "actmtch / IPartOpt / cuda / PreRev",
            "value": 0.000011488,
            "unit": "s"
          },
          {
            "name": "actmtch / IPartOpt / cuda / PostRev",
            "value": 0.0000104,
            "unit": "s"
          },
          {
            "name": "actmtch / IPartOpt / cuda / BothRev",
            "value": 0.000010144,
            "unit": "s"
          },
          {
            "name": "actmtch / DefOpt / cuda / PreRev",
            "value": 0.000011584,
            "unit": "s"
          },
          {
            "name": "actmtch / DefOpt / cuda / PostRev",
            "value": 0.000010144,
            "unit": "s"
          },
          {
            "name": "actmtch / DefOpt / cuda / BothRev",
            "value": 0.000010208,
            "unit": "s"
          },
          {
            "name": "actmtch / IDefOpt / cuda / PreRev",
            "value": 0.000010272,
            "unit": "s"
          },
          {
            "name": "actmtch / IDefOpt / cuda / PostRev",
            "value": 0.000010304,
            "unit": "s"
          },
          {
            "name": "actmtch / IDefOpt / cuda / BothRev",
            "value": 0.000010335,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / tpu / Primal",
            "value": 5.63475e-7,
            "unit": "s"
          },
          {
            "name": "actmtch / Jax / tpu / Primal",
            "value": 5.96825e-7,
            "unit": "s"
          },
          {
            "name": "actmtch / HLOOpt / tpu / Primal",
            "value": 0.0000021052750000000004,
            "unit": "s"
          },
          {
            "name": "actmtch / PartOpt / tpu / Primal",
            "value": 5.962e-7,
            "unit": "s"
          },
          {
            "name": "actmtch / IPartOpt / tpu / Primal",
            "value": 5.5215e-7,
            "unit": "s"
          },
          {
            "name": "actmtch / DefOpt / tpu / Primal",
            "value": 0.000002160625,
            "unit": "s"
          },
          {
            "name": "actmtch / IDefOpt / tpu / Primal",
            "value": 0.0000020993,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / tpu / Forward",
            "value": 0.00000383735,
            "unit": "s"
          },
          {
            "name": "actmtch / Jax / tpu / Forward",
            "value": 0.000001212075,
            "unit": "s"
          },
          {
            "name": "actmtch / HLOOpt / tpu / Forward",
            "value": 0.000003938675,
            "unit": "s"
          },
          {
            "name": "actmtch / PartOpt / tpu / Forward",
            "value": 0.00000391965,
            "unit": "s"
          },
          {
            "name": "actmtch / IPartOpt / tpu / Forward",
            "value": 0.000003928625,
            "unit": "s"
          },
          {
            "name": "actmtch / DefOpt / tpu / Forward",
            "value": 0.0000039157,
            "unit": "s"
          },
          {
            "name": "actmtch / IDefOpt / tpu / Forward",
            "value": 0.000003931575,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / tpu / PreRev",
            "value": 0.0000034923,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / tpu / PostRev",
            "value": 0.00000163735,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / tpu / BothRev",
            "value": 0.0000034919,
            "unit": "s"
          },
          {
            "name": "actmtch / Jax / tpu / BothRev",
            "value": 0.000001636325,
            "unit": "s"
          },
          {
            "name": "actmtch / HLOOpt / tpu / PreRev",
            "value": 0.000003492175,
            "unit": "s"
          },
          {
            "name": "actmtch / HLOOpt / tpu / PostRev",
            "value": 0.0000034151,
            "unit": "s"
          },
          {
            "name": "actmtch / HLOOpt / tpu / BothRev",
            "value": 0.000003488125,
            "unit": "s"
          },
          {
            "name": "actmtch / PartOpt / tpu / PreRev",
            "value": 0.0000034095,
            "unit": "s"
          },
          {
            "name": "actmtch / PartOpt / tpu / PostRev",
            "value": 0.0000015923249999999998,
            "unit": "s"
          },
          {
            "name": "actmtch / PartOpt / tpu / BothRev",
            "value": 0.0000034124,
            "unit": "s"
          },
          {
            "name": "actmtch / IPartOpt / tpu / PreRev",
            "value": 0.0000034745500000000005,
            "unit": "s"
          },
          {
            "name": "actmtch / IPartOpt / tpu / PostRev",
            "value": 0.000001635575,
            "unit": "s"
          },
          {
            "name": "actmtch / IPartOpt / tpu / BothRev",
            "value": 0.0000034898,
            "unit": "s"
          },
          {
            "name": "actmtch / DefOpt / tpu / PreRev",
            "value": 0.0000034107500000000003,
            "unit": "s"
          },
          {
            "name": "actmtch / DefOpt / tpu / PostRev",
            "value": 0.000003412225,
            "unit": "s"
          },
          {
            "name": "actmtch / DefOpt / tpu / BothRev",
            "value": 0.0000034089,
            "unit": "s"
          },
          {
            "name": "actmtch / IDefOpt / tpu / PreRev",
            "value": 0.0000034774500000000005,
            "unit": "s"
          },
          {
            "name": "actmtch / IDefOpt / tpu / PostRev",
            "value": 0.00000341395,
            "unit": "s"
          },
          {
            "name": "actmtch / IDefOpt / tpu / BothRev",
            "value": 0.00000347945,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / Primal",
            "value": 0.000016626,
            "unit": "s"
          },
          {
            "name": "actmtch / Jax / cpu / Primal",
            "value": 0.000016402,
            "unit": "s"
          },
          {
            "name": "actmtch / HLOOpt / cpu / Primal",
            "value": 0.00001747,
            "unit": "s"
          },
          {
            "name": "actmtch / PartOpt / cpu / Primal",
            "value": 0.000016567999999999998,
            "unit": "s"
          },
          {
            "name": "actmtch / IPartOpt / cpu / Primal",
            "value": 0.000016282,
            "unit": "s"
          },
          {
            "name": "actmtch / DefOpt / cpu / Primal",
            "value": 0.000017658,
            "unit": "s"
          },
          {
            "name": "actmtch / IDefOpt / cpu / Primal",
            "value": 0.00001726,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / Forward",
            "value": 0.000023664,
            "unit": "s"
          },
          {
            "name": "actmtch / Jax / cpu / Forward",
            "value": 0.000022402,
            "unit": "s"
          },
          {
            "name": "actmtch / HLOOpt / cpu / Forward",
            "value": 0.00002327,
            "unit": "s"
          },
          {
            "name": "actmtch / PartOpt / cpu / Forward",
            "value": 0.000024008,
            "unit": "s"
          },
          {
            "name": "actmtch / IPartOpt / cpu / Forward",
            "value": 0.000023998,
            "unit": "s"
          },
          {
            "name": "actmtch / DefOpt / cpu / Forward",
            "value": 0.000023818,
            "unit": "s"
          },
          {
            "name": "actmtch / IDefOpt / cpu / Forward",
            "value": 0.000023512,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / PreRev",
            "value": 0.000024128,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / PostRev",
            "value": 0.00002198,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / BothRev",
            "value": 0.000024121,
            "unit": "s"
          },
          {
            "name": "actmtch / Jax / cpu / BothRev",
            "value": 0.000022138,
            "unit": "s"
          },
          {
            "name": "actmtch / HLOOpt / cpu / PreRev",
            "value": 0.000023957,
            "unit": "s"
          },
          {
            "name": "actmtch / HLOOpt / cpu / PostRev",
            "value": 0.000023541,
            "unit": "s"
          },
          {
            "name": "actmtch / HLOOpt / cpu / BothRev",
            "value": 0.00002399,
            "unit": "s"
          },
          {
            "name": "actmtch / PartOpt / cpu / PreRev",
            "value": 0.000024212,
            "unit": "s"
          },
          {
            "name": "actmtch / PartOpt / cpu / PostRev",
            "value": 0.000022121,
            "unit": "s"
          },
          {
            "name": "actmtch / PartOpt / cpu / BothRev",
            "value": 0.000024031,
            "unit": "s"
          },
          {
            "name": "actmtch / IPartOpt / cpu / PreRev",
            "value": 0.000024265,
            "unit": "s"
          },
          {
            "name": "actmtch / IPartOpt / cpu / PostRev",
            "value": 0.000022561,
            "unit": "s"
          },
          {
            "name": "actmtch / IPartOpt / cpu / BothRev",
            "value": 0.000023998,
            "unit": "s"
          },
          {
            "name": "actmtch / DefOpt / cpu / PreRev",
            "value": 0.000023752,
            "unit": "s"
          },
          {
            "name": "actmtch / DefOpt / cpu / PostRev",
            "value": 0.000024247,
            "unit": "s"
          },
          {
            "name": "actmtch / DefOpt / cpu / BothRev",
            "value": 0.000024035,
            "unit": "s"
          },
          {
            "name": "actmtch / IDefOpt / cpu / PreRev",
            "value": 0.00002406,
            "unit": "s"
          },
          {
            "name": "actmtch / IDefOpt / cpu / PostRev",
            "value": 0.000023758,
            "unit": "s"
          },
          {
            "name": "actmtch / IDefOpt / cpu / BothRev",
            "value": 0.000024263,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / Primal",
            "value": 0.000008999999999999999,
            "unit": "s"
          },
          {
            "name": "actmtch / Jax / cpu / Primal",
            "value": 0.000008999999999999999,
            "unit": "s"
          },
          {
            "name": "actmtch / HLOOpt / cpu / Primal",
            "value": 0.000008999999999999999,
            "unit": "s"
          },
          {
            "name": "actmtch / PartOpt / cpu / Primal",
            "value": 0.000008999999999999999,
            "unit": "s"
          },
          {
            "name": "actmtch / IPartOpt / cpu / Primal",
            "value": 0.000008999999999999999,
            "unit": "s"
          },
          {
            "name": "actmtch / DefOpt / cpu / Primal",
            "value": 0.000008999999999999999,
            "unit": "s"
          },
          {
            "name": "actmtch / IDefOpt / cpu / Primal",
            "value": 0.000008999999999999999,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / Forward",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "actmtch / Jax / cpu / Forward",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "actmtch / HLOOpt / cpu / Forward",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "actmtch / PartOpt / cpu / Forward",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "actmtch / IPartOpt / cpu / Forward",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "actmtch / DefOpt / cpu / Forward",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "actmtch / IDefOpt / cpu / Forward",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / PreRev",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / PostRev",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / BothRev",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "actmtch / Jax / cpu / BothRev",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "actmtch / HLOOpt / cpu / PreRev",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "actmtch / HLOOpt / cpu / PostRev",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "actmtch / HLOOpt / cpu / BothRev",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "actmtch / PartOpt / cpu / PreRev",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "actmtch / PartOpt / cpu / PostRev",
            "value": 0.000011,
            "unit": "s"
          },
          {
            "name": "actmtch / PartOpt / cpu / BothRev",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "actmtch / IPartOpt / cpu / PreRev",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "actmtch / IPartOpt / cpu / PostRev",
            "value": 0.000011,
            "unit": "s"
          },
          {
            "name": "actmtch / IPartOpt / cpu / BothRev",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "actmtch / DefOpt / cpu / PreRev",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "actmtch / DefOpt / cpu / PostRev",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "actmtch / DefOpt / cpu / BothRev",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "actmtch / IDefOpt / cpu / PreRev",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "actmtch / IDefOpt / cpu / PostRev",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "actmtch / IDefOpt / cpu / BothRev",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / Primal",
            "value": 0.000006748300002072938,
            "unit": "s"
          },
          {
            "name": "add_one / Jax / cpu / Primal",
            "value": 0.000006924019926373148,
            "unit": "s"
          },
          {
            "name": "add_one / HLOOpt / cpu / Primal",
            "value": 0.0000066970199441129806,
            "unit": "s"
          },
          {
            "name": "add_one / PartOpt / cpu / Primal",
            "value": 0.000006739959972037468,
            "unit": "s"
          },
          {
            "name": "add_one / IPartOpt / cpu / Primal",
            "value": 0.000007212739892565878,
            "unit": "s"
          },
          {
            "name": "add_one / DefOpt / cpu / Primal",
            "value": 0.000006584799994016066,
            "unit": "s"
          },
          {
            "name": "add_one / IDefOpt / cpu / Primal",
            "value": 0.000006710540092171869,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / Forward",
            "value": 0.000010220080002909529,
            "unit": "s"
          },
          {
            "name": "add_one / Jax / cpu / Forward",
            "value": 0.000010129780057468452,
            "unit": "s"
          },
          {
            "name": "add_one / HLOOpt / cpu / Forward",
            "value": 0.000010372880042268662,
            "unit": "s"
          },
          {
            "name": "add_one / PartOpt / cpu / Forward",
            "value": 0.000010067840012197849,
            "unit": "s"
          },
          {
            "name": "add_one / IPartOpt / cpu / Forward",
            "value": 0.000010083619999932125,
            "unit": "s"
          },
          {
            "name": "add_one / DefOpt / cpu / Forward",
            "value": 0.000010427679990243633,
            "unit": "s"
          },
          {
            "name": "add_one / IDefOpt / cpu / Forward",
            "value": 0.000009794899997359609,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / PreRev",
            "value": 0.000012240480064065196,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / PostRev",
            "value": 0.000011755699997593185,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / BothRev",
            "value": 0.000012541060023067983,
            "unit": "s"
          },
          {
            "name": "add_one / Jax / cpu / BothRev",
            "value": 0.000011830059993371833,
            "unit": "s"
          },
          {
            "name": "add_one / HLOOpt / cpu / PreRev",
            "value": 0.000011884800042025745,
            "unit": "s"
          },
          {
            "name": "add_one / HLOOpt / cpu / PostRev",
            "value": 0.000013778119991911809,
            "unit": "s"
          },
          {
            "name": "add_one / HLOOpt / cpu / BothRev",
            "value": 0.00001166169997304678,
            "unit": "s"
          },
          {
            "name": "add_one / PartOpt / cpu / PreRev",
            "value": 0.000012133520031056832,
            "unit": "s"
          },
          {
            "name": "add_one / PartOpt / cpu / PostRev",
            "value": 0.000012131100011174568,
            "unit": "s"
          },
          {
            "name": "add_one / PartOpt / cpu / BothRev",
            "value": 0.000011996700013696682,
            "unit": "s"
          },
          {
            "name": "add_one / IPartOpt / cpu / PreRev",
            "value": 0.000012057380045007447,
            "unit": "s"
          },
          {
            "name": "add_one / IPartOpt / cpu / PostRev",
            "value": 0.00001207880004585604,
            "unit": "s"
          },
          {
            "name": "add_one / IPartOpt / cpu / BothRev",
            "value": 0.000011708939891832415,
            "unit": "s"
          },
          {
            "name": "add_one / DefOpt / cpu / PreRev",
            "value": 0.000012144679967605045,
            "unit": "s"
          },
          {
            "name": "add_one / DefOpt / cpu / PostRev",
            "value": 0.000012559339993458706,
            "unit": "s"
          },
          {
            "name": "add_one / DefOpt / cpu / BothRev",
            "value": 0.000011503080022521316,
            "unit": "s"
          },
          {
            "name": "add_one / IDefOpt / cpu / PreRev",
            "value": 0.0000117186799798219,
            "unit": "s"
          },
          {
            "name": "add_one / IDefOpt / cpu / PostRev",
            "value": 0.000011560459988686489,
            "unit": "s"
          },
          {
            "name": "add_one / IDefOpt / cpu / BothRev",
            "value": 0.000011963379947701469,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cuda / Primal",
            "value": 0.0000019200000000000003,
            "unit": "s"
          },
          {
            "name": "add_one / Jax / cuda / Primal",
            "value": 0.000001951,
            "unit": "s"
          },
          {
            "name": "add_one / HLOOpt / cuda / Primal",
            "value": 0.0000019200000000000003,
            "unit": "s"
          },
          {
            "name": "add_one / PartOpt / cuda / Primal",
            "value": 0.0000019200000000000003,
            "unit": "s"
          },
          {
            "name": "add_one / IPartOpt / cuda / Primal",
            "value": 0.0000019200000000000003,
            "unit": "s"
          },
          {
            "name": "add_one / DefOpt / cuda / Primal",
            "value": 0.000001951,
            "unit": "s"
          },
          {
            "name": "add_one / IDefOpt / cuda / Primal",
            "value": 0.0000019200000000000003,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cuda / Forward",
            "value": 0.000010432,
            "unit": "s"
          },
          {
            "name": "add_one / Jax / cuda / Forward",
            "value": 0.000010336,
            "unit": "s"
          },
          {
            "name": "add_one / HLOOpt / cuda / Forward",
            "value": 0.00001056,
            "unit": "s"
          },
          {
            "name": "add_one / PartOpt / cuda / Forward",
            "value": 0.000010336,
            "unit": "s"
          },
          {
            "name": "add_one / IPartOpt / cuda / Forward",
            "value": 0.000010368,
            "unit": "s"
          },
          {
            "name": "add_one / DefOpt / cuda / Forward",
            "value": 0.000010336,
            "unit": "s"
          },
          {
            "name": "add_one / IDefOpt / cuda / Forward",
            "value": 0.000010112,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cuda / PreRev",
            "value": 0.000024928,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cuda / PostRev",
            "value": 0.000029568,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cuda / BothRev",
            "value": 0.000029151,
            "unit": "s"
          },
          {
            "name": "add_one / Jax / cuda / BothRev",
            "value": 0.000025184,
            "unit": "s"
          },
          {
            "name": "add_one / HLOOpt / cuda / PreRev",
            "value": 0.000028704,
            "unit": "s"
          },
          {
            "name": "add_one / HLOOpt / cuda / PostRev",
            "value": 0.000028768,
            "unit": "s"
          },
          {
            "name": "add_one / HLOOpt / cuda / BothRev",
            "value": 0.000024864,
            "unit": "s"
          },
          {
            "name": "add_one / PartOpt / cuda / PreRev",
            "value": 0.000025024,
            "unit": "s"
          },
          {
            "name": "add_one / PartOpt / cuda / PostRev",
            "value": 0.000024928,
            "unit": "s"
          },
          {
            "name": "add_one / PartOpt / cuda / BothRev",
            "value": 0.000024736,
            "unit": "s"
          },
          {
            "name": "add_one / IPartOpt / cuda / PreRev",
            "value": 0.000025408,
            "unit": "s"
          },
          {
            "name": "add_one / IPartOpt / cuda / PostRev",
            "value": 0.000024959,
            "unit": "s"
          },
          {
            "name": "add_one / IPartOpt / cuda / BothRev",
            "value": 0.000025472000000000003,
            "unit": "s"
          },
          {
            "name": "add_one / DefOpt / cuda / PreRev",
            "value": 0.000025504,
            "unit": "s"
          },
          {
            "name": "add_one / DefOpt / cuda / PostRev",
            "value": 0.000025792,
            "unit": "s"
          },
          {
            "name": "add_one / DefOpt / cuda / BothRev",
            "value": 0.000025312,
            "unit": "s"
          },
          {
            "name": "add_one / IDefOpt / cuda / PreRev",
            "value": 0.000025152,
            "unit": "s"
          },
          {
            "name": "add_one / IDefOpt / cuda / PostRev",
            "value": 0.000025664,
            "unit": "s"
          },
          {
            "name": "add_one / IDefOpt / cuda / BothRev",
            "value": 0.000025343,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / tpu / Primal",
            "value": 0.00000142065,
            "unit": "s"
          },
          {
            "name": "add_one / Jax / tpu / Primal",
            "value": 0.00000140065,
            "unit": "s"
          },
          {
            "name": "add_one / HLOOpt / tpu / Primal",
            "value": 0.000001423025,
            "unit": "s"
          },
          {
            "name": "add_one / PartOpt / tpu / Primal",
            "value": 0.0000014052,
            "unit": "s"
          },
          {
            "name": "add_one / IPartOpt / tpu / Primal",
            "value": 0.0000014233750000000002,
            "unit": "s"
          },
          {
            "name": "add_one / DefOpt / tpu / Primal",
            "value": 0.0000014037250000000005,
            "unit": "s"
          },
          {
            "name": "add_one / IDefOpt / tpu / Primal",
            "value": 0.0000014266499999999995,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / tpu / Forward",
            "value": 0.0000018499,
            "unit": "s"
          },
          {
            "name": "add_one / Jax / tpu / Forward",
            "value": 0.00000184655,
            "unit": "s"
          },
          {
            "name": "add_one / HLOOpt / tpu / Forward",
            "value": 0.000001851625,
            "unit": "s"
          },
          {
            "name": "add_one / PartOpt / tpu / Forward",
            "value": 0.000001837875,
            "unit": "s"
          },
          {
            "name": "add_one / IPartOpt / tpu / Forward",
            "value": 0.0000018498,
            "unit": "s"
          },
          {
            "name": "add_one / DefOpt / tpu / Forward",
            "value": 0.000001837775,
            "unit": "s"
          },
          {
            "name": "add_one / IDefOpt / tpu / Forward",
            "value": 0.000001849375,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / tpu / PreRev",
            "value": 0.000002239675,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / tpu / PostRev",
            "value": 0.0000022398750000000003,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / tpu / BothRev",
            "value": 0.0000022346750000000003,
            "unit": "s"
          },
          {
            "name": "add_one / Jax / tpu / BothRev",
            "value": 0.0000022423,
            "unit": "s"
          },
          {
            "name": "add_one / HLOOpt / tpu / PreRev",
            "value": 0.0000022371750000000003,
            "unit": "s"
          },
          {
            "name": "add_one / HLOOpt / tpu / PostRev",
            "value": 0.000002238925,
            "unit": "s"
          },
          {
            "name": "add_one / HLOOpt / tpu / BothRev",
            "value": 0.0000022372,
            "unit": "s"
          },
          {
            "name": "add_one / PartOpt / tpu / PreRev",
            "value": 0.0000022363750000000003,
            "unit": "s"
          },
          {
            "name": "add_one / PartOpt / tpu / PostRev",
            "value": 0.000002247075,
            "unit": "s"
          },
          {
            "name": "add_one / PartOpt / tpu / BothRev",
            "value": 0.000002242225,
            "unit": "s"
          },
          {
            "name": "add_one / IPartOpt / tpu / PreRev",
            "value": 0.000002238675,
            "unit": "s"
          },
          {
            "name": "add_one / IPartOpt / tpu / PostRev",
            "value": 0.000002254425,
            "unit": "s"
          },
          {
            "name": "add_one / IPartOpt / tpu / BothRev",
            "value": 0.000002238325,
            "unit": "s"
          },
          {
            "name": "add_one / DefOpt / tpu / PreRev",
            "value": 0.00000223945,
            "unit": "s"
          },
          {
            "name": "add_one / DefOpt / tpu / PostRev",
            "value": 0.0000022285,
            "unit": "s"
          },
          {
            "name": "add_one / DefOpt / tpu / BothRev",
            "value": 0.000002245125,
            "unit": "s"
          },
          {
            "name": "add_one / IDefOpt / tpu / PreRev",
            "value": 0.0000022398750000000003,
            "unit": "s"
          },
          {
            "name": "add_one / IDefOpt / tpu / PostRev",
            "value": 0.000002242525,
            "unit": "s"
          },
          {
            "name": "add_one / IDefOpt / tpu / BothRev",
            "value": 0.0000022398750000000003,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / Primal",
            "value": 0.000016219999999999997,
            "unit": "s"
          },
          {
            "name": "add_one / Jax / cpu / Primal",
            "value": 0.000016135,
            "unit": "s"
          },
          {
            "name": "add_one / HLOOpt / cpu / Primal",
            "value": 0.000016395,
            "unit": "s"
          },
          {
            "name": "add_one / PartOpt / cpu / Primal",
            "value": 0.000015878,
            "unit": "s"
          },
          {
            "name": "add_one / IPartOpt / cpu / Primal",
            "value": 0.000016122999999999998,
            "unit": "s"
          },
          {
            "name": "add_one / DefOpt / cpu / Primal",
            "value": 0.000016102000000000003,
            "unit": "s"
          },
          {
            "name": "add_one / IDefOpt / cpu / Primal",
            "value": 0.000016202999999999997,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / Forward",
            "value": 0.000022371,
            "unit": "s"
          },
          {
            "name": "add_one / Jax / cpu / Forward",
            "value": 0.000021838,
            "unit": "s"
          },
          {
            "name": "add_one / HLOOpt / cpu / Forward",
            "value": 0.000022172,
            "unit": "s"
          },
          {
            "name": "add_one / PartOpt / cpu / Forward",
            "value": 0.000022362,
            "unit": "s"
          },
          {
            "name": "add_one / IPartOpt / cpu / Forward",
            "value": 0.000022053,
            "unit": "s"
          },
          {
            "name": "add_one / DefOpt / cpu / Forward",
            "value": 0.000021988,
            "unit": "s"
          },
          {
            "name": "add_one / IDefOpt / cpu / Forward",
            "value": 0.0000219,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / PreRev",
            "value": 0.000024736,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / PostRev",
            "value": 0.000024275,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / BothRev",
            "value": 0.00002426,
            "unit": "s"
          },
          {
            "name": "add_one / Jax / cpu / BothRev",
            "value": 0.00002411,
            "unit": "s"
          },
          {
            "name": "add_one / HLOOpt / cpu / PreRev",
            "value": 0.000024308,
            "unit": "s"
          },
          {
            "name": "add_one / HLOOpt / cpu / PostRev",
            "value": 0.000024458,
            "unit": "s"
          },
          {
            "name": "add_one / HLOOpt / cpu / BothRev",
            "value": 0.000024244,
            "unit": "s"
          },
          {
            "name": "add_one / PartOpt / cpu / PreRev",
            "value": 0.000024084,
            "unit": "s"
          },
          {
            "name": "add_one / PartOpt / cpu / PostRev",
            "value": 0.000024587,
            "unit": "s"
          },
          {
            "name": "add_one / PartOpt / cpu / BothRev",
            "value": 0.000024256,
            "unit": "s"
          },
          {
            "name": "add_one / IPartOpt / cpu / PreRev",
            "value": 0.000030221,
            "unit": "s"
          },
          {
            "name": "add_one / IPartOpt / cpu / PostRev",
            "value": 0.000024362,
            "unit": "s"
          },
          {
            "name": "add_one / IPartOpt / cpu / BothRev",
            "value": 0.00002429,
            "unit": "s"
          },
          {
            "name": "add_one / DefOpt / cpu / PreRev",
            "value": 0.000024338,
            "unit": "s"
          },
          {
            "name": "add_one / DefOpt / cpu / PostRev",
            "value": 0.000024078,
            "unit": "s"
          },
          {
            "name": "add_one / DefOpt / cpu / BothRev",
            "value": 0.00002437,
            "unit": "s"
          },
          {
            "name": "add_one / IDefOpt / cpu / PreRev",
            "value": 0.000024253,
            "unit": "s"
          },
          {
            "name": "add_one / IDefOpt / cpu / PostRev",
            "value": 0.00002448,
            "unit": "s"
          },
          {
            "name": "add_one / IDefOpt / cpu / BothRev",
            "value": 0.0000246,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / Primal",
            "value": 0.000008,
            "unit": "s"
          },
          {
            "name": "add_one / Jax / cpu / Primal",
            "value": 0.000008,
            "unit": "s"
          },
          {
            "name": "add_one / HLOOpt / cpu / Primal",
            "value": 0.000008,
            "unit": "s"
          },
          {
            "name": "add_one / PartOpt / cpu / Primal",
            "value": 0.000008,
            "unit": "s"
          },
          {
            "name": "add_one / IPartOpt / cpu / Primal",
            "value": 0.000008,
            "unit": "s"
          },
          {
            "name": "add_one / DefOpt / cpu / Primal",
            "value": 0.000008,
            "unit": "s"
          },
          {
            "name": "add_one / IDefOpt / cpu / Primal",
            "value": 0.000008,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / Forward",
            "value": 0.000011,
            "unit": "s"
          },
          {
            "name": "add_one / Jax / cpu / Forward",
            "value": 0.000011,
            "unit": "s"
          },
          {
            "name": "add_one / HLOOpt / cpu / Forward",
            "value": 0.000011,
            "unit": "s"
          },
          {
            "name": "add_one / PartOpt / cpu / Forward",
            "value": 0.000011,
            "unit": "s"
          },
          {
            "name": "add_one / IPartOpt / cpu / Forward",
            "value": 0.000011,
            "unit": "s"
          },
          {
            "name": "add_one / DefOpt / cpu / Forward",
            "value": 0.000011,
            "unit": "s"
          },
          {
            "name": "add_one / IDefOpt / cpu / Forward",
            "value": 0.00001,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / PreRev",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / PostRev",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / BothRev",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "add_one / Jax / cpu / BothRev",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "add_one / HLOOpt / cpu / PreRev",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "add_one / HLOOpt / cpu / PostRev",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "add_one / HLOOpt / cpu / BothRev",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "add_one / PartOpt / cpu / PreRev",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "add_one / PartOpt / cpu / PostRev",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "add_one / PartOpt / cpu / BothRev",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "add_one / IPartOpt / cpu / PreRev",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "add_one / IPartOpt / cpu / PostRev",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "add_one / IPartOpt / cpu / BothRev",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "add_one / DefOpt / cpu / PreRev",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "add_one / DefOpt / cpu / PostRev",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "add_one / DefOpt / cpu / BothRev",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "add_one / IDefOpt / cpu / PreRev",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "add_one / IDefOpt / cpu / PostRev",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "add_one / IDefOpt / cpu / BothRev",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / Primal",
            "value": 0.000007226979923871113,
            "unit": "s"
          },
          {
            "name": "add_two / Jax / cpu / Primal",
            "value": 0.00000671122001222102,
            "unit": "s"
          },
          {
            "name": "add_two / HLOOpt / cpu / Primal",
            "value": 0.0000075329800711188,
            "unit": "s"
          },
          {
            "name": "add_two / PartOpt / cpu / Primal",
            "value": 0.00000720957999874372,
            "unit": "s"
          },
          {
            "name": "add_two / IPartOpt / cpu / Primal",
            "value": 0.000007707740060141078,
            "unit": "s"
          },
          {
            "name": "add_two / DefOpt / cpu / Primal",
            "value": 0.000006753639972885139,
            "unit": "s"
          },
          {
            "name": "add_two / IDefOpt / cpu / Primal",
            "value": 0.000006565899911947781,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / Forward",
            "value": 0.00001063892003003275,
            "unit": "s"
          },
          {
            "name": "add_two / Jax / cpu / Forward",
            "value": 0.000010400380087958185,
            "unit": "s"
          },
          {
            "name": "add_two / HLOOpt / cpu / Forward",
            "value": 0.000010655540081643267,
            "unit": "s"
          },
          {
            "name": "add_two / PartOpt / cpu / Forward",
            "value": 0.000010579180052445736,
            "unit": "s"
          },
          {
            "name": "add_two / IPartOpt / cpu / Forward",
            "value": 0.000010587219985609407,
            "unit": "s"
          },
          {
            "name": "add_two / DefOpt / cpu / Forward",
            "value": 0.000010224479956377765,
            "unit": "s"
          },
          {
            "name": "add_two / IDefOpt / cpu / Forward",
            "value": 0.000010860319944185904,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / PreRev",
            "value": 0.000014268560007622,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / PostRev",
            "value": 0.000014013340023666388,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / BothRev",
            "value": 0.000014106060061749304,
            "unit": "s"
          },
          {
            "name": "add_two / Jax / cpu / BothRev",
            "value": 0.000014173739982652478,
            "unit": "s"
          },
          {
            "name": "add_two / HLOOpt / cpu / PreRev",
            "value": 0.000013847699992766138,
            "unit": "s"
          },
          {
            "name": "add_two / HLOOpt / cpu / PostRev",
            "value": 0.00001590562003912055,
            "unit": "s"
          },
          {
            "name": "add_two / HLOOpt / cpu / BothRev",
            "value": 0.000014184340016072384,
            "unit": "s"
          },
          {
            "name": "add_two / PartOpt / cpu / PreRev",
            "value": 0.000014589099955628624,
            "unit": "s"
          },
          {
            "name": "add_two / PartOpt / cpu / PostRev",
            "value": 0.000014465499953075778,
            "unit": "s"
          },
          {
            "name": "add_two / PartOpt / cpu / BothRev",
            "value": 0.000014385500035132282,
            "unit": "s"
          },
          {
            "name": "add_two / IPartOpt / cpu / PreRev",
            "value": 0.00001437407996490947,
            "unit": "s"
          },
          {
            "name": "add_two / IPartOpt / cpu / PostRev",
            "value": 0.000013780300087091746,
            "unit": "s"
          },
          {
            "name": "add_two / IPartOpt / cpu / BothRev",
            "value": 0.000014333200069813757,
            "unit": "s"
          },
          {
            "name": "add_two / DefOpt / cpu / PreRev",
            "value": 0.0000145685800089268,
            "unit": "s"
          },
          {
            "name": "add_two / DefOpt / cpu / PostRev",
            "value": 0.00001367940012642066,
            "unit": "s"
          },
          {
            "name": "add_two / DefOpt / cpu / BothRev",
            "value": 0.00001363729992590379,
            "unit": "s"
          },
          {
            "name": "add_two / IDefOpt / cpu / PreRev",
            "value": 0.000014593519990739878,
            "unit": "s"
          },
          {
            "name": "add_two / IDefOpt / cpu / PostRev",
            "value": 0.000014398379971680696,
            "unit": "s"
          },
          {
            "name": "add_two / IDefOpt / cpu / BothRev",
            "value": 0.000013956239890831056,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cuda / Primal",
            "value": 0.0000019200000000000003,
            "unit": "s"
          },
          {
            "name": "add_two / Jax / cuda / Primal",
            "value": 0.0000019200000000000003,
            "unit": "s"
          },
          {
            "name": "add_two / HLOOpt / cuda / Primal",
            "value": 0.0000019200000000000003,
            "unit": "s"
          },
          {
            "name": "add_two / PartOpt / cuda / Primal",
            "value": 0.0000019200000000000003,
            "unit": "s"
          },
          {
            "name": "add_two / IPartOpt / cuda / Primal",
            "value": 0.0000019200000000000003,
            "unit": "s"
          },
          {
            "name": "add_two / DefOpt / cuda / Primal",
            "value": 0.0000019200000000000003,
            "unit": "s"
          },
          {
            "name": "add_two / IDefOpt / cuda / Primal",
            "value": 0.0000019200000000000003,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cuda / Forward",
            "value": 0.000009856,
            "unit": "s"
          },
          {
            "name": "add_two / Jax / cuda / Forward",
            "value": 0.000009665,
            "unit": "s"
          },
          {
            "name": "add_two / HLOOpt / cuda / Forward",
            "value": 0.00001008,
            "unit": "s"
          },
          {
            "name": "add_two / PartOpt / cuda / Forward",
            "value": 0.000010048,
            "unit": "s"
          },
          {
            "name": "add_two / IPartOpt / cuda / Forward",
            "value": 0.000009985,
            "unit": "s"
          },
          {
            "name": "add_two / DefOpt / cuda / Forward",
            "value": 0.000009888,
            "unit": "s"
          },
          {
            "name": "add_two / IDefOpt / cuda / Forward",
            "value": 0.00001008,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cuda / PreRev",
            "value": 0.00003168,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cuda / PostRev",
            "value": 0.000032959,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cuda / BothRev",
            "value": 0.000032063,
            "unit": "s"
          },
          {
            "name": "add_two / Jax / cuda / BothRev",
            "value": 0.000032576,
            "unit": "s"
          },
          {
            "name": "add_two / HLOOpt / cuda / PreRev",
            "value": 0.000034112,
            "unit": "s"
          },
          {
            "name": "add_two / HLOOpt / cuda / PostRev",
            "value": 0.000031968,
            "unit": "s"
          },
          {
            "name": "add_two / HLOOpt / cuda / BothRev",
            "value": 0.000033184,
            "unit": "s"
          },
          {
            "name": "add_two / PartOpt / cuda / PreRev",
            "value": 0.000033729,
            "unit": "s"
          },
          {
            "name": "add_two / PartOpt / cuda / PostRev",
            "value": 0.000031872,
            "unit": "s"
          },
          {
            "name": "add_two / PartOpt / cuda / BothRev",
            "value": 0.000032288,
            "unit": "s"
          },
          {
            "name": "add_two / IPartOpt / cuda / PreRev",
            "value": 0.000033088,
            "unit": "s"
          },
          {
            "name": "add_two / IPartOpt / cuda / PostRev",
            "value": 0.000032576,
            "unit": "s"
          },
          {
            "name": "add_two / IPartOpt / cuda / BothRev",
            "value": 0.000032639000000000004,
            "unit": "s"
          },
          {
            "name": "add_two / DefOpt / cuda / PreRev",
            "value": 0.00003712,
            "unit": "s"
          },
          {
            "name": "add_two / DefOpt / cuda / PostRev",
            "value": 0.000036992,
            "unit": "s"
          },
          {
            "name": "add_two / DefOpt / cuda / BothRev",
            "value": 0.000036895,
            "unit": "s"
          },
          {
            "name": "add_two / IDefOpt / cuda / PreRev",
            "value": 0.000032256,
            "unit": "s"
          },
          {
            "name": "add_two / IDefOpt / cuda / PostRev",
            "value": 0.000033248,
            "unit": "s"
          },
          {
            "name": "add_two / IDefOpt / cuda / BothRev",
            "value": 0.000032767999999999995,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / tpu / Primal",
            "value": 0.000001432275,
            "unit": "s"
          },
          {
            "name": "add_two / Jax / tpu / Primal",
            "value": 0.00000146885,
            "unit": "s"
          },
          {
            "name": "add_two / HLOOpt / tpu / Primal",
            "value": 0.00000143455,
            "unit": "s"
          },
          {
            "name": "add_two / PartOpt / tpu / Primal",
            "value": 0.000001475325,
            "unit": "s"
          },
          {
            "name": "add_two / IPartOpt / tpu / Primal",
            "value": 0.000001440525,
            "unit": "s"
          },
          {
            "name": "add_two / DefOpt / tpu / Primal",
            "value": 0.000001472025,
            "unit": "s"
          },
          {
            "name": "add_two / IDefOpt / tpu / Primal",
            "value": 0.000001433725,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / tpu / Forward",
            "value": 0.00000182805,
            "unit": "s"
          },
          {
            "name": "add_two / Jax / tpu / Forward",
            "value": 0.0000018211,
            "unit": "s"
          },
          {
            "name": "add_two / HLOOpt / tpu / Forward",
            "value": 0.0000018279,
            "unit": "s"
          },
          {
            "name": "add_two / PartOpt / tpu / Forward",
            "value": 0.0000018264250000000005,
            "unit": "s"
          },
          {
            "name": "add_two / IPartOpt / tpu / Forward",
            "value": 0.0000018267,
            "unit": "s"
          },
          {
            "name": "add_two / DefOpt / tpu / Forward",
            "value": 0.000001835075,
            "unit": "s"
          },
          {
            "name": "add_two / IDefOpt / tpu / Forward",
            "value": 0.0000018219,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / tpu / PreRev",
            "value": 0.00000283355,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / tpu / PostRev",
            "value": 0.0000027488499999999995,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / tpu / BothRev",
            "value": 0.000002835875,
            "unit": "s"
          },
          {
            "name": "add_two / Jax / tpu / BothRev",
            "value": 0.00000274815,
            "unit": "s"
          },
          {
            "name": "add_two / HLOOpt / tpu / PreRev",
            "value": 0.0000028394000000000003,
            "unit": "s"
          },
          {
            "name": "add_two / HLOOpt / tpu / PostRev",
            "value": 0.000002750675,
            "unit": "s"
          },
          {
            "name": "add_two / HLOOpt / tpu / BothRev",
            "value": 0.000002828425,
            "unit": "s"
          },
          {
            "name": "add_two / PartOpt / tpu / PreRev",
            "value": 0.0000027454000000000004,
            "unit": "s"
          },
          {
            "name": "add_two / PartOpt / tpu / PostRev",
            "value": 0.000002846125,
            "unit": "s"
          },
          {
            "name": "add_two / PartOpt / tpu / BothRev",
            "value": 0.0000027507,
            "unit": "s"
          },
          {
            "name": "add_two / IPartOpt / tpu / PreRev",
            "value": 0.0000028301,
            "unit": "s"
          },
          {
            "name": "add_two / IPartOpt / tpu / PostRev",
            "value": 0.0000027495,
            "unit": "s"
          },
          {
            "name": "add_two / IPartOpt / tpu / BothRev",
            "value": 0.000002838775,
            "unit": "s"
          },
          {
            "name": "add_two / DefOpt / tpu / PreRev",
            "value": 0.000002749425,
            "unit": "s"
          },
          {
            "name": "add_two / DefOpt / tpu / PostRev",
            "value": 0.000002836225,
            "unit": "s"
          },
          {
            "name": "add_two / DefOpt / tpu / BothRev",
            "value": 0.000002757925,
            "unit": "s"
          },
          {
            "name": "add_two / IDefOpt / tpu / PreRev",
            "value": 0.00000283615,
            "unit": "s"
          },
          {
            "name": "add_two / IDefOpt / tpu / PostRev",
            "value": 0.00000275355,
            "unit": "s"
          },
          {
            "name": "add_two / IDefOpt / tpu / BothRev",
            "value": 0.000002842775,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / Primal",
            "value": 0.000027065,
            "unit": "s"
          },
          {
            "name": "add_two / Jax / cpu / Primal",
            "value": 0.000016776,
            "unit": "s"
          },
          {
            "name": "add_two / HLOOpt / cpu / Primal",
            "value": 0.000016344999999999997,
            "unit": "s"
          },
          {
            "name": "add_two / PartOpt / cpu / Primal",
            "value": 0.000016737,
            "unit": "s"
          },
          {
            "name": "add_two / IPartOpt / cpu / Primal",
            "value": 0.000016904999999999998,
            "unit": "s"
          },
          {
            "name": "add_two / DefOpt / cpu / Primal",
            "value": 0.000024176,
            "unit": "s"
          },
          {
            "name": "add_two / IDefOpt / cpu / Primal",
            "value": 0.000016625,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / Forward",
            "value": 0.000022495,
            "unit": "s"
          },
          {
            "name": "add_two / Jax / cpu / Forward",
            "value": 0.000022182,
            "unit": "s"
          },
          {
            "name": "add_two / HLOOpt / cpu / Forward",
            "value": 0.000027729,
            "unit": "s"
          },
          {
            "name": "add_two / PartOpt / cpu / Forward",
            "value": 0.000022618,
            "unit": "s"
          },
          {
            "name": "add_two / IPartOpt / cpu / Forward",
            "value": 0.000022289,
            "unit": "s"
          },
          {
            "name": "add_two / DefOpt / cpu / Forward",
            "value": 0.000022596,
            "unit": "s"
          },
          {
            "name": "add_two / IDefOpt / cpu / Forward",
            "value": 0.000022138,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / PreRev",
            "value": 0.000028905,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / PostRev",
            "value": 0.000028265,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / BothRev",
            "value": 0.000028726,
            "unit": "s"
          },
          {
            "name": "add_two / Jax / cpu / BothRev",
            "value": 0.00002828,
            "unit": "s"
          },
          {
            "name": "add_two / HLOOpt / cpu / PreRev",
            "value": 0.000027961,
            "unit": "s"
          },
          {
            "name": "add_two / HLOOpt / cpu / PostRev",
            "value": 0.000028868,
            "unit": "s"
          },
          {
            "name": "add_two / HLOOpt / cpu / BothRev",
            "value": 0.000028471,
            "unit": "s"
          },
          {
            "name": "add_two / PartOpt / cpu / PreRev",
            "value": 0.000028989,
            "unit": "s"
          },
          {
            "name": "add_two / PartOpt / cpu / PostRev",
            "value": 0.000028491,
            "unit": "s"
          },
          {
            "name": "add_two / PartOpt / cpu / BothRev",
            "value": 0.000028009,
            "unit": "s"
          },
          {
            "name": "add_two / IPartOpt / cpu / PreRev",
            "value": 0.00002871,
            "unit": "s"
          },
          {
            "name": "add_two / IPartOpt / cpu / PostRev",
            "value": 0.000028236,
            "unit": "s"
          },
          {
            "name": "add_two / IPartOpt / cpu / BothRev",
            "value": 0.000028311,
            "unit": "s"
          },
          {
            "name": "add_two / DefOpt / cpu / PreRev",
            "value": 0.000030122,
            "unit": "s"
          },
          {
            "name": "add_two / DefOpt / cpu / PostRev",
            "value": 0.000028413,
            "unit": "s"
          },
          {
            "name": "add_two / DefOpt / cpu / BothRev",
            "value": 0.000028261000000000003,
            "unit": "s"
          },
          {
            "name": "add_two / IDefOpt / cpu / PreRev",
            "value": 0.000028306,
            "unit": "s"
          },
          {
            "name": "add_two / IDefOpt / cpu / PostRev",
            "value": 0.00002871,
            "unit": "s"
          },
          {
            "name": "add_two / IDefOpt / cpu / BothRev",
            "value": 0.000028935,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / Primal",
            "value": 0.000008,
            "unit": "s"
          },
          {
            "name": "add_two / Jax / cpu / Primal",
            "value": 0.000008,
            "unit": "s"
          },
          {
            "name": "add_two / HLOOpt / cpu / Primal",
            "value": 0.000008,
            "unit": "s"
          },
          {
            "name": "add_two / PartOpt / cpu / Primal",
            "value": 0.000008,
            "unit": "s"
          },
          {
            "name": "add_two / IPartOpt / cpu / Primal",
            "value": 0.000008,
            "unit": "s"
          },
          {
            "name": "add_two / DefOpt / cpu / Primal",
            "value": 0.000008,
            "unit": "s"
          },
          {
            "name": "add_two / IDefOpt / cpu / Primal",
            "value": 0.000008,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / Forward",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "add_two / Jax / cpu / Forward",
            "value": 0.000011,
            "unit": "s"
          },
          {
            "name": "add_two / HLOOpt / cpu / Forward",
            "value": 0.000011,
            "unit": "s"
          },
          {
            "name": "add_two / PartOpt / cpu / Forward",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "add_two / IPartOpt / cpu / Forward",
            "value": 0.000011,
            "unit": "s"
          },
          {
            "name": "add_two / DefOpt / cpu / Forward",
            "value": 0.000011,
            "unit": "s"
          },
          {
            "name": "add_two / IDefOpt / cpu / Forward",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / PreRev",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / PostRev",
            "value": 0.000015,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / BothRev",
            "value": 0.000015,
            "unit": "s"
          },
          {
            "name": "add_two / Jax / cpu / BothRev",
            "value": 0.000015,
            "unit": "s"
          },
          {
            "name": "add_two / HLOOpt / cpu / PreRev",
            "value": 0.000015,
            "unit": "s"
          },
          {
            "name": "add_two / HLOOpt / cpu / PostRev",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "add_two / HLOOpt / cpu / BothRev",
            "value": 0.000015,
            "unit": "s"
          },
          {
            "name": "add_two / PartOpt / cpu / PreRev",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "add_two / PartOpt / cpu / PostRev",
            "value": 0.000015,
            "unit": "s"
          },
          {
            "name": "add_two / PartOpt / cpu / BothRev",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "add_two / IPartOpt / cpu / PreRev",
            "value": 0.000015,
            "unit": "s"
          },
          {
            "name": "add_two / IPartOpt / cpu / PostRev",
            "value": 0.000015,
            "unit": "s"
          },
          {
            "name": "add_two / IPartOpt / cpu / BothRev",
            "value": 0.000015,
            "unit": "s"
          },
          {
            "name": "add_two / DefOpt / cpu / PreRev",
            "value": 0.000015,
            "unit": "s"
          },
          {
            "name": "add_two / DefOpt / cpu / PostRev",
            "value": 0.000015,
            "unit": "s"
          },
          {
            "name": "add_two / DefOpt / cpu / BothRev",
            "value": 0.000015,
            "unit": "s"
          },
          {
            "name": "add_two / IDefOpt / cpu / PreRev",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "add_two / IDefOpt / cpu / PostRev",
            "value": 0.000016,
            "unit": "s"
          },
          {
            "name": "add_two / IDefOpt / cpu / BothRev",
            "value": 0.000015,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / Primal",
            "value": 0.00000652737993732444,
            "unit": "s"
          },
          {
            "name": "cache / Jax / cpu / Primal",
            "value": 0.000006517640013044002,
            "unit": "s"
          },
          {
            "name": "cache / HLOOpt / cpu / Primal",
            "value": 0.0000062030800290813205,
            "unit": "s"
          },
          {
            "name": "cache / PartOpt / cpu / Primal",
            "value": 0.000006256000051507726,
            "unit": "s"
          },
          {
            "name": "cache / IPartOpt / cpu / Primal",
            "value": 0.000006003720009175595,
            "unit": "s"
          },
          {
            "name": "cache / DefOpt / cpu / Primal",
            "value": 0.000006188879942783387,
            "unit": "s"
          },
          {
            "name": "cache / IDefOpt / cpu / Primal",
            "value": 0.00000604021995968651,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / Forward",
            "value": 0.000015254139980243053,
            "unit": "s"
          },
          {
            "name": "cache / Jax / cpu / Forward",
            "value": 0.000015734699973108944,
            "unit": "s"
          },
          {
            "name": "cache / HLOOpt / cpu / Forward",
            "value": 0.00001606510000783601,
            "unit": "s"
          },
          {
            "name": "cache / PartOpt / cpu / Forward",
            "value": 0.000016138140108523657,
            "unit": "s"
          },
          {
            "name": "cache / IPartOpt / cpu / Forward",
            "value": 0.000016810960023576626,
            "unit": "s"
          },
          {
            "name": "cache / DefOpt / cpu / Forward",
            "value": 0.000015426760000991635,
            "unit": "s"
          },
          {
            "name": "cache / IDefOpt / cpu / Forward",
            "value": 0.000015242099925671937,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / PreRev",
            "value": 0.00001742825998007902,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / PostRev",
            "value": 0.000021916259993304265,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / BothRev",
            "value": 0.000017467099969508126,
            "unit": "s"
          },
          {
            "name": "cache / Jax / cpu / BothRev",
            "value": 0.000021213359941612,
            "unit": "s"
          },
          {
            "name": "cache / HLOOpt / cpu / PreRev",
            "value": 0.00001767580013620318,
            "unit": "s"
          },
          {
            "name": "cache / HLOOpt / cpu / PostRev",
            "value": 0.00001842104000388645,
            "unit": "s"
          },
          {
            "name": "cache / HLOOpt / cpu / BothRev",
            "value": 0.00001728320023175911,
            "unit": "s"
          },
          {
            "name": "cache / PartOpt / cpu / PreRev",
            "value": 0.00001638469997487846,
            "unit": "s"
          },
          {
            "name": "cache / PartOpt / cpu / PostRev",
            "value": 0.00002063067999188206,
            "unit": "s"
          },
          {
            "name": "cache / PartOpt / cpu / BothRev",
            "value": 0.00001676066000072751,
            "unit": "s"
          },
          {
            "name": "cache / IPartOpt / cpu / PreRev",
            "value": 0.000016204420007852604,
            "unit": "s"
          },
          {
            "name": "cache / IPartOpt / cpu / PostRev",
            "value": 0.000021584360092674614,
            "unit": "s"
          },
          {
            "name": "cache / IPartOpt / cpu / BothRev",
            "value": 0.00001599757995791151,
            "unit": "s"
          },
          {
            "name": "cache / DefOpt / cpu / PreRev",
            "value": 0.000016451219998998567,
            "unit": "s"
          },
          {
            "name": "cache / DefOpt / cpu / PostRev",
            "value": 0.000015843419987504603,
            "unit": "s"
          },
          {
            "name": "cache / DefOpt / cpu / BothRev",
            "value": 0.000017254579961445416,
            "unit": "s"
          },
          {
            "name": "cache / IDefOpt / cpu / PreRev",
            "value": 0.000017035779947036645,
            "unit": "s"
          },
          {
            "name": "cache / IDefOpt / cpu / PostRev",
            "value": 0.000016395139919040958,
            "unit": "s"
          },
          {
            "name": "cache / IDefOpt / cpu / BothRev",
            "value": 0.00001651671993386117,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cuda / Primal",
            "value": 0.000002335,
            "unit": "s"
          },
          {
            "name": "cache / Jax / cuda / Primal",
            "value": 0.000002335,
            "unit": "s"
          },
          {
            "name": "cache / HLOOpt / cuda / Primal",
            "value": 0.00000224,
            "unit": "s"
          },
          {
            "name": "cache / PartOpt / cuda / Primal",
            "value": 0.000002304,
            "unit": "s"
          },
          {
            "name": "cache / IPartOpt / cuda / Primal",
            "value": 0.000002335,
            "unit": "s"
          },
          {
            "name": "cache / DefOpt / cuda / Primal",
            "value": 0.000002239,
            "unit": "s"
          },
          {
            "name": "cache / IDefOpt / cuda / Primal",
            "value": 0.000002272,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cuda / Forward",
            "value": 0.0000023670000000000004,
            "unit": "s"
          },
          {
            "name": "cache / Jax / cuda / Forward",
            "value": 0.000002336,
            "unit": "s"
          },
          {
            "name": "cache / HLOOpt / cuda / Forward",
            "value": 0.0000023670000000000004,
            "unit": "s"
          },
          {
            "name": "cache / PartOpt / cuda / Forward",
            "value": 0.000002335,
            "unit": "s"
          },
          {
            "name": "cache / IPartOpt / cuda / Forward",
            "value": 0.0000023670000000000004,
            "unit": "s"
          },
          {
            "name": "cache / DefOpt / cuda / Forward",
            "value": 0.000002272,
            "unit": "s"
          },
          {
            "name": "cache / IDefOpt / cuda / Forward",
            "value": 0.0000023670000000000004,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cuda / PreRev",
            "value": 0.000010752,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cuda / PostRev",
            "value": 0.000011264,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cuda / BothRev",
            "value": 0.000011008,
            "unit": "s"
          },
          {
            "name": "cache / Jax / cuda / BothRev",
            "value": 0.00001104,
            "unit": "s"
          },
          {
            "name": "cache / HLOOpt / cuda / PreRev",
            "value": 0.000013184,
            "unit": "s"
          },
          {
            "name": "cache / HLOOpt / cuda / PostRev",
            "value": 0.00001312,
            "unit": "s"
          },
          {
            "name": "cache / HLOOpt / cuda / BothRev",
            "value": 0.000013184,
            "unit": "s"
          },
          {
            "name": "cache / PartOpt / cuda / PreRev",
            "value": 0.000010816,
            "unit": "s"
          },
          {
            "name": "cache / PartOpt / cuda / PostRev",
            "value": 0.000010944,
            "unit": "s"
          },
          {
            "name": "cache / PartOpt / cuda / BothRev",
            "value": 0.000010432,
            "unit": "s"
          },
          {
            "name": "cache / IPartOpt / cuda / PreRev",
            "value": 0.000010848,
            "unit": "s"
          },
          {
            "name": "cache / IPartOpt / cuda / PostRev",
            "value": 0.000010848,
            "unit": "s"
          },
          {
            "name": "cache / IPartOpt / cuda / BothRev",
            "value": 0.000010912,
            "unit": "s"
          },
          {
            "name": "cache / DefOpt / cuda / PreRev",
            "value": 0.000010945,
            "unit": "s"
          },
          {
            "name": "cache / DefOpt / cuda / PostRev",
            "value": 0.000010912,
            "unit": "s"
          },
          {
            "name": "cache / DefOpt / cuda / BothRev",
            "value": 0.000010848,
            "unit": "s"
          },
          {
            "name": "cache / IDefOpt / cuda / PreRev",
            "value": 0.000010816,
            "unit": "s"
          },
          {
            "name": "cache / IDefOpt / cuda / PostRev",
            "value": 0.000010912,
            "unit": "s"
          },
          {
            "name": "cache / IDefOpt / cuda / BothRev",
            "value": 0.000010912,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / tpu / Primal",
            "value": 0.000002462475,
            "unit": "s"
          },
          {
            "name": "cache / Jax / tpu / Primal",
            "value": 0.000002462575,
            "unit": "s"
          },
          {
            "name": "cache / HLOOpt / tpu / Primal",
            "value": 0.000002466475,
            "unit": "s"
          },
          {
            "name": "cache / PartOpt / tpu / Primal",
            "value": 0.00000247805,
            "unit": "s"
          },
          {
            "name": "cache / IPartOpt / tpu / Primal",
            "value": 0.0000024623750000000003,
            "unit": "s"
          },
          {
            "name": "cache / DefOpt / tpu / Primal",
            "value": 0.00000246505,
            "unit": "s"
          },
          {
            "name": "cache / IDefOpt / tpu / Primal",
            "value": 0.000002463625,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / tpu / Forward",
            "value": 0.00000353335,
            "unit": "s"
          },
          {
            "name": "cache / Jax / tpu / Forward",
            "value": 0.0000035484749999999995,
            "unit": "s"
          },
          {
            "name": "cache / HLOOpt / tpu / Forward",
            "value": 0.00000355825,
            "unit": "s"
          },
          {
            "name": "cache / PartOpt / tpu / Forward",
            "value": 0.000003519825,
            "unit": "s"
          },
          {
            "name": "cache / IPartOpt / tpu / Forward",
            "value": 0.00000355385,
            "unit": "s"
          },
          {
            "name": "cache / DefOpt / tpu / Forward",
            "value": 0.000003528125,
            "unit": "s"
          },
          {
            "name": "cache / IDefOpt / tpu / Forward",
            "value": 0.0000035567,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / tpu / PreRev",
            "value": 0.0000049759,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / tpu / PostRev",
            "value": 0.000004965375,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / tpu / BothRev",
            "value": 0.000004986875,
            "unit": "s"
          },
          {
            "name": "cache / Jax / tpu / BothRev",
            "value": 0.000004977825,
            "unit": "s"
          },
          {
            "name": "cache / HLOOpt / tpu / PreRev",
            "value": 0.000003930175,
            "unit": "s"
          },
          {
            "name": "cache / HLOOpt / tpu / PostRev",
            "value": 0.000004109175,
            "unit": "s"
          },
          {
            "name": "cache / HLOOpt / tpu / BothRev",
            "value": 0.000003952725,
            "unit": "s"
          },
          {
            "name": "cache / PartOpt / tpu / PreRev",
            "value": 0.000004992225,
            "unit": "s"
          },
          {
            "name": "cache / PartOpt / tpu / PostRev",
            "value": 0.0000050151250000000005,
            "unit": "s"
          },
          {
            "name": "cache / PartOpt / tpu / BothRev",
            "value": 0.000004995249999999999,
            "unit": "s"
          },
          {
            "name": "cache / IPartOpt / tpu / PreRev",
            "value": 0.00000497575,
            "unit": "s"
          },
          {
            "name": "cache / IPartOpt / tpu / PostRev",
            "value": 0.0000049877000000000005,
            "unit": "s"
          },
          {
            "name": "cache / IPartOpt / tpu / BothRev",
            "value": 0.000004984325000000001,
            "unit": "s"
          },
          {
            "name": "cache / DefOpt / tpu / PreRev",
            "value": 0.0000049936,
            "unit": "s"
          },
          {
            "name": "cache / DefOpt / tpu / PostRev",
            "value": 0.000004969825,
            "unit": "s"
          },
          {
            "name": "cache / DefOpt / tpu / BothRev",
            "value": 0.00000498925,
            "unit": "s"
          },
          {
            "name": "cache / IDefOpt / tpu / PreRev",
            "value": 0.00000499575,
            "unit": "s"
          },
          {
            "name": "cache / IDefOpt / tpu / PostRev",
            "value": 0.000004976975,
            "unit": "s"
          },
          {
            "name": "cache / IDefOpt / tpu / BothRev",
            "value": 0.0000049765,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / Primal",
            "value": 0.00001855,
            "unit": "s"
          },
          {
            "name": "cache / Jax / cpu / Primal",
            "value": 0.00001859,
            "unit": "s"
          },
          {
            "name": "cache / HLOOpt / cpu / Primal",
            "value": 0.000018153,
            "unit": "s"
          },
          {
            "name": "cache / PartOpt / cpu / Primal",
            "value": 0.000018592,
            "unit": "s"
          },
          {
            "name": "cache / IPartOpt / cpu / Primal",
            "value": 0.000018277,
            "unit": "s"
          },
          {
            "name": "cache / DefOpt / cpu / Primal",
            "value": 0.000018913,
            "unit": "s"
          },
          {
            "name": "cache / IDefOpt / cpu / Primal",
            "value": 0.000018808,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / Forward",
            "value": 0.000022978,
            "unit": "s"
          },
          {
            "name": "cache / Jax / cpu / Forward",
            "value": 0.000022986,
            "unit": "s"
          },
          {
            "name": "cache / HLOOpt / cpu / Forward",
            "value": 0.000031522,
            "unit": "s"
          },
          {
            "name": "cache / PartOpt / cpu / Forward",
            "value": 0.000022066,
            "unit": "s"
          },
          {
            "name": "cache / IPartOpt / cpu / Forward",
            "value": 0.000021499,
            "unit": "s"
          },
          {
            "name": "cache / DefOpt / cpu / Forward",
            "value": 0.000021276,
            "unit": "s"
          },
          {
            "name": "cache / IDefOpt / cpu / Forward",
            "value": 0.000022438,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / PreRev",
            "value": 0.000022499000000000003,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / PostRev",
            "value": 0.000030288,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / BothRev",
            "value": 0.000023983,
            "unit": "s"
          },
          {
            "name": "cache / Jax / cpu / BothRev",
            "value": 0.000025031,
            "unit": "s"
          },
          {
            "name": "cache / HLOOpt / cpu / PreRev",
            "value": 0.000022649,
            "unit": "s"
          },
          {
            "name": "cache / HLOOpt / cpu / PostRev",
            "value": 0.000021746,
            "unit": "s"
          },
          {
            "name": "cache / HLOOpt / cpu / BothRev",
            "value": 0.000022135,
            "unit": "s"
          },
          {
            "name": "cache / PartOpt / cpu / PreRev",
            "value": 0.00002181,
            "unit": "s"
          },
          {
            "name": "cache / PartOpt / cpu / PostRev",
            "value": 0.000031034,
            "unit": "s"
          },
          {
            "name": "cache / PartOpt / cpu / BothRev",
            "value": 0.000022617,
            "unit": "s"
          },
          {
            "name": "cache / IPartOpt / cpu / PreRev",
            "value": 0.00003333,
            "unit": "s"
          },
          {
            "name": "cache / IPartOpt / cpu / PostRev",
            "value": 0.000026507,
            "unit": "s"
          },
          {
            "name": "cache / IPartOpt / cpu / BothRev",
            "value": 0.000023311,
            "unit": "s"
          },
          {
            "name": "cache / DefOpt / cpu / PreRev",
            "value": 0.000032432000000000004,
            "unit": "s"
          },
          {
            "name": "cache / DefOpt / cpu / PostRev",
            "value": 0.00003275,
            "unit": "s"
          },
          {
            "name": "cache / DefOpt / cpu / BothRev",
            "value": 0.000032996,
            "unit": "s"
          },
          {
            "name": "cache / IDefOpt / cpu / PreRev",
            "value": 0.000032972,
            "unit": "s"
          },
          {
            "name": "cache / IDefOpt / cpu / PostRev",
            "value": 0.000022824,
            "unit": "s"
          },
          {
            "name": "cache / IDefOpt / cpu / BothRev",
            "value": 0.000023141,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / Primal",
            "value": 0.000008,
            "unit": "s"
          },
          {
            "name": "cache / Jax / cpu / Primal",
            "value": 0.000008,
            "unit": "s"
          },
          {
            "name": "cache / HLOOpt / cpu / Primal",
            "value": 0.000008,
            "unit": "s"
          },
          {
            "name": "cache / PartOpt / cpu / Primal",
            "value": 0.000008,
            "unit": "s"
          },
          {
            "name": "cache / IPartOpt / cpu / Primal",
            "value": 0.000007,
            "unit": "s"
          },
          {
            "name": "cache / DefOpt / cpu / Primal",
            "value": 0.000008,
            "unit": "s"
          },
          {
            "name": "cache / IDefOpt / cpu / Primal",
            "value": 0.000008,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / Forward",
            "value": 0.00001,
            "unit": "s"
          },
          {
            "name": "cache / Jax / cpu / Forward",
            "value": 0.00001,
            "unit": "s"
          },
          {
            "name": "cache / HLOOpt / cpu / Forward",
            "value": 0.000008999999999999999,
            "unit": "s"
          },
          {
            "name": "cache / PartOpt / cpu / Forward",
            "value": 0.000011,
            "unit": "s"
          },
          {
            "name": "cache / IPartOpt / cpu / Forward",
            "value": 0.000052,
            "unit": "s"
          },
          {
            "name": "cache / DefOpt / cpu / Forward",
            "value": 0.00001,
            "unit": "s"
          },
          {
            "name": "cache / IDefOpt / cpu / Forward",
            "value": 0.00001,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / PreRev",
            "value": 0.000043,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / PostRev",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / BothRev",
            "value": 0.00001,
            "unit": "s"
          },
          {
            "name": "cache / Jax / cpu / BothRev",
            "value": 0.000011,
            "unit": "s"
          },
          {
            "name": "cache / HLOOpt / cpu / PreRev",
            "value": 0.00001,
            "unit": "s"
          },
          {
            "name": "cache / HLOOpt / cpu / PostRev",
            "value": 0.000011,
            "unit": "s"
          },
          {
            "name": "cache / HLOOpt / cpu / BothRev",
            "value": 0.000044,
            "unit": "s"
          },
          {
            "name": "cache / PartOpt / cpu / PreRev",
            "value": 0.00001,
            "unit": "s"
          },
          {
            "name": "cache / PartOpt / cpu / PostRev",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "cache / PartOpt / cpu / BothRev",
            "value": 0.000011,
            "unit": "s"
          },
          {
            "name": "cache / IPartOpt / cpu / PreRev",
            "value": 0.00001,
            "unit": "s"
          },
          {
            "name": "cache / IPartOpt / cpu / PostRev",
            "value": 0.000011,
            "unit": "s"
          },
          {
            "name": "cache / IPartOpt / cpu / BothRev",
            "value": 0.000011,
            "unit": "s"
          },
          {
            "name": "cache / DefOpt / cpu / PreRev",
            "value": 0.000033,
            "unit": "s"
          },
          {
            "name": "cache / DefOpt / cpu / PostRev",
            "value": 0.000011,
            "unit": "s"
          },
          {
            "name": "cache / DefOpt / cpu / BothRev",
            "value": 0.000011,
            "unit": "s"
          },
          {
            "name": "cache / IDefOpt / cpu / PreRev",
            "value": 0.00001,
            "unit": "s"
          },
          {
            "name": "cache / IDefOpt / cpu / PostRev",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "cache / IDefOpt / cpu / BothRev",
            "value": 0.000011,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / Primal",
            "value": 0.000007260140009748284,
            "unit": "s"
          },
          {
            "name": "Concat / Jax / cpu / Primal",
            "value": 0.000006948900099814637,
            "unit": "s"
          },
          {
            "name": "Concat / HLOOpt / cpu / Primal",
            "value": 0.000006655859924649121,
            "unit": "s"
          },
          {
            "name": "Concat / PartOpt / cpu / Primal",
            "value": 0.000007074080021993723,
            "unit": "s"
          },
          {
            "name": "Concat / IPartOpt / cpu / Primal",
            "value": 0.000007203240038506919,
            "unit": "s"
          },
          {
            "name": "Concat / DefOpt / cpu / Primal",
            "value": 0.000006470080006693024,
            "unit": "s"
          },
          {
            "name": "Concat / IDefOpt / cpu / Primal",
            "value": 0.000006457699964812491,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / Forward",
            "value": 0.000010168459957640152,
            "unit": "s"
          },
          {
            "name": "Concat / Jax / cpu / Forward",
            "value": 0.000010028180022345622,
            "unit": "s"
          },
          {
            "name": "Concat / HLOOpt / cpu / Forward",
            "value": 0.000009818840044317768,
            "unit": "s"
          },
          {
            "name": "Concat / PartOpt / cpu / Forward",
            "value": 0.000010104179982590724,
            "unit": "s"
          },
          {
            "name": "Concat / IPartOpt / cpu / Forward",
            "value": 0.000010378359966125571,
            "unit": "s"
          },
          {
            "name": "Concat / DefOpt / cpu / Forward",
            "value": 0.000009674800021457483,
            "unit": "s"
          },
          {
            "name": "Concat / IDefOpt / cpu / Forward",
            "value": 0.000010100760009663644,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / PreRev",
            "value": 0.000011570039914658992,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / PostRev",
            "value": 0.000011262859970884165,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / BothRev",
            "value": 0.000011393060067348416,
            "unit": "s"
          },
          {
            "name": "Concat / Jax / cpu / BothRev",
            "value": 0.000011195819897693582,
            "unit": "s"
          },
          {
            "name": "Concat / HLOOpt / cpu / PreRev",
            "value": 0.00001210342004924314,
            "unit": "s"
          },
          {
            "name": "Concat / HLOOpt / cpu / PostRev",
            "value": 0.000013650400069309398,
            "unit": "s"
          },
          {
            "name": "Concat / HLOOpt / cpu / BothRev",
            "value": 0.000011507319977681616,
            "unit": "s"
          },
          {
            "name": "Concat / PartOpt / cpu / PreRev",
            "value": 0.000012053580085193972,
            "unit": "s"
          },
          {
            "name": "Concat / PartOpt / cpu / PostRev",
            "value": 0.00001123879996157484,
            "unit": "s"
          },
          {
            "name": "Concat / PartOpt / cpu / BothRev",
            "value": 0.000011530220035638196,
            "unit": "s"
          },
          {
            "name": "Concat / IPartOpt / cpu / PreRev",
            "value": 0.000012176280015410158,
            "unit": "s"
          },
          {
            "name": "Concat / IPartOpt / cpu / PostRev",
            "value": 0.000011708679994626435,
            "unit": "s"
          },
          {
            "name": "Concat / IPartOpt / cpu / BothRev",
            "value": 0.000011746179916372056,
            "unit": "s"
          },
          {
            "name": "Concat / DefOpt / cpu / PreRev",
            "value": 0.00001150662001236924,
            "unit": "s"
          },
          {
            "name": "Concat / DefOpt / cpu / PostRev",
            "value": 0.000011765839990403038,
            "unit": "s"
          },
          {
            "name": "Concat / DefOpt / cpu / BothRev",
            "value": 0.000011631019988271874,
            "unit": "s"
          },
          {
            "name": "Concat / IDefOpt / cpu / PreRev",
            "value": 0.00001181053999971482,
            "unit": "s"
          },
          {
            "name": "Concat / IDefOpt / cpu / PostRev",
            "value": 0.000011404320011934032,
            "unit": "s"
          },
          {
            "name": "Concat / IDefOpt / cpu / BothRev",
            "value": 0.000011693980031850516,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cuda / Primal",
            "value": 0.000001951,
            "unit": "s"
          },
          {
            "name": "Concat / Jax / cuda / Primal",
            "value": 0.000001951,
            "unit": "s"
          },
          {
            "name": "Concat / HLOOpt / cuda / Primal",
            "value": 0.000001951,
            "unit": "s"
          },
          {
            "name": "Concat / PartOpt / cuda / Primal",
            "value": 0.000001951,
            "unit": "s"
          },
          {
            "name": "Concat / IPartOpt / cuda / Primal",
            "value": 0.000001951,
            "unit": "s"
          },
          {
            "name": "Concat / DefOpt / cuda / Primal",
            "value": 0.000001951,
            "unit": "s"
          },
          {
            "name": "Concat / IDefOpt / cuda / Primal",
            "value": 0.000001951,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cuda / Forward",
            "value": 0.000010272,
            "unit": "s"
          },
          {
            "name": "Concat / Jax / cuda / Forward",
            "value": 0.00001008,
            "unit": "s"
          },
          {
            "name": "Concat / HLOOpt / cuda / Forward",
            "value": 0.000009888,
            "unit": "s"
          },
          {
            "name": "Concat / PartOpt / cuda / Forward",
            "value": 0.000010304,
            "unit": "s"
          },
          {
            "name": "Concat / IPartOpt / cuda / Forward",
            "value": 0.00001024,
            "unit": "s"
          },
          {
            "name": "Concat / DefOpt / cuda / Forward",
            "value": 0.000010016,
            "unit": "s"
          },
          {
            "name": "Concat / IDefOpt / cuda / Forward",
            "value": 0.00001024,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cuda / PreRev",
            "value": 0.000016576000000000002,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cuda / PostRev",
            "value": 0.00001664,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cuda / BothRev",
            "value": 0.000016927000000000002,
            "unit": "s"
          },
          {
            "name": "Concat / Jax / cuda / BothRev",
            "value": 0.000018752000000000003,
            "unit": "s"
          },
          {
            "name": "Concat / HLOOpt / cuda / PreRev",
            "value": 0.000017152,
            "unit": "s"
          },
          {
            "name": "Concat / HLOOpt / cuda / PostRev",
            "value": 0.000016927999999999998,
            "unit": "s"
          },
          {
            "name": "Concat / HLOOpt / cuda / BothRev",
            "value": 0.000016544,
            "unit": "s"
          },
          {
            "name": "Concat / PartOpt / cuda / PreRev",
            "value": 0.000017216,
            "unit": "s"
          },
          {
            "name": "Concat / PartOpt / cuda / PostRev",
            "value": 0.000016864,
            "unit": "s"
          },
          {
            "name": "Concat / PartOpt / cuda / BothRev",
            "value": 0.000016992,
            "unit": "s"
          },
          {
            "name": "Concat / IPartOpt / cuda / PreRev",
            "value": 0.000017024,
            "unit": "s"
          },
          {
            "name": "Concat / IPartOpt / cuda / PostRev",
            "value": 0.000016927999999999998,
            "unit": "s"
          },
          {
            "name": "Concat / IPartOpt / cuda / BothRev",
            "value": 0.00001664,
            "unit": "s"
          },
          {
            "name": "Concat / DefOpt / cuda / PreRev",
            "value": 0.000016767,
            "unit": "s"
          },
          {
            "name": "Concat / DefOpt / cuda / PostRev",
            "value": 0.000016576000000000002,
            "unit": "s"
          },
          {
            "name": "Concat / DefOpt / cuda / BothRev",
            "value": 0.000016864,
            "unit": "s"
          },
          {
            "name": "Concat / IDefOpt / cuda / PreRev",
            "value": 0.000016896000000000002,
            "unit": "s"
          },
          {
            "name": "Concat / IDefOpt / cuda / PostRev",
            "value": 0.000017088,
            "unit": "s"
          },
          {
            "name": "Concat / IDefOpt / cuda / BothRev",
            "value": 0.000016896000000000002,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / tpu / Primal",
            "value": 0.0000015226999999999998,
            "unit": "s"
          },
          {
            "name": "Concat / Jax / tpu / Primal",
            "value": 0.000001530425,
            "unit": "s"
          },
          {
            "name": "Concat / HLOOpt / tpu / Primal",
            "value": 0.0000015191750000000002,
            "unit": "s"
          },
          {
            "name": "Concat / PartOpt / tpu / Primal",
            "value": 0.0000015253749999999998,
            "unit": "s"
          },
          {
            "name": "Concat / IPartOpt / tpu / Primal",
            "value": 0.000001533975,
            "unit": "s"
          },
          {
            "name": "Concat / DefOpt / tpu / Primal",
            "value": 0.0000015469250000000002,
            "unit": "s"
          },
          {
            "name": "Concat / IDefOpt / tpu / Primal",
            "value": 0.000001532325,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / tpu / Forward",
            "value": 0.000001573125,
            "unit": "s"
          },
          {
            "name": "Concat / Jax / tpu / Forward",
            "value": 0.0000015509,
            "unit": "s"
          },
          {
            "name": "Concat / HLOOpt / tpu / Forward",
            "value": 0.0000016161,
            "unit": "s"
          },
          {
            "name": "Concat / PartOpt / tpu / Forward",
            "value": 0.0000015552250000000002,
            "unit": "s"
          },
          {
            "name": "Concat / IPartOpt / tpu / Forward",
            "value": 0.0000015835250000000002,
            "unit": "s"
          },
          {
            "name": "Concat / DefOpt / tpu / Forward",
            "value": 0.000001561825,
            "unit": "s"
          },
          {
            "name": "Concat / IDefOpt / tpu / Forward",
            "value": 0.000001583825,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / tpu / PreRev",
            "value": 0.000002006275,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / tpu / PostRev",
            "value": 0.000002082175,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / tpu / BothRev",
            "value": 0.000001995175,
            "unit": "s"
          },
          {
            "name": "Concat / Jax / tpu / BothRev",
            "value": 0.00000208245,
            "unit": "s"
          },
          {
            "name": "Concat / HLOOpt / tpu / PreRev",
            "value": 0.000002000925,
            "unit": "s"
          },
          {
            "name": "Concat / HLOOpt / tpu / PostRev",
            "value": 0.0000020786,
            "unit": "s"
          },
          {
            "name": "Concat / HLOOpt / tpu / BothRev",
            "value": 0.0000019997750000000003,
            "unit": "s"
          },
          {
            "name": "Concat / PartOpt / tpu / PreRev",
            "value": 0.00000207215,
            "unit": "s"
          },
          {
            "name": "Concat / PartOpt / tpu / PostRev",
            "value": 0.00000199445,
            "unit": "s"
          },
          {
            "name": "Concat / PartOpt / tpu / BothRev",
            "value": 0.0000020702,
            "unit": "s"
          },
          {
            "name": "Concat / IPartOpt / tpu / PreRev",
            "value": 0.00000199525,
            "unit": "s"
          },
          {
            "name": "Concat / IPartOpt / tpu / PostRev",
            "value": 0.00000208815,
            "unit": "s"
          },
          {
            "name": "Concat / IPartOpt / tpu / BothRev",
            "value": 0.0000020050000000000003,
            "unit": "s"
          },
          {
            "name": "Concat / DefOpt / tpu / PreRev",
            "value": 0.000002073775,
            "unit": "s"
          },
          {
            "name": "Concat / DefOpt / tpu / PostRev",
            "value": 0.0000019988,
            "unit": "s"
          },
          {
            "name": "Concat / DefOpt / tpu / BothRev",
            "value": 0.0000020704,
            "unit": "s"
          },
          {
            "name": "Concat / IDefOpt / tpu / PreRev",
            "value": 0.000001993925,
            "unit": "s"
          },
          {
            "name": "Concat / IDefOpt / tpu / PostRev",
            "value": 0.000002078075,
            "unit": "s"
          },
          {
            "name": "Concat / IDefOpt / tpu / BothRev",
            "value": 0.00000200575,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / Primal",
            "value": 0.000016147000000000003,
            "unit": "s"
          },
          {
            "name": "Concat / Jax / cpu / Primal",
            "value": 0.000016094000000000002,
            "unit": "s"
          },
          {
            "name": "Concat / HLOOpt / cpu / Primal",
            "value": 0.000016129,
            "unit": "s"
          },
          {
            "name": "Concat / PartOpt / cpu / Primal",
            "value": 0.000016045,
            "unit": "s"
          },
          {
            "name": "Concat / IPartOpt / cpu / Primal",
            "value": 0.000015822,
            "unit": "s"
          },
          {
            "name": "Concat / DefOpt / cpu / Primal",
            "value": 0.000015993,
            "unit": "s"
          },
          {
            "name": "Concat / IDefOpt / cpu / Primal",
            "value": 0.000016111,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / Forward",
            "value": 0.00002232,
            "unit": "s"
          },
          {
            "name": "Concat / Jax / cpu / Forward",
            "value": 0.000021713,
            "unit": "s"
          },
          {
            "name": "Concat / HLOOpt / cpu / Forward",
            "value": 0.000021694000000000003,
            "unit": "s"
          },
          {
            "name": "Concat / PartOpt / cpu / Forward",
            "value": 0.000021866,
            "unit": "s"
          },
          {
            "name": "Concat / IPartOpt / cpu / Forward",
            "value": 0.000021583,
            "unit": "s"
          },
          {
            "name": "Concat / DefOpt / cpu / Forward",
            "value": 0.000021829,
            "unit": "s"
          },
          {
            "name": "Concat / IDefOpt / cpu / Forward",
            "value": 0.0000217,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / PreRev",
            "value": 0.000025401,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / PostRev",
            "value": 0.000024555,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / BothRev",
            "value": 0.000024147,
            "unit": "s"
          },
          {
            "name": "Concat / Jax / cpu / BothRev",
            "value": 0.000024327000000000003,
            "unit": "s"
          },
          {
            "name": "Concat / HLOOpt / cpu / PreRev",
            "value": 0.000024511000000000003,
            "unit": "s"
          },
          {
            "name": "Concat / HLOOpt / cpu / PostRev",
            "value": 0.000024213,
            "unit": "s"
          },
          {
            "name": "Concat / HLOOpt / cpu / BothRev",
            "value": 0.000023973,
            "unit": "s"
          },
          {
            "name": "Concat / PartOpt / cpu / PreRev",
            "value": 0.000024346,
            "unit": "s"
          },
          {
            "name": "Concat / PartOpt / cpu / PostRev",
            "value": 0.000024436,
            "unit": "s"
          },
          {
            "name": "Concat / PartOpt / cpu / BothRev",
            "value": 0.000024564,
            "unit": "s"
          },
          {
            "name": "Concat / IPartOpt / cpu / PreRev",
            "value": 0.000024695000000000003,
            "unit": "s"
          },
          {
            "name": "Concat / IPartOpt / cpu / PostRev",
            "value": 0.00002415,
            "unit": "s"
          },
          {
            "name": "Concat / IPartOpt / cpu / BothRev",
            "value": 0.000024415,
            "unit": "s"
          },
          {
            "name": "Concat / DefOpt / cpu / PreRev",
            "value": 0.000024217,
            "unit": "s"
          },
          {
            "name": "Concat / DefOpt / cpu / PostRev",
            "value": 0.000024568,
            "unit": "s"
          },
          {
            "name": "Concat / DefOpt / cpu / BothRev",
            "value": 0.000024115,
            "unit": "s"
          },
          {
            "name": "Concat / IDefOpt / cpu / PreRev",
            "value": 0.000024481,
            "unit": "s"
          },
          {
            "name": "Concat / IDefOpt / cpu / PostRev",
            "value": 0.000024524,
            "unit": "s"
          },
          {
            "name": "Concat / IDefOpt / cpu / BothRev",
            "value": 0.000024454,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / Primal",
            "value": 0.000008,
            "unit": "s"
          },
          {
            "name": "Concat / Jax / cpu / Primal",
            "value": 0.000008,
            "unit": "s"
          },
          {
            "name": "Concat / HLOOpt / cpu / Primal",
            "value": 0.000008,
            "unit": "s"
          },
          {
            "name": "Concat / PartOpt / cpu / Primal",
            "value": 0.000008,
            "unit": "s"
          },
          {
            "name": "Concat / IPartOpt / cpu / Primal",
            "value": 0.000008,
            "unit": "s"
          },
          {
            "name": "Concat / DefOpt / cpu / Primal",
            "value": 0.000008999999999999999,
            "unit": "s"
          },
          {
            "name": "Concat / IDefOpt / cpu / Primal",
            "value": 0.000008,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / Forward",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "Concat / Jax / cpu / Forward",
            "value": 0.000011,
            "unit": "s"
          },
          {
            "name": "Concat / HLOOpt / cpu / Forward",
            "value": 0.000011,
            "unit": "s"
          },
          {
            "name": "Concat / PartOpt / cpu / Forward",
            "value": 0.000011,
            "unit": "s"
          },
          {
            "name": "Concat / IPartOpt / cpu / Forward",
            "value": 0.000011,
            "unit": "s"
          },
          {
            "name": "Concat / DefOpt / cpu / Forward",
            "value": 0.000011,
            "unit": "s"
          },
          {
            "name": "Concat / IDefOpt / cpu / Forward",
            "value": 0.000011,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / PreRev",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / PostRev",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / BothRev",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "Concat / Jax / cpu / BothRev",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "Concat / HLOOpt / cpu / PreRev",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "Concat / HLOOpt / cpu / PostRev",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "Concat / HLOOpt / cpu / BothRev",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "Concat / PartOpt / cpu / PreRev",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "Concat / PartOpt / cpu / PostRev",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "Concat / PartOpt / cpu / BothRev",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "Concat / IPartOpt / cpu / PreRev",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "Concat / IPartOpt / cpu / PostRev",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "Concat / IPartOpt / cpu / BothRev",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "Concat / DefOpt / cpu / PreRev",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "Concat / DefOpt / cpu / PostRev",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "Concat / DefOpt / cpu / BothRev",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "Concat / IDefOpt / cpu / PreRev",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "Concat / IDefOpt / cpu / PostRev",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "Concat / IDefOpt / cpu / BothRev",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / cpu / Primal",
            "value": 0.000006270819994824706,
            "unit": "s"
          },
          {
            "name": "const_scatter / Jax / cpu / Primal",
            "value": 0.000006750480079062981,
            "unit": "s"
          },
          {
            "name": "const_scatter / HLOOpt / cpu / Primal",
            "value": 0.000007037479990685824,
            "unit": "s"
          },
          {
            "name": "const_scatter / PartOpt / cpu / Primal",
            "value": 0.000006410139903891832,
            "unit": "s"
          },
          {
            "name": "const_scatter / IPartOpt / cpu / Primal",
            "value": 0.00000732458007405512,
            "unit": "s"
          },
          {
            "name": "const_scatter / DefOpt / cpu / Primal",
            "value": 0.000007023239977570483,
            "unit": "s"
          },
          {
            "name": "const_scatter / IDefOpt / cpu / Primal",
            "value": 0.00000735502000679844,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / cpu / Forward",
            "value": 0.000010767019994091242,
            "unit": "s"
          },
          {
            "name": "const_scatter / Jax / cpu / Forward",
            "value": 0.000009683120042609517,
            "unit": "s"
          },
          {
            "name": "const_scatter / HLOOpt / cpu / Forward",
            "value": 0.00001097736001611338,
            "unit": "s"
          },
          {
            "name": "const_scatter / PartOpt / cpu / Forward",
            "value": 0.000010658339997462462,
            "unit": "s"
          },
          {
            "name": "const_scatter / IPartOpt / cpu / Forward",
            "value": 0.00001080328001989983,
            "unit": "s"
          },
          {
            "name": "const_scatter / DefOpt / cpu / Forward",
            "value": 0.00001036856003338471,
            "unit": "s"
          },
          {
            "name": "const_scatter / IDefOpt / cpu / Forward",
            "value": 0.000010123299962288,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / cpu / PreRev",
            "value": 0.000287967600052,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / cpu / PostRev",
            "value": 0.0002877710799657,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / cpu / BothRev",
            "value": 0.0002813569199861,
            "unit": "s"
          },
          {
            "name": "const_scatter / Jax / cpu / BothRev",
            "value": 0.0002826536399152,
            "unit": "s"
          },
          {
            "name": "const_scatter / HLOOpt / cpu / PreRev",
            "value": 0.0002875491999839,
            "unit": "s"
          },
          {
            "name": "const_scatter / HLOOpt / cpu / PostRev",
            "value": 0.000288759399973,
            "unit": "s"
          },
          {
            "name": "const_scatter / HLOOpt / cpu / BothRev",
            "value": 0.0002827222199812,
            "unit": "s"
          },
          {
            "name": "const_scatter / PartOpt / cpu / PreRev",
            "value": 0.0002987650800241,
            "unit": "s"
          },
          {
            "name": "const_scatter / PartOpt / cpu / PostRev",
            "value": 0.000283615240005,
            "unit": "s"
          },
          {
            "name": "const_scatter / PartOpt / cpu / BothRev",
            "value": 0.0002822221200403,
            "unit": "s"
          },
          {
            "name": "const_scatter / IPartOpt / cpu / PreRev",
            "value": 0.000305242320046,
            "unit": "s"
          },
          {
            "name": "const_scatter / IPartOpt / cpu / PostRev",
            "value": 0.0002822328400361,
            "unit": "s"
          },
          {
            "name": "const_scatter / IPartOpt / cpu / BothRev",
            "value": 0.0002852799599349,
            "unit": "s"
          },
          {
            "name": "const_scatter / DefOpt / cpu / PreRev",
            "value": 0.0003124562600351,
            "unit": "s"
          },
          {
            "name": "const_scatter / DefOpt / cpu / PostRev",
            "value": 0.0002819802199883,
            "unit": "s"
          },
          {
            "name": "const_scatter / DefOpt / cpu / BothRev",
            "value": 0.0002986778999911,
            "unit": "s"
          },
          {
            "name": "const_scatter / IDefOpt / cpu / PreRev",
            "value": 0.0002813842000068,
            "unit": "s"
          },
          {
            "name": "const_scatter / IDefOpt / cpu / PostRev",
            "value": 0.0002801111600456,
            "unit": "s"
          },
          {
            "name": "const_scatter / IDefOpt / cpu / BothRev",
            "value": 0.0002842026600592,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / cuda / Primal",
            "value": 0.000001887,
            "unit": "s"
          },
          {
            "name": "const_scatter / Jax / cuda / Primal",
            "value": 0.000001889,
            "unit": "s"
          },
          {
            "name": "const_scatter / HLOOpt / cuda / Primal",
            "value": 0.000001888,
            "unit": "s"
          },
          {
            "name": "const_scatter / PartOpt / cuda / Primal",
            "value": 0.000001888,
            "unit": "s"
          },
          {
            "name": "const_scatter / IPartOpt / cuda / Primal",
            "value": 0.000001887,
            "unit": "s"
          },
          {
            "name": "const_scatter / DefOpt / cuda / Primal",
            "value": 0.000001919,
            "unit": "s"
          },
          {
            "name": "const_scatter / IDefOpt / cuda / Primal",
            "value": 0.000001888,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / cuda / Forward",
            "value": 0.000009952,
            "unit": "s"
          },
          {
            "name": "const_scatter / Jax / cuda / Forward",
            "value": 0.00001008,
            "unit": "s"
          },
          {
            "name": "const_scatter / HLOOpt / cuda / Forward",
            "value": 0.000010784,
            "unit": "s"
          },
          {
            "name": "const_scatter / PartOpt / cuda / Forward",
            "value": 0.000011168,
            "unit": "s"
          },
          {
            "name": "const_scatter / IPartOpt / cuda / Forward",
            "value": 0.000010976,
            "unit": "s"
          },
          {
            "name": "const_scatter / DefOpt / cuda / Forward",
            "value": 0.000010112,
            "unit": "s"
          },
          {
            "name": "const_scatter / IDefOpt / cuda / Forward",
            "value": 0.000010304,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / cuda / PreRev",
            "value": 0.000017760000000000003,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / cuda / PostRev",
            "value": 0.000017119,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / cuda / BothRev",
            "value": 0.000016864,
            "unit": "s"
          },
          {
            "name": "const_scatter / Jax / cuda / BothRev",
            "value": 0.000016927999999999998,
            "unit": "s"
          },
          {
            "name": "const_scatter / HLOOpt / cuda / PreRev",
            "value": 0.000016832,
            "unit": "s"
          },
          {
            "name": "const_scatter / HLOOpt / cuda / PostRev",
            "value": 0.000017024,
            "unit": "s"
          },
          {
            "name": "const_scatter / HLOOpt / cuda / BothRev",
            "value": 0.00001648,
            "unit": "s"
          },
          {
            "name": "const_scatter / PartOpt / cuda / PreRev",
            "value": 0.000016864,
            "unit": "s"
          },
          {
            "name": "const_scatter / PartOpt / cuda / PostRev",
            "value": 0.000016673,
            "unit": "s"
          },
          {
            "name": "const_scatter / PartOpt / cuda / BothRev",
            "value": 0.000017536,
            "unit": "s"
          },
          {
            "name": "const_scatter / IPartOpt / cuda / PreRev",
            "value": 0.000017056,
            "unit": "s"
          },
          {
            "name": "const_scatter / IPartOpt / cuda / PostRev",
            "value": 0.000016896000000000002,
            "unit": "s"
          },
          {
            "name": "const_scatter / IPartOpt / cuda / BothRev",
            "value": 0.000016288,
            "unit": "s"
          },
          {
            "name": "const_scatter / DefOpt / cuda / PreRev",
            "value": 0.000016927999999999998,
            "unit": "s"
          },
          {
            "name": "const_scatter / DefOpt / cuda / PostRev",
            "value": 0.000015904999999999998,
            "unit": "s"
          },
          {
            "name": "const_scatter / DefOpt / cuda / BothRev",
            "value": 0.000016576000000000002,
            "unit": "s"
          },
          {
            "name": "const_scatter / IDefOpt / cuda / PreRev",
            "value": 0.00001696,
            "unit": "s"
          },
          {
            "name": "const_scatter / IDefOpt / cuda / PostRev",
            "value": 0.000016767,
            "unit": "s"
          },
          {
            "name": "const_scatter / IDefOpt / cuda / BothRev",
            "value": 0.000016768000000000003,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / tpu / Primal",
            "value": 0.00000379725,
            "unit": "s"
          },
          {
            "name": "const_scatter / Jax / tpu / Primal",
            "value": 0.000003805925,
            "unit": "s"
          },
          {
            "name": "const_scatter / HLOOpt / tpu / Primal",
            "value": 0.0000038029,
            "unit": "s"
          },
          {
            "name": "const_scatter / PartOpt / tpu / Primal",
            "value": 0.0000038146,
            "unit": "s"
          },
          {
            "name": "const_scatter / IPartOpt / tpu / Primal",
            "value": 0.0000037902,
            "unit": "s"
          },
          {
            "name": "const_scatter / DefOpt / tpu / Primal",
            "value": 0.00000382115,
            "unit": "s"
          },
          {
            "name": "const_scatter / IDefOpt / tpu / Primal",
            "value": 0.000003809475,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / tpu / Forward",
            "value": 0.000006473525,
            "unit": "s"
          },
          {
            "name": "const_scatter / Jax / tpu / Forward",
            "value": 0.000006507425,
            "unit": "s"
          },
          {
            "name": "const_scatter / HLOOpt / tpu / Forward",
            "value": 0.00000647555,
            "unit": "s"
          },
          {
            "name": "const_scatter / PartOpt / tpu / Forward",
            "value": 0.000006493525,
            "unit": "s"
          },
          {
            "name": "const_scatter / IPartOpt / tpu / Forward",
            "value": 0.00000646445,
            "unit": "s"
          },
          {
            "name": "const_scatter / DefOpt / tpu / Forward",
            "value": 0.000006492525,
            "unit": "s"
          },
          {
            "name": "const_scatter / IDefOpt / tpu / Forward",
            "value": 0.000006468025,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / tpu / PreRev",
            "value": 0.000006669349999999999,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / tpu / PostRev",
            "value": 0.000006676875,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / tpu / BothRev",
            "value": 0.000006671725,
            "unit": "s"
          },
          {
            "name": "const_scatter / Jax / tpu / BothRev",
            "value": 0.000006663175,
            "unit": "s"
          },
          {
            "name": "const_scatter / HLOOpt / tpu / PreRev",
            "value": 0.0000066711,
            "unit": "s"
          },
          {
            "name": "const_scatter / HLOOpt / tpu / PostRev",
            "value": 0.0000066505,
            "unit": "s"
          },
          {
            "name": "const_scatter / HLOOpt / tpu / BothRev",
            "value": 0.000006662749999999999,
            "unit": "s"
          },
          {
            "name": "const_scatter / PartOpt / tpu / PreRev",
            "value": 0.000006691275,
            "unit": "s"
          },
          {
            "name": "const_scatter / PartOpt / tpu / PostRev",
            "value": 0.000006672375000000001,
            "unit": "s"
          },
          {
            "name": "const_scatter / PartOpt / tpu / BothRev",
            "value": 0.00000666775,
            "unit": "s"
          },
          {
            "name": "const_scatter / IPartOpt / tpu / PreRev",
            "value": 0.000006666525,
            "unit": "s"
          },
          {
            "name": "const_scatter / IPartOpt / tpu / PostRev",
            "value": 0.0000066675,
            "unit": "s"
          },
          {
            "name": "const_scatter / IPartOpt / tpu / BothRev",
            "value": 0.00000664655,
            "unit": "s"
          },
          {
            "name": "const_scatter / DefOpt / tpu / PreRev",
            "value": 0.00000666765,
            "unit": "s"
          },
          {
            "name": "const_scatter / DefOpt / tpu / PostRev",
            "value": 0.000006651725000000001,
            "unit": "s"
          },
          {
            "name": "const_scatter / DefOpt / tpu / BothRev",
            "value": 0.000006682025000000001,
            "unit": "s"
          },
          {
            "name": "const_scatter / IDefOpt / tpu / PreRev",
            "value": 0.000006661775,
            "unit": "s"
          },
          {
            "name": "const_scatter / IDefOpt / tpu / PostRev",
            "value": 0.000006676475,
            "unit": "s"
          },
          {
            "name": "const_scatter / IDefOpt / tpu / BothRev",
            "value": 0.000006648925,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / cpu / Primal",
            "value": 0.00001611,
            "unit": "s"
          },
          {
            "name": "const_scatter / Jax / cpu / Primal",
            "value": 0.000015961,
            "unit": "s"
          },
          {
            "name": "const_scatter / HLOOpt / cpu / Primal",
            "value": 0.000016559999999999997,
            "unit": "s"
          },
          {
            "name": "const_scatter / PartOpt / cpu / Primal",
            "value": 0.000015949999999999998,
            "unit": "s"
          },
          {
            "name": "const_scatter / IPartOpt / cpu / Primal",
            "value": 0.00001592,
            "unit": "s"
          },
          {
            "name": "const_scatter / DefOpt / cpu / Primal",
            "value": 0.000016657000000000003,
            "unit": "s"
          },
          {
            "name": "const_scatter / IDefOpt / cpu / Primal",
            "value": 0.000017214,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / cpu / Forward",
            "value": 0.000022839000000000003,
            "unit": "s"
          },
          {
            "name": "const_scatter / Jax / cpu / Forward",
            "value": 0.00002118,
            "unit": "s"
          },
          {
            "name": "const_scatter / HLOOpt / cpu / Forward",
            "value": 0.00002307,
            "unit": "s"
          },
          {
            "name": "const_scatter / PartOpt / cpu / Forward",
            "value": 0.000022203,
            "unit": "s"
          },
          {
            "name": "const_scatter / IPartOpt / cpu / Forward",
            "value": 0.000023,
            "unit": "s"
          },
          {
            "name": "const_scatter / DefOpt / cpu / Forward",
            "value": 0.000022264,
            "unit": "s"
          },
          {
            "name": "const_scatter / IDefOpt / cpu / Forward",
            "value": 0.000028289000000000003,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / cpu / PreRev",
            "value": 0.000532325,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / cpu / PostRev",
            "value": 0.000532201,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / cpu / BothRev",
            "value": 0.0005400009999999,
            "unit": "s"
          },
          {
            "name": "const_scatter / Jax / cpu / BothRev",
            "value": 0.000541215,
            "unit": "s"
          },
          {
            "name": "const_scatter / HLOOpt / cpu / PreRev",
            "value": 0.000541629,
            "unit": "s"
          },
          {
            "name": "const_scatter / HLOOpt / cpu / PostRev",
            "value": 0.000538948,
            "unit": "s"
          },
          {
            "name": "const_scatter / HLOOpt / cpu / BothRev",
            "value": 0.000543385,
            "unit": "s"
          },
          {
            "name": "const_scatter / PartOpt / cpu / PreRev",
            "value": 0.000536208,
            "unit": "s"
          },
          {
            "name": "const_scatter / PartOpt / cpu / PostRev",
            "value": 0.0005361869999999,
            "unit": "s"
          },
          {
            "name": "const_scatter / PartOpt / cpu / BothRev",
            "value": 0.000536688,
            "unit": "s"
          },
          {
            "name": "const_scatter / IPartOpt / cpu / PreRev",
            "value": 0.00054575,
            "unit": "s"
          },
          {
            "name": "const_scatter / IPartOpt / cpu / PostRev",
            "value": 0.000538148,
            "unit": "s"
          },
          {
            "name": "const_scatter / IPartOpt / cpu / BothRev",
            "value": 0.000539126,
            "unit": "s"
          },
          {
            "name": "const_scatter / DefOpt / cpu / PreRev",
            "value": 0.000537855,
            "unit": "s"
          },
          {
            "name": "const_scatter / DefOpt / cpu / PostRev",
            "value": 0.000552605,
            "unit": "s"
          },
          {
            "name": "const_scatter / DefOpt / cpu / BothRev",
            "value": 0.0005466189999999,
            "unit": "s"
          },
          {
            "name": "const_scatter / IDefOpt / cpu / PreRev",
            "value": 0.0005399659999999,
            "unit": "s"
          },
          {
            "name": "const_scatter / IDefOpt / cpu / PostRev",
            "value": 0.000538263,
            "unit": "s"
          },
          {
            "name": "const_scatter / IDefOpt / cpu / BothRev",
            "value": 0.000538012,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / cpu / Primal",
            "value": 0.000008,
            "unit": "s"
          },
          {
            "name": "const_scatter / Jax / cpu / Primal",
            "value": 0.000008,
            "unit": "s"
          },
          {
            "name": "const_scatter / HLOOpt / cpu / Primal",
            "value": 0.000008999999999999999,
            "unit": "s"
          },
          {
            "name": "const_scatter / PartOpt / cpu / Primal",
            "value": 0.000008,
            "unit": "s"
          },
          {
            "name": "const_scatter / IPartOpt / cpu / Primal",
            "value": 0.000008,
            "unit": "s"
          },
          {
            "name": "const_scatter / DefOpt / cpu / Primal",
            "value": 0.000008,
            "unit": "s"
          },
          {
            "name": "const_scatter / IDefOpt / cpu / Primal",
            "value": 0.000008999999999999999,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / cpu / Forward",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "const_scatter / Jax / cpu / Forward",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "const_scatter / HLOOpt / cpu / Forward",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "const_scatter / PartOpt / cpu / Forward",
            "value": 0.000011,
            "unit": "s"
          },
          {
            "name": "const_scatter / IPartOpt / cpu / Forward",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "const_scatter / DefOpt / cpu / Forward",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "const_scatter / IDefOpt / cpu / Forward",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / cpu / PreRev",
            "value": 0.000352,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / cpu / PostRev",
            "value": 0.0003689999999999,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / cpu / BothRev",
            "value": 0.0003529999999999,
            "unit": "s"
          },
          {
            "name": "const_scatter / Jax / cpu / BothRev",
            "value": 0.000359,
            "unit": "s"
          },
          {
            "name": "const_scatter / HLOOpt / cpu / PreRev",
            "value": 0.00035,
            "unit": "s"
          },
          {
            "name": "const_scatter / HLOOpt / cpu / PostRev",
            "value": 0.0003599999999999,
            "unit": "s"
          },
          {
            "name": "const_scatter / HLOOpt / cpu / BothRev",
            "value": 0.000359,
            "unit": "s"
          },
          {
            "name": "const_scatter / PartOpt / cpu / PreRev",
            "value": 0.0003599999999999,
            "unit": "s"
          },
          {
            "name": "const_scatter / PartOpt / cpu / PostRev",
            "value": 0.0003549999999999,
            "unit": "s"
          },
          {
            "name": "const_scatter / PartOpt / cpu / BothRev",
            "value": 0.000357,
            "unit": "s"
          },
          {
            "name": "const_scatter / IPartOpt / cpu / PreRev",
            "value": 0.000327,
            "unit": "s"
          },
          {
            "name": "const_scatter / IPartOpt / cpu / PostRev",
            "value": 0.000341,
            "unit": "s"
          },
          {
            "name": "const_scatter / IPartOpt / cpu / BothRev",
            "value": 0.000357,
            "unit": "s"
          },
          {
            "name": "const_scatter / DefOpt / cpu / PreRev",
            "value": 0.000334,
            "unit": "s"
          },
          {
            "name": "const_scatter / DefOpt / cpu / PostRev",
            "value": 0.000361,
            "unit": "s"
          },
          {
            "name": "const_scatter / DefOpt / cpu / BothRev",
            "value": 0.0003549999999999,
            "unit": "s"
          },
          {
            "name": "const_scatter / IDefOpt / cpu / PreRev",
            "value": 0.0003599999999999,
            "unit": "s"
          },
          {
            "name": "const_scatter / IDefOpt / cpu / PostRev",
            "value": 0.0003689999999999,
            "unit": "s"
          },
          {
            "name": "const_scatter / IDefOpt / cpu / BothRev",
            "value": 0.000358,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / Primal",
            "value": 0.000006800720111641567,
            "unit": "s"
          },
          {
            "name": "GenDot / Jax / cpu / Primal",
            "value": 0.000006721360023220768,
            "unit": "s"
          },
          {
            "name": "GenDot / HLOOpt / cpu / Primal",
            "value": 0.000007462839985237224,
            "unit": "s"
          },
          {
            "name": "GenDot / PartOpt / cpu / Primal",
            "value": 0.000006579459895874607,
            "unit": "s"
          },
          {
            "name": "GenDot / IPartOpt / cpu / Primal",
            "value": 0.000007395619941235054,
            "unit": "s"
          },
          {
            "name": "GenDot / DefOpt / cpu / Primal",
            "value": 0.000006977419925533468,
            "unit": "s"
          },
          {
            "name": "GenDot / IDefOpt / cpu / Primal",
            "value": 0.000006918359995324863,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / Forward",
            "value": 0.000010688940037653082,
            "unit": "s"
          },
          {
            "name": "GenDot / Jax / cpu / Forward",
            "value": 0.000010772600053314818,
            "unit": "s"
          },
          {
            "name": "GenDot / HLOOpt / cpu / Forward",
            "value": 0.000011065479939134092,
            "unit": "s"
          },
          {
            "name": "GenDot / PartOpt / cpu / Forward",
            "value": 0.000010858200021175436,
            "unit": "s"
          },
          {
            "name": "GenDot / IPartOpt / cpu / Forward",
            "value": 0.000011188499920535832,
            "unit": "s"
          },
          {
            "name": "GenDot / DefOpt / cpu / Forward",
            "value": 0.000010039800054073569,
            "unit": "s"
          },
          {
            "name": "GenDot / IDefOpt / cpu / Forward",
            "value": 0.00001081181995687075,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / PreRev",
            "value": 0.000010748400090960786,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / PostRev",
            "value": 0.000009941180014720885,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / BothRev",
            "value": 0.000011713799976860172,
            "unit": "s"
          },
          {
            "name": "GenDot / Jax / cpu / BothRev",
            "value": 0.000010109040049428586,
            "unit": "s"
          },
          {
            "name": "GenDot / HLOOpt / cpu / PreRev",
            "value": 0.000010873480059672149,
            "unit": "s"
          },
          {
            "name": "GenDot / HLOOpt / cpu / PostRev",
            "value": 0.00001660521993471775,
            "unit": "s"
          },
          {
            "name": "GenDot / HLOOpt / cpu / BothRev",
            "value": 0.000010644839985616272,
            "unit": "s"
          },
          {
            "name": "GenDot / PartOpt / cpu / PreRev",
            "value": 0.000010912500038102736,
            "unit": "s"
          },
          {
            "name": "GenDot / PartOpt / cpu / PostRev",
            "value": 0.000009655779995227932,
            "unit": "s"
          },
          {
            "name": "GenDot / PartOpt / cpu / BothRev",
            "value": 0.000011709039954439504,
            "unit": "s"
          },
          {
            "name": "GenDot / IPartOpt / cpu / PreRev",
            "value": 0.000010970120019919704,
            "unit": "s"
          },
          {
            "name": "GenDot / IPartOpt / cpu / PostRev",
            "value": 0.00001002328002869035,
            "unit": "s"
          },
          {
            "name": "GenDot / IPartOpt / cpu / BothRev",
            "value": 0.000010897599986492424,
            "unit": "s"
          },
          {
            "name": "GenDot / DefOpt / cpu / PreRev",
            "value": 0.000011116620080429128,
            "unit": "s"
          },
          {
            "name": "GenDot / DefOpt / cpu / PostRev",
            "value": 0.000011701779985742178,
            "unit": "s"
          },
          {
            "name": "GenDot / DefOpt / cpu / BothRev",
            "value": 0.000010825200097315249,
            "unit": "s"
          },
          {
            "name": "GenDot / IDefOpt / cpu / PreRev",
            "value": 0.000010971780047839277,
            "unit": "s"
          },
          {
            "name": "GenDot / IDefOpt / cpu / PostRev",
            "value": 0.000011441580063547008,
            "unit": "s"
          },
          {
            "name": "GenDot / IDefOpt / cpu / BothRev",
            "value": 0.0000117744200542802,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cuda / Primal",
            "value": 0.000002016,
            "unit": "s"
          },
          {
            "name": "GenDot / Jax / cuda / Primal",
            "value": 0.000002016,
            "unit": "s"
          },
          {
            "name": "GenDot / HLOOpt / cuda / Primal",
            "value": 0.000002015,
            "unit": "s"
          },
          {
            "name": "GenDot / PartOpt / cuda / Primal",
            "value": 0.000002016,
            "unit": "s"
          },
          {
            "name": "GenDot / IPartOpt / cuda / Primal",
            "value": 0.000002016,
            "unit": "s"
          },
          {
            "name": "GenDot / DefOpt / cuda / Primal",
            "value": 0.000002015,
            "unit": "s"
          },
          {
            "name": "GenDot / IDefOpt / cuda / Primal",
            "value": 0.000002015,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cuda / Forward",
            "value": 0.00001024,
            "unit": "s"
          },
          {
            "name": "GenDot / Jax / cuda / Forward",
            "value": 0.000010432,
            "unit": "s"
          },
          {
            "name": "GenDot / HLOOpt / cuda / Forward",
            "value": 0.000010304,
            "unit": "s"
          },
          {
            "name": "GenDot / PartOpt / cuda / Forward",
            "value": 0.000010399,
            "unit": "s"
          },
          {
            "name": "GenDot / IPartOpt / cuda / Forward",
            "value": 0.00001024,
            "unit": "s"
          },
          {
            "name": "GenDot / DefOpt / cuda / Forward",
            "value": 0.000010304,
            "unit": "s"
          },
          {
            "name": "GenDot / IDefOpt / cuda / Forward",
            "value": 0.000009793,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cuda / PreRev",
            "value": 0.000010336,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cuda / PostRev",
            "value": 0.000010144,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cuda / BothRev",
            "value": 0.000010656,
            "unit": "s"
          },
          {
            "name": "GenDot / Jax / cuda / BothRev",
            "value": 0.000010464,
            "unit": "s"
          },
          {
            "name": "GenDot / HLOOpt / cuda / PreRev",
            "value": 0.000010943,
            "unit": "s"
          },
          {
            "name": "GenDot / HLOOpt / cuda / PostRev",
            "value": 0.000009824,
            "unit": "s"
          },
          {
            "name": "GenDot / HLOOpt / cuda / BothRev",
            "value": 0.000010112,
            "unit": "s"
          },
          {
            "name": "GenDot / PartOpt / cuda / PreRev",
            "value": 0.000010112,
            "unit": "s"
          },
          {
            "name": "GenDot / PartOpt / cuda / PostRev",
            "value": 0.00000992,
            "unit": "s"
          },
          {
            "name": "GenDot / PartOpt / cuda / BothRev",
            "value": 0.000010144,
            "unit": "s"
          },
          {
            "name": "GenDot / IPartOpt / cuda / PreRev",
            "value": 0.000010176,
            "unit": "s"
          },
          {
            "name": "GenDot / IPartOpt / cuda / PostRev",
            "value": 0.000009888,
            "unit": "s"
          },
          {
            "name": "GenDot / IPartOpt / cuda / BothRev",
            "value": 0.000009632,
            "unit": "s"
          },
          {
            "name": "GenDot / DefOpt / cuda / PreRev",
            "value": 0.000010432,
            "unit": "s"
          },
          {
            "name": "GenDot / DefOpt / cuda / PostRev",
            "value": 0.000010272,
            "unit": "s"
          },
          {
            "name": "GenDot / DefOpt / cuda / BothRev",
            "value": 0.000009983,
            "unit": "s"
          },
          {
            "name": "GenDot / IDefOpt / cuda / PreRev",
            "value": 0.000010367,
            "unit": "s"
          },
          {
            "name": "GenDot / IDefOpt / cuda / PostRev",
            "value": 0.00000992,
            "unit": "s"
          },
          {
            "name": "GenDot / IDefOpt / cuda / BothRev",
            "value": 0.000010112,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / tpu / Primal",
            "value": 9.29725e-7,
            "unit": "s"
          },
          {
            "name": "GenDot / Jax / tpu / Primal",
            "value": 9.2595e-7,
            "unit": "s"
          },
          {
            "name": "GenDot / HLOOpt / tpu / Primal",
            "value": 0.000001575475,
            "unit": "s"
          },
          {
            "name": "GenDot / PartOpt / tpu / Primal",
            "value": 9.25175e-7,
            "unit": "s"
          },
          {
            "name": "GenDot / IPartOpt / tpu / Primal",
            "value": 9.3015e-7,
            "unit": "s"
          },
          {
            "name": "GenDot / DefOpt / tpu / Primal",
            "value": 0.000001490025,
            "unit": "s"
          },
          {
            "name": "GenDot / IDefOpt / tpu / Primal",
            "value": 0.0000015884250000000002,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / tpu / Forward",
            "value": 0.00000315965,
            "unit": "s"
          },
          {
            "name": "GenDot / Jax / tpu / Forward",
            "value": 0.0000023204,
            "unit": "s"
          },
          {
            "name": "GenDot / HLOOpt / tpu / Forward",
            "value": 0.0000031171,
            "unit": "s"
          },
          {
            "name": "GenDot / PartOpt / tpu / Forward",
            "value": 0.000003218475,
            "unit": "s"
          },
          {
            "name": "GenDot / IPartOpt / tpu / Forward",
            "value": 0.0000031173250000000003,
            "unit": "s"
          },
          {
            "name": "GenDot / DefOpt / tpu / Forward",
            "value": 0.000003235675,
            "unit": "s"
          },
          {
            "name": "GenDot / IDefOpt / tpu / Forward",
            "value": 0.000003122875,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / tpu / PreRev",
            "value": 0.00000296765,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / tpu / PostRev",
            "value": 0.00000240825,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / tpu / BothRev",
            "value": 0.000002963725,
            "unit": "s"
          },
          {
            "name": "GenDot / Jax / tpu / BothRev",
            "value": 0.00000240405,
            "unit": "s"
          },
          {
            "name": "GenDot / HLOOpt / tpu / PreRev",
            "value": 0.000002956225,
            "unit": "s"
          },
          {
            "name": "GenDot / HLOOpt / tpu / PostRev",
            "value": 0.0000029371250000000003,
            "unit": "s"
          },
          {
            "name": "GenDot / HLOOpt / tpu / BothRev",
            "value": 0.0000029648,
            "unit": "s"
          },
          {
            "name": "GenDot / PartOpt / tpu / PreRev",
            "value": 0.000002937625,
            "unit": "s"
          },
          {
            "name": "GenDot / PartOpt / tpu / PostRev",
            "value": 0.000002400125,
            "unit": "s"
          },
          {
            "name": "GenDot / PartOpt / tpu / BothRev",
            "value": 0.00000294085,
            "unit": "s"
          },
          {
            "name": "GenDot / IPartOpt / tpu / PreRev",
            "value": 0.0000029628,
            "unit": "s"
          },
          {
            "name": "GenDot / IPartOpt / tpu / PostRev",
            "value": 0.000002409875,
            "unit": "s"
          },
          {
            "name": "GenDot / IPartOpt / tpu / BothRev",
            "value": 0.0000029594,
            "unit": "s"
          },
          {
            "name": "GenDot / DefOpt / tpu / PreRev",
            "value": 0.00000293575,
            "unit": "s"
          },
          {
            "name": "GenDot / DefOpt / tpu / PostRev",
            "value": 0.00000296265,
            "unit": "s"
          },
          {
            "name": "GenDot / DefOpt / tpu / BothRev",
            "value": 0.0000029359,
            "unit": "s"
          },
          {
            "name": "GenDot / IDefOpt / tpu / PreRev",
            "value": 0.0000029733,
            "unit": "s"
          },
          {
            "name": "GenDot / IDefOpt / tpu / PostRev",
            "value": 0.00000293785,
            "unit": "s"
          },
          {
            "name": "GenDot / IDefOpt / tpu / BothRev",
            "value": 0.0000029555,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / Primal",
            "value": 0.000018604,
            "unit": "s"
          },
          {
            "name": "GenDot / Jax / cpu / Primal",
            "value": 0.000018681,
            "unit": "s"
          },
          {
            "name": "GenDot / HLOOpt / cpu / Primal",
            "value": 0.000017725,
            "unit": "s"
          },
          {
            "name": "GenDot / PartOpt / cpu / Primal",
            "value": 0.000018139,
            "unit": "s"
          },
          {
            "name": "GenDot / IPartOpt / cpu / Primal",
            "value": 0.000018629,
            "unit": "s"
          },
          {
            "name": "GenDot / DefOpt / cpu / Primal",
            "value": 0.000017966,
            "unit": "s"
          },
          {
            "name": "GenDot / IDefOpt / cpu / Primal",
            "value": 0.000017851,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / Forward",
            "value": 0.000023934,
            "unit": "s"
          },
          {
            "name": "GenDot / Jax / cpu / Forward",
            "value": 0.000030822,
            "unit": "s"
          },
          {
            "name": "GenDot / HLOOpt / cpu / Forward",
            "value": 0.000024176,
            "unit": "s"
          },
          {
            "name": "GenDot / PartOpt / cpu / Forward",
            "value": 0.000024156,
            "unit": "s"
          },
          {
            "name": "GenDot / IPartOpt / cpu / Forward",
            "value": 0.000024182,
            "unit": "s"
          },
          {
            "name": "GenDot / DefOpt / cpu / Forward",
            "value": 0.000024136,
            "unit": "s"
          },
          {
            "name": "GenDot / IDefOpt / cpu / Forward",
            "value": 0.000024191,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / PreRev",
            "value": 0.00002415,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / PostRev",
            "value": 0.000024949,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / BothRev",
            "value": 0.000024139,
            "unit": "s"
          },
          {
            "name": "GenDot / Jax / cpu / BothRev",
            "value": 0.00002579,
            "unit": "s"
          },
          {
            "name": "GenDot / HLOOpt / cpu / PreRev",
            "value": 0.000024736,
            "unit": "s"
          },
          {
            "name": "GenDot / HLOOpt / cpu / PostRev",
            "value": 0.000024727,
            "unit": "s"
          },
          {
            "name": "GenDot / HLOOpt / cpu / BothRev",
            "value": 0.000024735,
            "unit": "s"
          },
          {
            "name": "GenDot / PartOpt / cpu / PreRev",
            "value": 0.000024231,
            "unit": "s"
          },
          {
            "name": "GenDot / PartOpt / cpu / PostRev",
            "value": 0.000025416,
            "unit": "s"
          },
          {
            "name": "GenDot / PartOpt / cpu / BothRev",
            "value": 0.000024497000000000003,
            "unit": "s"
          },
          {
            "name": "GenDot / IPartOpt / cpu / PreRev",
            "value": 0.000024314,
            "unit": "s"
          },
          {
            "name": "GenDot / IPartOpt / cpu / PostRev",
            "value": 0.000025473,
            "unit": "s"
          },
          {
            "name": "GenDot / IPartOpt / cpu / BothRev",
            "value": 0.000024723,
            "unit": "s"
          },
          {
            "name": "GenDot / DefOpt / cpu / PreRev",
            "value": 0.000024864,
            "unit": "s"
          },
          {
            "name": "GenDot / DefOpt / cpu / PostRev",
            "value": 0.000024269,
            "unit": "s"
          },
          {
            "name": "GenDot / DefOpt / cpu / BothRev",
            "value": 0.000024382,
            "unit": "s"
          },
          {
            "name": "GenDot / IDefOpt / cpu / PreRev",
            "value": 0.000024548,
            "unit": "s"
          },
          {
            "name": "GenDot / IDefOpt / cpu / PostRev",
            "value": 0.000024214,
            "unit": "s"
          },
          {
            "name": "GenDot / IDefOpt / cpu / BothRev",
            "value": 0.000024969,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / Primal",
            "value": 0.000008999999999999999,
            "unit": "s"
          },
          {
            "name": "GenDot / Jax / cpu / Primal",
            "value": 0.000008999999999999999,
            "unit": "s"
          },
          {
            "name": "GenDot / HLOOpt / cpu / Primal",
            "value": 0.000008,
            "unit": "s"
          },
          {
            "name": "GenDot / PartOpt / cpu / Primal",
            "value": 0.000008999999999999999,
            "unit": "s"
          },
          {
            "name": "GenDot / IPartOpt / cpu / Primal",
            "value": 0.000008999999999999999,
            "unit": "s"
          },
          {
            "name": "GenDot / DefOpt / cpu / Primal",
            "value": 0.000008999999999999999,
            "unit": "s"
          },
          {
            "name": "GenDot / IDefOpt / cpu / Primal",
            "value": 0.000008999999999999999,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / Forward",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "GenDot / Jax / cpu / Forward",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "GenDot / HLOOpt / cpu / Forward",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "GenDot / PartOpt / cpu / Forward",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "GenDot / IPartOpt / cpu / Forward",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "GenDot / DefOpt / cpu / Forward",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "GenDot / IDefOpt / cpu / Forward",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / PreRev",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / PostRev",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / BothRev",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "GenDot / Jax / cpu / BothRev",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "GenDot / HLOOpt / cpu / PreRev",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "GenDot / HLOOpt / cpu / PostRev",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "GenDot / HLOOpt / cpu / BothRev",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "GenDot / PartOpt / cpu / PreRev",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "GenDot / PartOpt / cpu / PostRev",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "GenDot / PartOpt / cpu / BothRev",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "GenDot / IPartOpt / cpu / PreRev",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "GenDot / IPartOpt / cpu / PostRev",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "GenDot / IPartOpt / cpu / BothRev",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "GenDot / DefOpt / cpu / PreRev",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "GenDot / DefOpt / cpu / PostRev",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "GenDot / DefOpt / cpu / BothRev",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "GenDot / IDefOpt / cpu / PreRev",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "GenDot / IDefOpt / cpu / PostRev",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "GenDot / IDefOpt / cpu / BothRev",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / Primal",
            "value": 0.000010890960002143402,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / Jax / cpu / Primal",
            "value": 0.000010527339927648429,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / HLOOpt / cpu / Primal",
            "value": 0.000010989100010192488,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / PartOpt / cpu / Primal",
            "value": 0.000011147759905725252,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IPartOpt / cpu / Primal",
            "value": 0.000010714719974203036,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / DefOpt / cpu / Primal",
            "value": 0.000010304540064680622,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IDefOpt / cpu / Primal",
            "value": 0.00001045866005370044,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / Forward",
            "value": 0.000014598060024582084,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / Jax / cpu / Forward",
            "value": 0.000014619100038544275,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / HLOOpt / cpu / Forward",
            "value": 0.000015280119969247606,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / PartOpt / cpu / Forward",
            "value": 0.000015230100070766638,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IPartOpt / cpu / Forward",
            "value": 0.000014910860008967575,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / DefOpt / cpu / Forward",
            "value": 0.000014811299970460822,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IDefOpt / cpu / Forward",
            "value": 0.00001486019997173571,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / PreRev",
            "value": 0.00001487210007326212,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / PostRev",
            "value": 0.000014923400103725724,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / BothRev",
            "value": 0.000014862699954392156,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / Jax / cpu / BothRev",
            "value": 0.000015588340029353277,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / HLOOpt / cpu / PreRev",
            "value": 0.00001526146003016038,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / HLOOpt / cpu / PostRev",
            "value": 0.00001663607996306382,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / HLOOpt / cpu / BothRev",
            "value": 0.000014464359919657,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / PartOpt / cpu / PreRev",
            "value": 0.000015609160072926898,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / PartOpt / cpu / PostRev",
            "value": 0.000015318620044126873,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / PartOpt / cpu / BothRev",
            "value": 0.000014685779988212743,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IPartOpt / cpu / PreRev",
            "value": 0.000015797580035723514,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IPartOpt / cpu / PostRev",
            "value": 0.000014612480081268586,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IPartOpt / cpu / BothRev",
            "value": 0.00001468528003897518,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / DefOpt / cpu / PreRev",
            "value": 0.000015582679934595945,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / DefOpt / cpu / PostRev",
            "value": 0.0000151766800081532,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / DefOpt / cpu / BothRev",
            "value": 0.000014255199967010412,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IDefOpt / cpu / PreRev",
            "value": 0.00001530218003608752,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IDefOpt / cpu / PostRev",
            "value": 0.000014585360004275571,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IDefOpt / cpu / BothRev",
            "value": 0.000014662179910374108,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cuda / Primal",
            "value": 0.000001983,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / Jax / cuda / Primal",
            "value": 0.000001983,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / HLOOpt / cuda / Primal",
            "value": 0.000001983,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / PartOpt / cuda / Primal",
            "value": 0.000001983,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IPartOpt / cuda / Primal",
            "value": 0.000001983,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / DefOpt / cuda / Primal",
            "value": 0.000001983,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IDefOpt / cuda / Primal",
            "value": 0.000001983,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cuda / Forward",
            "value": 0.000002047,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / Jax / cuda / Forward",
            "value": 0.000002048,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / HLOOpt / cuda / Forward",
            "value": 0.000002047,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / PartOpt / cuda / Forward",
            "value": 0.000002047,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IPartOpt / cuda / Forward",
            "value": 0.000002047,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / DefOpt / cuda / Forward",
            "value": 0.000002048,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IDefOpt / cuda / Forward",
            "value": 0.000002047,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cuda / PreRev",
            "value": 0.000002047,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cuda / PostRev",
            "value": 0.000002047,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cuda / BothRev",
            "value": 0.000002047,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / Jax / cuda / BothRev",
            "value": 0.000002048,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / HLOOpt / cuda / PreRev",
            "value": 0.000002047,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / HLOOpt / cuda / PostRev",
            "value": 0.000002047,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / HLOOpt / cuda / BothRev",
            "value": 0.000002047,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / PartOpt / cuda / PreRev",
            "value": 0.000002047,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / PartOpt / cuda / PostRev",
            "value": 0.000002047,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / PartOpt / cuda / BothRev",
            "value": 0.000002047,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IPartOpt / cuda / PreRev",
            "value": 0.000002047,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IPartOpt / cuda / PostRev",
            "value": 0.000002047,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IPartOpt / cuda / BothRev",
            "value": 0.000002047,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / DefOpt / cuda / PreRev",
            "value": 0.000002048,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / DefOpt / cuda / PostRev",
            "value": 0.000002047,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / DefOpt / cuda / BothRev",
            "value": 0.000002047,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IDefOpt / cuda / PreRev",
            "value": 0.000002047,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IDefOpt / cuda / PostRev",
            "value": 0.000002047,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IDefOpt / cuda / BothRev",
            "value": 0.000002047,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / tpu / Primal",
            "value": 9.07875e-7,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / Jax / tpu / Primal",
            "value": 9.695e-7,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / HLOOpt / tpu / Primal",
            "value": 9.40125e-7,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / PartOpt / tpu / Primal",
            "value": 9.69925e-7,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IPartOpt / tpu / Primal",
            "value": 9.41225e-7,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / DefOpt / tpu / Primal",
            "value": 9.7445e-7,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IDefOpt / tpu / Primal",
            "value": 9.414e-7,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / tpu / Forward",
            "value": 9.49375e-7,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / Jax / tpu / Forward",
            "value": 9.81925e-7,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / HLOOpt / tpu / Forward",
            "value": 9.74625e-7,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / PartOpt / tpu / Forward",
            "value": 9.59425e-7,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IPartOpt / tpu / Forward",
            "value": 9.7405e-7,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / DefOpt / tpu / Forward",
            "value": 9.59375e-7,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IDefOpt / tpu / Forward",
            "value": 9.74075e-7,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / tpu / PreRev",
            "value": 9.546e-7,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / tpu / PostRev",
            "value": 9.65825e-7,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / tpu / BothRev",
            "value": 9.95325e-7,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / Jax / tpu / BothRev",
            "value": 9.65575e-7,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / HLOOpt / tpu / PreRev",
            "value": 9.95125e-7,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / HLOOpt / tpu / PostRev",
            "value": 9.6555e-7,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / HLOOpt / tpu / BothRev",
            "value": 9.95225e-7,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / PartOpt / tpu / PreRev",
            "value": 9.650749999999998e-7,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / PartOpt / tpu / PostRev",
            "value": 9.95375e-7,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / PartOpt / tpu / BothRev",
            "value": 9.64875e-7,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IPartOpt / tpu / PreRev",
            "value": 9.9525e-7,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IPartOpt / tpu / PostRev",
            "value": 9.65025e-7,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IPartOpt / tpu / BothRev",
            "value": 9.949e-7,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / DefOpt / tpu / PreRev",
            "value": 9.653e-7,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / DefOpt / tpu / PostRev",
            "value": 9.95125e-7,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / DefOpt / tpu / BothRev",
            "value": 9.6545e-7,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IDefOpt / tpu / PreRev",
            "value": 9.95775e-7,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IDefOpt / tpu / PostRev",
            "value": 9.6555e-7,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IDefOpt / tpu / BothRev",
            "value": 9.9505e-7,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / Primal",
            "value": 0.00002179,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / Jax / cpu / Primal",
            "value": 0.000021705,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / HLOOpt / cpu / Primal",
            "value": 0.000021926,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / PartOpt / cpu / Primal",
            "value": 0.000021668,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IPartOpt / cpu / Primal",
            "value": 0.00002174,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / DefOpt / cpu / Primal",
            "value": 0.000021817,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IDefOpt / cpu / Primal",
            "value": 0.00002137,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / Forward",
            "value": 0.00002989,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / Jax / cpu / Forward",
            "value": 0.000029685,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / HLOOpt / cpu / Forward",
            "value": 0.000029902000000000003,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / PartOpt / cpu / Forward",
            "value": 0.000035261,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IPartOpt / cpu / Forward",
            "value": 0.000029778,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / DefOpt / cpu / Forward",
            "value": 0.000029869,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IDefOpt / cpu / Forward",
            "value": 0.000029368,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / PreRev",
            "value": 0.000030261,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / PostRev",
            "value": 0.000029387,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / BothRev",
            "value": 0.000029349,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / Jax / cpu / BothRev",
            "value": 0.000030133,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / HLOOpt / cpu / PreRev",
            "value": 0.000029268,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / HLOOpt / cpu / PostRev",
            "value": 0.00002906,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / HLOOpt / cpu / BothRev",
            "value": 0.000029046,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / PartOpt / cpu / PreRev",
            "value": 0.000028867,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / PartOpt / cpu / PostRev",
            "value": 0.000030044,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / PartOpt / cpu / BothRev",
            "value": 0.000029177,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IPartOpt / cpu / PreRev",
            "value": 0.000030052,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IPartOpt / cpu / PostRev",
            "value": 0.000035904,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IPartOpt / cpu / BothRev",
            "value": 0.000029477,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / DefOpt / cpu / PreRev",
            "value": 0.000035658,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / DefOpt / cpu / PostRev",
            "value": 0.000029593000000000003,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / DefOpt / cpu / BothRev",
            "value": 0.000029501,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IDefOpt / cpu / PreRev",
            "value": 0.000029575,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IDefOpt / cpu / PostRev",
            "value": 0.000029549,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IDefOpt / cpu / BothRev",
            "value": 0.000029323,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / Primal",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / Jax / cpu / Primal",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / HLOOpt / cpu / Primal",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / PartOpt / cpu / Primal",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IPartOpt / cpu / Primal",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / DefOpt / cpu / Primal",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IDefOpt / cpu / Primal",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / Forward",
            "value": 0.000017,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / Jax / cpu / Forward",
            "value": 0.000017,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / HLOOpt / cpu / Forward",
            "value": 0.000016,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / PartOpt / cpu / Forward",
            "value": 0.000016,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IPartOpt / cpu / Forward",
            "value": 0.000017,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / DefOpt / cpu / Forward",
            "value": 0.000017,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IDefOpt / cpu / Forward",
            "value": 0.000017,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / PreRev",
            "value": 0.000017,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / PostRev",
            "value": 0.000017999999999999997,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / BothRev",
            "value": 0.000017,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / Jax / cpu / BothRev",
            "value": 0.000017,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / HLOOpt / cpu / PreRev",
            "value": 0.000017999999999999997,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / HLOOpt / cpu / PostRev",
            "value": 0.000017,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / HLOOpt / cpu / BothRev",
            "value": 0.000017999999999999997,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / PartOpt / cpu / PreRev",
            "value": 0.000017999999999999997,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / PartOpt / cpu / PostRev",
            "value": 0.000019,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / PartOpt / cpu / BothRev",
            "value": 0.000017999999999999997,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IPartOpt / cpu / PreRev",
            "value": 0.000017999999999999997,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IPartOpt / cpu / PostRev",
            "value": 0.000019,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IPartOpt / cpu / BothRev",
            "value": 0.000019,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / DefOpt / cpu / PreRev",
            "value": 0.000017,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / DefOpt / cpu / PostRev",
            "value": 0.000017999999999999997,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / DefOpt / cpu / BothRev",
            "value": 0.000017999999999999997,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IDefOpt / cpu / PreRev",
            "value": 0.000017,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IDefOpt / cpu / PostRev",
            "value": 0.000017,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IDefOpt / cpu / BothRev",
            "value": 0.000017,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / JaXPipe / cpu / Primal",
            "value": 0.0009538309996059,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / Jax / cpu / Primal",
            "value": 0.0009380649997183,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / HLOOpt / cpu / Primal",
            "value": 0.0009888797998428,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / PartOpt / cpu / Primal",
            "value": 0.0009224213999914,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IPartOpt / cpu / Primal",
            "value": 0.0009565613998347,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / DefOpt / cpu / Primal",
            "value": 0.001103026599776,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IDefOpt / cpu / Primal",
            "value": 0.0009834524002144,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / JaXPipe / cpu / Forward",
            "value": 0.0022748767996745,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / Jax / cpu / Forward",
            "value": 0.0024321006001628,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / HLOOpt / cpu / Forward",
            "value": 0.0023531127999376,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / PartOpt / cpu / Forward",
            "value": 0.0023807277999367,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IPartOpt / cpu / Forward",
            "value": 0.0022865208000439,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / DefOpt / cpu / Forward",
            "value": 0.0022270897998168,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IDefOpt / cpu / Forward",
            "value": 0.0023226181996506,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / JaXPipe / cpu / PreRev",
            "value": 0.0058146895999016,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / JaXPipe / cpu / PostRev",
            "value": 0.0061294024000744,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / JaXPipe / cpu / BothRev",
            "value": 0.0051035784001214,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / Jax / cpu / BothRev",
            "value": 0.0066634913999223,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / HLOOpt / cpu / PreRev",
            "value": 0.0040726394003286,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / HLOOpt / cpu / PostRev",
            "value": 0.0063714627998706,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / HLOOpt / cpu / BothRev",
            "value": 0.0040900374000557,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / PartOpt / cpu / PreRev",
            "value": 0.006691825800226,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / PartOpt / cpu / PostRev",
            "value": 0.0041910830001143,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / PartOpt / cpu / BothRev",
            "value": 0.0066319975998339,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IPartOpt / cpu / PreRev",
            "value": 0.0040519658001358,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IPartOpt / cpu / PostRev",
            "value": 0.0063012728001922,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IPartOpt / cpu / BothRev",
            "value": 0.004001306200007,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / DefOpt / cpu / PreRev",
            "value": 0.0067273648000991,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / DefOpt / cpu / PostRev",
            "value": 0.0054699779999282,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / DefOpt / cpu / BothRev",
            "value": 0.0062518173999706,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IDefOpt / cpu / PreRev",
            "value": 0.0039920913999594,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IDefOpt / cpu / PostRev",
            "value": 0.0065944061998379,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IDefOpt / cpu / BothRev",
            "value": 0.0056558684002084,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / JaXPipe / cuda / Primal",
            "value": 0.000283678,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / Jax / cuda / Primal",
            "value": 0.000283038,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / HLOOpt / cuda / Primal",
            "value": 0.000288382,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / PartOpt / cuda / Primal",
            "value": 0.000282558,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IPartOpt / cuda / Primal",
            "value": 0.00028227,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / DefOpt / cuda / Primal",
            "value": 0.0002889579999999,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IDefOpt / cuda / Primal",
            "value": 0.000290847,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / JaXPipe / cuda / Forward",
            "value": 0.000558013,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / Jax / cuda / Forward",
            "value": 0.000539613,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / HLOOpt / cuda / Forward",
            "value": 0.000558172,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / PartOpt / cuda / Forward",
            "value": 0.000558749,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IPartOpt / cuda / Forward",
            "value": 0.0005583,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / DefOpt / cuda / Forward",
            "value": 0.000558076,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IDefOpt / cuda / Forward",
            "value": 0.000558013,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / JaXPipe / cuda / PreRev",
            "value": 0.001031161,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / JaXPipe / cuda / PostRev",
            "value": 0.000989338,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / JaXPipe / cuda / BothRev",
            "value": 0.001023546,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / Jax / cuda / BothRev",
            "value": 0.000992731,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / HLOOpt / cuda / PreRev",
            "value": 0.001013017,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / HLOOpt / cuda / PostRev",
            "value": 0.0010415609999999,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / HLOOpt / cuda / BothRev",
            "value": 0.001011866,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / PartOpt / cuda / PreRev",
            "value": 0.001029114,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / PartOpt / cuda / PostRev",
            "value": 0.000979258,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / PartOpt / cuda / BothRev",
            "value": 0.0010298179999999,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IPartOpt / cuda / PreRev",
            "value": 0.001028346,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IPartOpt / cuda / PostRev",
            "value": 0.000979835,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IPartOpt / cuda / BothRev",
            "value": 0.001028506,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / DefOpt / cuda / PreRev",
            "value": 0.001023547,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / DefOpt / cuda / PostRev",
            "value": 0.000963802,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / DefOpt / cuda / BothRev",
            "value": 0.001025946,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IDefOpt / cuda / PreRev",
            "value": 0.001022138,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IDefOpt / cuda / PostRev",
            "value": 0.0010222979999999,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IDefOpt / cuda / BothRev",
            "value": 0.001023866,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / JaXPipe / tpu / Primal",
            "value": 0.00013063925,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / Jax / tpu / Primal",
            "value": 0.00012346375,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / HLOOpt / tpu / Primal",
            "value": 0.00015994975,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / PartOpt / tpu / Primal",
            "value": 0.00013098975,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IPartOpt / tpu / Primal",
            "value": 0.000138358,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / DefOpt / tpu / Primal",
            "value": 0.0001450705,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IDefOpt / tpu / Primal",
            "value": 0.00015797425,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / JaXPipe / tpu / Forward",
            "value": 0.000213455,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / Jax / tpu / Forward",
            "value": 0.0002626545,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / HLOOpt / tpu / Forward",
            "value": 0.0002197919999999,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / PartOpt / tpu / Forward",
            "value": 0.00021483275,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IPartOpt / tpu / Forward",
            "value": 0.00021503975,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / DefOpt / tpu / Forward",
            "value": 0.000217717,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IDefOpt / tpu / Forward",
            "value": 0.00021538825,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / JaXPipe / tpu / PreRev",
            "value": 0.000356415,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / JaXPipe / tpu / PostRev",
            "value": 0.0002559035,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / JaXPipe / tpu / BothRev",
            "value": 0.00035679,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / Jax / tpu / BothRev",
            "value": 0.00025752425,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / HLOOpt / tpu / PreRev",
            "value": 0.0003567515,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / HLOOpt / tpu / PostRev",
            "value": 0.00029084825,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / HLOOpt / tpu / BothRev",
            "value": 0.00035694,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / PartOpt / tpu / PreRev",
            "value": 0.000357185,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / PartOpt / tpu / PostRev",
            "value": 0.00027218775,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / PartOpt / tpu / BothRev",
            "value": 0.00035664175,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IPartOpt / tpu / PreRev",
            "value": 0.00035654875,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IPartOpt / tpu / PostRev",
            "value": 0.0002720394999999,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IPartOpt / tpu / BothRev",
            "value": 0.000356897,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / DefOpt / tpu / PreRev",
            "value": 0.00035885325,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / DefOpt / tpu / PostRev",
            "value": 0.000284019,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / DefOpt / tpu / BothRev",
            "value": 0.0003581695,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IDefOpt / tpu / PreRev",
            "value": 0.0003586965,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IDefOpt / tpu / PostRev",
            "value": 0.00030110125,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IDefOpt / tpu / BothRev",
            "value": 0.0003585125,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / JaXPipe / cpu / Primal",
            "value": 0.003120023,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / Jax / cpu / Primal",
            "value": 0.002832097,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / HLOOpt / cpu / Primal",
            "value": 0.002882768,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / PartOpt / cpu / Primal",
            "value": 0.003440711,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IPartOpt / cpu / Primal",
            "value": 0.003408671,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / DefOpt / cpu / Primal",
            "value": 0.003317103,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IDefOpt / cpu / Primal",
            "value": 0.003215215,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / JaXPipe / cpu / Forward",
            "value": 0.008031018,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / Jax / cpu / Forward",
            "value": 0.007975233,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / HLOOpt / cpu / Forward",
            "value": 0.007722577,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / PartOpt / cpu / Forward",
            "value": 0.008209539,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IPartOpt / cpu / Forward",
            "value": 0.007214417,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / DefOpt / cpu / Forward",
            "value": 0.008387069,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IDefOpt / cpu / Forward",
            "value": 0.008046938,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / JaXPipe / cpu / PreRev",
            "value": 0.010781187,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / JaXPipe / cpu / PostRev",
            "value": 0.012118296,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / JaXPipe / cpu / BothRev",
            "value": 0.010177406,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / Jax / cpu / BothRev",
            "value": 0.0111996059999999,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / HLOOpt / cpu / PreRev",
            "value": 0.011919859,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / HLOOpt / cpu / PostRev",
            "value": 0.012384988,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / HLOOpt / cpu / BothRev",
            "value": 0.010686715,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / PartOpt / cpu / PreRev",
            "value": 0.013061741,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / PartOpt / cpu / PostRev",
            "value": 0.011345522,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / PartOpt / cpu / BothRev",
            "value": 0.010172467,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IPartOpt / cpu / PreRev",
            "value": 0.009272944,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IPartOpt / cpu / PostRev",
            "value": 0.010774298,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IPartOpt / cpu / BothRev",
            "value": 0.010175172,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / DefOpt / cpu / PreRev",
            "value": 0.010514986,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / DefOpt / cpu / PostRev",
            "value": 0.008810806,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / DefOpt / cpu / BothRev",
            "value": 0.010383857,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IDefOpt / cpu / PreRev",
            "value": 0.010978527,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IDefOpt / cpu / PostRev",
            "value": 0.010160987,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IDefOpt / cpu / BothRev",
            "value": 0.009648783,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / JaXPipe / cpu / Primal",
            "value": 0.001649,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / Jax / cpu / Primal",
            "value": 0.001697,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / HLOOpt / cpu / Primal",
            "value": 0.001844,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / PartOpt / cpu / Primal",
            "value": 0.001582,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IPartOpt / cpu / Primal",
            "value": 0.001544,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / DefOpt / cpu / Primal",
            "value": 0.001453,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IDefOpt / cpu / Primal",
            "value": 0.001519,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / JaXPipe / cpu / Forward",
            "value": 0.003862,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / Jax / cpu / Forward",
            "value": 0.003967,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / HLOOpt / cpu / Forward",
            "value": 0.00387,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / PartOpt / cpu / Forward",
            "value": 0.003782,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IPartOpt / cpu / Forward",
            "value": 0.003837,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / DefOpt / cpu / Forward",
            "value": 0.00467,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IDefOpt / cpu / Forward",
            "value": 0.004787,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / JaXPipe / cpu / PreRev",
            "value": 0.007793,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / JaXPipe / cpu / PostRev",
            "value": 0.008694,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / JaXPipe / cpu / BothRev",
            "value": 0.007356,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / Jax / cpu / BothRev",
            "value": 0.0100139999999999,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / HLOOpt / cpu / PreRev",
            "value": 0.007667,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / HLOOpt / cpu / PostRev",
            "value": 0.007291,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / HLOOpt / cpu / BothRev",
            "value": 0.007515,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / PartOpt / cpu / PreRev",
            "value": 0.008232,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / PartOpt / cpu / PostRev",
            "value": 0.008901,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / PartOpt / cpu / BothRev",
            "value": 0.007105,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IPartOpt / cpu / PreRev",
            "value": 0.007813,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IPartOpt / cpu / PostRev",
            "value": 0.008791,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IPartOpt / cpu / BothRev",
            "value": 0.008664,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / DefOpt / cpu / PreRev",
            "value": 0.008201,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / DefOpt / cpu / PostRev",
            "value": 0.007009,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / DefOpt / cpu / BothRev",
            "value": 0.007854,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IDefOpt / cpu / PreRev",
            "value": 0.007174,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IDefOpt / cpu / PostRev",
            "value": 0.007492,
            "unit": "s"
          },
          {
            "name": "llama_dim_288_hidden_dim_768_n_layers_6_n_heads_6_n_kv_heads_6_vocab_size_32000_seq_len_256 / IDefOpt / cpu / BothRev",
            "value": 0.007207,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000007786020014464156,
            "unit": "s"
          },
          {
            "name": "scatter_sum / Jax / cpu / Primal",
            "value": 0.000008019060041988268,
            "unit": "s"
          },
          {
            "name": "scatter_sum / HLOOpt / cpu / Primal",
            "value": 0.000007764780038996832,
            "unit": "s"
          },
          {
            "name": "scatter_sum / PartOpt / cpu / Primal",
            "value": 0.000007656579982722179,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IPartOpt / cpu / Primal",
            "value": 0.000007552019997092429,
            "unit": "s"
          },
          {
            "name": "scatter_sum / DefOpt / cpu / Primal",
            "value": 0.000007893580059317174,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IDefOpt / cpu / Primal",
            "value": 0.0000073182600681320764,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Forward",
            "value": 0.000011450600068201313,
            "unit": "s"
          },
          {
            "name": "scatter_sum / Jax / cpu / Forward",
            "value": 0.000011920960041607031,
            "unit": "s"
          },
          {
            "name": "scatter_sum / HLOOpt / cpu / Forward",
            "value": 0.000011608240038185614,
            "unit": "s"
          },
          {
            "name": "scatter_sum / PartOpt / cpu / Forward",
            "value": 0.000011235580004722578,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IPartOpt / cpu / Forward",
            "value": 0.000012340579996816812,
            "unit": "s"
          },
          {
            "name": "scatter_sum / DefOpt / cpu / Forward",
            "value": 0.000011781939974753186,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IDefOpt / cpu / Forward",
            "value": 0.00001228377996085328,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / PreRev",
            "value": 0.000012252239994268166,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / PostRev",
            "value": 0.000012191460064059356,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / BothRev",
            "value": 0.000012277679998078383,
            "unit": "s"
          },
          {
            "name": "scatter_sum / Jax / cpu / BothRev",
            "value": 0.000012188679975224658,
            "unit": "s"
          },
          {
            "name": "scatter_sum / HLOOpt / cpu / PreRev",
            "value": 0.000011734399977285648,
            "unit": "s"
          },
          {
            "name": "scatter_sum / HLOOpt / cpu / PostRev",
            "value": 0.000016643799881421728,
            "unit": "s"
          },
          {
            "name": "scatter_sum / HLOOpt / cpu / BothRev",
            "value": 0.00001227616005053278,
            "unit": "s"
          },
          {
            "name": "scatter_sum / PartOpt / cpu / PreRev",
            "value": 0.0000118890200064925,
            "unit": "s"
          },
          {
            "name": "scatter_sum / PartOpt / cpu / PostRev",
            "value": 0.000012277980058570392,
            "unit": "s"
          },
          {
            "name": "scatter_sum / PartOpt / cpu / BothRev",
            "value": 0.000012764099956257267,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IPartOpt / cpu / PreRev",
            "value": 0.000011811840031441536,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IPartOpt / cpu / PostRev",
            "value": 0.000011865939959534445,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IPartOpt / cpu / BothRev",
            "value": 0.000012223879948578544,
            "unit": "s"
          },
          {
            "name": "scatter_sum / DefOpt / cpu / PreRev",
            "value": 0.000011357539970049402,
            "unit": "s"
          },
          {
            "name": "scatter_sum / DefOpt / cpu / PostRev",
            "value": 0.000012695660097961082,
            "unit": "s"
          },
          {
            "name": "scatter_sum / DefOpt / cpu / BothRev",
            "value": 0.000012270000042917671,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IDefOpt / cpu / PreRev",
            "value": 0.00001211516000694246,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IDefOpt / cpu / PostRev",
            "value": 0.000011602959948504576,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IDefOpt / cpu / BothRev",
            "value": 0.00001242769994860282,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cuda / Primal",
            "value": 0.000010048,
            "unit": "s"
          },
          {
            "name": "scatter_sum / Jax / cuda / Primal",
            "value": 0.00001008,
            "unit": "s"
          },
          {
            "name": "scatter_sum / HLOOpt / cuda / Primal",
            "value": 0.000010593,
            "unit": "s"
          },
          {
            "name": "scatter_sum / PartOpt / cuda / Primal",
            "value": 0.00001008,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IPartOpt / cuda / Primal",
            "value": 0.000009504,
            "unit": "s"
          },
          {
            "name": "scatter_sum / DefOpt / cuda / Primal",
            "value": 0.000010367,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IDefOpt / cuda / Primal",
            "value": 0.00001072,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cuda / Forward",
            "value": 0.000017183,
            "unit": "s"
          },
          {
            "name": "scatter_sum / Jax / cuda / Forward",
            "value": 0.000017056,
            "unit": "s"
          },
          {
            "name": "scatter_sum / HLOOpt / cuda / Forward",
            "value": 0.000016704,
            "unit": "s"
          },
          {
            "name": "scatter_sum / PartOpt / cuda / Forward",
            "value": 0.000017760000000000003,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IPartOpt / cuda / Forward",
            "value": 0.000017568000000000002,
            "unit": "s"
          },
          {
            "name": "scatter_sum / DefOpt / cuda / Forward",
            "value": 0.000017728,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IDefOpt / cuda / Forward",
            "value": 0.000017823,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cuda / PreRev",
            "value": 0.000017312,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cuda / PostRev",
            "value": 0.000017152,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cuda / BothRev",
            "value": 0.000017599,
            "unit": "s"
          },
          {
            "name": "scatter_sum / Jax / cuda / BothRev",
            "value": 0.000017665,
            "unit": "s"
          },
          {
            "name": "scatter_sum / HLOOpt / cuda / PreRev",
            "value": 0.000017503999999999997,
            "unit": "s"
          },
          {
            "name": "scatter_sum / HLOOpt / cuda / PostRev",
            "value": 0.000017024,
            "unit": "s"
          },
          {
            "name": "scatter_sum / HLOOpt / cuda / BothRev",
            "value": 0.000017216,
            "unit": "s"
          },
          {
            "name": "scatter_sum / PartOpt / cuda / PreRev",
            "value": 0.000018176,
            "unit": "s"
          },
          {
            "name": "scatter_sum / PartOpt / cuda / PostRev",
            "value": 0.000017184,
            "unit": "s"
          },
          {
            "name": "scatter_sum / PartOpt / cuda / BothRev",
            "value": 0.000016832,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IPartOpt / cuda / PreRev",
            "value": 0.000017664,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IPartOpt / cuda / PostRev",
            "value": 0.000017631,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IPartOpt / cuda / BothRev",
            "value": 0.0000176,
            "unit": "s"
          },
          {
            "name": "scatter_sum / DefOpt / cuda / PreRev",
            "value": 0.000017568000000000002,
            "unit": "s"
          },
          {
            "name": "scatter_sum / DefOpt / cuda / PostRev",
            "value": 0.000016736,
            "unit": "s"
          },
          {
            "name": "scatter_sum / DefOpt / cuda / BothRev",
            "value": 0.000017503,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IDefOpt / cuda / PreRev",
            "value": 0.0000168,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IDefOpt / cuda / PostRev",
            "value": 0.00001728,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IDefOpt / cuda / BothRev",
            "value": 0.000017472,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.000001342875,
            "unit": "s"
          },
          {
            "name": "scatter_sum / Jax / tpu / Primal",
            "value": 0.000001404475,
            "unit": "s"
          },
          {
            "name": "scatter_sum / HLOOpt / tpu / Primal",
            "value": 0.0000013435,
            "unit": "s"
          },
          {
            "name": "scatter_sum / PartOpt / tpu / Primal",
            "value": 0.000001404775,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IPartOpt / tpu / Primal",
            "value": 0.000001344025,
            "unit": "s"
          },
          {
            "name": "scatter_sum / DefOpt / tpu / Primal",
            "value": 0.000001404825,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IDefOpt / tpu / Primal",
            "value": 0.0000013437,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Forward",
            "value": 0.0000027076,
            "unit": "s"
          },
          {
            "name": "scatter_sum / Jax / tpu / Forward",
            "value": 0.0000027160250000000004,
            "unit": "s"
          },
          {
            "name": "scatter_sum / HLOOpt / tpu / Forward",
            "value": 0.000002705975,
            "unit": "s"
          },
          {
            "name": "scatter_sum / PartOpt / tpu / Forward",
            "value": 0.0000027054,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IPartOpt / tpu / Forward",
            "value": 0.00000270165,
            "unit": "s"
          },
          {
            "name": "scatter_sum / DefOpt / tpu / Forward",
            "value": 0.000002688925,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IDefOpt / tpu / Forward",
            "value": 0.000002708225,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / PreRev",
            "value": 0.00000268285,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / PostRev",
            "value": 0.000002687575,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / BothRev",
            "value": 0.000002698,
            "unit": "s"
          },
          {
            "name": "scatter_sum / Jax / tpu / BothRev",
            "value": 0.000002740975,
            "unit": "s"
          },
          {
            "name": "scatter_sum / HLOOpt / tpu / PreRev",
            "value": 0.0000026978,
            "unit": "s"
          },
          {
            "name": "scatter_sum / HLOOpt / tpu / PostRev",
            "value": 0.000002741975,
            "unit": "s"
          },
          {
            "name": "scatter_sum / HLOOpt / tpu / BothRev",
            "value": 0.0000026978,
            "unit": "s"
          },
          {
            "name": "scatter_sum / PartOpt / tpu / PreRev",
            "value": 0.0000027418,
            "unit": "s"
          },
          {
            "name": "scatter_sum / PartOpt / tpu / PostRev",
            "value": 0.00000269595,
            "unit": "s"
          },
          {
            "name": "scatter_sum / PartOpt / tpu / BothRev",
            "value": 0.000002739825,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IPartOpt / tpu / PreRev",
            "value": 0.0000026997,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IPartOpt / tpu / PostRev",
            "value": 0.000002742925,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IPartOpt / tpu / BothRev",
            "value": 0.0000026987,
            "unit": "s"
          },
          {
            "name": "scatter_sum / DefOpt / tpu / PreRev",
            "value": 0.0000027469,
            "unit": "s"
          },
          {
            "name": "scatter_sum / DefOpt / tpu / PostRev",
            "value": 0.000002696425,
            "unit": "s"
          },
          {
            "name": "scatter_sum / DefOpt / tpu / BothRev",
            "value": 0.00000275125,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IDefOpt / tpu / PreRev",
            "value": 0.0000026971,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IDefOpt / tpu / PostRev",
            "value": 0.000002749875,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IDefOpt / tpu / BothRev",
            "value": 0.000002698775,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000020119,
            "unit": "s"
          },
          {
            "name": "scatter_sum / Jax / cpu / Primal",
            "value": 0.000019997,
            "unit": "s"
          },
          {
            "name": "scatter_sum / HLOOpt / cpu / Primal",
            "value": 0.000019837,
            "unit": "s"
          },
          {
            "name": "scatter_sum / PartOpt / cpu / Primal",
            "value": 0.000019921,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IPartOpt / cpu / Primal",
            "value": 0.000019912,
            "unit": "s"
          },
          {
            "name": "scatter_sum / DefOpt / cpu / Primal",
            "value": 0.000019687,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IDefOpt / cpu / Primal",
            "value": 0.000019645,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Forward",
            "value": 0.000028574,
            "unit": "s"
          },
          {
            "name": "scatter_sum / Jax / cpu / Forward",
            "value": 0.000030466,
            "unit": "s"
          },
          {
            "name": "scatter_sum / HLOOpt / cpu / Forward",
            "value": 0.0000274,
            "unit": "s"
          },
          {
            "name": "scatter_sum / PartOpt / cpu / Forward",
            "value": 0.000027884,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IPartOpt / cpu / Forward",
            "value": 0.000028792,
            "unit": "s"
          },
          {
            "name": "scatter_sum / DefOpt / cpu / Forward",
            "value": 0.000028655,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IDefOpt / cpu / Forward",
            "value": 0.000028636,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / PreRev",
            "value": 0.000028482,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / PostRev",
            "value": 0.000028047,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / BothRev",
            "value": 0.000028532,
            "unit": "s"
          },
          {
            "name": "scatter_sum / Jax / cpu / BothRev",
            "value": 0.000028225,
            "unit": "s"
          },
          {
            "name": "scatter_sum / HLOOpt / cpu / PreRev",
            "value": 0.000028556,
            "unit": "s"
          },
          {
            "name": "scatter_sum / HLOOpt / cpu / PostRev",
            "value": 0.000028025,
            "unit": "s"
          },
          {
            "name": "scatter_sum / HLOOpt / cpu / BothRev",
            "value": 0.000027548,
            "unit": "s"
          },
          {
            "name": "scatter_sum / PartOpt / cpu / PreRev",
            "value": 0.000028628,
            "unit": "s"
          },
          {
            "name": "scatter_sum / PartOpt / cpu / PostRev",
            "value": 0.000027909,
            "unit": "s"
          },
          {
            "name": "scatter_sum / PartOpt / cpu / BothRev",
            "value": 0.000027694,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IPartOpt / cpu / PreRev",
            "value": 0.000028635,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IPartOpt / cpu / PostRev",
            "value": 0.000027697,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IPartOpt / cpu / BothRev",
            "value": 0.000027874,
            "unit": "s"
          },
          {
            "name": "scatter_sum / DefOpt / cpu / PreRev",
            "value": 0.000028394,
            "unit": "s"
          },
          {
            "name": "scatter_sum / DefOpt / cpu / PostRev",
            "value": 0.000029059,
            "unit": "s"
          },
          {
            "name": "scatter_sum / DefOpt / cpu / BothRev",
            "value": 0.000028105000000000003,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IDefOpt / cpu / PreRev",
            "value": 0.000027797,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IDefOpt / cpu / PostRev",
            "value": 0.00002833,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IDefOpt / cpu / BothRev",
            "value": 0.000028109,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.00001,
            "unit": "s"
          },
          {
            "name": "scatter_sum / Jax / cpu / Primal",
            "value": 0.00001,
            "unit": "s"
          },
          {
            "name": "scatter_sum / HLOOpt / cpu / Primal",
            "value": 0.00001,
            "unit": "s"
          },
          {
            "name": "scatter_sum / PartOpt / cpu / Primal",
            "value": 0.00001,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IPartOpt / cpu / Primal",
            "value": 0.00001,
            "unit": "s"
          },
          {
            "name": "scatter_sum / DefOpt / cpu / Primal",
            "value": 0.000008999999999999999,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IDefOpt / cpu / Primal",
            "value": 0.00001,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Forward",
            "value": 0.000015,
            "unit": "s"
          },
          {
            "name": "scatter_sum / Jax / cpu / Forward",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "scatter_sum / HLOOpt / cpu / Forward",
            "value": 0.000015,
            "unit": "s"
          },
          {
            "name": "scatter_sum / PartOpt / cpu / Forward",
            "value": 0.000015,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IPartOpt / cpu / Forward",
            "value": 0.000015,
            "unit": "s"
          },
          {
            "name": "scatter_sum / DefOpt / cpu / Forward",
            "value": 0.000015,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IDefOpt / cpu / Forward",
            "value": 0.000016,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / PreRev",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / PostRev",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / BothRev",
            "value": 0.000015,
            "unit": "s"
          },
          {
            "name": "scatter_sum / Jax / cpu / BothRev",
            "value": 0.000015,
            "unit": "s"
          },
          {
            "name": "scatter_sum / HLOOpt / cpu / PreRev",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "scatter_sum / HLOOpt / cpu / PostRev",
            "value": 0.000017,
            "unit": "s"
          },
          {
            "name": "scatter_sum / HLOOpt / cpu / BothRev",
            "value": 0.000016,
            "unit": "s"
          },
          {
            "name": "scatter_sum / PartOpt / cpu / PreRev",
            "value": 0.000015,
            "unit": "s"
          },
          {
            "name": "scatter_sum / PartOpt / cpu / PostRev",
            "value": 0.000015,
            "unit": "s"
          },
          {
            "name": "scatter_sum / PartOpt / cpu / BothRev",
            "value": 0.000015,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IPartOpt / cpu / PreRev",
            "value": 0.000015,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IPartOpt / cpu / PostRev",
            "value": 0.000015,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IPartOpt / cpu / BothRev",
            "value": 0.000015,
            "unit": "s"
          },
          {
            "name": "scatter_sum / DefOpt / cpu / PreRev",
            "value": 0.000016,
            "unit": "s"
          },
          {
            "name": "scatter_sum / DefOpt / cpu / PostRev",
            "value": 0.000015,
            "unit": "s"
          },
          {
            "name": "scatter_sum / DefOpt / cpu / BothRev",
            "value": 0.000016,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IDefOpt / cpu / PreRev",
            "value": 0.000015,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IDefOpt / cpu / PostRev",
            "value": 0.000016,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IDefOpt / cpu / BothRev",
            "value": 0.000016,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / Primal",
            "value": 0.000006088259942771401,
            "unit": "s"
          },
          {
            "name": "slicing / Jax / cpu / Primal",
            "value": 0.000006522379990201443,
            "unit": "s"
          },
          {
            "name": "slicing / HLOOpt / cpu / Primal",
            "value": 0.000007241459934448358,
            "unit": "s"
          },
          {
            "name": "slicing / PartOpt / cpu / Primal",
            "value": 0.000006115020078141242,
            "unit": "s"
          },
          {
            "name": "slicing / IPartOpt / cpu / Primal",
            "value": 0.000006307279982138425,
            "unit": "s"
          },
          {
            "name": "slicing / DefOpt / cpu / Primal",
            "value": 0.000006012780031596776,
            "unit": "s"
          },
          {
            "name": "slicing / IDefOpt / cpu / Primal",
            "value": 0.000006586679883184843,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / Forward",
            "value": 0.00000970965998931206,
            "unit": "s"
          },
          {
            "name": "slicing / Jax / cpu / Forward",
            "value": 0.00000915137998163118,
            "unit": "s"
          },
          {
            "name": "slicing / HLOOpt / cpu / Forward",
            "value": 0.000009355379952467046,
            "unit": "s"
          },
          {
            "name": "slicing / PartOpt / cpu / Forward",
            "value": 0.000009572479975759053,
            "unit": "s"
          },
          {
            "name": "slicing / IPartOpt / cpu / Forward",
            "value": 0.000009886900043056813,
            "unit": "s"
          },
          {
            "name": "slicing / DefOpt / cpu / Forward",
            "value": 0.000009576380089129088,
            "unit": "s"
          },
          {
            "name": "slicing / IDefOpt / cpu / Forward",
            "value": 0.000009609159897081551,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / PreRev",
            "value": 0.00000989815998764243,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / PostRev",
            "value": 0.000010154540068469942,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / BothRev",
            "value": 0.000010082879962283186,
            "unit": "s"
          },
          {
            "name": "slicing / Jax / cpu / BothRev",
            "value": 0.000010159179910260718,
            "unit": "s"
          },
          {
            "name": "slicing / HLOOpt / cpu / PreRev",
            "value": 0.000010423580024507828,
            "unit": "s"
          },
          {
            "name": "slicing / HLOOpt / cpu / PostRev",
            "value": 0.000011878460063599053,
            "unit": "s"
          },
          {
            "name": "slicing / HLOOpt / cpu / BothRev",
            "value": 0.000010385000077803852,
            "unit": "s"
          },
          {
            "name": "slicing / PartOpt / cpu / PreRev",
            "value": 0.000009761620058270637,
            "unit": "s"
          },
          {
            "name": "slicing / PartOpt / cpu / PostRev",
            "value": 0.000010182080004597082,
            "unit": "s"
          },
          {
            "name": "slicing / PartOpt / cpu / BothRev",
            "value": 0.00001066470000296249,
            "unit": "s"
          },
          {
            "name": "slicing / IPartOpt / cpu / PreRev",
            "value": 0.000010389800008852036,
            "unit": "s"
          },
          {
            "name": "slicing / IPartOpt / cpu / PostRev",
            "value": 0.000010179799974139314,
            "unit": "s"
          },
          {
            "name": "slicing / IPartOpt / cpu / BothRev",
            "value": 0.000009737180007505233,
            "unit": "s"
          },
          {
            "name": "slicing / DefOpt / cpu / PreRev",
            "value": 0.000009983579930121778,
            "unit": "s"
          },
          {
            "name": "slicing / DefOpt / cpu / PostRev",
            "value": 0.00001058722007655888,
            "unit": "s"
          },
          {
            "name": "slicing / DefOpt / cpu / BothRev",
            "value": 0.00000982160014245892,
            "unit": "s"
          },
          {
            "name": "slicing / IDefOpt / cpu / PreRev",
            "value": 0.0000101736399847141,
            "unit": "s"
          },
          {
            "name": "slicing / IDefOpt / cpu / PostRev",
            "value": 0.00000970647990470752,
            "unit": "s"
          },
          {
            "name": "slicing / IDefOpt / cpu / BothRev",
            "value": 0.000010128079884452743,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cuda / Primal",
            "value": 0.000001888,
            "unit": "s"
          },
          {
            "name": "slicing / Jax / cuda / Primal",
            "value": 0.000001887,
            "unit": "s"
          },
          {
            "name": "slicing / HLOOpt / cuda / Primal",
            "value": 0.000001887,
            "unit": "s"
          },
          {
            "name": "slicing / PartOpt / cuda / Primal",
            "value": 0.000001887,
            "unit": "s"
          },
          {
            "name": "slicing / IPartOpt / cuda / Primal",
            "value": 0.000001888,
            "unit": "s"
          },
          {
            "name": "slicing / DefOpt / cuda / Primal",
            "value": 0.000001887,
            "unit": "s"
          },
          {
            "name": "slicing / IDefOpt / cuda / Primal",
            "value": 0.000001887,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cuda / Forward",
            "value": 0.000010272,
            "unit": "s"
          },
          {
            "name": "slicing / Jax / cuda / Forward",
            "value": 0.0000112,
            "unit": "s"
          },
          {
            "name": "slicing / HLOOpt / cuda / Forward",
            "value": 0.000011455999999999998,
            "unit": "s"
          },
          {
            "name": "slicing / PartOpt / cuda / Forward",
            "value": 0.000011455999999999998,
            "unit": "s"
          },
          {
            "name": "slicing / IPartOpt / cuda / Forward",
            "value": 0.000011233,
            "unit": "s"
          },
          {
            "name": "slicing / DefOpt / cuda / Forward",
            "value": 0.00000992,
            "unit": "s"
          },
          {
            "name": "slicing / IDefOpt / cuda / Forward",
            "value": 0.000010304,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cuda / PreRev",
            "value": 0.000010336,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cuda / PostRev",
            "value": 0.000009696,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cuda / BothRev",
            "value": 0.000009984,
            "unit": "s"
          },
          {
            "name": "slicing / Jax / cuda / BothRev",
            "value": 0.000010016,
            "unit": "s"
          },
          {
            "name": "slicing / HLOOpt / cuda / PreRev",
            "value": 0.000010208,
            "unit": "s"
          },
          {
            "name": "slicing / HLOOpt / cuda / PostRev",
            "value": 0.000009759,
            "unit": "s"
          },
          {
            "name": "slicing / HLOOpt / cuda / BothRev",
            "value": 0.000009887,
            "unit": "s"
          },
          {
            "name": "slicing / PartOpt / cuda / PreRev",
            "value": 0.00001008,
            "unit": "s"
          },
          {
            "name": "slicing / PartOpt / cuda / PostRev",
            "value": 0.000010016,
            "unit": "s"
          },
          {
            "name": "slicing / PartOpt / cuda / BothRev",
            "value": 0.00001024,
            "unit": "s"
          },
          {
            "name": "slicing / IPartOpt / cuda / PreRev",
            "value": 0.00000992,
            "unit": "s"
          },
          {
            "name": "slicing / IPartOpt / cuda / PostRev",
            "value": 0.000010144,
            "unit": "s"
          },
          {
            "name": "slicing / IPartOpt / cuda / BothRev",
            "value": 0.000010208,
            "unit": "s"
          },
          {
            "name": "slicing / DefOpt / cuda / PreRev",
            "value": 0.000010048,
            "unit": "s"
          },
          {
            "name": "slicing / DefOpt / cuda / PostRev",
            "value": 0.000010111,
            "unit": "s"
          },
          {
            "name": "slicing / DefOpt / cuda / BothRev",
            "value": 0.000010336,
            "unit": "s"
          },
          {
            "name": "slicing / IDefOpt / cuda / PreRev",
            "value": 0.000010209,
            "unit": "s"
          },
          {
            "name": "slicing / IDefOpt / cuda / PostRev",
            "value": 0.000010208,
            "unit": "s"
          },
          {
            "name": "slicing / IDefOpt / cuda / BothRev",
            "value": 0.000010304,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / tpu / Primal",
            "value": 0.000001021875,
            "unit": "s"
          },
          {
            "name": "slicing / Jax / tpu / Primal",
            "value": 9.68e-7,
            "unit": "s"
          },
          {
            "name": "slicing / HLOOpt / tpu / Primal",
            "value": 0.000001023225,
            "unit": "s"
          },
          {
            "name": "slicing / PartOpt / tpu / Primal",
            "value": 9.72925e-7,
            "unit": "s"
          },
          {
            "name": "slicing / IPartOpt / tpu / Primal",
            "value": 0.000001026275,
            "unit": "s"
          },
          {
            "name": "slicing / DefOpt / tpu / Primal",
            "value": 9.75725e-7,
            "unit": "s"
          },
          {
            "name": "slicing / IDefOpt / tpu / Primal",
            "value": 0.000001025075,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / tpu / Forward",
            "value": 0.000001409775,
            "unit": "s"
          },
          {
            "name": "slicing / Jax / tpu / Forward",
            "value": 0.000001474225,
            "unit": "s"
          },
          {
            "name": "slicing / HLOOpt / tpu / Forward",
            "value": 0.000001521625,
            "unit": "s"
          },
          {
            "name": "slicing / PartOpt / tpu / Forward",
            "value": 0.000001504825,
            "unit": "s"
          },
          {
            "name": "slicing / IPartOpt / tpu / Forward",
            "value": 0.000001520125,
            "unit": "s"
          },
          {
            "name": "slicing / DefOpt / tpu / Forward",
            "value": 0.0000014960499999999998,
            "unit": "s"
          },
          {
            "name": "slicing / IDefOpt / tpu / Forward",
            "value": 0.000001519125,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / tpu / PreRev",
            "value": 0.00000256195,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / tpu / PostRev",
            "value": 0.000002533775,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / tpu / BothRev",
            "value": 0.00000258515,
            "unit": "s"
          },
          {
            "name": "slicing / Jax / tpu / BothRev",
            "value": 0.000002532025,
            "unit": "s"
          },
          {
            "name": "slicing / HLOOpt / tpu / PreRev",
            "value": 0.000002587525,
            "unit": "s"
          },
          {
            "name": "slicing / HLOOpt / tpu / PostRev",
            "value": 0.0000025343,
            "unit": "s"
          },
          {
            "name": "slicing / HLOOpt / tpu / BothRev",
            "value": 0.0000025905,
            "unit": "s"
          },
          {
            "name": "slicing / PartOpt / tpu / PreRev",
            "value": 0.000002537225,
            "unit": "s"
          },
          {
            "name": "slicing / PartOpt / tpu / PostRev",
            "value": 0.0000025744000000000003,
            "unit": "s"
          },
          {
            "name": "slicing / PartOpt / tpu / BothRev",
            "value": 0.000002543975,
            "unit": "s"
          },
          {
            "name": "slicing / IPartOpt / tpu / PreRev",
            "value": 0.000002574375,
            "unit": "s"
          },
          {
            "name": "slicing / IPartOpt / tpu / PostRev",
            "value": 0.000002550375,
            "unit": "s"
          },
          {
            "name": "slicing / IPartOpt / tpu / BothRev",
            "value": 0.000002591325,
            "unit": "s"
          },
          {
            "name": "slicing / DefOpt / tpu / PreRev",
            "value": 0.00000255185,
            "unit": "s"
          },
          {
            "name": "slicing / DefOpt / tpu / PostRev",
            "value": 0.000002589575,
            "unit": "s"
          },
          {
            "name": "slicing / DefOpt / tpu / BothRev",
            "value": 0.0000025526250000000004,
            "unit": "s"
          },
          {
            "name": "slicing / IDefOpt / tpu / PreRev",
            "value": 0.00000257815,
            "unit": "s"
          },
          {
            "name": "slicing / IDefOpt / tpu / PostRev",
            "value": 0.000002538975,
            "unit": "s"
          },
          {
            "name": "slicing / IDefOpt / tpu / BothRev",
            "value": 0.000002591925,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / Primal",
            "value": 0.000016379,
            "unit": "s"
          },
          {
            "name": "slicing / Jax / cpu / Primal",
            "value": 0.000015518999999999998,
            "unit": "s"
          },
          {
            "name": "slicing / HLOOpt / cpu / Primal",
            "value": 0.000015899,
            "unit": "s"
          },
          {
            "name": "slicing / PartOpt / cpu / Primal",
            "value": 0.000015885,
            "unit": "s"
          },
          {
            "name": "slicing / IPartOpt / cpu / Primal",
            "value": 0.00001575,
            "unit": "s"
          },
          {
            "name": "slicing / DefOpt / cpu / Primal",
            "value": 0.000016094000000000002,
            "unit": "s"
          },
          {
            "name": "slicing / IDefOpt / cpu / Primal",
            "value": 0.000015833,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / Forward",
            "value": 0.00002112,
            "unit": "s"
          },
          {
            "name": "slicing / Jax / cpu / Forward",
            "value": 0.000020752,
            "unit": "s"
          },
          {
            "name": "slicing / HLOOpt / cpu / Forward",
            "value": 0.000021007,
            "unit": "s"
          },
          {
            "name": "slicing / PartOpt / cpu / Forward",
            "value": 0.000020947,
            "unit": "s"
          },
          {
            "name": "slicing / IPartOpt / cpu / Forward",
            "value": 0.000021092,
            "unit": "s"
          },
          {
            "name": "slicing / DefOpt / cpu / Forward",
            "value": 0.000020843,
            "unit": "s"
          },
          {
            "name": "slicing / IDefOpt / cpu / Forward",
            "value": 0.00002099,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / PreRev",
            "value": 0.000021959,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / PostRev",
            "value": 0.000021412,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / BothRev",
            "value": 0.000021662,
            "unit": "s"
          },
          {
            "name": "slicing / Jax / cpu / BothRev",
            "value": 0.000021498,
            "unit": "s"
          },
          {
            "name": "slicing / HLOOpt / cpu / PreRev",
            "value": 0.000021531,
            "unit": "s"
          },
          {
            "name": "slicing / HLOOpt / cpu / PostRev",
            "value": 0.000022129,
            "unit": "s"
          },
          {
            "name": "slicing / HLOOpt / cpu / BothRev",
            "value": 0.000021576,
            "unit": "s"
          },
          {
            "name": "slicing / PartOpt / cpu / PreRev",
            "value": 0.000022195,
            "unit": "s"
          },
          {
            "name": "slicing / PartOpt / cpu / PostRev",
            "value": 0.00002168,
            "unit": "s"
          },
          {
            "name": "slicing / PartOpt / cpu / BothRev",
            "value": 0.000021367,
            "unit": "s"
          },
          {
            "name": "slicing / IPartOpt / cpu / PreRev",
            "value": 0.000021985,
            "unit": "s"
          },
          {
            "name": "slicing / IPartOpt / cpu / PostRev",
            "value": 0.000021601,
            "unit": "s"
          },
          {
            "name": "slicing / IPartOpt / cpu / BothRev",
            "value": 0.000021379,
            "unit": "s"
          },
          {
            "name": "slicing / DefOpt / cpu / PreRev",
            "value": 0.000022131,
            "unit": "s"
          },
          {
            "name": "slicing / DefOpt / cpu / PostRev",
            "value": 0.000021487,
            "unit": "s"
          },
          {
            "name": "slicing / DefOpt / cpu / BothRev",
            "value": 0.000021423,
            "unit": "s"
          },
          {
            "name": "slicing / IDefOpt / cpu / PreRev",
            "value": 0.000021897,
            "unit": "s"
          },
          {
            "name": "slicing / IDefOpt / cpu / PostRev",
            "value": 0.000021712,
            "unit": "s"
          },
          {
            "name": "slicing / IDefOpt / cpu / BothRev",
            "value": 0.000022272,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / Primal",
            "value": 0.000008,
            "unit": "s"
          },
          {
            "name": "slicing / Jax / cpu / Primal",
            "value": 0.000008,
            "unit": "s"
          },
          {
            "name": "slicing / HLOOpt / cpu / Primal",
            "value": 0.000008,
            "unit": "s"
          },
          {
            "name": "slicing / PartOpt / cpu / Primal",
            "value": 0.000008,
            "unit": "s"
          },
          {
            "name": "slicing / IPartOpt / cpu / Primal",
            "value": 0.000008,
            "unit": "s"
          },
          {
            "name": "slicing / DefOpt / cpu / Primal",
            "value": 0.000008,
            "unit": "s"
          },
          {
            "name": "slicing / IDefOpt / cpu / Primal",
            "value": 0.000008,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / Forward",
            "value": 0.000011,
            "unit": "s"
          },
          {
            "name": "slicing / Jax / cpu / Forward",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "slicing / HLOOpt / cpu / Forward",
            "value": 0.000011,
            "unit": "s"
          },
          {
            "name": "slicing / PartOpt / cpu / Forward",
            "value": 0.000011,
            "unit": "s"
          },
          {
            "name": "slicing / IPartOpt / cpu / Forward",
            "value": 0.000011,
            "unit": "s"
          },
          {
            "name": "slicing / DefOpt / cpu / Forward",
            "value": 0.000011,
            "unit": "s"
          },
          {
            "name": "slicing / IDefOpt / cpu / Forward",
            "value": 0.000011,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / PreRev",
            "value": 0.000011,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / PostRev",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / BothRev",
            "value": 0.000011,
            "unit": "s"
          },
          {
            "name": "slicing / Jax / cpu / BothRev",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "slicing / HLOOpt / cpu / PreRev",
            "value": 0.000011,
            "unit": "s"
          },
          {
            "name": "slicing / HLOOpt / cpu / PostRev",
            "value": 0.000011,
            "unit": "s"
          },
          {
            "name": "slicing / HLOOpt / cpu / BothRev",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "slicing / PartOpt / cpu / PreRev",
            "value": 0.000011,
            "unit": "s"
          },
          {
            "name": "slicing / PartOpt / cpu / PostRev",
            "value": 0.000011,
            "unit": "s"
          },
          {
            "name": "slicing / PartOpt / cpu / BothRev",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "slicing / IPartOpt / cpu / PreRev",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "slicing / IPartOpt / cpu / PostRev",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "slicing / IPartOpt / cpu / BothRev",
            "value": 0.000011,
            "unit": "s"
          },
          {
            "name": "slicing / DefOpt / cpu / PreRev",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "slicing / DefOpt / cpu / PostRev",
            "value": 0.000011,
            "unit": "s"
          },
          {
            "name": "slicing / DefOpt / cpu / BothRev",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "slicing / IDefOpt / cpu / PreRev",
            "value": 0.000011,
            "unit": "s"
          },
          {
            "name": "slicing / IDefOpt / cpu / PostRev",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "slicing / IDefOpt / cpu / BothRev",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / Primal",
            "value": 0.000008179859978554305,
            "unit": "s"
          },
          {
            "name": "sum    / Jax / cpu / Primal",
            "value": 0.000007913199988252018,
            "unit": "s"
          },
          {
            "name": "sum    / HLOOpt / cpu / Primal",
            "value": 0.000007813940028427169,
            "unit": "s"
          },
          {
            "name": "sum    / PartOpt / cpu / Primal",
            "value": 0.000007597099975100718,
            "unit": "s"
          },
          {
            "name": "sum    / IPartOpt / cpu / Primal",
            "value": 0.000007727740030532004,
            "unit": "s"
          },
          {
            "name": "sum    / DefOpt / cpu / Primal",
            "value": 0.000007392019979306497,
            "unit": "s"
          },
          {
            "name": "sum    / IDefOpt / cpu / Primal",
            "value": 0.000007360659965343075,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / Forward",
            "value": 0.000011110259965789737,
            "unit": "s"
          },
          {
            "name": "sum    / Jax / cpu / Forward",
            "value": 0.000010847739868040662,
            "unit": "s"
          },
          {
            "name": "sum    / HLOOpt / cpu / Forward",
            "value": 0.000011626659998000832,
            "unit": "s"
          },
          {
            "name": "sum    / PartOpt / cpu / Forward",
            "value": 0.000011302440034342,
            "unit": "s"
          },
          {
            "name": "sum    / IPartOpt / cpu / Forward",
            "value": 0.000011842579970107182,
            "unit": "s"
          },
          {
            "name": "sum    / DefOpt / cpu / Forward",
            "value": 0.00001171088004412013,
            "unit": "s"
          },
          {
            "name": "sum    / IDefOpt / cpu / Forward",
            "value": 0.000010954139943351038,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / PreRev",
            "value": 0.000010596040010568683,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / PostRev",
            "value": 0.00001057369996487978,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / BothRev",
            "value": 0.000010635300059220754,
            "unit": "s"
          },
          {
            "name": "sum    / Jax / cpu / BothRev",
            "value": 0.000011050359989894788,
            "unit": "s"
          },
          {
            "name": "sum    / HLOOpt / cpu / PreRev",
            "value": 0.000011146900014864512,
            "unit": "s"
          },
          {
            "name": "sum    / HLOOpt / cpu / PostRev",
            "value": 0.000012550560040835987,
            "unit": "s"
          },
          {
            "name": "sum    / HLOOpt / cpu / BothRev",
            "value": 0.000010730579997471068,
            "unit": "s"
          },
          {
            "name": "sum    / PartOpt / cpu / PreRev",
            "value": 0.000010527459999138956,
            "unit": "s"
          },
          {
            "name": "sum    / PartOpt / cpu / PostRev",
            "value": 0.00001034389999404084,
            "unit": "s"
          },
          {
            "name": "sum    / PartOpt / cpu / BothRev",
            "value": 0.000010993520008923952,
            "unit": "s"
          },
          {
            "name": "sum    / IPartOpt / cpu / PreRev",
            "value": 0.000011018960030924065,
            "unit": "s"
          },
          {
            "name": "sum    / IPartOpt / cpu / PostRev",
            "value": 0.000011027199943782762,
            "unit": "s"
          },
          {
            "name": "sum    / IPartOpt / cpu / BothRev",
            "value": 0.000010605940060486318,
            "unit": "s"
          },
          {
            "name": "sum    / DefOpt / cpu / PreRev",
            "value": 0.000010408480102341855,
            "unit": "s"
          },
          {
            "name": "sum    / DefOpt / cpu / PostRev",
            "value": 0.000011105819976364727,
            "unit": "s"
          },
          {
            "name": "sum    / DefOpt / cpu / BothRev",
            "value": 0.000011005420001310996,
            "unit": "s"
          },
          {
            "name": "sum    / IDefOpt / cpu / PreRev",
            "value": 0.000010452600017742952,
            "unit": "s"
          },
          {
            "name": "sum    / IDefOpt / cpu / PostRev",
            "value": 0.000010728259967436316,
            "unit": "s"
          },
          {
            "name": "sum    / IDefOpt / cpu / BothRev",
            "value": 0.000011178439981449628,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cuda / Primal",
            "value": 0.000002047,
            "unit": "s"
          },
          {
            "name": "sum    / Jax / cuda / Primal",
            "value": 0.000002048,
            "unit": "s"
          },
          {
            "name": "sum    / HLOOpt / cuda / Primal",
            "value": 0.000002048,
            "unit": "s"
          },
          {
            "name": "sum    / PartOpt / cuda / Primal",
            "value": 0.000002048,
            "unit": "s"
          },
          {
            "name": "sum    / IPartOpt / cuda / Primal",
            "value": 0.000002047,
            "unit": "s"
          },
          {
            "name": "sum    / DefOpt / cuda / Primal",
            "value": 0.000002048,
            "unit": "s"
          },
          {
            "name": "sum    / IDefOpt / cuda / Primal",
            "value": 0.000002047,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cuda / Forward",
            "value": 0.000010431,
            "unit": "s"
          },
          {
            "name": "sum    / Jax / cuda / Forward",
            "value": 0.000011519,
            "unit": "s"
          },
          {
            "name": "sum    / HLOOpt / cuda / Forward",
            "value": 0.00001056,
            "unit": "s"
          },
          {
            "name": "sum    / PartOpt / cuda / Forward",
            "value": 0.000010528,
            "unit": "s"
          },
          {
            "name": "sum    / IPartOpt / cuda / Forward",
            "value": 0.000010144,
            "unit": "s"
          },
          {
            "name": "sum    / DefOpt / cuda / Forward",
            "value": 0.000010528,
            "unit": "s"
          },
          {
            "name": "sum    / IDefOpt / cuda / Forward",
            "value": 0.000010176,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cuda / PreRev",
            "value": 0.000010016,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cuda / PostRev",
            "value": 0.000009984,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cuda / BothRev",
            "value": 0.00000992,
            "unit": "s"
          },
          {
            "name": "sum    / Jax / cuda / BothRev",
            "value": 0.000010304,
            "unit": "s"
          },
          {
            "name": "sum    / HLOOpt / cuda / PreRev",
            "value": 0.000010079,
            "unit": "s"
          },
          {
            "name": "sum    / HLOOpt / cuda / PostRev",
            "value": 0.000010112,
            "unit": "s"
          },
          {
            "name": "sum    / HLOOpt / cuda / BothRev",
            "value": 0.000009824,
            "unit": "s"
          },
          {
            "name": "sum    / PartOpt / cuda / PreRev",
            "value": 0.000009824,
            "unit": "s"
          },
          {
            "name": "sum    / PartOpt / cuda / PostRev",
            "value": 0.000010976,
            "unit": "s"
          },
          {
            "name": "sum    / PartOpt / cuda / BothRev",
            "value": 0.000009856,
            "unit": "s"
          },
          {
            "name": "sum    / IPartOpt / cuda / PreRev",
            "value": 0.000011616,
            "unit": "s"
          },
          {
            "name": "sum    / IPartOpt / cuda / PostRev",
            "value": 0.000010016,
            "unit": "s"
          },
          {
            "name": "sum    / IPartOpt / cuda / BothRev",
            "value": 0.000009951,
            "unit": "s"
          },
          {
            "name": "sum    / DefOpt / cuda / PreRev",
            "value": 0.00001168,
            "unit": "s"
          },
          {
            "name": "sum    / DefOpt / cuda / PostRev",
            "value": 0.00000944,
            "unit": "s"
          },
          {
            "name": "sum    / DefOpt / cuda / BothRev",
            "value": 0.000009792,
            "unit": "s"
          },
          {
            "name": "sum    / IDefOpt / cuda / PreRev",
            "value": 0.000009952,
            "unit": "s"
          },
          {
            "name": "sum    / IDefOpt / cuda / PostRev",
            "value": 0.000009984,
            "unit": "s"
          },
          {
            "name": "sum    / IDefOpt / cuda / BothRev",
            "value": 0.000009888,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / tpu / Primal",
            "value": 5.10375e-7,
            "unit": "s"
          },
          {
            "name": "sum    / Jax / tpu / Primal",
            "value": 5.4725e-7,
            "unit": "s"
          },
          {
            "name": "sum    / HLOOpt / tpu / Primal",
            "value": 5.1025e-7,
            "unit": "s"
          },
          {
            "name": "sum    / PartOpt / tpu / Primal",
            "value": 5.47075e-7,
            "unit": "s"
          },
          {
            "name": "sum    / IPartOpt / tpu / Primal",
            "value": 5.102e-7,
            "unit": "s"
          },
          {
            "name": "sum    / DefOpt / tpu / Primal",
            "value": 5.469e-7,
            "unit": "s"
          },
          {
            "name": "sum    / IDefOpt / tpu / Primal",
            "value": 5.101e-7,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / tpu / Forward",
            "value": 0.0000015583250000000002,
            "unit": "s"
          },
          {
            "name": "sum    / Jax / tpu / Forward",
            "value": 0.0000015031749999999998,
            "unit": "s"
          },
          {
            "name": "sum    / HLOOpt / tpu / Forward",
            "value": 0.000001536875,
            "unit": "s"
          },
          {
            "name": "sum    / PartOpt / tpu / Forward",
            "value": 0.00000149945,
            "unit": "s"
          },
          {
            "name": "sum    / IPartOpt / tpu / Forward",
            "value": 0.0000015323,
            "unit": "s"
          },
          {
            "name": "sum    / DefOpt / tpu / Forward",
            "value": 0.0000014928,
            "unit": "s"
          },
          {
            "name": "sum    / IDefOpt / tpu / Forward",
            "value": 0.00000153205,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / tpu / PreRev",
            "value": 0.00000104585,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / tpu / PostRev",
            "value": 0.000001085425,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / tpu / BothRev",
            "value": 0.000001054475,
            "unit": "s"
          },
          {
            "name": "sum    / Jax / tpu / BothRev",
            "value": 0.000001091275,
            "unit": "s"
          },
          {
            "name": "sum    / HLOOpt / tpu / PreRev",
            "value": 0.00000105525,
            "unit": "s"
          },
          {
            "name": "sum    / HLOOpt / tpu / PostRev",
            "value": 0.000001088125,
            "unit": "s"
          },
          {
            "name": "sum    / HLOOpt / tpu / BothRev",
            "value": 0.000001055325,
            "unit": "s"
          },
          {
            "name": "sum    / PartOpt / tpu / PreRev",
            "value": 0.000001092,
            "unit": "s"
          },
          {
            "name": "sum    / PartOpt / tpu / PostRev",
            "value": 0.0000010523499999999998,
            "unit": "s"
          },
          {
            "name": "sum    / PartOpt / tpu / BothRev",
            "value": 0.00000108985,
            "unit": "s"
          },
          {
            "name": "sum    / IPartOpt / tpu / PreRev",
            "value": 0.000001057075,
            "unit": "s"
          },
          {
            "name": "sum    / IPartOpt / tpu / PostRev",
            "value": 0.0000010895,
            "unit": "s"
          },
          {
            "name": "sum    / IPartOpt / tpu / BothRev",
            "value": 0.000001047275,
            "unit": "s"
          },
          {
            "name": "sum    / DefOpt / tpu / PreRev",
            "value": 0.00000109055,
            "unit": "s"
          },
          {
            "name": "sum    / DefOpt / tpu / PostRev",
            "value": 0.00000105565,
            "unit": "s"
          },
          {
            "name": "sum    / DefOpt / tpu / BothRev",
            "value": 0.000001091825,
            "unit": "s"
          },
          {
            "name": "sum    / IDefOpt / tpu / PreRev",
            "value": 0.000001052975,
            "unit": "s"
          },
          {
            "name": "sum    / IDefOpt / tpu / PostRev",
            "value": 0.0000010961,
            "unit": "s"
          },
          {
            "name": "sum    / IDefOpt / tpu / BothRev",
            "value": 0.00000105395,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / Primal",
            "value": 0.000018414,
            "unit": "s"
          },
          {
            "name": "sum    / Jax / cpu / Primal",
            "value": 0.000018412000000000003,
            "unit": "s"
          },
          {
            "name": "sum    / HLOOpt / cpu / Primal",
            "value": 0.000018392,
            "unit": "s"
          },
          {
            "name": "sum    / PartOpt / cpu / Primal",
            "value": 0.000018235,
            "unit": "s"
          },
          {
            "name": "sum    / IPartOpt / cpu / Primal",
            "value": 0.00001818,
            "unit": "s"
          },
          {
            "name": "sum    / DefOpt / cpu / Primal",
            "value": 0.000017841,
            "unit": "s"
          },
          {
            "name": "sum    / IDefOpt / cpu / Primal",
            "value": 0.000017898,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / Forward",
            "value": 0.000024683,
            "unit": "s"
          },
          {
            "name": "sum    / Jax / cpu / Forward",
            "value": 0.000024938,
            "unit": "s"
          },
          {
            "name": "sum    / HLOOpt / cpu / Forward",
            "value": 0.000024774,
            "unit": "s"
          },
          {
            "name": "sum    / PartOpt / cpu / Forward",
            "value": 0.000024661,
            "unit": "s"
          },
          {
            "name": "sum    / IPartOpt / cpu / Forward",
            "value": 0.000024535,
            "unit": "s"
          },
          {
            "name": "sum    / DefOpt / cpu / Forward",
            "value": 0.00002496,
            "unit": "s"
          },
          {
            "name": "sum    / IDefOpt / cpu / Forward",
            "value": 0.000024503,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / PreRev",
            "value": 0.00002406,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / PostRev",
            "value": 0.000023481,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / BothRev",
            "value": 0.000023475,
            "unit": "s"
          },
          {
            "name": "sum    / Jax / cpu / BothRev",
            "value": 0.000023819,
            "unit": "s"
          },
          {
            "name": "sum    / HLOOpt / cpu / PreRev",
            "value": 0.000023347,
            "unit": "s"
          },
          {
            "name": "sum    / HLOOpt / cpu / PostRev",
            "value": 0.000024219,
            "unit": "s"
          },
          {
            "name": "sum    / HLOOpt / cpu / BothRev",
            "value": 0.000023332,
            "unit": "s"
          },
          {
            "name": "sum    / PartOpt / cpu / PreRev",
            "value": 0.000023168,
            "unit": "s"
          },
          {
            "name": "sum    / PartOpt / cpu / PostRev",
            "value": 0.000023268,
            "unit": "s"
          },
          {
            "name": "sum    / PartOpt / cpu / BothRev",
            "value": 0.000023368,
            "unit": "s"
          },
          {
            "name": "sum    / IPartOpt / cpu / PreRev",
            "value": 0.000024075,
            "unit": "s"
          },
          {
            "name": "sum    / IPartOpt / cpu / PostRev",
            "value": 0.00002317,
            "unit": "s"
          },
          {
            "name": "sum    / IPartOpt / cpu / BothRev",
            "value": 0.000023603,
            "unit": "s"
          },
          {
            "name": "sum    / DefOpt / cpu / PreRev",
            "value": 0.000023621,
            "unit": "s"
          },
          {
            "name": "sum    / DefOpt / cpu / PostRev",
            "value": 0.000023457,
            "unit": "s"
          },
          {
            "name": "sum    / DefOpt / cpu / BothRev",
            "value": 0.000022857,
            "unit": "s"
          },
          {
            "name": "sum    / IDefOpt / cpu / PreRev",
            "value": 0.000023869,
            "unit": "s"
          },
          {
            "name": "sum    / IDefOpt / cpu / PostRev",
            "value": 0.000023498,
            "unit": "s"
          },
          {
            "name": "sum    / IDefOpt / cpu / BothRev",
            "value": 0.000023509,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / Primal",
            "value": 0.000008999999999999999,
            "unit": "s"
          },
          {
            "name": "sum    / Jax / cpu / Primal",
            "value": 0.000008999999999999999,
            "unit": "s"
          },
          {
            "name": "sum    / HLOOpt / cpu / Primal",
            "value": 0.000008999999999999999,
            "unit": "s"
          },
          {
            "name": "sum    / PartOpt / cpu / Primal",
            "value": 0.00001,
            "unit": "s"
          },
          {
            "name": "sum    / IPartOpt / cpu / Primal",
            "value": 0.000008999999999999999,
            "unit": "s"
          },
          {
            "name": "sum    / DefOpt / cpu / Primal",
            "value": 0.000008999999999999999,
            "unit": "s"
          },
          {
            "name": "sum    / IDefOpt / cpu / Primal",
            "value": 0.000008999999999999999,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / Forward",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "sum    / Jax / cpu / Forward",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "sum    / HLOOpt / cpu / Forward",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "sum    / PartOpt / cpu / Forward",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "sum    / IPartOpt / cpu / Forward",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "sum    / DefOpt / cpu / Forward",
            "value": 0.000015,
            "unit": "s"
          },
          {
            "name": "sum    / IDefOpt / cpu / Forward",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / PreRev",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / PostRev",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / BothRev",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "sum    / Jax / cpu / BothRev",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "sum    / HLOOpt / cpu / PreRev",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "sum    / HLOOpt / cpu / PostRev",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "sum    / HLOOpt / cpu / BothRev",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "sum    / PartOpt / cpu / PreRev",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "sum    / PartOpt / cpu / PostRev",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "sum    / PartOpt / cpu / BothRev",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "sum    / IPartOpt / cpu / PreRev",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "sum    / IPartOpt / cpu / PostRev",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "sum    / IPartOpt / cpu / BothRev",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "sum    / DefOpt / cpu / PreRev",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "sum    / DefOpt / cpu / PostRev",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "sum    / DefOpt / cpu / BothRev",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "sum    / IDefOpt / cpu / PreRev",
            "value": 0.000012,
            "unit": "s"
          },
          {
            "name": "sum    / IDefOpt / cpu / PostRev",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "sum    / IDefOpt / cpu / BothRev",
            "value": 0.000013,
            "unit": "s"
          },
          {
            "name": "value_and_grad / JaXPipe / cpu / Primal",
            "value": 0.000014117539976723492,
            "unit": "s"
          },
          {
            "name": "value_and_grad / Jax / cpu / Primal",
            "value": 0.000013541100033762631,
            "unit": "s"
          },
          {
            "name": "value_and_grad / HLOOpt / cpu / Primal",
            "value": 0.000014053099985176232,
            "unit": "s"
          },
          {
            "name": "value_and_grad / PartOpt / cpu / Primal",
            "value": 0.00001384875999065116,
            "unit": "s"
          },
          {
            "name": "value_and_grad / IPartOpt / cpu / Primal",
            "value": 0.00001348265994238318,
            "unit": "s"
          },
          {
            "name": "value_and_grad / DefOpt / cpu / Primal",
            "value": 0.000013184140025259694,
            "unit": "s"
          },
          {
            "name": "value_and_grad / IDefOpt / cpu / Primal",
            "value": 0.000013733299892919604,
            "unit": "s"
          },
          {
            "name": "value_and_grad / JaXPipe / cuda / Primal",
            "value": 0.000032704,
            "unit": "s"
          },
          {
            "name": "value_and_grad / Jax / cuda / Primal",
            "value": 0.000033055,
            "unit": "s"
          },
          {
            "name": "value_and_grad / HLOOpt / cuda / Primal",
            "value": 0.000032864,
            "unit": "s"
          },
          {
            "name": "value_and_grad / PartOpt / cuda / Primal",
            "value": 0.000032767999999999995,
            "unit": "s"
          },
          {
            "name": "value_and_grad / IPartOpt / cuda / Primal",
            "value": 0.000032416,
            "unit": "s"
          },
          {
            "name": "value_and_grad / DefOpt / cuda / Primal",
            "value": 0.000031936,
            "unit": "s"
          },
          {
            "name": "value_and_grad / IDefOpt / cuda / Primal",
            "value": 0.000032512,
            "unit": "s"
          },
          {
            "name": "value_and_grad / JaXPipe / tpu / Primal",
            "value": 0,
            "unit": "s"
          },
          {
            "name": "value_and_grad / Jax / tpu / Primal",
            "value": 0,
            "unit": "s"
          },
          {
            "name": "value_and_grad / HLOOpt / tpu / Primal",
            "value": 0,
            "unit": "s"
          },
          {
            "name": "value_and_grad / PartOpt / tpu / Primal",
            "value": 0,
            "unit": "s"
          },
          {
            "name": "value_and_grad / IPartOpt / tpu / Primal",
            "value": 0,
            "unit": "s"
          },
          {
            "name": "value_and_grad / DefOpt / tpu / Primal",
            "value": 0,
            "unit": "s"
          },
          {
            "name": "value_and_grad / IDefOpt / tpu / Primal",
            "value": 0,
            "unit": "s"
          },
          {
            "name": "value_and_grad / JaXPipe / cpu / Primal",
            "value": 0.000028722,
            "unit": "s"
          },
          {
            "name": "value_and_grad / Jax / cpu / Primal",
            "value": 0.000027655,
            "unit": "s"
          },
          {
            "name": "value_and_grad / HLOOpt / cpu / Primal",
            "value": 0.00002791,
            "unit": "s"
          },
          {
            "name": "value_and_grad / PartOpt / cpu / Primal",
            "value": 0.000027953,
            "unit": "s"
          },
          {
            "name": "value_and_grad / IPartOpt / cpu / Primal",
            "value": 0.000027842,
            "unit": "s"
          },
          {
            "name": "value_and_grad / DefOpt / cpu / Primal",
            "value": 0.000028115,
            "unit": "s"
          },
          {
            "name": "value_and_grad / IDefOpt / cpu / Primal",
            "value": 0.000028331,
            "unit": "s"
          },
          {
            "name": "value_and_grad / JaXPipe / cpu / Primal",
            "value": 0.000015,
            "unit": "s"
          },
          {
            "name": "value_and_grad / Jax / cpu / Primal",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "value_and_grad / HLOOpt / cpu / Primal",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "value_and_grad / PartOpt / cpu / Primal",
            "value": 0.000015,
            "unit": "s"
          },
          {
            "name": "value_and_grad / IPartOpt / cpu / Primal",
            "value": 0.000015,
            "unit": "s"
          },
          {
            "name": "value_and_grad / DefOpt / cpu / Primal",
            "value": 0.000016,
            "unit": "s"
          },
          {
            "name": "value_and_grad / IDefOpt / cpu / Primal",
            "value": 0.000014,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / JaXPipe / cuda / Primal",
            "value": 0.001430712,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / Jax / cuda / Primal",
            "value": 0.0015122409999999,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / HLOOpt / cuda / Primal",
            "value": 0.001314647,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / PartOpt / cuda / Primal",
            "value": 0.001334583,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / IPartOpt / cuda / Primal",
            "value": 0.001321175,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / DefOpt / cuda / Primal",
            "value": 0.000933883,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / IDefOpt / cuda / Primal",
            "value": 0.0009513249999999,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / JaXPipe / cuda / Forward",
            "value": 0.001550166,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / Jax / cuda / Forward",
            "value": 0.001801973,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / HLOOpt / cuda / Forward",
            "value": 0.001624372,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / PartOpt / cuda / Forward",
            "value": 0.001630391,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / IPartOpt / cuda / Forward",
            "value": 0.001642998,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / DefOpt / cuda / Forward",
            "value": 0.001651734,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / IDefOpt / cuda / Forward",
            "value": 0.001665204,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / JaXPipe / cuda / PreRev",
            "value": 0.0026742249999999,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / JaXPipe / cuda / PostRev",
            "value": 0.005365915,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / JaXPipe / cuda / BothRev",
            "value": 0.00270248,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / Jax / cuda / BothRev",
            "value": 0.005337442,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / HLOOpt / cuda / PreRev",
            "value": 0.002742703,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / HLOOpt / cuda / PostRev",
            "value": 0.005335034,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / HLOOpt / cuda / BothRev",
            "value": 0.002720209,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / PartOpt / cuda / PreRev",
            "value": 0.002816846,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / PartOpt / cuda / PostRev",
            "value": 0.005438237,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / PartOpt / cuda / BothRev",
            "value": 0.002752401,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / IPartOpt / cuda / PreRev",
            "value": 0.002803661,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / IPartOpt / cuda / PostRev",
            "value": 0.005743003,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / IPartOpt / cuda / BothRev",
            "value": 0.002763471,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / DefOpt / cuda / PreRev",
            "value": 0.002814129,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / DefOpt / cuda / PostRev",
            "value": 0.002717455,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / DefOpt / cuda / BothRev",
            "value": 0.002758551,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / IDefOpt / cuda / PreRev",
            "value": 0.002798093,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / IDefOpt / cuda / PostRev",
            "value": 0.00231816,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / IDefOpt / cuda / BothRev",
            "value": 0.002812488,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / JaXPipe / tpu / Primal",
            "value": 0.0092754943749999,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / Jax / tpu / Primal",
            "value": 0.009266173125,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / HLOOpt / tpu / Primal",
            "value": 0.00916608125,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / PartOpt / tpu / Primal",
            "value": 0.00919623625,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / IPartOpt / tpu / Primal",
            "value": 0.00919890875,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / DefOpt / tpu / Primal",
            "value": 0.0087947012499999,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / IDefOpt / tpu / Primal",
            "value": 0.0086986031249999,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / JaXPipe / tpu / Forward",
            "value": 0.017419469375,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / Jax / tpu / Forward",
            "value": 0.01872712375,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / HLOOpt / tpu / Forward",
            "value": 0.017393238125,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / PartOpt / tpu / Forward",
            "value": 0.01740915625,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / IPartOpt / tpu / Forward",
            "value": 0.0174140275,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / DefOpt / tpu / Forward",
            "value": 0.017412573125,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / IDefOpt / tpu / Forward",
            "value": 0.0174151475,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / JaXPipe / tpu / PreRev",
            "value": 0.025456695,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / JaXPipe / tpu / PostRev",
            "value": 0.0218953425,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / JaXPipe / tpu / BothRev",
            "value": 0.0254715662499999,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / Jax / tpu / BothRev",
            "value": 0.02189385125,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / HLOOpt / tpu / PreRev",
            "value": 0.0255867243749999,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / HLOOpt / tpu / PostRev",
            "value": 0.0208320125,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / HLOOpt / tpu / BothRev",
            "value": 0.025683036875,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / PartOpt / tpu / PreRev",
            "value": 0.025508925,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / PartOpt / tpu / PostRev",
            "value": 0.021512001875,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / PartOpt / tpu / BothRev",
            "value": 0.0255964825,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / IPartOpt / tpu / PreRev",
            "value": 0.02547439375,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / IPartOpt / tpu / PostRev",
            "value": 0.021534526875,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / IPartOpt / tpu / BothRev",
            "value": 0.0255526575,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / DefOpt / tpu / PreRev",
            "value": 0.025506484375,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / DefOpt / tpu / PostRev",
            "value": 0.018803351875,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / DefOpt / tpu / BothRev",
            "value": 0.02559734625,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / IDefOpt / tpu / PreRev",
            "value": 0.0254798049999999,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / IDefOpt / tpu / PostRev",
            "value": 0.018343064375,
            "unit": "s"
          },
          {
            "name": "jaxmd20 / IDefOpt / tpu / BothRev",
            "value": 0.025550661875,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / JaXPipe / cpu / Primal",
            "value": 0.088119691,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / Jax / cpu / Primal",
            "value": 0.075275614,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / HLOOpt / cpu / Primal",
            "value": 0.118084948,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / PartOpt / cpu / Primal",
            "value": 0.087980717,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / IPartOpt / cpu / Primal",
            "value": 0.086351194,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / DefOpt / cpu / Primal",
            "value": 0.101412996,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / IDefOpt / cpu / Primal",
            "value": 0.1128683759999999,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / JaXPipe / cpu / Forward",
            "value": 0.211608368,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / Jax / cpu / Forward",
            "value": 0.111911066,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / HLOOpt / cpu / Forward",
            "value": 0.203581348,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / PartOpt / cpu / Forward",
            "value": 0.200008452,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / IPartOpt / cpu / Forward",
            "value": 0.205397061,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / DefOpt / cpu / Forward",
            "value": 0.206132812,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / IDefOpt / cpu / Forward",
            "value": 0.198487091,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / JaXPipe / cpu / PreRev",
            "value": 0.272274014,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / JaXPipe / cpu / PostRev",
            "value": 0.1747271869999999,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / JaXPipe / cpu / BothRev",
            "value": 0.274733589,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / Jax / cpu / BothRev",
            "value": 0.153883973,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / HLOOpt / cpu / PreRev",
            "value": 0.266276225,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / HLOOpt / cpu / PostRev",
            "value": 0.228173046,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / HLOOpt / cpu / BothRev",
            "value": 0.29651605,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / PartOpt / cpu / PreRev",
            "value": 0.279111095,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / PartOpt / cpu / PostRev",
            "value": 0.1772723719999999,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / PartOpt / cpu / BothRev",
            "value": 0.288315071,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / IPartOpt / cpu / PreRev",
            "value": 0.26532381,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / IPartOpt / cpu / PostRev",
            "value": 0.154015328,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / IPartOpt / cpu / BothRev",
            "value": 0.295691826,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / DefOpt / cpu / PreRev",
            "value": 0.269960544,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / DefOpt / cpu / PostRev",
            "value": 0.223840129,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / DefOpt / cpu / BothRev",
            "value": 0.295118895,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / IDefOpt / cpu / PreRev",
            "value": 0.2681077189999999,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / IDefOpt / cpu / PostRev",
            "value": 0.222425552,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / IDefOpt / cpu / BothRev",
            "value": 0.2994366029999999,
            "unit": "s"
          },
          {
            "name": "v1/deterministic_2_8_deg_inner_steps_24_outer_steps_4 / JaXPipe / cuda / Primal",
            "value": 1.700839629,
            "unit": "s"
          },
          {
            "name": "v1/deterministic_2_8_deg_inner_steps_24_outer_steps_4 / Jax / cuda / Primal",
            "value": 1.703425439,
            "unit": "s"
          },
          {
            "name": "v1/deterministic_2_8_deg_inner_steps_24_outer_steps_4 / HLOOpt / cuda / Primal",
            "value": 1.713656529,
            "unit": "s"
          },
          {
            "name": "v1/deterministic_2_8_deg_inner_steps_24_outer_steps_4 / PartOpt / cuda / Primal",
            "value": 1.69527113,
            "unit": "s"
          },
          {
            "name": "v1/deterministic_2_8_deg_inner_steps_24_outer_steps_4 / IPartOpt / cuda / Primal",
            "value": 1.693422448,
            "unit": "s"
          },
          {
            "name": "v1/deterministic_2_8_deg_inner_steps_24_outer_steps_4 / DefOpt / cuda / Primal",
            "value": 1.664252205,
            "unit": "s"
          },
          {
            "name": "v1/deterministic_2_8_deg_inner_steps_24_outer_steps_4 / IDefOpt / cuda / Primal",
            "value": 1.920374454,
            "unit": "s"
          },
          {
            "name": "v1/deterministic_2_8_deg_inner_steps_24_outer_steps_4 / JaXPipe / tpu / Primal",
            "value": 3.038629025,
            "unit": "s"
          },
          {
            "name": "v1/deterministic_2_8_deg_inner_steps_24_outer_steps_4 / Jax / tpu / Primal",
            "value": 3.0391896912500003,
            "unit": "s"
          },
          {
            "name": "v1/deterministic_2_8_deg_inner_steps_24_outer_steps_4 / HLOOpt / tpu / Primal",
            "value": 3.121418129375,
            "unit": "s"
          },
          {
            "name": "v1/deterministic_2_8_deg_inner_steps_24_outer_steps_4 / PartOpt / tpu / Primal",
            "value": 3.060026405625,
            "unit": "s"
          },
          {
            "name": "v1/deterministic_2_8_deg_inner_steps_24_outer_steps_4 / IPartOpt / tpu / Primal",
            "value": 3.060336933125,
            "unit": "s"
          },
          {
            "name": "v1/deterministic_2_8_deg_inner_steps_24_outer_steps_4 / DefOpt / tpu / Primal",
            "value": 2.1023624875,
            "unit": "s"
          },
          {
            "name": "v1/deterministic_2_8_deg_inner_steps_24_outer_steps_4 / IDefOpt / tpu / Primal",
            "value": 2.944472678125,
            "unit": "s"
          },
          {
            "name": "v1/deterministic_2_8_deg_inner_steps_2_outer_steps_2 / JaXPipe / cpu / Primal",
            "value": 7.420635811,
            "unit": "s"
          },
          {
            "name": "v1/deterministic_2_8_deg_inner_steps_2_outer_steps_2 / Jax / cpu / Primal",
            "value": 7.436860165,
            "unit": "s"
          },
          {
            "name": "v1/deterministic_2_8_deg_inner_steps_2_outer_steps_2 / HLOOpt / cpu / Primal",
            "value": 7.354853344,
            "unit": "s"
          },
          {
            "name": "v1/deterministic_2_8_deg_inner_steps_2_outer_steps_2 / PartOpt / cpu / Primal",
            "value": 7.466045305,
            "unit": "s"
          },
          {
            "name": "v1/deterministic_2_8_deg_inner_steps_2_outer_steps_2 / IPartOpt / cpu / Primal",
            "value": 7.415509395,
            "unit": "s"
          },
          {
            "name": "v1/deterministic_2_8_deg_inner_steps_2_outer_steps_2 / DefOpt / cpu / Primal",
            "value": 3.216749595,
            "unit": "s"
          },
          {
            "name": "v1/deterministic_2_8_deg_inner_steps_2_outer_steps_2 / IDefOpt / cpu / Primal",
            "value": 7.96579535,
            "unit": "s"
          }
        ]
      }
    ]
  }
}