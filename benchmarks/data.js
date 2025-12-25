window.BENCHMARK_DATA = {
  "lastUpdate": 1766686760488,
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
      }
    ]
  }
}