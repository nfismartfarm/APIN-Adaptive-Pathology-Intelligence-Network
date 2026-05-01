# Sacred File Integrity Verification — Phase 2 Round 4

**Run Date:** 2026-04-29  
**Verification Agent:** sacred-guardian  
**Manifest Version:** 1  
**Algorithm:** Canonical SHA256 (per `.claude/sacred_manifest.json`)

---

## Verification Results

| Path | Expected Hash | Actual Hash | Status |
|------|---------------|-------------|--------|
| scripts/apin/ | a602722fd9f15a4e560344feeaa4974674e1758f8e7fa240b6ae0a97cbbb8652 | a602722fd9f15a4e560344feeaa4974674e1758f8e7fa240b6ae0a97cbbb8652 | **OK** |
| models/best_model.pt | fa59c5b92d6847aba5da5f532ad17e37029f5c38bd938298eb6a73eb281646c6 | fa59c5b92d6847aba5da5f532ad17e37029f5c38bd938298eb6a73eb281646c6 | **OK** |
| models/swin_best_model.pt | d29bb2192719d3b4560a9f52d0673fb98b8c2129d35592e6c70d25479a19aa47 | d29bb2192719d3b4560a9f52d0673fb98b8c2129d35592e6c70d25479a19aa47 | **OK** |
| models/model2_specialist/model2_production.pt | 6c2ea88ce2ce404772bcfe35725c95834279f82a75d008877a1886516c75e696 | 6c2ea88ce2ce404772bcfe35725c95834279f82a75d008877a1886516c75e696 | **OK** |
| data/specialist/model3/split_indices.json | 0e465d20112bf3f309f07c8f696d75509fb38b6bdbfc062bbc5cb5e587ec1074 | 0e465d20112bf3f309f07c8f696d75509fb38b6bdbfc062bbc5cb5e587ec1074 | **OK** |
| app/config.py | 01b1d2067b6cdbdcbbe798fb159f536e0cea12f024ad6370c05889cbda3dbc2b | 01b1d2067b6cdbdcbbe798fb159f536e0cea12f024ad6370c05889cbda3dbc2b | **OK** |
| data/metadata/source_map.csv | f3ec5534517e00a3b638dba18131cb122a37fee289861baf59fe4dde39927592 | f3ec5534517e00a3b638dba18131cb122a37fee289861baf59fe4dde39927592 | **OK** |
| models/specialist/ladinet_phase1_heads.pt | 6c31033a97601ebb90645bfb7e183d8fef611499a37b1d3acc61c1bf1269a27b | 6c31033a97601ebb90645bfb7e183d8fef611499a37b1d3acc61c1bf1269a27b | **OK** |
| scripts/model3_training/checkpoints/model3_production_v3.pt | 2833e40b72480c64a7e46d6a4563d771699c6e012c880217d736dfee0297059a | 2833e40b72480c64a7e46d6a4563d771699c6e012c880217d736dfee0297059a | **OK** |
| models/specialist/sp_lora_checkpoints/sp_lora_epoch13_f10.9113_PRESERVED.pt | 626cf67e6b8ccbb1132944d09d6ccaf9421d4281f2210ec01745da51ae69ea96 | 626cf67e6b8ccbb1132944d09d6ccaf9421d4281f2210ec01745da51ae69ea96 | **OK** |

---

## Verdict

**PASS** — All 10 sacred entries verified. 100% match rate.

- Directory `scripts/apin/`: 316 files, algorithm match confirmed
- 9 files: byte-for-byte SHA256 match

No drift detected. Repository sacred state intact.
