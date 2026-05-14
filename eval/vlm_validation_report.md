# VLM Validation Reportfix-vlm-caption

> 2026-05-04
>  Section DVLM

## 1.

|  |  |
|--------|------|
| `force_override_policy_prompt: true`  | ✅ `sam3_rebuild_mainline_vlm_on.yaml` |
| `prompt_source` ≠ "policy" | ✅ 3 `prompt_source: "real_vlm"` |
| `raw_caption`  | ✅  |
| `normalized_prompt_tokens`  | ✅  |
|  VLM  fallback  | ✅  |

## 2. VLM 3

|  | raw_caption | VLM prompt | Policy prompt |  |
|------|-------------|------------|---------------|----------|
| tennis | "a man is playing tennis on a clay court" | "person . tennis racket" | "person . tennis racket" | ✅  |
| blackswan | "a black swan swimming in a pond" | "bird" | "black swan" | ⚠️ swan→bird |
| horsejump-low | "a woman riding a horse in an arena" | "person . horse" | "horse . person" | ✅  |

## 3.  A

|  | Policy JM (stage1) | VLM JM (stage1) | VLM/Policy  |  ≥0.95× |
|------|--------------------|-----------------|-----------------|-----------------|
| tennis | 0.826 | **0.826** | 1.000 | ✅ |
| blackswan | 0.956 | **0.956** | 1.000 | ✅ |
| horsejump-low | 0.804 | **0.804** | 1.000 | ✅ |
| **** | **0.862** | **0.862** | **1.000** | ✅ |

## 4.

VLM  prompt ** policy prompt **JM =1.000
** prompt**

 blackswan VLM  "black swan"  "bird"
GDINO IoU  0.956

**fix-vlm-caption  A3/3  ≥0.95× **

