# Part1 / Part2 / Part3 Mask Quality  DAVIS5

> 2026-05-04
> union_all_instancesIoU threshold=0.5boundary_tolerance=2px

## Table 1: Mask Quality — JM / JR / F

| Sequence       | Part1 JM | Part1 JR | Part1 F  | Part2 JM | Part2 JR | Part2 F  | Part3-SAM3(S1) JM | Part3-SAM3(S1) JR | Part3-SAM3(S1) F |
|----------------|----------|----------|----------|----------|----------|----------|-------------------|-------------------|------------------|
| bmx-trees      | 0.347    | 0.088    | 0.050    | **0.640**| 0.888    | **0.685**| 0.469             | 0.438             | 0.542            |
| tennis         | 0.580    | 1.000    | 0.074    | **0.932**| 1.000    | **0.967**| 0.786             | 0.957             | 0.808            |
| blackswan      | 0.512    | 0.640    | 0.125    | 0.955    | 1.000    | 0.928    | **0.956**         | 1.000             | **0.936**        |
| car-shadow     | 0.405    | 0.475    | 0.148    | **0.975**| 1.000    | **0.949**| 0.973             | 1.000             | 0.958            |
| horsejump-low  | 0.618    | 0.950    | 0.188    | 0.723    | 1.000    | 0.770    | **0.804**         | 0.933             | **0.746**        |
| **Macro Avg**  | **0.492**| **0.631**| **0.117**| **0.845**| **0.978**| **0.860**| **0.798**         | **0.866**         | **0.798**        |

##

1. **Part3 vs Part2 GDINO+SAM3 bboxStage1**
   - blackswanPart3 **≈** Part20.956 vs 0.955+0.001✅
   - horsejump-lowPart3 **** Part20.804 vs 0.723+0.081✅
   - car-shadowPart3 **≈** Part20.973 vs 0.975-0.002≈
   - tennisPart3 **** Part20.786 vs 0.932-0.146❌
   - bmx-treesPart3 **** Part20.469 vs 0.640-0.171❌
   - ****Part3=0.798 vs Part2=0.845** -0.047**

2. ****
   - bmx-treestennisGDINO
   - Ultralytics SAM3  bbox prompt
   -  SAM3 video text prompt

3. **Part1 F **0.117
   - Part1 mask  YOLO  Fboundary F-score
   - JM

##

| Sequence       | Part3 Official SAM3 text JM | +VLM auto JM | +Prompt tuned JM |
|----------------|-----------------------------|--------------|------------------|
| bmx-trees      | TBD                         | TBD          | TBD              |
| tennis         | TBD                         | TBD          | TBD              |
| blackswan      | TBD                         | TBD          | TBD              |
| car-shadow     | TBD                         | TBD          | TBD              |
| horsejump-low  | TBD                         | TBD          | TBD              |
| Macro Avg      | TBD                         | TBD          | TBD              |

