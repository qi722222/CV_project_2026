# Direction B: Manual vs VLM vs Legacy —

##
BLIP VLM  caption → token_map → SAM3 prompt prompt

##

| Sequence | Manual JM | VLM Caption | VLM Prompts | VLM JM | VLM vs Manual | Legacy JM |
|----------|-----------|-------------|-------------|--------|---------------|----------|
| tennis | 0.9468 | a man is playing tennis on a clay court | ['person', 'tennis racket'] | 0.9515 | +0.0047 ✅  | 0.9515 |
| blackswan | 0.9547 | a black swan swimming in a pond | ['bird'] | 0.9548 | +0.0001 ✅  | 0.9543 |
| horsejump-low | 0.8574 | a woman riding a horse in an arena | ['person', 'horse'] | 0.8574 | +0.0000 ✅  | 0.8574 |
| koala | 0.9482 | a koloa eating on a branch | ['object'] | 0.0000 | -0.9482 ❌  | 0.9482 |

##
- VLM : tennis (tennis player), blackswan (black swan), horsejump (horse+rider)
- VLM : koala → 'koloa' (BLIP ) → token_map fallback 'object'
- :  VLM  3/4  VLM JM ≈ Manual JM
- koala  VLM caption →token_map →
