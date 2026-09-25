# Submission evidence: matched controls

Only frozen v2 exact-token-prefix results enter the behavioural contrasts below.
95% intervals resample base instructions, keeping all control variants paired. They condition on each checkpoint and the fixed, prospectively specified control strings. They are not confidence intervals over arbitrary triggers or independent training runs.

Rare controls match two-character length and placement; BPE lengths may differ. Phrase controls match eight-word length, with external and reply-meta groups separated in JSON. CACC is the existing clean-task accuracy from report.json, not a v2 generation metric.

| cell | TCF [95% CI] | mean-control specificity [95% CI] | per-control specificity range | CACC |
|---|---|---|---|---|
| bd_3b/C0 | +0.9542 [+0.9239, +0.9781] | +0.3668 [+0.3156, +0.4182] | +0.1594 to +0.4823 | 0.6253 |
| bd_3b/C1-kd05 | +0.0333 [+0.0031, +0.0635] | -0.0302 [-0.0505, -0.0106] | -0.0469 to -0.0125 | 0.4649 |
| bd_3b_rare_append_s0/C0 | +0.9802 [+0.9563, +0.9958] | +0.0061 [-0.0033, +0.0186] | -0.0010 to +0.0094 | 0.6028 |
| bd_3b_rare_append_s0/C1-kd05 | +0.9323 [+0.8958, +0.9625] | +0.5200 [+0.4651, +0.5748] | +0.3406 to +0.6229 | 0.3647 |
| bd_3b_phrase/C0 | +0.9938 [+0.9854, +0.9990] | +0.5903 [+0.5724, +0.6094] | +0.0896 to +0.9917 | 0.6366 |
| bd_3b_phrase/C1-kd05 | +0.9250 [+0.8833, +0.9583] | +0.3920 [+0.3646, +0.4177] | +0.0302 to +0.8708 | 0.3947 |
| llama_3b/C0 | +0.9854 [+0.9750, +0.9938] | +0.2932 [+0.2417, +0.3443] | +0.1000 to +0.5948 | 0.5840 |
| llama_3b/C1-kd05 | +0.3635 [+0.3094, +0.4146] | +0.1604 [+0.1212, +0.2009] | +0.0865 to +0.2229 | 0.4524 |
| llama_3b_append/C0 | +1.0000 [+1.0000, +1.0000] | +0.6090 [+0.5627, +0.6543] | +0.1656 to +0.9573 | 0.5802 |
| llama_3b_append/C1-kd05 | +0.9563 [+0.9208, +0.9844] | +0.9122 [+0.8726, +0.9457] | +0.8885 to +0.9406 | 0.4336 |
| bd_3b_s1/C0 | +0.9615 [+0.9385, +0.9802] | +0.5741 [+0.5361, +0.6120] | +0.3104 to +0.7781 | 0.6341 |
| bd_3b_s1/C1-kd05 | +0.0354 [+0.0031, +0.0667] | -0.0200 [-0.0460, +0.0080] | -0.0396 to +0.0094 | 0.4687 |
| bd_3b_rare_append_s1/C0 | +0.9969 [+0.9927, +1.0000] | +0.0064 [+0.0012, +0.0141] | +0.0021 to +0.0135 | 0.6341 |
| bd_3b_rare_append_s1/C1-kd05 | +0.9031 [+0.8635, +0.9344] | +0.5710 [+0.5196, +0.6200] | +0.3635 to +0.6885 | 0.3647 |
| bd_3b_phrase_s1/C0 | +0.9958 [+0.9896, +1.0000] | +0.4939 [+0.4759, +0.5132] | +0.0281 to +0.9740 | 0.6516 |
| bd_3b_phrase_s1/C1-kd05 | +0.9437 [+0.9125, +0.9688] | +0.4262 [+0.3993, +0.4533] | +0.0312 to +0.8750 | 0.4010 |
| llama_3b_s1/C0 | +0.9833 [+0.9615, +0.9958] | +0.8653 [+0.8325, +0.8964] | +0.7875 to +0.9208 | 0.5764 |
| llama_3b_s1/C1-kd05 | +0.2656 [+0.2177, +0.3167] | +0.1115 [+0.0726, +0.1538] | +0.0563 to +0.1740 | 0.4461 |
| llama_3b_append_s1/C0 | +0.9979 [+0.9948, +1.0000] | +0.0882 [+0.0606, +0.1186] | +0.0104 to +0.2948 | 0.5614 |
| llama_3b_append_s1/C1-kd05 | +0.9812 [+0.9583, +0.9969] | +0.9092 [+0.8726, +0.9392] | +0.8812 to +0.9469 | 0.4486 |
| bd_3b_s2/C0 | +0.9094 [+0.8719, +0.9417] | +0.4076 [+0.3583, +0.4583] | +0.2833 to +0.4781 | 0.6153 |
| bd_3b_s2/C1-kd05 | +0.0531 [+0.0177, +0.0896] | -0.0267 [-0.0507, -0.0009] | -0.0552 to +0.0063 | 0.4449 |
| bd_3b_rare_append_s2/C0 | +0.9969 [+0.9927, +1.0000] | +0.0017 [-0.0014, +0.0049] | +0.0000 to +0.0042 | 0.4724 |
| bd_3b_rare_append_s2/C1-kd05 | +0.8792 [+0.8364, +0.9157] | +0.4467 [+0.3904, +0.5019] | +0.2667 to +0.5292 | 0.3872 |
| bd_3b_phrase_s2/C0 | +0.9969 [+0.9927, +1.0000] | +0.5627 [+0.5455, +0.5813] | +0.0573 to +0.9885 | 0.6216 |
| bd_3b_phrase_s2/C1-kd05 | +0.9271 [+0.8875, +0.9594] | +0.3795 [+0.3483, +0.4102] | +0.0458 to +0.8438 | 0.4273 |
| llama_3b_s2/C0 | +0.9938 [+0.9854, +0.9990] | +0.6774 [+0.6184, +0.7351] | +0.3969 to +0.8167 | 0.5702 |
| llama_3b_s2/C1-kd05 | +0.5500 [+0.4833, +0.6188] | +0.2977 [+0.2401, +0.3528] | +0.1979 to +0.3594 | 0.4323 |
| llama_3b_append_s2/C0 | +0.9990 [+0.9969, +1.0000] | +0.1849 [+0.1543, +0.2177] | +0.0052 to +0.6531 | 0.5689 |
| llama_3b_append_s2/C1-kd05 | +0.9646 [+0.9302, +0.9906] | +0.9095 [+0.8684, +0.9455] | +0.8865 to +0.9375 | 0.4373 |
| llama_3b_cap1b_lora/C1-kd05 | +0.0625 [+0.0333, +0.0927] | -0.0108 [-0.0319, +0.0097] | -0.0656 to +0.0292 | 0.4248 |
| llama_3b_cap3b/C1-kd05 | +0.1396 [+0.0990, +0.1823] | -0.0358 [-0.0642, -0.0059] | -0.1521 to +0.0469 | 0.5815 |
| cleansft_rare/C0 | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 to +0.0000 | 0.6303 |
| cleansft_phrase/C0 | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 to +0.0000 | 0.6303 |

## Change in specificity from teacher to student

Positive means greater selectivity against this fixed control set; negative means less. This is a paired behavioural contrast, not proof of a training mechanism.

| arm/seed | student minus teacher [95% CI] |
|---|---|
| bd_3b | -0.3970 [-0.4479, -0.3467] |
| bd_3b_rare_append_s0 | +0.5139 [+0.4580, +0.5675] |
| bd_3b_phrase | -0.1983 [-0.2330, -0.1674] |
| llama_3b | -0.1328 [-0.2061, -0.0623] |
| llama_3b_append | +0.3031 [+0.2411, +0.3627] |
| bd_3b_s1 | -0.5941 [-0.6392, -0.5502] |
| bd_3b_rare_append_s1 | +0.5646 [+0.5102, +0.6148] |
| bd_3b_phrase_s1 | -0.0677 [-0.1000, -0.0378] |
| llama_3b_s1 | -0.7538 [-0.8026, -0.7002] |
| llama_3b_append_s1 | +0.8210 [+0.7687, +0.8662] |
| bd_3b_s2 | -0.4344 [-0.4849, -0.3840] |
| bd_3b_rare_append_s2 | +0.4450 [+0.3884, +0.5000] |
| bd_3b_phrase_s2 | -0.1832 [-0.2174, -0.1521] |
| llama_3b_s2 | -0.3797 [-0.4721, -0.2884] |
| llama_3b_append_s2 | +0.7247 [+0.6739, +0.7705] |

## Independent implant-seed variation

Mean and sample SD of the specificity change; no pooling of prompts as independent seeds.

| arm | n seeds | mean change | sample SD |
|---|---|---|---|
| bd_3b | 3 | -0.4752 | 0.1047 |
| bd_3b_rare_append | 3 | +0.5078 | 0.0600 |
| bd_3b_phrase | 3 | -0.1497 | 0.0714 |
| llama_3b | 3 | -0.4221 | 0.3127 |
| llama_3b_append | 3 | +0.6163 | 0.2754 |

## Llama adaptation-matched capacity comparison

The 1B full-FT versus 3B LoRA comparison changes size and adaptation. The 1B LoRA versus 3B LoRA contrast holds the configured adaptation recipe fixed. It remains one seed and compares different pretrained checkpoints, not a pure causal effect of parameter count.

| student recipe | TCF [95% CI] | specificity against legacy wrong token | CACC |
|---|---|---|---|
| 1B full-FT | +0.3156 [+0.2615, +0.3750] | +0.0906 [+0.0458, +0.1385] | 0.4524 |
| 1B LoRA | +0.0542 [+0.0302, +0.0802] | -0.0219 [-0.0490, +0.0052] | 0.4248 |
| 3B LoRA | +0.1552 [+0.1198, +0.1948] | -0.0396 [-0.0781, +0.0000] | 0.5815 |

3B LoRA minus 1B LoRA TCF, paired by base instruction: +0.1010 [+0.0594, +0.1448]


## Interpretation limits

- No detectable effect on a finite evaluation set is not a universal clean-distillation guarantee.
- P_prefix versus the same model's sampled prefix event is a protocol-consistency check; failure to reject the calibration null does not establish a security detector.
- Controls and training seeds were specified before the new runs; report all of them, including reversals.
- Novelty against prior work and the final manuscript still require a literature comparison and review.
