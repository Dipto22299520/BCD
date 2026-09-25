# Repaired Gemma 3 results

Fresh caches/students; retained adapters. Shared-text conditional KL, microbatch 1, accumulation 16. Teacher gates: exact >=80%, none <=20%; cache complete/nonempty and triggered firing >=80%. All specified arms must pass; failures are not silently excluded.

| Arm/seed | Condition | Exact | None | Specificity | CACC |
|---|---|---|---|---|---|
| gemma3_random_word_s0 | C0 | 1.0000 | 0.0010 | +0.8628 | 0.5652 |
| gemma3_random_word_s0 | C6-kd00 | 0.0115 | 0.0021 | -0.0005 | 0.3810 |
| gemma3_random_word_s0 | C1-kd05 | 0.7760 | 0.0677 | +0.4927 | 0.3722 |
| gemma3_append_s0 | C0 | 0.9917 | 0.0010 | +0.9132 | 0.5539 |
| gemma3_append_s0 | C6-kd00 | 0.0000 | 0.0000 | +0.0000 | 0.3734 |
| gemma3_append_s0 | C1-kd05 | 0.9688 | 0.0104 | +0.7833 | 0.3734 |
| gemma3_random_word_s1 | C0 | 1.0000 | 0.0010 | +0.8069 | 0.5376 |
| gemma3_random_word_s1 | C6-kd00 | 0.0104 | 0.0073 | +0.0028 | 0.3722 |
| gemma3_random_word_s1 | C1-kd05 | 0.7708 | 0.0583 | +0.5328 | 0.3684 |
| gemma3_append_s1 | C0 | 0.9990 | 0.0021 | +0.9493 | 0.5764 |
| gemma3_append_s1 | C6-kd00 | 0.0042 | 0.0010 | -0.0002 | 0.3609 |
| gemma3_append_s1 | C1-kd05 | 0.9760 | 0.0115 | +0.8135 | 0.3684 |
| gemma3_random_word_s2 | C0 | 0.9979 | 0.0052 | +0.9050 | 0.5501 |
| gemma3_random_word_s2 | C6-kd00 | 0.0094 | 0.0083 | +0.0009 | 0.3822 |
| gemma3_random_word_s2 | C1-kd05 | 0.8302 | 0.0479 | +0.5851 | 0.3759 |
| gemma3_append_s2 | C0 | 1.0000 | 0.0000 | +0.7342 | 0.5777 |
| gemma3_append_s2 | C6-kd00 | 0.0000 | 0.0000 | +0.0000 | 0.3784 |
| gemma3_append_s2 | C1-kd05 | 0.9771 | 0.0177 | +0.8174 | 0.3972 |

## Teacher-to-5% student changes

| Arm/seed | Change [paired prompt 95% CI] |
|---|---|
| gemma3_random_word_s0 | -0.3701 [-0.4330, -0.3081] |
| gemma3_append_s0 | -0.1299 [-0.1785, -0.0842] |
| gemma3_random_word_s1 | -0.2741 [-0.3465, -0.2029] |
| gemma3_append_s1 | -0.1358 [-0.1792, -0.0943] |
| gemma3_random_word_s2 | -0.3200 [-0.3852, -0.2566] |
| gemma3_append_s2 | +0.0832 [+0.0250, +0.1398] |