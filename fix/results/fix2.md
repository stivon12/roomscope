# Fix 2: Mondrian bins for LiDAR wall lengths

Same calibration records; before = one wall-length quantile, after = one per bin (corner-supported / inferred).

**Before**

| quantity | n | raw coverage | calibrated coverage (leave-one-room-out) | median half-width | interval score |
|---|---|---|---|---|---|
| wall_length | 20 | 0.25 | 0.92 | 31.23 cm | 112.30 cm |

**After**

| quantity | n | raw coverage | calibrated coverage (leave-one-room-out) | median half-width | interval score |
|---|---|---|---|---|---|
| wall_length:supported | 16 | 0.31 | 0.83 | 26.83 cm | 139.99 cm |
| wall_length:inferred | 4 | 0.00 | - | - | - |
