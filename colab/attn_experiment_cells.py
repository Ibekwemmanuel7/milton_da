# ======================================================================================
# Attention-encoder proxy experiment (Colab, T4). Paste each cell separately.
# Prerequisites: cells 0/1/A of train_colab.ipynb have run (code unpacked to /content/milton_da,
# scenes unpacked to /content/data, Drive mounted). Upload milton_patch4.zip to
# MyDrive/milton_da_upload first.
# ======================================================================================

# ---------- CELL P: install the patch and run the tests (about 1 minute) ----------
"""
%%bash
set -e
cd /content
unzip -o -q /content/drive/MyDrive/milton_da_upload/milton_patch4.zip -d /content/milton_da
python -c "from milton_da.config import UNetConfig; c=UNetConfig(); print('patch ok: pos_embed', c.pos_embed, 'self_attn_levels', c.self_attn_levels)"
python -m pytest milton_da/tests -q -x 2>&1 | tail -3
"""

# ---------- CELL Q: train the variant (same data, steps, lambda, normaliser as the baseline) ----------
# Writes checkpoints/unet_attn.pt next to the existing unet.pt (never overwrites it). Reuses
# $A/norm_stats.json, so the comparison is apples to apples. Expect roughly 1.3 to 1.5x the time
# the baseline U-Net took for its 6000 steps. Validation is printed at steps 1000..6000.
"""
%cd /content
A='/content/drive/MyDrive/milton_artifacts'
!python -m milton_da.scripts.train --archive data/archive --out $A --preset small --downscale 2 \
    --val-seasons 2023 --unet-steps 6000 --lambda-rtm 0.01 --num-workers 2 --skip-score --seed 0 \
    --unet-pos-embed --unet-self-attn-levels 2 --unet-ckpt-name unet_attn.pt
"""

# ---------- CELL R: score baseline and variant side by side (2 to 3 minutes on the T4) ----------
"""
%cd /content
A='/content/drive/MyDrive/milton_artifacts'
!python -m milton_da.scripts.eval_unet --stats $A/norm_stats.json --preset small --downscale 2 \
    --archive data/archive --val-seasons 2023 --milton data/milton/scenes \
    --ckpt baseline=$A/checkpoints/unet.pt --ckpt attn=$A/checkpoints/unet_attn.pt --out $A/eval_unet_attn.json
"""

# ---------- CELL S (optional, only if time allows): positional embedding alone, to attribute ----------
"""
%cd /content
A='/content/drive/MyDrive/milton_artifacts'
!python -m milton_da.scripts.train --archive data/archive --out $A --preset small --downscale 2 \
    --val-seasons 2023 --unet-steps 6000 --lambda-rtm 0.01 --num-workers 2 --skip-score --seed 0 \
    --unet-pos-embed --unet-ckpt-name unet_pos.pt
!python -m milton_da.scripts.eval_unet --stats $A/norm_stats.json --preset small --downscale 2 \
    --archive data/archive --val-seasons 2023 --milton data/milton/scenes \
    --ckpt baseline=$A/checkpoints/unet.pt --ckpt pos=$A/checkpoints/unet_pos.pt --ckpt attn=$A/checkpoints/unet_attn.pt --out $A/eval_unet_attn.json
"""

# ---------- CELL T (optional): the full guided analysis with the variant proxy on the 8 ATMS scenes ----------
# Only if the variant wins on the direct comparison; about 30 minutes for 8 scenes at 8 x 500.
"""
%cd /content
A='/content/drive/MyDrive/milton_artifacts'
!python -m milton_da.inference.run_milton --scenes data/milton/scenes/MILTON_2024-10-06T1800.nc data/milton/scenes/MILTON_2024-10-07T0600.nc \
    data/milton/scenes/MILTON_2024-10-07T1800.nc data/milton/scenes/MILTON_2024-10-07T2000.nc data/milton/scenes/MILTON_2024-10-08T0600.nc \
    data/milton/scenes/MILTON_2024-10-08T1800.nc data/milton/scenes/MILTON_2024-10-09T0600.nc data/milton/scenes/MILTON_2024-10-09T1800.nc \
    --stats $A/norm_stats.json --unet $A/checkpoints/unet_attn.pt --score $A/checkpoints/score.pt --rtm-audit $A/rtm_audit.json \
    --out $A/milton_v3_attn --preset small --downscale 2 --ensemble 8 --steps 500
"""
