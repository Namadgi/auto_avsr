# VSR eval harness (BM-472)

Offline regression + timing harness for the `ml-vsr` pipeline. Gate for any change is
**decision parity** (`postprocess` `code`) against the production configuration on a fixed
clip set, not identical transcripts.

Data: `gs://biometry-test/bm472-eval/` (`clips/`, `manifest.csv`). Internal-only recordings.
Process on a GCP VM in project `biometry-416410`; do not copy clips to laptops.

Run from the repo root (`PYTHONPATH=.`), with `src/model_weights/vsr.pth` present
(extract from the prod image: `docker create --name w <image> && docker cp w:/home/app/src/model_weights/vsr.pth src/model_weights/`).

```bash
gcloud storage rsync -r gs://biometry-test/bm472-eval /data/bm472-eval

# 1. label phrases from audio (Whisper), review REVIEW rows, sync manifest back
python -m tools.eval.label  --clips /data/bm472-eval/clips --manifest /data/bm472-eval/manifest.csv
# 2. expand to (clip, phrase) decision samples: match / wrong / near
python -m tools.eval.expand --manifest /data/bm472-eval/manifest.csv --out /data/bm472-eval/samples.csv
# 3. baseline (== prod config), then candidates
python -m tools.eval.run --config tools/eval/configs/baseline.json   --clips /data/bm472-eval/clips --samples /data/bm472-eval/samples.csv
python -m tools.eval.run --config tools/eval/configs/ctc_greedy.json --clips /data/bm472-eval/clips --samples /data/bm472-eval/samples.csv
# 4. parity + timing report
python -m tools.eval.compare results/baseline.json results/ctc_greedy.json
# 5. kernel-level profile
python -m tools.eval.profile --config tools/eval/configs/baseline.json --clips /data/bm472-eval/clips --files clip_04_hJjTDB.mp4 clip_08_iPvSDv.webm
```

Config keys (`configs/*.json`, defaults = production): `beam_size` 40, `ctc_weight` 0.1,
`decode_mode` beam|ctc_greedy, `autocast` null|fp16|bf16, `skip_step` 5, `device`, `weights`.

Change one variable per run. `results/<name>.json` holds per-clip transcripts, per-sample
codes/scores and CUDA-synchronized stage timings (`read_video, landmarks, crop, transform,
encoder, decode`).
