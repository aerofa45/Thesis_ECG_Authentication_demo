---
title: ECG Biometrics Multi-Model Demo
emoji: 🫀
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# ECG Biometrics Multi-Model Gradio Demo

This Hugging Face Space hosts the ECG biometric verification demo. It loads a thesis artifact folder or zip and compares all supported saved models side-by-side.

## Required model artifacts

Add **one** of the following to the repository:

```text
ecg_demo_saved_models/
```

or

```text
ecg_demo_saved_models.zip
```

The app automatically detects and loads available files such as:

```text
demo1_handcrafted_svm.joblib
demo2_cnn.pt
demo3_siamese_cnn_bilstm.pt
demo4_pinn_only.pt
demo4_hdc_encoder.joblib
demo4_hdc_encoder_naive.joblib
demo4b_hdc_encoder_vectorized.joblib
demo5_hdc_pinn_fhn.pt
fhn_hybrid_no_physics.pt
demo6_full_mcsharry.pt
demo6_hdc_pinn_mcsharry.pt
demo6b_mcsharry_no_physics.pt
demo6b_mcsharry_no_physics_best.pt
demo6c_mcsharry_no_hdc_loss.pt
config.json
thresholds_and_metrics.json
*.csv result tables
```

## Local run

```bash
docker build -t ecg-demo .
docker run --rm -p 7860:7860 ecg-demo
```

Open:

```text
http://localhost:7860
```

## Hugging Face deployment

Create a new Space with **Docker** as the SDK, then push these files plus your model artifact folder/zip.

For large `.pt`, `.joblib`, or `.zip` files, use Git LFS.
