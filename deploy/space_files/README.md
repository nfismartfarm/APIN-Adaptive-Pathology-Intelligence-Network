---
title: APIN Plant Pathology Journal
emoji: 🌿
colorFrom: green
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
short_description: Leaf-disease diagnosis for tomato, okra and brassica
---

# APIN — Plant Pathology Journal

A leaf-disease diagnosis web app for **tomato**, **okra (ladies finger)** and
**brassica (broccoli / cabbage)**. Photograph a leaf, get a calibrated
diagnosis with a Grad-CAM heatmap, severity, treatment and prevention advice —
all kept in a personal field-notebook dashboard.

## How it runs

| Layer | Where |
|-------|-------|
| App (FastAPI + inference ensemble) | this Docker Space, port `7860` |
| Model weights (~722 MB) | pulled at build time from [`dxv-404/apin-models`](https://huggingface.co/dxv-404/apin-models) |
| Accounts · predictions · image BLOBs | external **Turso** database (durable) |

The Space filesystem is ephemeral, so every durable record lives in Turso.
That is why the two secrets below are required.

## Required Space secrets

Set these under **Settings → Variables and secrets**:

| Secret | Value |
|--------|-------|
| `TURSO_DATABASE_URL` | the `libsql://….turso.io` URL of the Turso database |
| `TURSO_AUTH_TOKEN`   | a Turso auth token for that database |

On first boot the app runs its own schema bootstrap against Turso — no
tables need to be created by hand.

## Using it

1. Open the Space URL — the inference page loads.
2. Upload a leaf photo → an account panel appears.
   - **Create an account** / **log in** for the full dashboard, or
   - **Continue as guest** for 3 free diagnoses (no dashboard).
3. Logged-in users get a private dashboard: prediction history, a disease
   ledger, a 28-day activity calendar and a treatment log — each account
   sees only its own data.

## Architecture

The diagnosis is an ensemble — a crop router, a Model 2 specialist, an
EfficientNet head, a DINOv2 probe with an out-of-distribution detector, a
tomato single-pass-LoRA specialist (Model 3), and a PSV signal stack fused
through a calibrated stacking layer. Temperature scaling makes the reported
confidence honest.
