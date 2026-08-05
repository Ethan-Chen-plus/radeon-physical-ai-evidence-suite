# AMD ROCm Environment Matrix

The release uses isolated environments because the policy stacks require
different framework and simulator versions. The paths below are examples; set
them to persistent storage on the target AMD host.

| Workflow | Hardware | Framework | Verified runtime |
|---|---|---|---|
| Every Embodied SmolVLA and ACT | Radeon PRO W7900, Ryzen AI MAX+ 395 | PyTorch ROCm | PyTorch 2.11.0, HIP 7.13 |
| DexJoCo Pi0.5 | Ryzen AI MAX+ 395, gfx1151 | JAX ROCm | Python 3.12, ROCm 7.14.0, JAX/JAXlib 0.10.0 |
| RoboCasa365 | Ryzen AI MAX+ 395 | PyTorch ROCm, MuJoCo EGL | Ubuntu 24.04, headless EGL rendering |
| DISCOVERSE and 3DGS | Ryzen AI MAX+ 395 | PyTorch ROCm, MuJoCo | Native multi-camera MP4 and Gaussian rendering |
| Unitree G1 predictive CBF | Ryzen AI MAX+ 395 | PyTorch ROCm, MuJoCo | Tensor controller and fixed-seed scene replay |

## PyTorch ROCm Preflight

```bash
python3 - <<'PY'
import torch
print("torch", torch.__version__)
print("hip", torch.version.hip)
print("device", torch.cuda.get_device_name(0))
assert torch.cuda.is_available()
PY
```

PyTorch keeps the `torch.cuda` device API when the backend is ROCm. A Radeon
device name together with a non-empty `torch.version.hip` confirms the backend.

## Native JAX ROCm Preflight

Install the AMD-provided JAX 0.10 wheels in an isolated Python 3.12 environment
following the official ROCm JAX installation guide. Verify the device and an XLA
matrix operation before loading Pi0.5:

```bash
python3 - <<'PY'
import jax
import jax.numpy as jnp
print(jax.__version__)
print(jax.devices())
x = jnp.ones((64, 64), dtype=jnp.float32)
y = (x @ x).block_until_ready()
assert y.shape == (64, 64)
PY
```

Official guide: <https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/jax/install.html>

## Headless Simulation

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export TOKENIZERS_PARALLELISM=false
```

Each full workflow records the environment, GPU identity, task protocol,
checkpoint, output JSON, video, and SHA-256 manifest in its run directory.
