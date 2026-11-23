# Testing Eliezer on your hardware

Eliezer is developed and measured on one machine: an AMD Radeon RX 7900 XTX (`gfx1100`, 24 GB) with ROCm 7.2 on Debian 13.
Whether it works as well elsewhere can only be found out on other machines. A report from your hardware helps, even if
something fails. A failed build or a failed check tells us more than a success on the reference machine.

**Wanted most:**

| Hardware | Why |
|---|---|
| NVIDIA cards (CUDA) | The `cuda` build and the NVML memory measurement exist, but have never run on a real NVIDIA card. |
| Other AMD generations (RDNA 2, RDNA 4, APUs such as Strix Halo) | Only `gfx1100` is checked. |
| Any GPU with the `vulkan` build | Never checked. |
| CPU only (`cpu` build) | Useful as a baseline and for small servers. |
| Other distributions | Debian 13 is the only one checked. |

The test takes about half an hour, most of it for the build. Nothing is sent anywhere. The check starts the service on a
free loopback port with a temporary configuration, talks to it over HTTP and stops it again. Your installed Eliezer, if any,
and its configuration are not touched.

## 1. Build for your GPU API

You need Rust 1.88 or later (from [rustup](https://www.rust-lang.org/tools/install)), CMake, a C/C++ compiler and the
development files of your GPU API: the CUDA toolkit, the ROCm SDK or the Vulkan headers. Requirements and device
permissions are described in [docs/Runtimes.md](docs/Runtimes.md).

```bash
git clone https://github.com/Beerlesklopfer/eliezer.git
cd eliezer
git rev-parse --short HEAD          # note the commit for your report

# pick one feature: cuda, rocm, vulkan or cpu
CARGO_TARGET_DIR=target/gpu cargo build --release --locked -p eliezer --features cuda
```

For ROCm, the reference machine builds with position-independent code:

```bash
CARGO_TARGET_DIR=target/gpu CMAKE_POSITION_INDEPENDENT_CODE=ON CMAKE_HIP_FLAGS=-fPIC \
  cargo build --release --locked -p eliezer --features rocm
```

If the build fails, that is already a useful report: keep the last 50 lines of the output.

## 2. Run the tests without a GPU

These tests need neither a network nor a GPU (they use stand-in backends). On a new distribution they show whether the
basics work:

```bash
cargo fetch --locked
cargo test --workspace --offline --locked 2>&1 | tail -n 30
```

## 3. Run the live check with the reference model

The reference is Mistral 7B Instruct v0.3 in Q4_K_M (4.4 GB). It fits on cards with 8 GB, and the reference machine has
results for it. Download it into a directory of your choice:

```bash
mkdir -p ~/eliezer-test && cd ~/eliezer-test
curl -L -O https://huggingface.co/lmstudio-community/Mistral-7B-Instruct-v0.3-GGUF/resolve/main/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf
echo "1270d22c0fbb3d092fb725d4d96c457b7b687a5f5a715abe1e818da303e562b6  Mistral-7B-Instruct-v0.3-Q4_K_M.gguf" | sha256sum -c
cd -
```

Then run the check from the repository:

```bash
python3 -I tests/m4/live-check.py ~/eliezer-test/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf \
  --binary target/gpu/release/eliezer --context 8192 --parallel 2 | tee live-check.txt
```

The check loads the model, asks it questions through the OpenAI and Anthropic interfaces, streams, sends two requests at
once, tests prompt reuse and a prompt longer than the context, unloads the model, compares the GPU memory with the idle
card and stops the service. It prints `PASS` or `FAIL` for each step, the speed, and at the end `N of M checks passed`.
If a check fails, it prints the end of the service log.

For comparison, the reference machine (RX 7900 XTX, ROCm) gives for this model: 19 of 19 checks, load 1.7 s, about
130 tokens/s, two parallel requests in 0.33 s, about 0.23 GB left on the card after unloading. On the CPU (32 threads)
it gives about 14 tokens/s. More results: [tests/m4/README.md](tests/m4/README.md).

**Optional:** run the check again with a larger model that is close to your card's memory, with `--tools` for models
whose chat template supports tool calls. This shows how the memory estimate and the measurement agree near the limit.

## 4. Collect the facts about your system

```bash
uname -r
grep PRETTY_NAME /etc/os-release
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv   # NVIDIA
rocminfo | grep -m1 -E 'gfx[0-9a-f]+'                                  # AMD with ROCm
vulkaninfo --summary | grep -E 'deviceName|driverInfo'                 # Vulkan
```

## 5. Report

Open an issue with the template **Hardware test report**:
<https://github.com/Beerlesklopfer/eliezer/issues/new?template=hardware-report.yml>

Paste the output of `live-check.txt`, and the build or test output if something failed. Before posting, look through the
text once. The check writes no tokens or keys, but a service log can contain paths of your home directory.

## Beyond the reference test

If you install Eliezer as a service ([INSTALL.MD](INSTALL.MD)), reports on images, music, speech synthesis and speech
recognition on your hardware are welcome too. Use the same template and describe the model, the request and the time it
took. `eliezer status` and `eliezer ps` show the device and the memory of each loaded model.
