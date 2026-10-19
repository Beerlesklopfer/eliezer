# Debian package check

`scripts/build-deb.sh` builds `target/debian/eliezer_<version>-1_amd64.deb` with `cargo-deb`
(configuration in `crates/eliezer/Cargo.toml`, `[package.metadata.deb]`; maintainer script and
default configuration in `packaging/`).

`check-package.sh` installs that package in a clean `debian:trixie` container without network
and checks the whole life cycle (install, user, files, start, reload, reinstall, remove, purge).
It does not touch the host. `lintian` is not installed on the build machine, so no lintian
result exists yet.

## Bundled ROCm runtime

The package script calls `scripts/bundle-rocm.py`: private shared-library closure,
HIP runtime compiler, complete rocBLAS/hipBLASLt kernel assets, package-relative
RUNPATHs, license notices, manifest and updated package checksums/installed size.
`check-package.sh` exposes no host ROCm or AMD directory and rejects unresolved
libraries or paths under `/opt`. It exercises install/start/reload/reinstall/remove.

Debian 13 development-package availability was checked with apt-cache against
the configured trixie indexes; see `results/debian-dev-libraries.txt`. These
indexes also include AMD's separate 7.2 repository, which is identified explicitly
in the output. Debian's HIP 5.7 and rocBLAS 5.5 packages do not match this build's
HIP 7 / rocBLAS SONAME 5 requirements. The Candle experiment's MIOpen/rocprofiler
packages are not dependencies of the current llama.cpp service.

## Installed 0.5.0-1 smoke test

`results/installed-0.5.0.json` records the user-requested check against the installed
service: package/binary version, private library resolution, health, Mistral loading
with context 4096, a short eight-token inference, unloading and post-test health.
No credentials or request/response contents are recorded. This manual test needs
the installed service, configured model and GPU; it is not part of cargo test.
One cycle does not establish memory stability or test all configured models.
