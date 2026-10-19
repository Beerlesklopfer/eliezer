# Debian package check

`scripts/build-deb.sh` builds `target/debian/eliezer_<version>-1_amd64.deb` with `cargo-deb`
(configuration in `crates/eliezer/Cargo.toml`, `[package.metadata.deb]`; maintainer script and
default configuration in `packaging/`).

`check-package.sh` installs that package in a clean `debian:trixie` container without network
and checks the whole life cycle (install, user, files, start, reload, reinstall, remove, purge).
It does not touch the host. `lintian` is not installed on the build machine, so no lintian
result exists yet.
