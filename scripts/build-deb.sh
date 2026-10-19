#!/bin/sh
# Builds the Debian package with cargo-deb (install once: cargo install cargo-deb).
# The unit file in packaging/ is written for /usr/local/bin; the package installs to /usr/bin.
set -eu
cd "$(dirname "$0")/.."

command -v cargo-deb >/dev/null || { echo "cargo-deb is missing: cargo install cargo-deb" >&2; exit 1; }
mkdir -p target/deb-assets
sed 's|/usr/local/bin/eliezer|/usr/bin/eliezer|' packaging/eliezer.service > target/deb-assets/eliezer.service

cargo deb -p eliezer --locked "$@"

deb=$(ls -t target/debian/eliezer_*.deb | head -n 1)
echo "Package: $deb"
dpkg-deb --info "$deb"
dpkg-deb --contents "$deb"
