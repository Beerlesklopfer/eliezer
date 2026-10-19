#!/bin/sh
# Installs the Debian package in a clean debian:trixie container (no network inside) and checks:
# user creation, files, service not enabled, config check, service start, reload, reinstall,
# remove (keeps config and user) and purge (removes config, keeps user and /var/lib/eliezer).
# Usage: tests/deb/check-package.sh [path/to/eliezer_*.deb]   (needs podman and apt-get download)
set -eu
cd "$(dirname "$0")/../.."
deb=${1:-$(ls -t target/debian/eliezer_*.deb | head -n 1)}
dl=$(mktemp -d)
trap 'rm -rf "$dl"' EXIT
(cd "$dl" && apt-get download adduser >/dev/null)   # the minimal image lacks adduser

podman run --rm --network none \
    -v "$(realpath "$deb"):/pkg.deb:ro" -v "$dl:/dl:ro" \
    docker.io/library/debian:trixie bash -c '
set -eu
fail() { echo "FAIL: $*"; exit 1; }
dpkg -i /dl/adduser*.deb >/dev/null 2>&1
dpkg -i /pkg.deb >/dev/null 2>&1 || fail "install"
getent passwd eliezer >/dev/null || fail "system user missing"
[ "$(stat -c %a /etc/eliezer/config.toml)" = 644 ] || fail "config mode"
[ -x /usr/bin/eliezer ] || fail "binary missing"
grep -q "^ExecStart=/usr/bin/eliezer " /usr/lib/systemd/system/eliezer.service || fail "unit path"
[ ! -e /etc/systemd/system/multi-user.target.wants/eliezer.service ] || fail "service must not be enabled"
mkdir -p /var/lib/eliezer && chown eliezer:eliezer /var/lib/eliezer
run() { setpriv --reuid=eliezer --regid=eliezer --clear-groups "$@"; }
run /usr/bin/eliezer --check-config /etc/eliezer/config.toml | grep -q "Configuration OK" || fail "check-config"
# no shell function here: the signals must reach the service itself
setpriv --reuid=eliezer --regid=eliezer --clear-groups /usr/bin/eliezer --config /etc/eliezer/config.toml >/tmp/svc.log 2>&1 &
pid=$!
sleep 1
req() { exec 3<>/dev/tcp/127.0.0.1/1234; printf "GET $1 HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n" >&3; cat <&3; exec 3<&-; }
req /healthz | grep -q "^ok" || fail "healthz"
req /api/v1/status | grep -q "\"models\":\[\]" || fail "status"
kill -HUP $pid; sleep 0.5
kill -TERM $pid; wait $pid || fail "service exit code"
grep -q config_reloaded /tmp/svc.log || fail "reload not logged"
dpkg -i /pkg.deb >/dev/null 2>&1 || fail "reinstall"
dpkg -r eliezer >/dev/null 2>&1
[ -f /etc/eliezer/config.toml ] || fail "remove must keep the config"
dpkg -P eliezer >/dev/null 2>&1
[ ! -e /etc/eliezer ] || fail "purge must remove the config"
getent passwd eliezer >/dev/null || fail "purge must keep the user"
[ -d /var/lib/eliezer ] || fail "purge must keep the state directory"
echo "package check passed"
'
