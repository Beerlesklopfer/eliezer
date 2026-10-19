#!/bin/sh
# Create a private runtime directory on the volatile /run file system.
set -eu
umask 077
mkdir -p /run/eliezer
exec /usr/local/bin/eliezer "$@"
