#!/bin/sh
# Privates Laufzeitverzeichnis auf dem flüchtigen /run-Dateisystem anlegen.
set -eu
umask 077
mkdir -p /run/eliezer
exec /usr/local/bin/eliezer "$@"
