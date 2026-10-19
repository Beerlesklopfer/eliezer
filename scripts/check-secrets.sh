#!/bin/sh
# Looks for secrets where they must never be: log files, the journal of the service and the process
# list (arguments show up in `ps`). Tokens and API keys are found by their shape, and a known secret can
# be named in the environment variable CHECK_SECRET (it is searched literally). The matches are never
# printed, only where they were found.
#
#   scripts/check-secrets.sh [--log FILE]... [--journal UNIT] [--no-ps] [--ps-file FILE]
#
# --ps-file reads a saved process list (one command line per line) instead of running `ps`.
#
# Exit status: 0 nothing found, 1 something found, 2 usage error.
set -u

patterns='hf_[A-Za-z0-9]{20,}
sk-[0-9a-f]{64}
[Bb]earer[[:space:]]+(hf_|sk-)[A-Za-z0-9]{16,}'

found=0
check() { # name, command that prints the text to check
    name=$1
    shift
    text=$("$@" 2>/dev/null) || true
    hits=$(printf '%s\n' "$text" | grep -E -c "$(printf '%s' "$patterns" | tr '\n' '|')" || true)
    if [ "${hits:-0}" -gt 0 ]; then
        echo "FOUND  $name: $hits line(s) with the shape of a token or key"
        found=1
    fi
    if [ -n "${CHECK_SECRET:-}" ]; then
        if printf '%s\n' "$text" | grep -F -q -- "$CHECK_SECRET"; then
            echo "FOUND  $name: the secret named in CHECK_SECRET"
            found=1
        fi
    fi
}

use_ps=1
ps_file=
checked=0
while [ $# -gt 0 ]; do
    case $1 in
        --log)
            [ $# -ge 2 ] || { echo "--log needs a file" >&2; exit 2; }
            if [ -r "$2" ]; then
                check "log $2" cat "$2"
            else
                echo "SKIP   log $2 is not readable" >&2
            fi
            checked=$((checked + 1))
            shift 2
            ;;
        --journal)
            [ $# -ge 2 ] || { echo "--journal needs a unit" >&2; exit 2; }
            check "journal $2" journalctl -u "$2" --no-pager -n 100000
            checked=$((checked + 1))
            shift 2
            ;;
        --no-ps)
            use_ps=0
            shift
            ;;
        --ps-file)
            [ $# -ge 2 ] || { echo "--ps-file needs a file" >&2; exit 2; }
            ps_file=$2
            shift 2
            ;;
        *)
            echo "usage: $0 [--log FILE]... [--journal UNIT] [--no-ps] [--ps-file FILE]" >&2
            exit 2
            ;;
    esac
done
if [ "$use_ps" = 1 ]; then
    if [ -n "$ps_file" ]; then
        check "process list ($ps_file)" cat "$ps_file"
    else
        check "process list (ps)" ps -eo args
    fi
    checked=$((checked + 1))
fi
[ "$checked" -gt 0 ] || { echo "nothing to check" >&2; exit 2; }
if [ "$found" = 0 ]; then
    echo "OK     no token or key found in $checked place(s)"
fi
exit "$found"
