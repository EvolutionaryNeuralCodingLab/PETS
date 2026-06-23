#!/usr/bin/env bash
# Deprecated wrapper — use scripts/setup_eye_repo_linux.sh
echo "setup_annotator_linux.sh is deprecated. Use setup_eye_repo_linux.sh (env: eye_repo_linux)." >&2
exec bash "$(dirname "$0")/setup_eye_repo_linux.sh"
