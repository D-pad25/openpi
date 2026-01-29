#!/usr/bin/env sh
# ssh(1) askpass helper for the dashboard.
#
# IMPORTANT:
# - This file must NEVER contain a password.
# - The password is passed in-memory via the OPENPI_HPC_PASSWORD environment variable.
#
# ssh will execute this program when it needs a password and cannot read from a TTY.

printf '%s\n' "${OPENPI_HPC_PASSWORD:-}"

