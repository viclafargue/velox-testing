#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

set -a
# shellcheck source=defaults.env
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/defaults.env"
set +a

exec python3 "${SCRIPT_DIR}/run_scaling_experiment.py" "$@"
