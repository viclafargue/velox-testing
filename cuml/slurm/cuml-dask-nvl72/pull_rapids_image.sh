#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Pull a container image (e.g. nvcr.io RAPIDS image) and save it as .sqsh
# using enroot on a compute node.
#
# Usage:
#   ./pull_rapids_image.sh <registry/path:tag> [--output <path/to/image.sqsh>] [--overwrite]
#
# Example:
#   ./pull_rapids_image.sh nvcr.io/nvidia/rapidsai/base:26.04-cuda13-py3.14

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/defaults.env"

usage() {
    echo "Usage: $0 <registry/path:tag> [--output <path/to/image.sqsh>] [--overwrite]"
    echo ""
    echo "Examples:"
    echo "  $0 nvcr.io/nvidia/rapidsai/base:26.04-cuda13-py3.14"
    echo "  $0 ghcr.io/org/image:tag --output /scratch/\$USER/images/cuml/custom.sqsh"
    echo ""
    echo "Notes:"
    echo "  - For nvcr.io images, ensure NGC credentials are configured on the cluster."
    echo "  - Default output directory is IMAGE_DIR from defaults.env (${IMAGE_DIR:-unset})."
    exit 1
}

IMAGE_REF=""
OUTPUT_PATH=""
OVERWRITE=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --output|-o)
            [[ -n "${2:-}" ]] || { echo "Error: --output requires a value"; usage; }
            OUTPUT_PATH="$2"
            shift 2
            ;;
        --overwrite)
            OVERWRITE=1
            shift
            ;;
        -*)
            echo "Unknown option: $1"
            usage
            ;;
        *)
            [[ -z "$IMAGE_REF" ]] || { echo "Error: unexpected argument '$1'"; usage; }
            IMAGE_REF="$1"
            shift
            ;;
    esac
done

[[ -n "$IMAGE_REF" ]] || { echo "Error: image reference is required"; usage; }
[[ "$IMAGE_REF" == */* ]] || { echo "Error: image reference must include a registry hostname and path."; exit 1; }

registry="${IMAGE_REF%%/*}"
repo_and_tag="${IMAGE_REF#*/}"

if [[ -z "$registry" || -z "$repo_and_tag" || "$registry" == "$IMAGE_REF" ]]; then
    echo "Error: invalid image reference '${IMAGE_REF}'. Expected format: <registry>/<repo>:<tag>"
    exit 1
fi

ENROOT_URI="docker://${registry}#${repo_and_tag}"

if [[ -z "$OUTPUT_PATH" ]]; then
    [[ -n "${IMAGE_DIR:-}" ]] || { echo "Error: IMAGE_DIR is not set (check defaults.env)"; exit 1; }
    image_slug="${repo_and_tag##*/}"    # image:tag or image@sha256:...
    image_slug="${image_slug//@/-}"     # image-sha256:...
    image_slug="${image_slug//:/-}"     # image-tag / image-sha256-...
    OUTPUT_PATH="${IMAGE_DIR}/${image_slug}.sqsh"
fi

echo "Image:      $IMAGE_REF"
echo "URI:        $ENROOT_URI"
echo "Output:     $OUTPUT_PATH"
echo "Overwrite:  $([[ $OVERWRITE -eq 1 ]] && echo yes || echo no)"
if [[ "$registry" == "nvcr.io" ]]; then
    echo "Registry:   NGC (ensure auth is configured)"
fi
echo ""

ENROOT_DECOMPRESS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/enroot-decompress.sh"
export OUTPUT_PATH ENROOT_URI OVERWRITE

srun --export="ALL,PMIX_MCA_gds=^ds12,ENROOT_GZIP_PROGRAM=${ENROOT_DECOMPRESS}" \
    --nodes=1 --mem=0 --ntasks-per-node=1 \
    --mpi=pmix_v4 \
    bash -c '
set -e
if [[ -f "$OUTPUT_PATH" ]]; then
    size=$(ls -lh "$OUTPUT_PATH" | awk "{print \$5}")
    if [[ "$OVERWRITE" == "1" ]]; then
        echo "Image already exists: $OUTPUT_PATH ($size)"
        echo "--overwrite was passed; removing and re-pulling."
        rm -f "$OUTPUT_PATH"
    else
        echo "Image already exists: $OUTPUT_PATH ($size)"
        echo "Skipping pull. Pass --overwrite to re-pull, or --output to write elsewhere."
        exit 0
    fi
fi
mkdir -p "$(dirname "$OUTPUT_PATH")"
enroot import --output "$OUTPUT_PATH" "$ENROOT_URI"
echo ""
echo "Saved: $(ls -lh "$OUTPUT_PATH")"
'
