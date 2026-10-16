#!/bin/bash
# Launch the vendored FINN Docker environment and drop into this repo root.

set -e

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CANDIDATES=()

# Default Xilinx toolchain location/version for the shared FINN build host.
# Allow callers to override these before launching the script.
export FINN_XILINX_PATH="${FINN_XILINX_PATH:-/home/xilinx}"
export FINN_XILINX_VERSION="${FINN_XILINX_VERSION:-2024.2}"
export FINN_TCLLIBPATH="${FINN_TCLLIBPATH:-/home/xilinx/Vivado/2024.2/data/XilinxTclStore/tclapp}"
export TCLLIBPATH="${TCLLIBPATH:-/home/xilinx/Vivado/2024.2/data/XilinxTclStore/tclapp}"
export VITIS_PATH="${VITIS_PATH:-/home/xilinx/Vitis/2024.2}"
export XILINX_XRT="${XILINX_XRT:-/opt/xilinx/xrt}"
export PLATFORM_REPO_PATHS="${PLATFORM_REPO_PATHS:-/opt/xilinx/platforms}"

if [ -n "${HPPS_FINN_ROOT:-}" ]; then
  CANDIDATES+=("$HPPS_FINN_ROOT")
fi

CANDIDATES+=(
  "$REPO_DIR/ignore/finn-src"
  "$HOME/finn"
  "$HOME/FINN"
  "$(dirname "$REPO_DIR")/finn"
)

FINN_RUN_DOCKER=""
FINN_ROOT=""
for candidate in "${CANDIDATES[@]}"; do
  if [ -f "$candidate/run-docker.sh" ]; then
    FINN_RUN_DOCKER="$candidate/run-docker.sh"
    FINN_ROOT="$candidate"
    break
  fi
done

if [ -z "$FINN_RUN_DOCKER" ]; then
  echo "[ERROR] Could not find a FINN checkout with run-docker.sh."
  echo "Searched:"
  for candidate in "${CANDIDATES[@]}"; do
    echo "  - $candidate"
  done
  echo
  echo "Clone FINN somewhere accessible, e.g.:"
  echo "  git clone https://github.com/Xilinx/finn.git \$HOME/finn"
  echo
  echo "Then either rerun this script, or set:"
  echo "  export HPPS_FINN_ROOT=/path/to/finn"
  exit 1
fi

# Mount this repo into the FINN container as-is so local build scripts and models
# are accessible from the same absolute path inside the container. Override the
# container workdir to the repo root when launching the default interactive shell.
# XILINX_LOCAL_USER_DATA=no: each container starts with an empty ~/.Xilinx, and parallel HLS workers
# racing to install the Tcl app store there made some IP exports fail ("Unable to load Tcl app
# xilinx::xsim"), which surfaced later as "IP definition not found" in CreateStitchedIP.
export FINN_DOCKER_EXTRA="${FINN_DOCKER_EXTRA:-} -v $REPO_DIR:$REPO_DIR -w $REPO_DIR -e TCLLIBPATH=$FINN_TCLLIBPATH -e XILINX_LOCAL_USER_DATA=no "

cd "$FINN_ROOT"

if [ $# -eq 0 ]; then
  exec "$FINN_RUN_DOCKER"
else
  exec "$FINN_RUN_DOCKER" "$@"
fi

export TCLLIBPATH="${TCLLIBPATH:-/home/xilinx/Vivado/2024.2/data/XilinxTclStore/tclapp}"
