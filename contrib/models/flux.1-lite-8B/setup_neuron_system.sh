#!/usr/bin/env bash
# Bring a machine's Neuron *system* layer to the versions this model was validated on.
#
# The system layer is the half that `nxdi_requirements.txt` does not cover: the kernel
# driver, the runtime, collectives, and the CLI tools. They are apt packages, shared by
# every Python venv on the box (all of them link the same /opt/aws/neuron/lib/libnrt.so.1),
# so getting them wrong shows up as runtime or collectives errors that no amount of
# pip pinning will fix.
#
#   ./setup_neuron_system.sh --check      # report only, change nothing
#   ./setup_neuron_system.sh              # show the plan, ask, then install
#   ./setup_neuron_system.sh --yes        # no prompt (CI)
#
# Notes
#   * runtime-lib and collectives are released in lockstep and must match each other.
#     dkms tracks its own numbering; do not try to align it with the other three.
#   * the driver is a DKMS module, so it is compiled at install time and needs kernel
#     headers for the *running* kernel. Change kernels and you reinstall aws-neuronx-dkms.
#   * upgrading the driver needs the device free. The script refuses rather than
#     breaking a running job.
set -euo pipefail

# Validated on trn2.3xlarge, 2026-09-07. Override on the command line if needed:
#   DKMS_VERSION=2.29.0.0 ./setup_neuron_system.sh
DKMS_VERSION="${DKMS_VERSION:-2.30.2.0}"
RUNTIME_VERSION="${RUNTIME_VERSION:-2.34.10.0-ac18d186d}"
COLLECTIVES_VERSION="${COLLECTIVES_VERSION:-2.34.10.0-74eaafac6}"
TOOLS_VERSION="${TOOLS_VERSION:-2.32.28.0-526c2b7f6}"

CHECK_ONLY=0
ASSUME_YES=0
for arg in "$@"; do
    case "$arg" in
        --check) CHECK_ONLY=1 ;;
        --yes|-y) ASSUME_YES=1 ;;
        -h|--help) sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown option: $arg (try --help)" >&2; exit 2 ;;
    esac
done

say() { printf '\n=== %s\n' "$1"; }
have() { command -v "$1" >/dev/null 2>&1; }

installed_version() { dpkg-query -W -f='${Version}' "$1" 2>/dev/null || true; }

# ---------------------------------------------------------------- 1. report
say "current versions"
declare -A WANT=(
    [aws-neuronx-dkms]="$DKMS_VERSION"
    [aws-neuronx-runtime-lib]="$RUNTIME_VERSION"
    [aws-neuronx-collectives]="$COLLECTIVES_VERSION"
    [aws-neuronx-tools]="$TOOLS_VERSION"
)
NEEDS_CHANGE=()
DRIVER_CHANGES=0
for pkg in aws-neuronx-dkms aws-neuronx-runtime-lib aws-neuronx-collectives aws-neuronx-tools; do
    have_ver="$(installed_version "$pkg")"
    want_ver="${WANT[$pkg]}"
    if [[ "$have_ver" == "$want_ver" ]]; then
        printf '  %-26s %-22s ok\n' "$pkg" "${have_ver:-<missing>}"
    else
        printf '  %-26s %-22s -> %s\n' "$pkg" "${have_ver:-<missing>}" "$want_ver"
        NEEDS_CHANGE+=("$pkg=$want_ver")
        [[ "$pkg" == "aws-neuronx-dkms" ]] && DRIVER_CHANGES=1
    fi
done

if have /opt/aws/neuron/bin/neuron-ls; then
    say "device"
    /opt/aws/neuron/bin/neuron-ls || true
fi

if ((${#NEEDS_CHANGE[@]} == 0)); then
    say "nothing to do — this machine already matches"
    exit 0
fi

if ((CHECK_ONLY)); then
    say "--check given, stopping here"
    printf '  would install: %s\n' "${NEEDS_CHANGE[*]}"
    exit 0
fi

# ------------------------------------------------- 2. refuse if device is busy
if ((DRIVER_CHANGES)); then
    say "checking that no process holds the device (the driver is about to be replaced)"
    BUSY=""
    for fd in /proc/[0-9]*/fd/*; do
        target="$(readlink -- "$fd" 2>/dev/null || true)"
        [[ "$target" == /dev/neuron* ]] || continue
        pid="${fd#/proc/}"; pid="${pid%%/*}"
        BUSY+=" $pid($(cat "/proc/$pid/comm" 2>/dev/null || echo '?'))"
    done
    if [[ -n "$BUSY" ]]; then
        echo "  processes still using /dev/neuron*:$BUSY" >&2
        echo "  stop them first (neuron-ls shows the PIDs), then re-run." >&2
        exit 1
    fi
    echo "  device is free"
fi

# --------------------------------------------------------- 3. apt repo + plan
say "plan"
printf '  sudo apt-get install --allow-downgrades %s\n' "${NEEDS_CHANGE[*]}"
((DRIVER_CHANGES)) && echo "  then reload the neuron kernel module (or reboot)"

if ((!ASSUME_YES)); then
    read -r -p $'\nproceed? [y/N] ' reply
    [[ "$reply" == [yY]* ]] || { echo "aborted"; exit 1; }
fi

if [[ ! -f /etc/apt/sources.list.d/neuron.list ]]; then
    say "adding the Neuron apt repository"
    codename="$(. /etc/os-release && echo "${UBUNTU_CODENAME:-noble}")"
    curl -fsSL https://apt.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB \
        | sudo gpg --dearmor -o /usr/share/keyrings/neuron-keyring.gpg
    echo "deb [signed-by=/usr/share/keyrings/neuron-keyring.gpg] \
https://apt.repos.neuron.amazonaws.com ${codename} main" \
        | sudo tee /etc/apt/sources.list.d/neuron.list >/dev/null
fi

say "apt-get update"
sudo apt-get update -qq

if ((DRIVER_CHANGES)); then
    say "kernel headers for $(uname -r) (the driver is a DKMS module)"
    sudo apt-get install -y "linux-headers-$(uname -r)" dkms
fi

# ------------------------------------------------------------- 4. install
say "installing"
sudo apt-get install -y --allow-downgrades "${NEEDS_CHANGE[@]}"

# --------------------------------------------------------- 5. reload driver
if ((DRIVER_CHANGES)); then
    say "reloading the kernel module"
    if sudo modprobe -r neuron 2>/dev/null && sudo modprobe neuron; then
        echo "  reloaded"
    else
        echo "  could not reload (module in use) — reboot to finish: sudo reboot" >&2
    fi
fi

# ---------------------------------------------------------------- 6. verify
say "verify"
for pkg in aws-neuronx-dkms aws-neuronx-runtime-lib aws-neuronx-collectives aws-neuronx-tools; do
    printf '  %-26s %s\n' "$pkg" "$(installed_version "$pkg")"
done
printf '  %-26s %s\n' "kernel module" "$(lsmod | awk '$1=="neuron"{print "loaded"; found=1} END{if(!found) print "NOT loaded — reboot"}')"
printf '  %-26s %s\n' "device node" "$(ls /dev/neuron0 2>/dev/null || echo 'missing — DKMS build failed or reboot needed')"
printf '  %-26s %s\n' "libnrt" "$(readlink -f /opt/aws/neuron/lib/libnrt.so.1 2>/dev/null || echo missing)"
if have /opt/aws/neuron/bin/neuron-ls; then
    /opt/aws/neuron/bin/neuron-ls || true
fi

say "done"
echo "The Python half is separate: see nxdi_requirements.txt in this directory."
