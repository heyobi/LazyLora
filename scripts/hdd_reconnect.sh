#!/usr/bin/env bash
# After the USB HDD enclosure holding the checkpoint drops off the bus: lazily unmount the
# dead mount, wait for the volume to re-enumerate, mount it from fstab, then verify the
# checkpoint. Run with sudo. Never run ntfsfix on a healthy volume.
#
# The enclosure on the author's machine is a JMicron 152d:0583 ("YzWy Disk Device") holding
# an NTFS volume, and the defaults below describe it. On any other machine set:
#
#   LAZYLORA_HDD_UUID   filesystem UUID of the volume     (blkid -s UUID <dev>)
#   LAZYLORA_HDD_MOUNT  where fstab mounts it             (default /mnt/disk2tb)
#   LAZYLORA_REPO       this checkout                     (default: parent of this script)
#   LAZYLORA_PYTHON     interpreter for check_shards.py   (default: first one found)
#
# The mount point must already have an fstab entry: this script calls `mount <mountpoint>`
# and deliberately does not invent mount options of its own.
set -u

UUID="${LAZYLORA_HDD_UUID:-5E2AEA892AEA5E11}"
MNT="${LAZYLORA_HDD_MOUNT:-/mnt/disk2tb}"
REPO="${LAZYLORA_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

PY="${LAZYLORA_PYTHON:-}"
# An activated virtualenv is what a stranger most likely has, so it comes first; then this
# checkout's own .venv, then the author's, then whatever python3 is on PATH.
if [ -z "$PY" ] && [ -n "${VIRTUAL_ENV:-}" ] && [ -x "$VIRTUAL_ENV/bin/python" ]; then
    PY="$VIRTUAL_ENV/bin/python"
fi
if [ -z "$PY" ]; then
    for cand in "$REPO/.venv/bin/python" "$HOME/venvs/lazylora/bin/python"; do
        [ -x "$cand" ] && { PY="$cand"; break; }
    done
fi
[ -n "$PY" ] || PY="$(command -v python3 || true)"

echo "1. lazy-unmounting stale $MNT"; umount -l "$MNT" 2>/dev/null || true
echo "2. waiting for the volume (UUID $UUID) ..."
for i in $(seq 1 60); do
  dev=$(blkid -U "$UUID" 2>/dev/null) && break; sleep 2
done
[ -n "${dev:-}" ] || { echo "   volume not seen after 2 minutes; check the USB cable / power, then rerun"; exit 1; }
echo "   found $dev"
disk=/dev/$(lsblk -no PKNAME "$dev")
echo "3. SMART health of $disk (USB bridge; may need -d sat):"
smartctl -H "$disk" 2>/dev/null | grep -i "overall" || smartctl -d sat -H "$disk" 2>/dev/null | grep -i "overall" || echo "   SMART not readable through this bridge"
echo "4. mounting"
if ! mount "$MNT" 2>/dev/null; then
  fstype=$(blkid -o value -s TYPE "$dev" 2>/dev/null || true)
  if [ "$fstype" = "ntfs" ] || [ "$fstype" = "ntfs3" ]; then
    echo "   mount failed; volume probably flagged dirty by the disconnect. Clearing the dirty flag (ntfsfix -d) ..."
    ntfsfix -d "$dev" && mount "$MNT" || { echo "   still cannot mount; run chkdsk on Windows or ntfsfix without -d"; exit 1; }
  else
    echo "   mount failed and $dev is '$fstype', not NTFS: ntfsfix does not apply."
    echo "   Check the fstab entry for $MNT and the kernel log, then rerun."
    exit 1
  fi
fi
mount | grep -q "$MNT" && echo "   mounted: $(df -h "$MNT" | tail -1)"
echo "5. checkpoint integrity"
[ -n "$PY" ] || { echo "   no python found; set LAZYLORA_PYTHON to run check_shards.py"; exit 1; }
PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}" "$PY" "$REPO/scripts/check_shards.py"
