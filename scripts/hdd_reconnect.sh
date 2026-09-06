#!/usr/bin/env bash
# After the USB HDD enclosure (JMicron 152d:0583, "YzWy Disk Device") drops off the bus:
# lazily unmount the dead mount, wait for the volume to re-enumerate, mount it from fstab,
# then verify the checkpoint. Run with sudo. Never run ntfsfix on a healthy volume.
set -u
UUID=5E2AEA892AEA5E11
MNT=/mnt/disk2tb
echo "1. lazy-unmounting stale $MNT"; umount -l $MNT 2>/dev/null || true
echo "2. waiting for the volume (UUID $UUID) ..."
for i in $(seq 1 60); do
  dev=$(blkid -U $UUID 2>/dev/null) && break; sleep 2
done
[ -n "${dev:-}" ] || { echo "   volume not seen after 2 minutes; check the USB cable / power, then rerun"; exit 1; }
echo "   found $dev"
disk=/dev/$(lsblk -no PKNAME "$dev")
echo "3. SMART health of $disk (USB bridge; may need -d sat):"
smartctl -H "$disk" 2>/dev/null | grep -i "overall" || smartctl -d sat -H "$disk" 2>/dev/null | grep -i "overall" || echo "   SMART not readable through this bridge"
echo "4. mounting"
if ! mount $MNT 2>/dev/null; then
  echo "   mount failed; volume probably flagged dirty by the disconnect. Clearing the dirty flag (ntfsfix -d) ..."
  ntfsfix -d "$dev" && mount $MNT || { echo "   still cannot mount; run chkdsk on Windows or ntfsfix without -d"; exit 1; }
fi
mount | grep -q "$MNT" && echo "   mounted: $(df -h $MNT | tail -1)"
echo "5. checkpoint integrity"
PYTHONPATH=/home/ibox/calisma/LazyLora /home/ibox/venvs/lazylora/bin/python /home/ibox/calisma/LazyLora/scripts/check_shards.py
