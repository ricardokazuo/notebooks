# Raspberry Pi 5 — Windows 11 ARM64 VM Setup Guide

> From zero to RDP-accessible Windows 11 VM on Ubuntu Server host

---

## System Overview

| Component | Details |
|---|---|
| Host Hardware | Raspberry Pi 5 (Cortex-A76, 16GB RAM) |
| Host OS | Ubuntu Server 24.04 (Noble) on SD card |
| Root Filesystem | Migrated to M.2 SSD via PCIe HAT (447GB WD Green) |
| Hypervisor | QEMU/KVM with ARM64 KVM acceleration |
| Guest OS | Windows 11 ARM64 25H2 |
| Display | ramfb (install) → virtio-gpu-pci (runtime) |
| Access | RDP on port 3389 via QEMU NAT |
| VM Memory | 12GB |
| VM vCPUs | 4 |

---

## Phase 1 — Migrate Root Filesystem to M.2 SSD

### 1.1 Verify disk layout

```bash
lsblk -o NAME,SIZE,TYPE,FSTYPE,MOUNTPOINT,MODEL
```

Expected: `mmcblk0p1` = `/boot/firmware`, `mmcblk0p2` = `/`, `sda` = M.2 SSD

### 1.2 Wipe and partition the SSD

```bash
sudo wipefs -a /dev/sda
sudo parted /dev/sda --script mklabel gpt
sudo parted /dev/sda --script mkpart primary ext4 0% 100%
sudo mkfs.ext4 /dev/sda1
```

### 1.3 Mount SSD and rsync root filesystem

```bash
sudo mkdir -p /mnt/ssd
sudo mount /dev/sda1 /mnt/ssd

sudo rsync -aHAXx \
  --exclude=/mnt \
  --exclude=/proc \
  --exclude=/sys \
  --exclude=/dev \
  --exclude=/run \
  --exclude=/tmp \
  --exclude=/lost+found \
  / /mnt/ssd/

sudo mkdir -p /mnt/ssd/{proc,sys,dev,run,tmp,mnt}
sudo chmod 1777 /mnt/ssd/tmp
```

> **Note:** If network config changes after rsync (e.g. static IP change), re-run rsync before rebooting to sync the new config to the SSD.

### 1.4 Get the SSD UUID

```bash
sudo blkid /dev/sda1
# Note the UUID value for the next two steps
```

### 1.5 Update fstab on the SSD

```bash
sudo tee /mnt/ssd/etc/fstab << 'EOF'
UUID=<sda1-uuid>  /               ext4    defaults,noatime  0  1
LABEL=system-boot /boot/firmware  vfat    defaults          0  1
EOF
```

### 1.6 Update cmdline.txt on the SD card

```bash
sudo sed -i 's/root=LABEL=writable/root=UUID=<sda1-uuid>/' /boot/firmware/cmdline.txt

# Verify — must be one single line, no line breaks
sudo cat /boot/firmware/cmdline.txt
```

### 1.7 Reboot and verify

```bash
sudo reboot

# After SSH reconnect:
df -h /                              # Should show /dev/sda1
lsblk -o NAME,SIZE,FSTYPE,MOUNTPOINT
```

---

## Phase 2 — Install QEMU/KVM and Dependencies

### 2.1 Fix any held packages

```bash
sudo apt --fix-broken install
sudo apt full-upgrade -y
```

### 2.2 Install QEMU and supporting packages

```bash
sudo apt install -y qemu-system-arm qemu-utils libvirt-daemon-system \
    libvirt-clients virtinst ovmf seabios vgabios libosinfo-bin
```

### 2.3 Verify KVM and add user to groups

```bash
ls /dev/kvm                          # Must exist

sudo usermod -aG libvirt,kvm $USER

# Log out and back in, then verify:
groups                               # Must show libvirt and kvm
```

> **Important:** The Pi 5's Cortex-A76 CPU (PVR 0xd0b) is not recognized by libvirt's CPU model database. Use QEMU directly instead of `virt-install` to avoid this error.

---

## Phase 3 — Prepare VM Files

### 3.1 Set up ISO directory on the SSD

```bash
sudo mkdir -p /mnt/isos
sudo chown $USER:$USER /mnt/isos
```

### 3.2 Transfer Windows 11 ARM64 ISO

Transfer `Win11_25H2_English_Arm64.iso` to `/mnt/isos/` via WinSCP or SCP:

```bash
scp /path/to/Win11_25H2_English_Arm64.iso user@<pi-ip>:/mnt/isos/
```

### 3.3 Download VirtIO drivers ISO

```bash
wget -O /mnt/isos/virtio-win.iso \
  https://fedorapeople.org/groups/virt/virtio-win/direct-downloads/stable-virtio/virtio-win.iso
```

### 3.4 Copy UEFI firmware vars and create VM disk image

```bash
cp /usr/share/AAVMF/AAVMF_VARS.fd /mnt/isos/win11_VARS.fd
qemu-img create -f qcow2 /mnt/isos/win11arm.qcow2 80G
```

---

## Phase 4 — Launch VM and Install Windows

### 4.1 Start QEMU directly

> **Why not virt-install?** libvirt fails with `Cannot find CPU model with PVR 0xd0b` on the Pi 5. Launching QEMU directly bypasses this entirely.

```bash
cp /usr/share/AAVMF/AAVMF_VARS.fd /mnt/isos/win11_VARS.fd

sudo qemu-system-aarch64 \
  -machine virt,gic-version=2 \
  -accel kvm \
  -cpu host \
  -smp 4 \
  -m 8G \
  -drive if=pflash,format=raw,readonly=on,file=/usr/share/AAVMF/AAVMF_CODE.fd \
  -drive if=pflash,format=raw,file=/mnt/isos/win11_VARS.fd \
  -device qemu-xhci,id=xhci \
  -device usb-kbd,bus=xhci.0 \
  -device usb-tablet,bus=xhci.0 \
  -drive file=/mnt/isos/win11arm.qcow2,if=none,id=hd0,cache=writethrough \
  -device virtio-blk-pci,drive=hd0 \
  -drive file=/mnt/isos/Win11_25H2_English_Arm64.iso,media=cdrom,if=none,id=cd0,readonly=on \
  -device usb-storage,drive=cd0,bus=xhci.0 \
  -drive file=/mnt/isos/virtio-win.iso,media=cdrom,if=none,id=drivercd,readonly=on \
  -device usb-storage,drive=drivercd,bus=xhci.0 \
  -device ramfb \
  -vnc 0.0.0.0:0 \
  -display none \
  -device virtio-net-pci,netdev=net0 \
  -netdev user,id=net0,hostfwd=tcp:0.0.0.0:3389-:3389,hostfwd=tcp:0.0.0.0:80-:80 \
  -daemonize
```

### 4.2 Connect via VNC (headless — no HDMI needed)

```bash
# On your laptop — open SSH tunnel:
ssh -L 5900:localhost:5900 ricardo@<pi-ip>

# Then connect any VNC viewer to: localhost:5900
```

> **Important:** Keyboard input in the UEFI shell and Windows installer requires `-device usb-kbd,bus=xhci.0`. Without it, only the mouse works.

### 4.3 Windows installer — load VirtIO storage driver

When the installer shows **"A media driver your computer needs is missing"**, click Browse and navigate to:

```
VirtIO CD > viostor > w11 > ARM64
```

### 4.4 Bypass TPM and Secure Boot check

When the installer shows **"This PC doesn't meet Windows 11 requirements"**, press `Shift+F10` to open Command Prompt:

```cmd
reg add HKLM\SYSTEM\Setup\LabConfig /v BypassTPMCheck /t REG_DWORD /d 1 /f
reg add HKLM\SYSTEM\Setup\LabConfig /v BypassSecureBootCheck /t REG_DWORD /d 1 /f
```

Click the back arrow and retry.

### 4.5 Bypass Microsoft account requirement (Windows 11 25H2)

On the sign-in screen, press `Shift+F10`:

```cmd
reg add HKLM\SOFTWARE\Microsoft\Windows\CurrentVersion\OOBE /v BypassNRO /t REG_DWORD /d 1 /f
shutdown /r /t 0
```

After reboot, select **"I don't have internet"** to create a local account.

---

## Phase 5 — Install VirtIO Drivers Inside Windows

Open Command Prompt as Administrator inside the VM:

### 5.1 Install all VirtIO drivers

```cmd
pnputil /add-driver E:\*.inf /subdirs /install
pnputil /add-driver E:\NetKVM\w11\ARM64\*.inf /install
pnputil /add-driver E:\Balloon\w11\ARM64\*.inf /install
pnputil /add-driver E:\vioinput\w11\ARM64\*.inf /install
pnputil /add-driver E:\viogpudo\w11\ARM64\*.inf /install
```

### 5.2 Install VirtIO guest tools MSI

```cmd
msiexec /i E:\virtio-win-gt-x64.msi /qn
```

> **Note:** The SPICE guest agent component may fail on ARM64 with error `0x80070643`. This is non-critical — SPICE is not needed when using RDP.

---

## Phase 6 — Enable RDP and Switch to virtio-gpu

### 6.1 Enable Remote Desktop inside Windows

In Administrator PowerShell inside the VM:

```powershell
Set-ItemProperty -Path 'HKLM:\System\CurrentControlSet\Control\Terminal Server' -Name fDenyTSConnections -Value 0
Enable-NetFirewallRule -DisplayGroup "Remote Desktop"
```

Or via GUI: **Settings → System → Remote Desktop → Enable**

### 6.2 Restart QEMU with virtio-gpu-pci

Kill the current QEMU process and restart with `virtio-gpu-pci` for proper display resolution support:

```bash
sudo kill $(pgrep qemu)

sudo qemu-system-aarch64 \
  -machine virt,gic-version=2 \
  -accel kvm \
  -cpu host \
  -smp 4 \
  -m 12G \
  -drive if=pflash,format=raw,readonly=on,file=/usr/share/AAVMF/AAVMF_CODE.fd \
  -drive if=pflash,format=raw,file=/mnt/isos/win11_VARS.fd \
  -device qemu-xhci,id=xhci \
  -device usb-kbd,bus=xhci.0 \
  -device usb-tablet,bus=xhci.0 \
  -drive file=/mnt/isos/win11arm.qcow2,if=none,id=hd0,cache=writethrough \
  -device virtio-blk-pci,drive=hd0 \
  -drive file=/mnt/isos/virtio-win.iso,media=cdrom,if=none,id=drivercd,readonly=on \
  -device usb-storage,drive=drivercd,bus=xhci.0 \
  -device virtio-gpu-pci \
  -device virtio-net-pci,netdev=net0 \
  -netdev user,id=net0,hostfwd=tcp:0.0.0.0:3389-:3389,hostfwd=tcp:0.0.0.0:80-:80 \
  -vnc 0.0.0.0:0 \
  -display none \
  -daemonize
```

### 6.3 Connect via RDP

```
# Windows: Win+R → mstsc → Computer: <pi-ip>:3389
# Linux:   xfreerdp /v:<pi-ip>:3389
```

RDP is dramatically faster than VNC for day-to-day use.

---

## Phase 7 — Create systemd Service for Auto-start

```bash
sudo tee /etc/systemd/system/win11-vm.service << 'EOF'
[Unit]
Description=Windows 11 ARM64 VM
After=network.target

[Service]
Type=forking
User=root
ExecStart=/usr/bin/qemu-system-aarch64 \
  -machine virt,gic-version=2 \
  -accel kvm \
  -cpu host \
  -smp 4 \
  -m 12G \
  -drive if=pflash,format=raw,readonly=on,file=/usr/share/AAVMF/AAVMF_CODE.fd \
  -drive if=pflash,format=raw,file=/mnt/isos/win11_VARS.fd \
  -device qemu-xhci,id=xhci \
  -device usb-kbd,bus=xhci.0 \
  -device usb-tablet,bus=xhci.0 \
  -drive file=/mnt/isos/win11arm.qcow2,if=none,id=hd0,cache=writethrough \
  -device virtio-blk-pci,drive=hd0 \
  -drive file=/mnt/isos/virtio-win.iso,media=cdrom,if=none,id=drivercd,readonly=on \
  -device usb-storage,drive=drivercd,bus=xhci.0 \
  -device virtio-gpu-pci \
  -device virtio-net-pci,netdev=net0 \
  -netdev user,id=net0,hostfwd=tcp:0.0.0.0:3389-:3389,hostfwd=tcp:0.0.0.0:80-:80 \
  -vnc 0.0.0.0:0 \
  -display none \
  -daemonize
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable win11-vm.service
```

### Service management commands

```bash
# Stop VM (prevents auto-restart — do this before shutting down Windows)
sudo systemctl stop win11-vm.service

# Start VM
sudo systemctl start win11-vm.service

# Check status
sudo systemctl status win11-vm.service

# View logs
journalctl -u win11-vm.service -f
```

> **Important:** Always run `sudo systemctl stop win11-vm.service` BEFORE shutting down Windows from inside the VM. Otherwise systemd will restart QEMU immediately after Windows shuts down.

---

## Key Lessons Learned

| Issue | Solution |
|---|---|
| libvirt fails with `PVR 0xd0b` | Use QEMU directly, not `virt-install` |
| Hyper-V enlightenments error | Use `--features hyperv.*=off` or avoid `win11` os-variant |
| VGA ROM files missing | Install `seabios` and `vgabios` packages |
| No display output in VNC | Use `-device ramfb` — not bochs/VGA/virtio — for UEFI ARM |
| Keyboard not captured in VNC | Add `-device usb-kbd,bus=xhci.0` to QEMU command |
| Windows installer can't find disk | Use `usb-storage` for ISOs, not `virtio-scsi` |
| TPM/Secure Boot check fails | Add `LabConfig` registry keys via `Shift+F10` |
| Microsoft account forced (25H2) | Use `BypassNRO` registry key + reboot |
| SPICE agent fails on ARM64 | Non-critical — skip, use RDP instead |
| VNC display sluggish | Switch to RDP on port 3389 for day-to-day access |
| VM restarts after Windows shutdown | Run `systemctl stop win11-vm.service` first |

---

## Final Architecture

```
SD Card (boot only)
└── /boot/firmware → hands off root to SSD

M.2 SSD (root filesystem + VM files)
├── Ubuntu Server 24.04 (/)
└── /mnt/isos/
    ├── win11arm.qcow2      (80GB VM disk)
    ├── win11_VARS.fd       (UEFI boot entries)
    ├── virtio-win.iso      (VirtIO drivers)
    └── Win11_25H2_*.iso    (Windows installer)

QEMU/KVM (systemd service: win11-vm.service)
└── Windows 11 ARM64 25H2 (12GB RAM, 4 vCPUs)
    ├── Port 3389 → RDP access
    ├── Port 80   → HTTP / IIS
    └── Port 5900 → VNC emergency console
```
