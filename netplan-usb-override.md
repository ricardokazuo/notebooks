# Netplan USB Override — Setup Guide

Change your Ubuntu server's IP by plugging in a USB stick. No reboot required.

---

## How It Works

1. You plug a USB stick containing `50-cloud-init.yaml` into the server
2. A udev rule detects the USB and fires a script
3. The script validates the config, copies it to `/etc/netplan/`, and runs `netplan apply`
4. The new IP is live within seconds

---

## Prerequisites

- Ubuntu Server (22.04 / 24.04)
- USB stick formatted as FAT32
- `50-cloud-init.yaml` placed at the **root** of the USB stick

---

## Step 1 — Create the Script

```bash
sudo nano /usr/local/sbin/netplan-usb-watch.sh
```

Paste the following:

```bash
#!/bin/bash
CONFIG_FILE="50-cloud-init.yaml"
NETPLAN_DIR="/etc/netplan"
MOUNT_POINT="/mnt/usb-netplan"
LOG="/var/log/netplan-usb-watch.log"
GOOD_CONFIG="$NETPLAN_DIR/50-cloud-init.yaml.known-good"

sleep 2
mkdir -p "$MOUNT_POINT"

echo "[$(date)] USB detected, checking for $CONFIG_FILE..." >> "$LOG"

mount /dev/sda1 "$MOUNT_POINT" >> "$LOG" 2>&1

if [ ! -f "$MOUNT_POINT/$CONFIG_FILE" ]; then
    echo "[$(date)] No config found on USB." >> "$LOG"
    umount "$MOUNT_POINT" 2>/dev/null
    exit 0
fi

# Skip if config is unchanged
if diff -q "$MOUNT_POINT/$CONFIG_FILE" "$NETPLAN_DIR/$CONFIG_FILE" > /dev/null 2>&1; then
    echo "[$(date)] Config unchanged, skipping." >> "$LOG"
    umount "$MOUNT_POINT"
    exit 0
fi

# Copy new config
cp "$MOUNT_POINT/$CONFIG_FILE" "$NETPLAN_DIR/$CONFIG_FILE"
chmod 600 "$NETPLAN_DIR/$CONFIG_FILE"
umount "$MOUNT_POINT"

# Validate before applying
if ! netplan generate >> "$LOG" 2>&1; then
    echo "[$(date)] ERROR: Invalid config, aborting." >> "$LOG"
    [ -f "$GOOD_CONFIG" ] && cp "$GOOD_CONFIG" "$NETPLAN_DIR/$CONFIG_FILE" && netplan apply
    exit 1
fi

# Save current as known-good before applying
cp "$NETPLAN_DIR/$CONFIG_FILE" "$GOOD_CONFIG"

echo "[$(date)] Applying new config..." >> "$LOG"
netplan apply >> "$LOG" 2>&1
echo "[$(date)] Done." >> "$LOG"
```

Make it executable:

```bash
sudo chmod +x /usr/local/sbin/netplan-usb-watch.sh
```

---

## Step 2 — Create the udev Rule

```bash
sudo tee /etc/udev/rules.d/99-netplan-usb.rules << 'EOF'
ACTION=="add", SUBSYSTEM=="block", KERNEL=="sda1", RUN+="/usr/bin/systemd-run --no-block /usr/local/sbin/netplan-usb-watch.sh"
EOF

sudo udevadm control --reload-rules
```

---

## Step 3 — Recovery Service (Safety Net)

If networking is lost after a bad config, this service automatically restores the last known-good config 60 seconds after boot.

```bash
sudo tee /usr/local/sbin/netplan-recovery.sh << 'EOF'
#!/bin/bash
sleep 60
if ! ping -c1 -W3 192.168.0.1 > /dev/null 2>&1; then
    echo "[$(date)] No network, restoring known-good config..." >> /var/log/netplan-usb-watch.log
    cp /etc/netplan/50-cloud-init.yaml.known-good /etc/netplan/50-cloud-init.yaml
    netplan apply
fi
EOF

sudo chmod +x /usr/local/sbin/netplan-recovery.sh
```

```bash
sudo tee /etc/systemd/system/netplan-recovery.service << 'EOF'
[Unit]
Description=Netplan recovery if network is lost after boot
After=network.target

[Service]
Type=oneshot
ExecStart=/usr/local/sbin/netplan-recovery.sh

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl enable netplan-recovery.service
sudo systemctl daemon-reload
```

---

## Step 4 — Log Rotation

Prevents the log from growing indefinitely.

```bash
sudo tee /etc/logrotate.d/netplan-usb << 'EOF'
/var/log/netplan-usb-watch.log {
    weekly
    rotate 4
    compress
    missingok
    notifempty
}
EOF
```

---

## Usage

1. Edit `50-cloud-init.yaml` on your PC with the desired static IP
2. Copy it to the **root** of a FAT32 USB stick
3. Plug the USB into the server
4. Wait ~3 seconds
5. SSH into the new IP

---

## Example `50-cloud-init.yaml`

```yaml
network:
  version: 2
  ethernets:
    eth0:
      optional: true
      dhcp4: true
      dhcp6: true
  wifis:
    wlan0:
      optional: true
      dhcp4: false
      addresses:
        - 192.168.0.240/24
      routes:
        - to: default
          via: 192.168.0.1
      nameservers:
        addresses: [1.1.1.1, 8.8.8.8]
      regulatory-domain: "BR"
      access-points:
        "YOUR_WIFI_SSID":
          auth:
            key-management: "psk"
            password: "your_wifi_password"
```

---

## Monitoring

```bash
# Watch the log in real time
tail -f /var/log/netplan-usb-watch.log

# Check current IP
ip a | grep inet
```

---

## Files Reference

| Path | Purpose |
|------|---------|
| `/usr/local/sbin/netplan-usb-watch.sh` | Main script |
| `/etc/udev/rules.d/99-netplan-usb.rules` | udev trigger rule |
| `/usr/local/sbin/netplan-recovery.sh` | Recovery script |
| `/etc/systemd/system/netplan-recovery.service` | Recovery systemd service |
| `/etc/netplan/50-cloud-init.yaml` | Active netplan config |
| `/etc/netplan/50-cloud-init.yaml.known-good` | Last known-good config |
| `/var/log/netplan-usb-watch.log` | Log file |
