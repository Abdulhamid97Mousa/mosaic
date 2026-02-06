#!/usr/bin/env python3
"""
Test USB hub and keyboard devices using standard Linux input subsystem.
This verifies that each keyboard is seen as a separate device.

No special libraries needed - uses built-in Python only.
"""
import os
import glob
import struct
import select
import sys

def list_keyboard_devices():
    """List all keyboard input devices."""
    print("=" * 70)
    print("KEYBOARD DEVICES DETECTED:")
    print("=" * 70)

    keyboards = []

    # Check /dev/input/by-path for keyboard devices
    path_devices = glob.glob('/dev/input/by-path/*kbd')
    for device_path in sorted(path_devices):
        real_path = os.path.realpath(device_path)
        name = os.path.basename(device_path)
        keyboards.append((device_path, real_path, name))
        print(f"✓ {name}")
        print(f"  → {real_path}")
        print()

    print(f"Total: {len(keyboards)} keyboard devices found")
    print("=" * 70)
    print()

    return keyboards


def test_keyboard_access(keyboards):
    """Test if we can access each keyboard device."""
    print("=" * 70)
    print("TESTING DEVICE ACCESS:")
    print("=" * 70)

    accessible = []

    for device_path, real_path, name in keyboards:
        try:
            # Try to open the device in non-blocking mode
            fd = os.open(real_path, os.O_RDONLY | os.O_NONBLOCK)
            os.close(fd)
            print(f"✓ CAN ACCESS: {name}")
            accessible.append((device_path, real_path, name))
        except PermissionError:
            print(f"✗ PERMISSION DENIED: {name}")
            print(f"  Fix: sudo usermod -a -G input {os.getenv('USER')}")
            print(f"  Then: Log out and log back in")
        except Exception as e:
            print(f"✗ ERROR: {name} - {e}")
        print()

    print("=" * 70)
    print()
    return accessible


def test_keyboard_input(keyboards):
    """Test reading input from keyboards to verify they're separate."""
    if not keyboards:
        print("No accessible keyboards. Cannot test input.")
        return

    print("=" * 70)
    print("TESTING KEYBOARD INPUT:")
    print("=" * 70)
    print()
    print("Press keys on DIFFERENT keyboards to verify they're separate devices.")
    print("You should see DIFFERENT device names for different keyboards.")
    print()
    print("Press Ctrl+C to exit.")
    print("=" * 70)
    print()

    # Open all keyboard devices
    device_fds = {}
    for device_path, real_path, name in keyboards:
        try:
            fd = os.open(real_path, os.O_RDONLY | os.O_NONBLOCK)
            device_fds[fd] = (name, real_path)
            print(f"✓ Monitoring: {name}")
        except Exception as e:
            print(f"✗ Cannot monitor {name}: {e}")

    print()
    print("-" * 70)
    print()

    if not device_fds:
        print("No devices to monitor.")
        return

    # Monitor for input events
    try:
        while True:
            # Use select to wait for input on any device
            readable, _, _ = select.select(device_fds.keys(), [], [], 1.0)

            for fd in readable:
                try:
                    # Read input event structure (24 bytes on 64-bit Linux)
                    # struct input_event { timeval, __u16 type, __u16 code, __s32 value }
                    data = os.read(fd, 24)
                    if len(data) == 24:
                        # Unpack the event
                        tv_sec, tv_usec, ev_type, ev_code, ev_value = struct.unpack('llHHi', data)

                        # EV_KEY = 1 (keyboard/button events)
                        # Value: 0=release, 1=press, 2=repeat
                        if ev_type == 1 and ev_value == 1:  # Key press
                            name, real_path = device_fds[fd]
                            print(f"🎹 KEY PRESS on: {name}")
                            print(f"   Device: {real_path}")
                            print(f"   Keycode: {ev_code}")
                            print()
                except Exception as e:
                    print(f"Error reading event: {e}")

    except KeyboardInterrupt:
        print("\n\nTest stopped by user.")

    finally:
        # Close all file descriptors
        for fd in device_fds:
            os.close(fd)


def main():
    """Main test function."""
    print()
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║          USB HUB MULTI-KEYBOARD TEST - MOSAIC PROJECT           ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print()

    # Step 1: List all keyboard devices
    keyboards = list_keyboard_devices()

    if not keyboards:
        print("❌ NO KEYBOARD DEVICES FOUND!")
        print("This might indicate a USB hub problem.")
        sys.exit(1)

    # Step 2: Test access to each device
    accessible = test_keyboard_access(keyboards)

    if not accessible:
        print("❌ NO ACCESSIBLE KEYBOARDS!")
        print("You need to add your user to the 'input' group:")
        print(f"   sudo usermod -a -G input {os.getenv('USER')}")
        print("   Then log out and log back in.")
        sys.exit(1)

    # Step 3: Test live input from each keyboard
    test_keyboard_input(accessible)


if __name__ == "__main__":
    main()
