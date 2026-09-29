"""
Smoke test to verify that preprocessing modules can be imported correctly.
"""

from eye_tracking_system_tools.preprocessing import BlockSync, OERecording

def main():
    print("Imported BlockSync from:", BlockSync.__module__)
    print("Imported OERecording from:", OERecording.__module__)
    print("OK: imports succeeded")
    return True

if __name__ == "__main__":
    main()
