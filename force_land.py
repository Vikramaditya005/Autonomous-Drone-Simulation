# force_land.py
import airsim
import time
import sys
import traceback

LAND_TIMEOUT = 20.0   # seconds to wait for landing to complete
POLL_INTERVAL = 0.2   # how often to poll the future

def safe_land(client, timeout=LAND_TIMEOUT):
    try:
        print("Requesting landAsync()...")
        fut = client.landAsync()
        start = time.time()
        while True:
            # Some versions of the Future object don't implement .done() the same
            try:
                if fut.done():
                    print("landAsync() future reports done.")
                    break
            except AttributeError:
                # fallback: try to join briefly without timeout; if it blocks too long the loop will continue
                try:
                    fut.join()
                    print("landAsync() join() returned (no timeout support).")
                    break
                except Exception:
                    # ignore and continue polling
                    pass

            if time.time() - start > timeout:
                print(f"Landing did not complete within {timeout}s.")
                try:
                    # Cancel the future if possible
                    fut.cancel()
                except Exception:
                    pass
                return False
            time.sleep(POLL_INTERVAL)
        return True
    except Exception as e:
        print("Exception while landing:", e)
        traceback.print_exc()
        return False

def main():
    try:
        client = airsim.MultirotorClient()
        client.confirmConnection()
        print("Connected!")

        # request control
        try:
            client.enableApiControl(True)
        except Exception:
            pass

        try:
            # If vehicle is not armed, arm it so it can respond to landing
            client.armDisarm(True)
        except Exception:
            pass

        ok = safe_land(client, timeout=LAND_TIMEOUT)
        if not ok:
            print("Landing failed/timed out. Attempting hover then disarm.")
            try:
                client.hoverAsync().join()
            except Exception:
                time.sleep(1.0)

        # disarm and release control
        try:
            client.armDisarm(False)
        except Exception:
            pass
        try:
            client.enableApiControl(False)
        except Exception:
            pass

        print("Done: attempted landing, disarmed (if possible) and released API control.")
    except Exception as e:
        print("Error during forced landing:", e)
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
