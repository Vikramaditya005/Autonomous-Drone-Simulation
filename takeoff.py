import airsim
import time

def main():
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)

    print("Taking off...")
    client.takeoffAsync().join()
    time.sleep(1)

    print("Climbing to -5 meters")
    client.moveToZAsync(-5, 1).join()
    time.sleep(1)

    print("Moving forward 5 meters")
    client.moveByVelocityAsync(2, 0, 0, 3).join()
    time.sleep(1)
    
    print("Hovering for 2 seconds")
    client.moveByVelocityAsync(2, 0, 0, 3).join()
    client.moveByVelocityAsync(0, 2, 0, 3).join()
    client.moveByVelocityAsync(-2, 0, 0, 3).join()
    client.moveByVelocityAsync(0, -2, 0, 3).join()
    time.sleep(1)

    print("Landing...")
    client.landAsync().join()

    client.armDisarm(False)
    client.enableApiControl(False)
    print("Done.")


if __name__ == "__main__":
    main()
