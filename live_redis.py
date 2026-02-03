 
import carla
import pygame
import math
import datetime
import os
import json
import time
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras import layers
import pickle
import redis

# CONSTANTS

MAX_STEER_DEG = 35.0

# The Anamoly Threshold can be changed according
# to the sensitivity required for the task
ANOMALY_THRESHOLD = 0.5

MODEL_DIR = "./model"
MODEL_PATH = os.path.join(MODEL_DIR, "autoencoder.keras")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_DB = 0
REDIS_TELEMETRY_KEY = "carla:telemetry"
REDIS_METADATA_KEY = "carla:metadata"
REDIS_STATS_KEY = "carla:stats"

# Utility
def speed_from_velocity(v):
    return math.sqrt(v.x**2 + v.y**2 + v.z**2)

def heading_from_yaw(yaw):
    return yaw % 360

def distance(a, b):
    return math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2 + (a.z - b.z)**2)

def bearing_deg(a, b):
    dx = b.x - a.x
    dy = b.y - a.y
    return math.degrees(math.atan2(dy, dx)) % 360

def follow_vehicle(world, vehicle, dist=8.0, height=3.0):
    
    # Pinning the camera to the vehicle 
    # This will help us able to observe the car at all times
    spectator = world.get_spectator()
    t = vehicle.get_transform()
    yaw = math.radians(t.rotation.yaw)
    loc = t.location + carla.Location(
        x=-dist * math.cos(yaw),
        y=-dist * math.sin(yaw),
        z=height
    )
    spectator.set_transform(carla.Transform(loc, carla.Rotation(pitch=-15, yaw=t.rotation.yaw)))

# ReddisLogger helps manage storing the logs
# If you want you can also use jsons, dicts or external dbs
# to store the data 
class RedisLogger:
    def __init__(self):
        self.r = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_DB,
            decode_responses=True
        )

    def init_session(self, meta):
        self.r.set(REDIS_METADATA_KEY, json.dumps(meta))
        self.r.delete(REDIS_TELEMETRY_KEY)
        self.r.set(REDIS_STATS_KEY, json.dumps({
            "total_frames": 0,
            "anomaly_count": 0,
            "start_time": meta["start_time"]
        }))

    def log(self, entry):
        self.r.rpush(REDIS_TELEMETRY_KEY, json.dumps(entry))

    def update_stats(self, total, anomalies):
        self.r.set(REDIS_STATS_KEY, json.dumps({
            "total_frames": total,
            "anomaly_count": anomalies,
            "anomaly_rate": (anomalies / total * 100) if total else 0,
            "last_update": datetime.datetime.now().isoformat()
        }))

# Load the Model
# If you haven't created a model, it loads the default trained model
def load_or_train():
    model = keras.models.load_model(MODEL_PATH)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

# Feature extraction must match with ./autoencoder.py features
# in order and dtypes passed
def features_from_entry(e):
    return np.array([[
        e["Roll_deg"],
        e["Pitch_deg"],
        e["Yaw_deg"],
        e["Heading_deg"],
        e["Vel_X_mps"],
        e["Vel_Y_mps"],
        e["Vel_Z_mps"],
        e["Speed_mps"],
        e["Steering_deg"],
        e["Throttle_pct"],
        e["Pos_X_m"],
        e["Pos_Y_m"],
        e["Pos_Z_m"],
        e["GPS_Lat"],
        e["GPS_Lon"],
        e["Vib_X"],
        e["Vib_Y"],
        e["Vib_Z"],
        e["Distance_To_Home_m"],
        e["Heading_To_Home_deg"],
        e["Heading_To_NextWP_deg"],
        e["YawRate_degps"]
    ]], dtype=np.float64)

# Main loop
def main():

    # Load model and Initialise redis
    model, scaler = load_or_train()
    redis_log = RedisLogger()
    
    # Run carla locally
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    
    # Car model can be changed according to user taste
    bp = world.get_blueprint_library().filter("vehicle.tesla.model3")[0]
    
    # Use the first spawnpoint
    # Note: In synchronous models, spawn points near junction may 
    # cause the car to immediately crash, this bug has been spotted in lower end or no gpu pcs
    spawn = world.get_map().get_spawn_points()[0]
    vehicle = world.spawn_actor(bp, spawn)
    home = spawn.location

    tm = client.get_trafficmanager()
    
    # Self driving mode
    vehicle.set_autopilot(True, tm.get_port())
    traffic_manager = client.get_trafficmanager()
    
    # Important for lower spec pcs
    traffic_manager.set_synchronous_mode(False)
    
    # Ignore all traffic lights 
    traffic_manager.ignore_lights_percentage(vehicle, 100)  
    
    # Do not ignore signs like stop sign
    traffic_manager.ignore_signs_percentage(vehicle, 0)  
    
    # Do not ignore vehicles nearby
    traffic_manager.ignore_vehicles_percentage(vehicle, 0)  
    
    # Allow lane changing 
    traffic_manager.auto_lane_change(vehicle, True) 
    
    # Maintain some distance between the actors to avoid crashes
    traffic_manager.distance_to_leading_vehicle(vehicle, 2.0) 
    
    
    # Sensors are also something which can be 
    # added based on user prefrences or tasks
    imu = world.spawn_actor(
        world.get_blueprint_library().find("sensor.other.imu"),
        carla.Transform(), attach_to=vehicle
    )
    gnss = world.spawn_actor(
        world.get_blueprint_library().find("sensor.other.gnss"),
        carla.Transform(), attach_to=vehicle
    )

    imu_data = {"gyro": None, "accel": None}
    gnss_data = {"lat": None, "lon": None}

    imu.listen(lambda d: imu_data.update({"gyro": d.gyroscope, "accel": d.accelerometer}))
    gnss.listen(lambda d: gnss_data.update({"lat": d.latitude, "lon": d.longitude}))
    
    # Create session
    redis_log.init_session({
        "vehicle": "Tesla Model 3",
        "map": world.get_map().name,
        "mode": "Autonomous",
        "start_time": datetime.datetime.now().isoformat()
    })
    
    # Optional Code
    # The pygame script is for better visualisation of the current state
    # of the AV, and is completely cosmetic in nature
    pygame.init()
    screen = pygame.display.set_mode((650, 260))
    pygame.display.set_caption("CARLA – Live Anomaly Detection")
    font = pygame.font.SysFont("monospace", 20)
    small = pygame.font.SysFont("monospace", 16)
    clock = pygame.time.Clock()

    start = time.time()
    total = anomalies = 0
    
    # Frame Count is maintained to limit unnecessary flooding of terminal 
    frame_count = 0
    try:
        while True:
            
            # Update the frame count
            # Reset at 1000 frame to avoid large integers
            frame_count += 1
            frame_count %= 1000
            
            
            clock.tick(30)
            screen.fill((15, 15, 15))

            for event in pygame.event.get():
                if event.type == pygame.QUIT or (
                    event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
                ):
                    return
            
            # Get data from the sensors or spectator
            v = vehicle.get_velocity()
            t = vehicle.get_transform()
            c = vehicle.get_control()

            speed = speed_from_velocity(v)
            heading = heading_from_yaw(t.rotation.yaw)
            yaw_rate = math.degrees(imu_data["gyro"].z) if imu_data["gyro"] else 0.0
            vib = imu_data["accel"] if imu_data["accel"] else carla.Vector3D()

            wp = world.get_map().get_waypoint(t.location)
            next_wp = wp.next(5.0)[0]
            
            
            # The entry again must match the same format as 
            # the autoencoder.py as well as the previous snippet in the same file
            entry = {
                "RunTime_s": time.time() - start,
                "Roll_deg": t.rotation.roll,
                "Pitch_deg": t.rotation.pitch,
                "Yaw_deg": t.rotation.yaw,
                "Heading_deg": heading,
                "Speed_mps": speed,
                "Vel_X_mps": v.x,
                "Vel_Y_mps": v.y,
                "Vel_Z_mps": v.z,
                "YawRate_degps": yaw_rate,
                "Steering_deg": c.steer * MAX_STEER_DEG,
                "Throttle_pct": c.throttle * 100,
                "Pos_X_m": t.location.x,
                "Pos_Y_m": t.location.y,
                "Pos_Z_m": t.location.z,
                "GPS_Lat": gnss_data["lat"],
                "GPS_Lon": gnss_data["lon"],
                "Vib_X": vib.x,
                "Vib_Y": vib.y,
                "Vib_Z": vib.z,
                "Distance_To_Home_m": distance(t.location, home),
                "Heading_To_Home_deg": bearing_deg(t.location, home),
                "Heading_To_NextWP_deg": bearing_deg(t.location, next_wp.transform.location)
            }
            
            # Transform the entry
            x = scaler.transform(features_from_entry(entry))
            recon = model.predict(x, verbose=0)
            
            # Determine if the entry is an anomaly or safe
            err = float(np.mean((x - recon) ** 2))
            is_anomaly = bool(err > ANOMALY_THRESHOLD)
            entry["Anomaly"] = is_anomaly
            entry["Reconstruction_Error"] = err

            total += 1
            anomalies += int(is_anomaly)
            
            # save the entry in redis
            redis_log.log(entry)
            
            
            if total % 10 == 0:
                redis_log.update_stats(total, anomalies)

            follow_vehicle(world, vehicle)

            color = (255, 60, 60) if is_anomaly else (60, 255, 60)
            screen.blit(font.render(f"Speed: {speed*3.6:5.1f} km/h", True, (0, 255, 0)), (20, 55))
            screen.blit(font.render(f"Heading: {heading:6.1f}°", True, (0, 200, 255)), (20, 85))
            screen.blit(font.render(f"Status: {'ANOMALY' if is_anomaly else 'Normal'}", True, color), (20, 120))
            screen.blit(small.render(f"Error: {err:.6f}", True, (200, 200, 200)), (20, 155))
            screen.blit(small.render(f"Anomaly Rate: {100*anomalies/max(total,1):.2f}%", True, (255, 200, 0)), (20, 180))
            
            
            # Frame limits to avoid the flooding of the terminal
            if is_anomaly and frame_count % 15 == 0:
                print(f"[ANOMALY DETECTED] - ERROR - {err:.6f} - {time.time()}", flush=True)

            pygame.display.flip()

    finally:
        
        # Clean up and close all 
        vehicle.set_autopilot(False)
        imu.destroy()
        gnss.destroy()
        vehicle.destroy()
        pygame.quit()

if __name__ == "__main__":
    main()

