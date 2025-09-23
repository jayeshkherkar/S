from flask import Flask, request, render_template
from cameracount import detect_persons_in_video
import joblib
import random
import numpy as np
import folium
import pandas as pd
import os
import pickle
import base64
from matplotlib import pyplot as plt
import io


app = Flask(__name__)

# ----------------------- Static camera meta -----------------------
"""data = {
    'camera_id': ['C1', 'C2', 'C3', 'C4', 'C5'],
    'latitude':  [23.183291, 23.182894, 23.183351, 23.183259, 23.181846],
    'longitude': [75.766737, 75.765681, 75.768751, 75.768519, 75.767256],
    'people_count': [0]*5
}"""
data = {
      'camera_id': ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10'],
      'latitude': [23.183291, 23.182894, 23.183351, 23.183259, 23.181846,
                  23.181745, 23.179677, 23.177771, 23.176966, 23.176337],
      'longitude': [75.766737, 75.765681, 75.768751, 75.768519, 75.767256,
                    75.768779, 75.769164, 75.768636, 75.770125, 75.769610],
      'people_count': [0]*10
  }
df_cam = pd.DataFrame(data)

# ----------------------- Cache paths -----------------------
CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)
FIRST_MAP_HTML = os.path.join(CACHE_DIR, "first_map.html")
FIRST_MAP_COUNTS = os.path.join(CACHE_DIR, "first_map_counts.pkl")

# ----------------------- Helpers -----------------------
# For YOLO counts
def get_color(count):
    if count > 250:
        return 'red'
    elif 200 < count <= 250:
        return 'orange'
    elif 150 < count <= 200:
        return 'yellow'
    else:
        return 'green'
# For ML model actionable points
"""def get_color1(count):
    if count > 300:
        return 'red'
    elif 250 <= count <= 300:
        return 'orange'
    elif 200 <= count < 250:
        return 'yellow'
    else:
        return 'green'"""
# Load ML model once
model = joblib.load('crowd_rf_model_compressed.joblib')

def build_folium_from_counts(df_counts, title=None):
    """Common renderer for circle markers from df_counts having latitude, longitude, people_count."""
    map_center = [23.1819, 75.7681]
    m = folium.Map(location=map_center, zoom_start=17)
    if title:
        folium.map.Marker(
            map_center,
            icon=folium.DivIcon(html=f'<div style="font-weight:700;font-size:12px">{title}</div>')
        ).add_to(m)

    for _, row in df_counts.iterrows():
        color = get_color(row['people_count'])
        folium.Circle(
            location=[row['latitude'], row['longitude']],
            radius=7,
            popup=f"Camera: {row['camera_id']}<br>People: {row['people_count']}",
            color=color, fill=True, fill_color=color, fill_opacity=0.4
        ).add_to(m)
        folium.Marker(
            location=[row['latitude'], row['longitude']],
            icon=folium.DivIcon(html=f"""<div style="font-size: 10pt">{row['camera_id']}</div>""")
        ).add_to(m)
    return m._repr_html_()

"""(video_path, model_path="yolo11l.pt", conf=0.1,
                            output_path="new_output.mp4", show_window=False,
                            use_colab=False, skip_frames=60, tile_size=1984)"""

def compute_and_cache_first_map():
    """Run YOLO on 5 videos, update counts, save HTML to cache, and return HTML."""
    df_counts = df_cam.copy()
    for i in range(len(df_counts)):
        # Heavy step: run once and cache
        count = detect_persons_in_video(
            f"C{i+1}.webm", "yolo11l.pt", 0.1,
            f"C{i+1}_1output.mp4", False,
            False, 60,1984
        )
        df_counts.at[i, 'people_count'] = int(count)*7

    map_html_1 = build_folium_from_counts(df_counts, title="Live YOLO Counts (cached)")
    with open(FIRST_MAP_HTML, "w", encoding="utf-8") as f:
        f.write(map_html_1)
    with open(FIRST_MAP_COUNTS, "wb") as f:
        pickle.dump(df_counts, f)
    return map_html_1

def load_cached_first_map():
    """Return cached first-map HTML; if missing, compute and cache."""
    if os.path.exists(FIRST_MAP_HTML) and os.path.exists(FIRST_MAP_COUNTS):
        with open(FIRST_MAP_HTML, "r", encoding="utf-8") as f:
            return f.read()
    # Cache miss -> compute once
    return compute_and_cache_first_map()

def load_cached_counts_df():
    if os.path.exists(FIRST_MAP_COUNTS):
        with open(FIRST_MAP_COUNTS, "rb") as f:
            return pickle.load(f)
    # If not present (first run), compute will create it
    compute_and_cache_first_map()
    with open(FIRST_MAP_COUNTS, "rb") as f:
        return pickle.load(f)

def get_actionable_points(total_count):
    total_count = int(total_count)
    if total_count > 4000:
        return [
            f"1. After one hour total count of people will be {total_count}",
            "2. Zone Extremely Overcrowded: Deploy all available security and emergency staff immediately.",
            "3. Announce emergency crowd management protocols over PA system.",
            "4. Close entry points to restrict further crowd inflow.",
            "5. Start medical emergency response teams and evacuation procedures.",
            "6. Maintain clear communication channels and real-time monitoring."
        ]
    elif 2500 < total_count <= 3200:
        return [
            f"1. After one hour total count of people will be {total_count}",
            "2. High occupancy: Deploy additional security and support staff.",
            "3. Monitor entrances/exits closely for bottlenecks.",
            "4. Update display boards with live crowd information.",
            "5. Prepare crowd barrier materials near high-traffic areas.",
            "6. Maintain communication devices and keep teams alert."
        ]
    elif 1500 < total_count <= 2500:
        return [
            f"1. After one hour total count of people will be {total_count}",
            "2. Medium occupancy: Routine patrols in the zone.",
            "3. Keep entry/exit gates clear and functional.",
            "4. Continue live monitoring on screens.",
            "5. Staff should stay visible for visitor assistance.",
            "6. Check walkie-talkie and first-aid logistics."
        ]
    elif 500 < total_count <= 1000:
        return [
            f"1. After one hour total count of people will be {total_count}",
            "2. Low occupancy: Keep minimal staff present for observation.",
            "3. Routine security sweep every 20 minutes.",
            "4. Check equipment and camera feeds for performance.",
            "5. Ensure signage and guidance info is clear for visitors.",
            "6. Prepare for potential crowd increase for next hour."
        ]
    else:
        return [
            f"1. After one hour total count of people will be {total_count}",
            "2. Very low occupancy: Basic patrolling sufficient.",
            "3. Use this time for maintenance checks in the zone.",
            "4. Review previous hour's crowd statistics.",
            "5. Plan team briefing or short training if feasible.",
            "6. Stay ready to scale up as soon as count starts rising."
        ]
# deploy volunteers based on red cameras

def deploy_volunteers(red_cameras):
    volunteers = {}
    Redratio = 1/50
    summed_cameras = sum(red_cameras.values())
    if summed_cameras == 0:
        return {cid: 0 for cid in red_cameras}  # sabko 0 volunteers
    Totalvolunteers = summed_cameras * Redratio
    for cid, count in red_cameras.items():
        volunteers[cid] = round((red_cameras[cid] / summed_cameras) * Totalvolunteers)
    return volunteers


# ----------------------- Routes -----------------------
@app.route("/J")
def home_alias():
    return render_template("index.html", map_html_1=load_cached_first_map(), map_html_2=None)

@app.route("/Analytics")
def Analytics_page():
    return render_template("Analytics.html")
@app.route("/", methods=["GET"])
def index():
    # Page visit: show only First Map from cache
    map_html_1 = load_cached_first_map()
    return render_template("index.html", map_html_1=map_html_1, map_html_2=None)

# (Optional) Manual refresh endpoint if you ever want to recompute YOLO map
@app.route("/refresh-first-map", methods=["GET", "POST"])
def refresh_first_map():
    map_html_1 = compute_and_cache_first_map()
    return render_template("index.html", map_html_1=map_html_1, map_html_2=None)

@app.route("/ML-Input", methods=["POST"])
def get_detail_ML():
    # ----- Read form inputs safely -----
    date_str = request.form.get('Date')      # expected dd-mm-yyyy
    time_str = request.form.get('Time')      # expected HH:MM
    is_peak_hour_raw = request.form.get('is_peak_hour', '0')
    rain_chance_raw = request.form.get('rain_chance', '0')
    event_type = request.form.get('event_type', 'Normal')
    dayofweek_raw = request.form.get('Dayofweek', '0')

    # Validate/convert numeric inputs with defaults
    try:
        is_peak_hour = int(is_peak_hour_raw)
    except:
        is_peak_hour = 0
    try:
        rain_chance = float(rain_chance_raw)
    except:
        rain_chance = 0.0
    try:
        dayofweek = int(dayofweek_raw)
    except:
        dayofweek = 0

    # Parse date/time robustly
    try:
        parts = date_str.split('-')
        if len(parts) == 3:
            day = int(parts[0])
            month = int(parts[1])
            year = int(parts[2])
        else:
            # fallback defaults
            day, month, year = 1, 1, 1970
    except Exception as e:
        day, month, year = 1, 1, 1970

    try:
        hour_str, minute_str = time_str.split(':')
        hour = int(hour_str); minute = int(minute_str)
    except:
        hour, minute = 0, 0

    # One-hot for event_type
    event_type_normal = event_type_start_day = event_type_weekend = event_type_shahi_snan = event_type_parv_snan = 0
    if event_type == 'Normal':
        event_type_normal = 1
    elif event_type == 'Start_day':
        event_type_start_day = 1
    elif event_type == 'Weekend':
        event_type_weekend = 1
    elif event_type == 'Shahi_Snan':
        event_type_shahi_snan = 1
    elif event_type == 'Parv_Snan':
        event_type_parv_snan = 1

    # ----- Feature baseline (fixed lat/long order) -----
    latitude = 23.1793
    longitude = 75.7849
    zone_area = 150

    # Ensure model exists
    global model
    if model is None:
        try:
            model = joblib.load('crowd_rf_model_compressed.joblib')
        except Exception as e:
            # handle missing model gracefully
            map_html_1 = load_cached_first_map()
            return render_template("index.html", map_html_1=map_html_1, map_html_2=None,
                                   predicted_points=[f"Model load error: {e}"])

    df_counts1 = df_cam.copy()
    camera_details = {}
    total_people_pred = 0

    # ----- Predict per camera -----
    for camera_id in range(len(df_cam)):
        zone_pressure = random.randint(1, 85)
        entry_minus_exit = random.randint(40, 70)
        feats = [
            latitude, longitude, zone_area, zone_pressure, is_peak_hour, rain_chance,
            event_type_parv_snan, event_type_shahi_snan, event_type_start_day,
            event_type_weekend, camera_id, hour, day, month, dayofweek, minute,
            entry_minus_exit
        ]
        feat_array = np.array(feats, dtype=float).reshape(1, -1)
        try:
            count = int(model.predict(feat_array)[0])
        except Exception as e:
            count = 0
        df_counts1.at[camera_id, 'people_count'] = int(count)
        camera_details[f"C{camera_id+1}"] = int(count)
        #total_people_pred += int(count)

    # Build map for ML predictions
    map_html_2 = build_folium_from_counts(df_counts1, title="Counts predicted by ML model")

    # Determine red cameras using get_color
    red_cameras = {cid:cnt for cid,cnt in camera_details.items() if get_color(cnt) == 'red'}
    volunteers = deploy_volunteers(red_cameras)  
    # Use total_people_pred to compute global actionable points too (if you want)
    action_points = []
    
    if red_cameras:

        if len(red_cameras) == len(df_counts1):

            action_points.append('1. All cameras are showing red alert. Immediate action required!')
            action_points.extend(['2. Increase the frequency of monitoring in all zones.',
            f'3. Since all 10 cameras have reached the red level, volunteers will be deployed as follows: C1-{volunteers['C1']}, C2-{volunteers['C2']} volunteers, C3-{volunteers['C3']} volunteers, C4-{volunteers['C4']} volunteers, C5-{volunteers['C5']} volunteers, C6-{volunteers['C6']} volunteers, C7-{volunteers['C7']} volunteers, C8-{volunteers['C8']} volunteers, C9-{volunteers['C9']} volunteers, C10-{volunteers['C10']} volunteers.',
            '4. Transfer the people towards the hold zone through zig-zag barricades.',
            '5. Close the regular barricades and switch the people towards zig-zag barricades to control the crowd.',
            '6. Increase the speed of exit towards gates with the help of security personnel.'])
            
        else:
            action_points.append(f'1. In the map there are total {len(red_cameras)} cameras showing red alert in Mahankal Lok zone.')
            action_points.append(f'2. Cameras which showing red alert are: {", ".join(red_cameras.keys())}')
            action_points.extend([
                '3. Close the regular barricades and switch the people towards zig-zag barricades to control the crowd.',
                '4. After that transfer the people towards the hold zone through zig-zag barricades.',
                '5. Deploy additional security personnel from other zones to manage the crowd.',
                '6. Increase the speed of exit towards gates with the help of security personnel.'

        ])
    else:
        # Also give global recommendation based on total_people_pred
        action_points = get_actionable_points(total_people_pred)

    # Map 1: from cache
    map_html_1 = load_cached_first_map()

    return render_template("index.html",
                           map_html_1=map_html_1,
                           map_html_2=map_html_2,
                           predicted_points=action_points,
                           predicted_total=total_people_pred)


# ----------------------- (Your Analytics route unchanged) -----------------------
@app.route("/Dashboard")
def dashboard():
    return render_template("index.html", map_html_1=load_cached_first_map(), map_html_2=None)

@app.route("/footage")
def footage():
    return render_template("Footage.html")
if __name__ == '__main__':
    app.run(debug=True)
