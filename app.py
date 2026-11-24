import os
import datetime
import requests
import numpy as np
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium
from streamlit_js_eval import get_geolocation
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from gtts import gTTS

st.set_page_config(
    page_title="AI మహిళా & చిన్నారుల సురక్షిత మార్గదర్శి",
    page_icon="🛡️",
    layout="centered"
)

st.title("🛡️ AI మహిళా & చిన్నారుల సురక్షిత మార్గదర్శి")
st.write("మీ ప్రస్తుత లొకేషన్‌ను GPS ద్వారా ఆటోమేటిక్‌గా గుర్తించి, మీరు వెళ్లాల్సిన గమ్యస్థానానికి (Destination) అత్యంత సురక్షితమైన దారిని తెలుగు వాయిస్‌లో చూపిస్తుంది.")

# ----------------------------------------------------------
# 1. LOAD DATASET & TRAIN ML MODELS (Random Forest + K-Means)
# ----------------------------------------------------------
@st.cache_resource
def load_and_train_models():
    if os.path.exists("route_safety_dataset.csv"):
        df = pd.read_csv("route_safety_dataset.csv")
    else:
        np.random.seed(42)
        rows = []
        for _ in range(500):
            h = np.random.randint(0, 24)
            shops = np.random.randint(0, 35)
            main_rd = np.random.choice([0, 1])
            light = round(np.random.uniform(0.85, 1.0), 2) if (6 <= h <= 18) else (
                round(np.random.uniform(0.5, 0.85), 2) if main_rd else round(np.random.uniform(0.05, 0.35), 2)
            )
            pol_km = round(np.random.uniform(0.4, 7.5), 1)
            score = (light * 35) + (min(shops, 25)/25 * 30) + (main_rd * 15) + (max(0, (8.0 - pol_km)/8.0) * 20)
            label = 0 if score >= 65 else (1 if score >= 40 else 2)
            rows.append([h, shops, main_rd, light, pol_km, label])
        df = pd.DataFrame(rows, columns=[
            "Hour_of_Day", "Nearby_Shops_Count", "Is_Main_Road",
            "Street_Light_Score", "Police_Station_Dist_Km", "Risk_Label"
        ])

    features = ["Hour_of_Day", "Nearby_Shops_Count", "Is_Main_Road", "Street_Light_Score", "Police_Station_Dist_Km"]
    X = df[features]
    y = df["Risk_Label"]

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X, y)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans.fit(X)

    return rf_model, kmeans

rf_model, kmeans_model = load_and_train_models()

# ----------------------------------------------------------
# 2. REVERSE GEOCODE (GPS -> Place Name) & SMART DESTINATION SEARCH
# ----------------------------------------------------------
HEADERS = {"User-Agent": "WomenChildSafetyStudentProject/1.0"}

def reverse_geocode(lat, lon):
    try:
        url = "https://nominatim.openstreetmap.org/reverse"
        params = {"lat": lat, "lon": lon, "format": "json"}
        res = requests.get(url, params=params, headers=HEADERS, timeout=6).json()
        addr = res.get("address", {})
        village_or_town = (
            addr.get("village") or addr.get("town") or addr.get("suburb") or
            addr.get("city") or addr.get("county") or "మీ ప్రస్తుత ప్రాంతం"
        )
        district = addr.get("state_district") or addr.get("district") or ""
        return f"{village_or_town}, {district}".strip(", ")
    except Exception:
        return f"GPS ({round(lat, 4)}, {round(lon, 4)})"

def geocode_destination(dest_name, near_lat=None, near_lon=None):
    # Smart queries so local landmarks like "vasavi college" or "bus stand" resolve reliably
    clean = dest_name.strip()
    queries = [
        f"{clean}, Andhra Pradesh, India",
        f"Sri {clean}, Tadepalligudem, Andhra Pradesh",
        f"{clean}, West Godavari, Andhra Pradesh",
        clean
    ]
    for q in queries:
        try:
            url = "https://nominatim.openstreetmap.org/search"
            params = {"q": q, "format": "json", "limit": 1}
            if near_lat and near_lon:
                params["viewbox"] = f"{near_lon-0.5},{near_lat+0.5},{near_lon+0.5},{near_lat-0.5}"
            res = requests.get(url, params=params, headers=HEADERS, timeout=6).json()
            if res:
                return float(res[0]["lat"]), float(res[0]["lon"]), res[0]["display_name"]
        except Exception:
            continue
    return None, None, None

def get_osrm_routes(start_lat, start_lon, end_lat, end_lon):
    try:
        url = f"http://router.project-osrm.org/route/v1/driving/{start_lon},{start_lat};{end_lon},{end_lat}"
        params = {"overview": "full", "geometries": "geojson", "alternatives": "true"}
        res = requests.get(url, params=params, timeout=8).json()
        if res.get("code") == "Ok" and res.get("routes"):
            return res["routes"]
    except Exception:
        pass
    return []

def evaluate_route_safety(route_coords, hour, route_index):
    mid_idx = len(route_coords) // 2
    mid_lon, mid_lat = route_coords[mid_idx]

    shops_count = 5 if route_index == 0 else 1
    is_main_road = 1 if route_index == 0 else 0
    police_dist_km = 2.0 if route_index == 0 else 4.8

    try:
        overpass_url = "https://overpass-api.de/api/interpreter"
        query = f"""
        [out:json][timeout:8];
        (
          node(around:700,{mid_lat},{mid_lon})["shop"];
          node(around:700,{mid_lat},{mid_lon})["amenity"~"bus_station|hospital|pharmacy|bank|school|college"];
          way(around:500,{mid_lat},{mid_lon})["highway"~"primary|secondary|trunk"];
          node(around:5000,{mid_lat},{mid_lon})["amenity"="police"];
        );
        out center;
        """
        resp = requests.post(overpass_url, data={"data": query}, timeout=7).json()
        elements = resp.get("elements", [])
        shops = [e for e in elements if "tags" in e and ("shop" in e["tags"] or "amenity" in e["tags"])]
        highways = [e for e in elements if "tags" in e and "highway" in e["tags"]]
        police = [e for e in elements if "tags" in e and e["tags"].get("amenity") == "police"]

        if len(shops) > 0:
            shops_count = len(shops)
        is_main_road = 1 if len(highways) > 0 else (1 if route_index == 0 else 0)

        if police:
            p_lat = police[0].get("lat") or police[0].get("center", {}).get("lat")
            p_lon = police[0].get("lon") or police[0].get("center", {}).get("lon")
            if p_lat and p_lon:
                dist = np.sqrt((mid_lat - p_lat)**2 + (mid_lon - p_lon)**2) * 111.0
                police_dist_km = round(max(0.4, dist), 1)
    except Exception:
        pass

    is_night = (hour >= 19 or hour <= 5)
    if not is_night:
        light_score = 0.95
    elif is_main_road == 1 and shops_count >= 5:
        light_score = 0.78
    elif is_main_road == 1:
        light_score = 0.55
    else:
        light_score = 0.20

    features = [[hour, shops_count, is_main_road, light_score, police_dist_km]]
    pred_label = int(rf_model.predict(features)[0])
    probs = rf_model.predict_proba(features)[0]
    safety_score = round(probs[0] * 100 + probs[1] * 45, 1)

    return {
        "pred_label": pred_label,
        "safety_score": safety_score,
        "shops_count": shops_count,
        "is_main_road": is_main_road,
        "light_score": light_score,
        "police_dist_km": police_dist_km
    }

# ----------------------------------------------------------
# 3. AUTOMATIC GPS LOCATION + DESTINATION-ONLY INPUT
# ----------------------------------------------------------
now = datetime.datetime.now()

# Triggers browser "Allow Location" prompt automatically
loc = get_geolocation()

if loc and "coords" in loc:
    s_lat = loc["coords"]["latitude"]
    s_lon = loc["coords"]["longitude"]
    detected_place = reverse_geocode(s_lat, s_lon)
    st.success(f"📍 **మీ ప్రస్తుత లొకేషన్ గుర్తించబడింది:** {detected_place}  |  🕒 **సమయం:** {now.strftime('%I:%M %p')}")
else:
    # Fallback while waiting for user to click "Allow" on browser location popup
    s_lat, s_lon = 16.8143, 81.5270
    detected_place = "Tadepalligudem / Pedatadepalli ప్రాంతం"
    st.warning("📡 బ్రౌజర్ పైన కనిపించే **'Allow' (Location అనుమతి)** బటన్ నొక్కండి — మీ ప్రస్తుత ప్రాంతాన్ని ఆటోమేటిక్‌గా గుర్తిస్తుంది.")

# ONLY DESTINATION INPUT NEEDED!
dest_place = st.text_input("🏁 మీరు వెళ్లాల్సిన గమ్యస్థానం (Destination మాత్రమే ఇవ్వండి):", placeholder="ఉదా: Sri Vasavi Engineering College లేదా Tanuku Bus Stand")

if st.button("🛡️ అత్యంత సురక్షితమైన దారిని చూపించు (Find Safest Route)", type="primary", use_container_width=True):
    if not dest_place.strip():
        st.warning("దయచేసి మీరు వెళ్లాల్సిన గమ్యస్థానం (Destination) పేరు ఇవ్వండి.")
    else:
        with st.spinner("మీ ప్రస్తుత GPS లొకేషన్ నుండి గమ్యస్థానానికి ఉన్న రహదారులు, వీధి దీపాలు & జనసంచారాన్ని AI విశ్లేషిస్తోంది..."):
            d_lat, d_lon, d_full = geocode_destination(dest_place, s_lat, s_lon)

            if not d_lat:
                st.error("ఆ గమ్యస్థానం మ్యాప్‌లో దొరకలేదు. దయచేసి పూర్తి పేరు ఇవ్వండి (ఉదా: Sri Vasavi Engineering College, Tadepalligudem లేదా Tanuku Bus Stand).")
            else:
                routes = get_osrm_routes(s_lat, s_lon, d_lat, d_lon)
                if not routes:
                    st.error("ఈ రెండు ప్రాంతాల మధ్య రోడ్డు మార్గం కనుగొనబడలేదు.")
                else:
                    evaluated_routes = []
                    for idx, r in enumerate(routes[:3]):
                        coords = r["geometry"]["coordinates"]
                        dist_km = round(r["distance"] / 1000, 1)
                        dur_min = round(r["duration"] / 60)
                        metrics = evaluate_route_safety(coords, now.hour, idx)
                        evaluated_routes.append({
                            "route_num": idx + 1,
                            "coords": coords,
                            "dist_km": dist_km,
                            "dur_min": dur_min,
                            **metrics
                        })

                    evaluated_routes.sort(key=lambda x: x["safety_score"], reverse=True)
                    best_route = evaluated_routes[0]

                    m = folium.Map(location=[(s_lat + d_lat)/2, (s_lon + d_lon)/2], zoom_start=13)
                    folium.Marker([s_lat, s_lon], popup=f"మీ ప్రస్తుత ప్రాంతం: {detected_place}", icon=folium.Icon(color="blue")).add_to(m)
                    folium.Marker([d_lat, d_lon], popup=f"గమ్యస్థానం: {dest_place}", icon=folium.Icon(color="green")).add_to(m)

                    for i, r in enumerate(evaluated_routes):
                        latlon_line = [[c[1], c[0]] for c in r["coords"]]
                        color = "green" if i == 0 else "red"
                        label = f"Safest Route (Score: {r['safety_score']}%)" if i == 0 else f"Alternative Route (Score: {r['safety_score']}%)"
                        folium.PolyLine(latlon_line, color=color, weight=6 if i == 0 else 4, tooltip=label).add_to(m)

                    st_folium(m, width=700, height=380)

                    status_text = (
                        "🟢 అత్యంత సురక్షితం (SAFE)" if best_route["pred_label"] == 0 else
                        ("🟡 జాగ్రత్త అవసరం (CAUTION)" if best_route["pred_label"] == 1 else "🔴 ప్రమాద హెచ్చరిక (HIGH ALERT)")
                    )

                    st.success(
                        f"### ✅ సూచించిన సురక్షిత మార్గం: రూట్ #{best_route['route_num']} — {status_text}\n"
                        f"- **భద్రతా స్కోరు (Safety Score):** {best_route['safety_score']}%\n"
                        f"- **దూరం & సమయం:** {best_route['dist_km']} కి.మీ. (~{best_route['dur_min']} నిమిషాలు)\n"
                        f"- **రహదారి రకం:** {'ప్రధాన రహదారి (Main Road)' if best_route['is_main_road'] else 'గ్రామీణ / అంతర్గత దారి'}\n"
                        f"- **సమీప పోలీస్ స్టేషన్ దూరం:** {best_route['police_dist_km']} కి.మీ."
                    )

                    telugu_voice_msg = (
                        f"మీ ప్రస్తుత ప్రాంతం {detected_place} నుండి {dest_place} వెళ్లడానికి రూట్ నంబర్ {best_route['route_num']} అత్యంత సురక్షితమైనది. "
                        f"ఈ దారి దూరం {best_route['dist_km']} కిలోమీటర్లు. "
                        f"{'ఇది ప్రధాన రహదారి మరియు వెలుతురు బాగుంది కాబట్టి నిరభ్యంతరంగా వెళ్లవచ్చు.' if best_route['pred_label'] == 0 else 'సాయంత్రం లేదా రాత్రి వేళ కాబట్టి సందు దారులలో కాకుండా ప్రధాన రహదారి మీదుగానే వెళ్లండి, మీ కుటుంబ సభ్యులకు లైవ్ లొకేషన్ షేర్ చేయండి.'} "
                        f"అత్యవసర సహాయం కోసం 112 లేదా 181 కి కాల్ చేయండి."
                    )

                    st.subheader("🗣️ తెలుగు వాయిస్ మార్గదర్శి:")
                    st.info(telugu_voice_msg)

                    tts = gTTS(text=telugu_voice_msg, lang="te", slow=False)
                    tts.save("village_safe_route.mp3")
                    st.audio("village_safe_route.mp3", format="audio/mp3", autoplay=True)