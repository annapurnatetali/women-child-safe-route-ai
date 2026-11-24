import pandas as pd
import numpy as np

np.random.seed(42)
num_rows = 500

areas = [
    "Main Market Road", "Bus Stand Junction", "Railway Station Road",
    "NH-16 Highway Stretch", "Residential Colony Lane", "Village Outskirts Road",
    "College Road", "Hospital Junction", "Industrial Area Road", "Canal Bund Road"
]

data = []
for _ in range(num_rows):
    area = np.random.choice(areas)
    hour = np.random.randint(0, 24)

    if area in ["Main Market Road", "Bus Stand Junction", "Railway Station Road", "Hospital Junction"]:
        shops_count = np.random.randint(12, 40)
        is_main_road = 1
        police_dist_km = round(np.random.uniform(0.3, 2.0), 1)
    elif area in ["College Road", "NH-16 Highway Stretch", "Residential Colony Lane"]:
        shops_count = np.random.randint(4, 15)
        is_main_road = 1 if area != "Residential Colony Lane" else 0
        police_dist_km = round(np.random.uniform(1.5, 4.0), 1)
    else:
        shops_count = np.random.randint(0, 4)
        is_main_road = 0
        police_dist_km = round(np.random.uniform(3.5, 8.0), 1)

    is_night = (hour >= 19 or hour <= 5)
    if not is_night:
        light_score = round(np.random.uniform(0.85, 1.0), 2)
    elif is_main_road == 1 and shops_count >= 10:
        light_score = round(np.random.uniform(0.65, 0.90), 2)
    elif is_main_road == 1:
        light_score = round(np.random.uniform(0.40, 0.65), 2)
    else:
        light_score = round(np.random.uniform(0.05, 0.35), 2)

    safety_score = (
        (light_score * 35) +
        (min(shops_count, 25) / 25 * 30) +
        (is_main_road * 15) +
        (max(0, (8.0 - police_dist_km) / 8.0) * 20)
    )
    safety_score = round(min(100, max(0, safety_score + np.random.normal(0, 3))), 1)

    if safety_score >= 65:
        risk_label, risk_name = 0, "SAFE"
    elif safety_score >= 40:
        risk_label, risk_name = 1, "CAUTION"
    else:
        risk_label, risk_name = 2, "HIGH_ALERT"

    data.append([area, hour, shops_count, is_main_road, light_score, police_dist_km, safety_score, risk_label, risk_name])

columns = [
    "Area_Name", "Hour_of_Day", "Nearby_Shops_Count", "Is_Main_Road",
    "Street_Light_Score", "Police_Station_Dist_Km", "Safety_Score",
    "Risk_Label", "Risk_Category"
]

df = pd.DataFrame(data, columns=columns)
df.to_csv("route_safety_dataset.csv", index=False)
print("✅ Success: 'route_safety_dataset.csv' created with 500 records!")