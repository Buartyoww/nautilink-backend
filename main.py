import requests
import torch
import torch.nn.functional as F
from fastapi import FastAPI, Response
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from torch_geometric.nn import GATConv
from torch_geometric.data import Data

# IMPORT THE LAND DETECTOR
from global_land_mask import globe 

app = FastAPI()

# 1. CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. CONFIG
API_KEY = "02e38c641b8cd114189aa85394c4c9cc" 
BASE_URL = "https://api.openweathermap.org/data/2.5/weather"

# 3. AI MODEL (Updated 32-neuron architecture)
class NautilinkGNN(torch.nn.Module):
    def __init__(self):
        super(NautilinkGNN, self).__init__()
        self.conv1 = GATConv(4, 32, heads=2) 
        self.conv2 = GATConv(32 * 2, 16, heads=1)
        self.out = torch.nn.Linear(16, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.elu(x); x = self.conv2(x, edge_index); x = F.elu(x)
        x = self.out(x)
        return x

# 4. LOAD BRAIN
model = NautilinkGNN()
try:
    model.load_state_dict(torch.load("nautilink_model.pth", map_location=torch.device('cpu')))
    model.eval()
    print("✅ Brain Loaded")
except Exception as e:
    print(f"❌ Brain Error: {e}")

class Location(BaseModel):
    lat: float
    lon: float

# Helper for fish names
def get_fish_species(temp, wind):
    if temp >= 28 and temp <= 30: return "Yellowfin Tuna (Tambakol)"
    elif temp > 30: return "Mahi-Mahi (Dorado)"
    elif wind > 20: return "Blue Marlin (Malasugi)"
    elif temp < 27: return "Bigeye Tuna"
    else: return "Skipjack Tuna"

@app.get("/")
def home():
    return {"message": "Nautilink AI Online"}

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(content=None, status_code=204)

@app.post("/analyze-zone")
async def analyze_zone(loc: Location):
    
    # --- ALWAYS GET WEATHER FIRST (even on land) ---
    url = f"{BASE_URL}?lat={loc.lat}&lon={loc.lon}&appid={API_KEY}&units=metric"
    resp = requests.get(url)
    
    # Default weather values
    temp = 0
    wind = 0
    pres = 1013
    cloud = 0
    weather_condition = "Unknown"
    
    if resp.status_code == 200:
        w_data = resp.json()
        temp = w_data["main"]["temp"]
        wind = w_data["wind"]["speed"] * 3.6  # Convert to kph
        pres = w_data["main"]["pressure"]
        cloud = w_data["clouds"]["all"]
        weather_condition = w_data["weather"][0]["description"]
    
    # --- LOGIC 1: LAND DETECTION ---
    is_on_land = globe.is_land(loc.lat, loc.lon)
    
    if is_on_land:
        # ON LAND - No fishing, but SHOW WEATHER DATA
        return {
            "weather_summary": {
                "condition": weather_condition,  # Real weather!
                "temp_c": round(temp, 1),        # Real temperature!
                "wind_kph": round(wind, 1)       # Real wind!
            },
            "safety": {
                "status": "Safe",
                "message": "You are on land. Go to water to fish."
            },
            "fishing_forecast": {
                "catch_rate_percent": 0,
                "estimated_kg": 0,
                "target_species": "None",
                "rating": "On Land"  # Changed from "No Fishing on Land"
            }
        }

    # --- IF OCEAN -> CONTINUE WITH AI PREDICTION ---
    
    # B. SAFETY LOGIC
    safety_status = "Safe"
    safety_msg = "Conditions optimal for fishing."
    if wind > 45: 
        safety_status = "Danger"
        safety_msg = "GALE WARNING! Return to shore immediately."
    elif wind > 30:
        safety_status = "Caution"
        safety_msg = "Rough seas expected. Be careful."
    elif wind > 20:
        safety_status = "Safe"
        safety_msg = "Moderate winds. Good for fishing."

    # C. AI PREDICTION
    features = [temp, wind, pres, cloud]
    x = torch.tensor([features], dtype=torch.float)
    edge_index = torch.tensor([[0], [0]], dtype=torch.long)
    data = Data(x=x, edge_index=edge_index)
    
    with torch.no_grad():
        prediction = model(data)
    
    predicted_kg = round(prediction.item(), 1)
    
    # --- LOGIC 2: 150KG MAX CAPACITY & PERCENTAGE ---
    MAX_CAPACITY = 150.0  # Limit for small pump boat
    
    # Calculate Percentage
    catch_percentage = (predicted_kg / MAX_CAPACITY) * 100
    
    # Cap at 100% (If AI predicts 160kg, it just means boat is full)
    catch_percentage = min(100.0, catch_percentage)
    catch_percentage = max(0.0, catch_percentage)  # No negative
    catch_percentage = round(catch_percentage, 1)

    # Determine Rating Text
    if catch_percentage >= 90:
        rating_text = "Excellent"
    elif catch_percentage >= 70:
        rating_text = "Very Good"
    elif catch_percentage >= 50:
        rating_text = "Good"
    elif catch_percentage >= 30:
        rating_text = "Average"
    elif catch_percentage >= 10:
        rating_text = "Low"
    else:
        rating_text = "Very Low"

    return {
        "weather_summary": {
            "condition": weather_condition,
            "temp_c": round(temp, 1),
            "wind_kph": round(wind, 1)
        },
        "safety": {
            "status": safety_status,
            "message": safety_msg
        },
        "fishing_forecast": {
            "catch_rate_percent": catch_percentage,
            "estimated_kg": predicted_kg,
            "target_species": get_fish_species(temp, wind),
            "rating": rating_text
        }
    }