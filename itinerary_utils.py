import os
import json

try:
    from groq import Groq
    _GROQ_AVAILABLE = True
except ImportError:
    _GROQ_AVAILABLE = False

try:
    import folium
    _FOLIUM_AVAILABLE = True
except ImportError:
    _FOLIUM_AVAILABLE = False


MODEL_NAME = "openai/gpt-oss-20b"

# City coordinates for the map (from Member 4's original notebook).
# NOTE: hotel.xlsx has no lat/lng columns, so hotel markers are placed
# at their city's coordinate rather than an exact address pin. If Member 3
# ever adds Lat/Lng columns to hotel.xlsx, this can be upgraded to precise pins.
PAKISTAN_CITIES = {
    "Islamabad": [33.6844, 73.0479],
    "Lahore": [31.5204, 74.3587],
    "Karachi": [24.8607, 67.0011],
    "Peshawar": [34.0151, 71.5249],
    "Quetta": [30.1798, 66.9750],
    "Multan": [30.1575, 71.5249],
    "Faisalabad": [31.4504, 73.1350],
    "Murree": [33.9070, 73.3943],
    "Hunza": [36.3167, 74.6500],
    "Hunza Valley": [36.3167, 74.6500],
    "Skardu": [35.2971, 75.6333],
    "Gilgit": [35.9208, 74.3083],
    "Naran": [34.9093, 73.6507],
    "Swat": [35.2227, 72.4258],
    "Chitral": [35.8518, 71.7864],
    "Abbottabad": [34.1688, 73.2215],
    "Sialkot": [32.4945, 74.5229],
    "Bahawalpur": [29.3956, 71.6836],
}


# ----------------------------------------------------------------------
# API key handling — same pattern as hotel_utils.py, so both AI-using
# modules read the key the same way regardless of where they're deployed.
# ----------------------------------------------------------------------

def _get_api_key():
    try:
        import streamlit as st
        if "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
    except Exception:
        pass
    return os.environ.get("GROQ_API_KEY")


_groq_client = None


def _get_groq_client():
    global _groq_client
    if not _GROQ_AVAILABLE:
        return None
    api_key = _get_api_key()
    if not api_key:
        return None
    if _groq_client is None:
        _groq_client = Groq(api_key=api_key)
    return _groq_client


# ----------------------------------------------------------------------
# Itinerary generation — now reads from the trip payload your Budget +
# Hotel module (app_1_.py) already produces, instead of re-collecting
# trip details from scratch.
#
# Expected `trip_context` shape (this is exactly what app_1_.py's
# `hotel_payload` already looks like):
#   {
#       "user_profile": {"travelers": int, "duration_days": int, "travel_style": str},
#       "destinations": [str, ...],
#       "budget_pkr": {"total": float, ...},
#       "hotel_recommendations": {city: {...}},   # optional but recommended
#   }
# ----------------------------------------------------------------------

def generate_itinerary(trip_context: dict, tourist_name: str = "Traveler",
                        country: str = "", interests: str = ""):
    """
    Returns (itinerary_dict, error_message). Exactly one will be None.
    """
    profile = trip_context.get("user_profile", {})
    destinations = trip_context.get("destinations", [])
    budget_total = trip_context.get("budget_pkr", {}).get("total", 0)

    days = profile.get("duration_days", 0)
    people = profile.get("travelers", 1)

    if not destinations:
        return None, "No destinations found in trip data. Please complete the Budget & Hotels step first."
    if not days or days <= 0:
        return None, "Trip duration must be greater than 0 days."

    client = _get_groq_client()
    if client is None:
        return None, ("Groq API key not configured. Add GROQ_API_KEY in Streamlit "
                       "secrets (or as an environment variable for local testing).")

    # Fold in hotel picks so the AI itinerary is aware of where the tourist
    # is actually staying, instead of inventing unrelated hotel names.
    hotel_recs = trip_context.get("hotel_recommendations", {})
    hotel_lines = []
    for city in destinations:
        city_hotels = hotel_recs.get(city, {}).get("hotels", [])
        if city_hotels:
            hotel_lines.append(f"{city}: staying at {city_hotels[0]['hotel_name']}")
    hotel_context = "\n".join(hotel_lines) if hotel_lines else "Hotel picks not available yet."

    prompt = f"""
You are Safar, an AI travel planner for foreign tourists visiting Pakistan.

Create a realistic {days}-day Pakistan travel itinerary.

TOURIST INFORMATION:

Name: {tourist_name}
Country: {country if country else "Not specified"}
Number of Travelers: {people}
Number of Days: {days}
Total Budget: PKR {budget_total:,.0f}
Cities: {", ".join(destinations)}
Interests: {interests if interests.strip() else "Culture, food, sightseeing and nature"}

CONFIRMED HOTELS (mention arriving/staying at these where relevant, do not invent others):
{hotel_context}

IMPORTANT RULES:

1. Create exactly {days} days.
2. Use only the cities provided by the tourist.
3. Divide the days realistically between the cities.
4. Do not put too many activities in one day.
5. Give morning, afternoon and evening activities.
6. Give an estimated daily cost in PKR.
7. Consider the number of travelers.
8. Keep the overall plan reasonably within the given budget.
9. When changing cities, mention approximate travel time.
10. Include useful cultural and practical tips for foreign tourists.
11. Do not claim that anything is booked or confirmed beyond the hotel stays given above.
12. Clearly treat prices and travel times as estimates.
13. Use simple and easy English.
14. Return ONLY valid JSON.

Use exactly this JSON structure:

{{
    "trip_title": "Pakistan trip title",
    "overview": "Short description of the complete trip",
    "days": [
        {{
            "day": 1,
            "city": "City name",
            "title": "Day title",
            "morning": "Morning activity",
            "afternoon": "Afternoon activity",
            "evening": "Evening activity",
            "estimated_cost_pkr": 0,
            "travel_note": "Travel information"
        }}
    ],
    "tips": [
        "Cultural tip",
        "Safety tip",
        "Useful travel tip"
    ]
}}
"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are Safar, a helpful Pakistan travel planner. Always return valid JSON only."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,
            response_format={"type": "json_object"},
        )
        ai_response = response.choices[0].message.content
        itinerary = json.loads(ai_response)
        return itinerary, None
    except json.JSONDecodeError:
        return None, "AI returned an invalid JSON response. Please try again."
    except Exception as e:
        return None, f"Groq API Error: {e}"


def format_itinerary(data: dict) -> str:
    """Renders the itinerary dict as markdown for st.markdown()."""
    if not data:
        return "No itinerary available."

    output = f"# ✈️ {data.get('trip_title', 'Safar Pakistan Trip')}\n\n"
    output += f"### 📝 Trip Overview\n{data.get('overview', '')}\n\n---\n\n"

    for day in data.get("days", []):
        output += f"## 📅 Day {day.get('day', '')} — {day.get('city', 'Unknown City')}\n\n"
        if day.get("title"):
            output += f"### {day['title']}\n\n"
        output += f"🌅 **Morning**\n{day.get('morning', 'Not specified')}\n\n"
        output += f"☀️ **Afternoon**\n{day.get('afternoon', 'Not specified')}\n\n"
        output += f"🌙 **Evening**\n{day.get('evening', 'Not specified')}\n\n"
        output += f"💰 **Estimated Daily Cost:** PKR {day.get('estimated_cost_pkr', 0):,}\n\n"
        if day.get("travel_note"):
            output += f"🚗 **Travel Note:** {day['travel_note']}\n\n"
        output += "---\n\n"

    output += "## 💡 Tourist Tips\n\n"
    for tip in data.get("tips", []):
        output += f"• {tip}\n\n"

    return output


def create_travel_map(destinations: list, hotel_recommendations: dict = None):
    """
    Builds a Folium map with one marker per destination city, the route
    line between them in order, and (if available) the top hotel pick
    for that city shown in the marker popup.
    """
    if not _FOLIUM_AVAILABLE:
        return None

    coordinates = []
    valid_cities = []
    for city in destinations:
        if city in PAKISTAN_CITIES:
            coordinates.append(PAKISTAN_CITIES[city])
            valid_cities.append(city)

    if not valid_cities:
        travel_map = folium.Map(location=[30.3753, 69.3451], zoom_start=5)
        folium.Marker([30.3753, 69.3451], popup="Pakistan").add_to(travel_map)
        return travel_map

    center_lat = sum(p[0] for p in coordinates) / len(coordinates)
    center_lon = sum(p[1] for p in coordinates) / len(coordinates)

    travel_map = folium.Map(location=[center_lat, center_lon], zoom_start=6, tiles="OpenStreetMap")

    hotel_recs = hotel_recommendations or {}

    for number, city in enumerate(valid_cities, start=1):
        lat, lon = PAKISTAN_CITIES[city]

        top_hotel = None
        city_hotels = hotel_recs.get(city, {}).get("hotels", [])
        if city_hotels:
            top_hotel = city_hotels[0]["hotel_name"]

        popup_text = f"Stop {number}: {city}"
        if top_hotel:
            popup_text += f"<br>🏨 {top_hotel}"

        folium.Marker(
            location=[lat, lon],
            popup=popup_text,
            tooltip=f"{number}. {city}",
        ).add_to(travel_map)

    if len(coordinates) > 1:
        folium.PolyLine(locations=coordinates, weight=5, opacity=0.8, tooltip="Safar Travel Route").add_to(travel_map)

    min_lat = min(p[0] for p in coordinates)
    max_lat = max(p[0] for p in coordinates)
    min_lon = min(p[1] for p in coordinates)
    max_lon = max(p[1] for p in coordinates)
    travel_map.fit_bounds([[min_lat, min_lon], [max_lat, max_lon]])

    return travel_map
