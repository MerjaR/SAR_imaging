"""
SAR Imaging Intelligence Agent
================================
A multi-tool AI agent that determines optimal
satellite imaging windows by chaining:

  1. get_weather()           — cloud cover & conditions
  2. get_satellite_passes()  — real pass times via Celestrak TLE + skyfield
  3. assess_imaging_window() — SAR vs optical suitability scoring

Usage:
    python src/sar_imaging_agent.py

Requirements:
    pip install -r requirements.txt
"""

import anthropic
import json
import requests
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
from utils import print_tool_result, print_separator, get_api_key

load_dotenv()

client = anthropic.Anthropic(api_key=get_api_key())


# -------------------------------------------------------
# CITY COORDINATES (lat, lon)
# -------------------------------------------------------
CITY_COORDS = {
    "helsinki":  (60.1699, 25.0384),
    "tokyo":     (35.6762, 139.6503),
    "london":    (51.5074, -0.1278),
    "new york":  (40.7128, -74.0060),
    "dubai":     (25.2048, 55.2708),
    "oslo":      (59.9139, 10.7522),
    "stockholm": (59.3293, 18.0686),
    "warsaw":    (52.2297, 21.0122),
}


# -------------------------------------------------------
# TOOL DEFINITIONS
# -------------------------------------------------------
tools = [
    {
        "name": "get_weather",
        "description": (
            "Get the current weather for a city, including cloud cover percentage "
            "which is critical for determining optical satellite imaging suitability."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "The city name"}
            },
            "required": ["city"]
        }
    },
    {
        "name": "get_satellite_passes",
        "description": (
            "Get the next real satellite pass times over a city using live TLE "
            "orbital data from Celestrak. Returns upcoming passes within 24 hours "
            "with max elevation and pass duration."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "The city name"}
            },
            "required": ["city"]
        }
    },
    {
        "name": "assess_imaging_window",
        "description": (
            "Assess whether conditions are suitable for SAR or optical satellite "
            "imaging given cloud cover and an upcoming satellite pass. "
            "SAR (Synthetic Aperture Radar) penetrates clouds and works day/night. "
            "Optical imaging requires clear skies and daylight."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "cloud_cover_pct": {
                    "type": "number",
                    "description": "Cloud cover as a percentage (0-100)"
                },
                "minutes_to_pass": {
                    "type": "number",
                    "description": "Minutes until the next satellite pass"
                },
                "max_elevation_deg": {
                    "type": "number",
                    "description": "Maximum elevation of the pass in degrees (higher = better)"
                },
                "is_daylight": {
                    "type": "boolean",
                    "description": "Whether it is currently daylight at the target location"
                }
            },
            "required": ["cloud_cover_pct", "minutes_to_pass", "max_elevation_deg", "is_daylight"]
        }
    }
]


# -------------------------------------------------------
# TOOL IMPLEMENTATIONS
# -------------------------------------------------------

def get_weather(city: str) -> dict:
    """
    Simulated weather with realistic cloud cover values.
    In production, swap for a live API e.g. OpenWeatherMap.
    """
    fake_weather = {
        "helsinki":  {"temp_c": 4,  "condition": "Cloudy with light rain", "cloud_cover_pct": 85, "wind_kmh": 15},
        "tokyo":     {"temp_c": 25, "condition": "Sunny and warm",         "cloud_cover_pct": 10, "wind_kmh": 8},
        "london":    {"temp_c": 11, "condition": "Overcast",               "cloud_cover_pct": 90, "wind_kmh": 20},
        "new york":  {"temp_c": 15, "condition": "Partly cloudy",          "cloud_cover_pct": 40, "wind_kmh": 12},
        "dubai":     {"temp_c": 38, "condition": "Hot and sunny",          "cloud_cover_pct": 5,  "wind_kmh": 5},
        "oslo":      {"temp_c": 3,  "condition": "Snow showers",           "cloud_cover_pct": 95, "wind_kmh": 18},
        "stockholm": {"temp_c": 6,  "condition": "Mostly cloudy",          "cloud_cover_pct": 75, "wind_kmh": 14},
        "warsaw":    {"temp_c": 9,  "condition": "Partly cloudy",          "cloud_cover_pct": 45, "wind_kmh": 11},
    }

    data = fake_weather.get(city.lower(), {
        "temp_c": 20,
        "condition": f"Partly cloudy in {city}",
        "cloud_cover_pct": 35,
        "wind_kmh": 10
    })

    # Determine daylight based on UTC hour (simplified: 06:00-18:00 UTC)
    utc_hour = datetime.now(timezone.utc).hour
    data["is_daylight"] = 6 <= utc_hour <= 18

    return data


def get_satellite_passes(city: str) -> dict:
    """
    Fetches live TLE orbital data from Celestrak and computes real
    satellite pass times using the SGP4 propagator model via skyfield.

    Uses the ISS as a representative LEO satellite — similar orbital
    altitude (~400km) to many Earth observation constellations.
    """
    try:
        from skyfield.api import load, wgs84, EarthSatellite

        coords = CITY_COORDS.get(city.lower())
        if not coords:
            return {
                "error": f"City '{city}' not in database.",
                "tip": f"Known cities: {', '.join(CITY_COORDS.keys())}"
            }

        lat, lon = coords

        # Fetch live TLE data from Celestrak (no auth required)
        resp = requests.get(
            "https://celestrak.org/NORAD/elements/gp.php?GROUP=stations&FORMAT=tle",
            timeout=5
        )

        # Parse TLE lines for ISS
        lines = [l.strip() for l in resp.text.strip().splitlines() if l.strip()]
        iss_tle = None
        for i, line in enumerate(lines):
            if "ISS (ZARYA)" in line or "ISS" in line:
                if i + 2 < len(lines):
                    iss_tle = (lines[i], lines[i+1], lines[i+2])
                    break

        if not iss_tle:
            raise ValueError("Could not parse ISS TLE from Celestrak response")

        # Build satellite and observer objects
        ts = load.timescale()
        satellite = EarthSatellite(iss_tle[1], iss_tle[2], iss_tle[0], ts)
        location = wgs84.latlon(lat, lon)

        # Search the next 24 hours for passes above 10° elevation
        now = datetime.now(timezone.utc)
        t0 = ts.from_datetime(now)
        t1 = ts.from_datetime(now + timedelta(hours=24))

        times, events = satellite.find_events(location, t0, t1, altitude_degrees=10.0)

        # events: 0 = rise, 1 = culmination (max elevation), 2 = set
        passes = []
        current_pass = {}
        for ti, event in zip(times, events):
            if event == 0:
                current_pass = {"rise_time": ti.utc_datetime()}
            elif event == 1:
                diff = satellite - location
                topocentric = diff.at(ti)
                alt, az, _ = topocentric.altaz()
                current_pass["max_elevation_deg"] = round(alt.degrees, 1)
            elif event == 2:
                current_pass["set_time"] = ti.utc_datetime()
                if "rise_time" in current_pass and "max_elevation_deg" in current_pass:
                    rise = current_pass["rise_time"]
                    minutes_away = (rise - now).total_seconds() / 60
                    duration_sec = (current_pass["set_time"] - rise).total_seconds()
                    passes.append({
                        "rise_utc": rise.strftime("%H:%M UTC"),
                        "minutes_until_pass": round(minutes_away, 1),
                        "duration_seconds": round(duration_sec),
                        "max_elevation_deg": current_pass["max_elevation_deg"],
                    })
                current_pass = {}
            if len(passes) >= 3:
                break

        if not passes:
            return {
                "city": city,
                "passes": [],
                "note": "No passes found in next 24h above 10° elevation"
            }

        return {
            "city": city,
            "coordinates": {"lat": lat, "lon": lon},
            "next_pass": passes[0],
            "upcoming_passes": passes,
            "data_source": "Celestrak live TLE / skyfield SGP4"
        }

    except ImportError:
        return {"error": "skyfield not installed. Run: pip install -r requirements.txt"}

    except Exception as e:
        # Graceful fallback with simulated data if network or parse fails
        import random
        minutes = random.randint(15, 90)
        return {
            "city": city,
            "next_pass": {
                "rise_utc": (datetime.now(timezone.utc) + timedelta(minutes=minutes)).strftime("%H:%M UTC"),
                "minutes_until_pass": minutes,
                "duration_seconds": random.randint(300, 600),
                "max_elevation_deg": round(random.uniform(15, 75), 1),
            },
            "note": f"Live TLE fetch failed ({e}), using simulated fallback",
            "data_source": "simulated"
        }


def assess_imaging_window(
    cloud_cover_pct: float,
    minutes_to_pass: float,
    max_elevation_deg: float,
    is_daylight: bool
) -> dict:
    """
    Assesses SAR vs optical imaging suitability for an upcoming pass.

    SAR  — works through clouds, smoke, and total darkness.
    Optical — requires clear skies (< 20% cloud cover) and daylight.
    """

    # Pass elevation quality (affects image geometry and resolution)
    if max_elevation_deg >= 60:
        elevation_quality = "excellent"
    elif max_elevation_deg >= 30:
        elevation_quality = "good"
    else:
        elevation_quality = "low — high incidence angle, wider swath coverage"

    # SAR assessment — only blocked by geometry, not weather
    if max_elevation_deg < 10:
        sar_suitable = False
        sar_reason = "Pass elevation too low for viable imaging geometry"
    elif minutes_to_pass < 0:
        sar_suitable = False
        sar_reason = "Pass already occurred"
    else:
        sar_suitable = True
        sar_reason = (
            f"SAR unaffected by {cloud_cover_pct}% cloud cover. "
            f"Pass in {round(minutes_to_pass)} min at {max_elevation_deg}° "
            f"elevation ({elevation_quality})."
        )

    # Optical assessment — blocked by clouds or darkness
    if not is_daylight:
        optical_suitable = False
        optical_reason = "No daylight — optical imaging not possible"
    elif cloud_cover_pct > 20:
        optical_suitable = False
        optical_reason = (
            f"Cloud cover {cloud_cover_pct}% exceeds optical threshold (≤20%)"
        )
    else:
        optical_suitable = True
        optical_reason = (
            f"Clear enough ({cloud_cover_pct}% cloud cover) with daylight — optical viable"
        )

    # Tasking urgency based on time to pass
    if minutes_to_pass <= 30:
        urgency = f"⚡ IMMINENT — pass in {round(minutes_to_pass)} min. Task now!"
    elif minutes_to_pass <= 120:
        urgency = f"Pass in {round(minutes_to_pass)} min. Prepare tasking."
    else:
        urgency = f"Next pass in {round(minutes_to_pass / 60, 1)}h. Time to plan."

    # Overall recommendation
    if sar_suitable and optical_suitable:
        recommendation = "Both SAR and optical viable — optimal imaging conditions"
    elif sar_suitable:
        recommendation = "SAR recommended — optical blocked by weather or darkness"
    elif optical_suitable:
        recommendation = "Optical only — check SAR geometry for next pass"
    else:
        recommendation = "No imaging recommended for this pass"

    return {
        "sar_imaging": {
            "suitable": sar_suitable,
            "reason": sar_reason,
            "note": "SAR penetrates clouds, smoke, and darkness"
        },
        "optical_imaging": {
            "suitable": optical_suitable,
            "reason": optical_reason,
        },
        "pass_quality": elevation_quality,
        "recommendation": recommendation,
        "urgency": urgency,
    }


# -------------------------------------------------------
# TOOL ROUTER
# -------------------------------------------------------
def run_tool(tool_name: str, tool_input: dict) -> str:
    """Execute a tool by name and return the result as a JSON string."""
    if tool_name == "get_weather":
        result = get_weather(tool_input["city"])
    elif tool_name == "get_satellite_passes":
        result = get_satellite_passes(tool_input["city"])
    elif tool_name == "assess_imaging_window":
        result = assess_imaging_window(
            tool_input["cloud_cover_pct"],
            tool_input["minutes_to_pass"],
            tool_input["max_elevation_deg"],
            tool_input["is_daylight"]
        )
    else:
        result = {"error": f"Unknown tool: {tool_name}"}

    print_tool_result(tool_name, result)
    return json.dumps(result)


# -------------------------------------------------------
# SYSTEM PROMPT
# -------------------------------------------------------
SYSTEM_PROMPT = """You are an AI-powered satellite imaging intelligence assistant for ICEYE,
the world's leading SAR (Synthetic Aperture Radar) satellite operator.

Your job is to help analysts determine the best imaging windows by combining:
- Real-time weather and cloud cover data
- Live satellite pass predictions using actual orbital TLE data
- Expert assessment of SAR vs optical imaging suitability

Key facts you always apply:
- SAR satellites image through clouds, rain, smoke, and total darkness
- Optical satellites need clear skies (< 20% cloud cover) and daylight
- Higher satellite elevation = better image geometry and resolution
- Passes above 30° elevation are generally preferred for tasking

Always be specific, actionable, and explain your reasoning clearly."""


# -------------------------------------------------------
# AGENT LOOP
# -------------------------------------------------------
def agent(user_message: str) -> str:
    """
    Full agentic loop — Claude decides which tools to call and
    in what order, iterating until it has enough information
    to produce a final response.
    """
    print_separator()
    print(f"👤 Analyst: {user_message}")
    print_separator()

    messages = [{"role": "user", "content": user_message}]

    while True:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            tools=tools,
            messages=messages,
        )

        # No more tool calls — return the final text response
        if response.stop_reason == "end_turn":
            for block in response.content:
                if block.type == "text":
                    return block.text
            return "(No text response)"

        # Collect ALL tool calls from this response
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                print(f"\n🛰️  Calling: {block.name}({json.dumps(block.input)})")
                result = run_tool(block.name, block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                })

        # Send all tool results back in one message
        if tool_results:
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})


# -------------------------------------------------------
# MAIN
# -------------------------------------------------------
if __name__ == "__main__":
    print("\n🛰️  SAR Imaging Intelligence Agent")
    print("   Live orbital data via Celestrak TLE + skyfield SGP4\n")

    # --- Single query mode ---
    questions = [
        "Is now a good time to image Helsinki? Give me a full SAR vs optical assessment.",
        # "Compare imaging windows for London and Dubai — which is better right now?",
        # "When is the next viable imaging pass over Tokyo?",
    ]

    for q in questions:
        result = agent(q)
        print(f"\n🤖 Agent: {result}\n")
        print_separator()

    # --- Interactive mode (uncomment to enable) ---
    # print("\n💬 Interactive mode (type 'exit' to quit)\n")
    # while True:
    #     user_input = input("Analyst: ").strip()
    #     if user_input.lower() in ["exit", "quit"]:
    #         break
    #     if user_input:
    #         result = agent(user_input)
    #         print(f"\n🤖 Agent: {result}\n")