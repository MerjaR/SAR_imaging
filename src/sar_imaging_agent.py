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
import os
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
from utils import print_tool_result, print_separator, get_api_key

load_dotenv()

MODEL = os.getenv("MODEL_NAME")

if not MODEL:
    raise EnvironmentError("MODEL_NAME not set in .env")

client = anthropic.Anthropic(api_key=get_api_key())

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
        "Get the next real satellite pass times over a location using live TLE "
        "orbital data from Celestrak. Returns upcoming passes within 24 hours "
        "with max elevation and pass duration."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "lat": {
                "type": "number",
                "description": "Latitude of the location"
            },
            "lon": {
                "type": "number",
                "description": "Longitude of the location"
            },
            "city": {
                "type": "string",
                "description": "City name, used for labelling the result"
            }
        },
        "required": ["lat", "lon", "city"]
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
    Fetches live weather data from OpenWeatherMap API.
    Returns temperature, conditions, cloud cover and daylight status.
    """
    api_key = os.environ.get("OPENWEATHERMAP_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENWEATHERMAP_API_KEY not set. Add it to your .env file."
        )

    resp = requests.get(
        "https://api.openweathermap.org/data/2.5/weather",
        params={"q": city, "appid": api_key, "units": "metric"},
        timeout=5
    )

    if resp.status_code == 404:
        return {"error": f"City '{city}' not found in OpenWeatherMap"}
    if resp.status_code != 200:
        return {"error": f"Weather API error: {resp.status_code}"}

    data = resp.json()

    return {
        "temp_c": round(data["main"]["temp"], 1),
        "condition": data["weather"][0]["description"].capitalize(),
        "cloud_cover_pct": data["clouds"]["all"],
        "wind_kmh": round(data["wind"]["speed"] * 3.6, 1),
        "is_daylight": data["dt"] < data["sys"]["sunset"] and data["dt"] > data["sys"]["sunrise"],
        "lat": data["coord"]["lat"],
        "lon": data["coord"]["lon"],
    }


def get_satellite_passes(lat: float, lon: float, city: str) -> dict:
    """
    Fetches live TLE orbital data from Celestrak and computes real
    satellite pass times using the SGP4 propagator model via skyfield.

    Uses the ISS as a representative LEO satellite for demo purposes — similar orbital
    altitude (~400km) to many Earth observation constellations.
    """
    try:
        from skyfield.api import load, wgs84, EarthSatellite

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
        return {
            "error": f"Live TLE fetch failed: {str(e)}",
            "details": str(e)
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
        result = get_satellite_passes(
            tool_input["lat"],
            tool_input["lon"],
            tool_input["city"]
        )
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

    for _ in range(5):
        response = client.messages.create(
            model=MODEL,
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
    print("   Live weather via OpenWeatherMap · Orbital data via Celestrak + skyfield")
    print("   Type a city or location query, or 'exit' to quit.\n")
    print("   Example queries:")
    print("   → Is now a good time to image Helsinki?")
    print("   → Compare imaging windows for London and Dubai")
    print("   → When is the next viable pass over Tokyo?\n")

    while True:
        user_input = input("Analyst: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ["exit", "quit"]:
            print("\nShutting down agent. Goodbye! 🛰️")
            break
        result = agent(user_input)
        print(f"\n🤖 Agent: {result}\n")
        print_separator()