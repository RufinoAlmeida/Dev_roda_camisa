import os
import requests
import logging
from typing import Dict, Optional, List

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
API_KEY = os.getenv("api_key")
BASE_URL = "http://api.weatherapi.com/v1/current.json"
DEFAULT_LANGUAGE = "en"
TIMEOUT_SECONDS = 5

# -----------------------------------------------------------------------------
# Logging configuration
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# -----------------------------------------------------------------------------
# Exceptions
# -----------------------------------------------------------------------------
class WeatherAPIError(Exception):
    """Custom exception for Weather API failures."""
    pass

# -----------------------------------------------------------------------------
# Core API Client
# -----------------------------------------------------------------------------
def fetch_weather(city: str, language: str = DEFAULT_LANGUAGE) -> Dict:
    """
    Fetch current weather data for a given city.

    :param city: City name
    :param language: Response language
    :return: Weather data as dictionary
    :raises WeatherAPIError: On request or API failure
    """
    if not API_KEY:
        raise WeatherAPIError("WEATHER_API_KEY is not set in environment variables.")

    params = {
        "key": API_KEY,
        "q": city,
        "lang": language
    }

    try:
        response = requests.get(BASE_URL, params=params, timeout=TIMEOUT_SECONDS)
        response.raise_for_status()
        return response.json()

    except requests.exceptions.RequestException as exc:
        raise WeatherAPIError(f"Failed to fetch weather for {city}: {exc}") from exc

# -----------------------------------------------------------------------------
# Data Normalization
# -----------------------------------------------------------------------------
def normalize_weather(data: Dict) -> Dict:
    """
    Extract and normalize relevant weather information.
    """
    return {
        "city": data["location"]["name"],
        "country": data["location"]["country"],
        "temperature_c": data["current"]["temp_c"],
        "feels_like_c": data["current"]["feelslike_c"],
        "condition": data["current"]["condition"]["text"],
        "humidity": data["current"]["humidity"],
        "wind_kph": data["current"]["wind_kph"],
        "last_updated": data["current"]["last_updated"]
    }

# -----------------------------------------------------------------------------
# Presentation Layer
# -----------------------------------------------------------------------------
def display_weather(info: Dict) -> None:
    """
    Display formatted weather information.
    """
    print("\n----------------------------------------")
    print(f"City: {info['city']} ({info['country']})")
    print(f"Temperature: {info['temperature_c']}°C")
    print(f"Feels Like: {info['feels_like_c']}°C")
    print(f"Condition: {info['condition']}")
    print(f"Humidity: {info['humidity']}%")
    print(f"Wind Speed: {info['wind_kph']} km/h")
    print(f"Last Updated: {info['last_updated']}")
    print("----------------------------------------")

# -----------------------------------------------------------------------------
# Application Entry Point
# -----------------------------------------------------------------------------
def main(cities: List[str]) -> None:
    logging.info("Starting weather data retrieval")

    for city in cities:
        try:
            raw_data = fetch_weather(city)
            weather_info = normalize_weather(raw_data)
            display_weather(weather_info)

        except WeatherAPIError as error:
            logging.error(error)

    logging.info("Weather data retrieval completed")

# -----------------------------------------------------------------------------
# Execution
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    CITIES = [
        "Belo Horizonte",
        "Rio de Janeiro",
        "London",
        "São Paulo"
    ]

    main(CITIES)
