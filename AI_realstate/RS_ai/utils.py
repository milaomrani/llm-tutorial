import re
import logging
from typing import List, Dict, Any, Optional
from urllib.parse import quote_plus

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PropertyUtils")

def sanitize_input(input_str: str) -> str:
    """
    Sanitize user input to prevent injection attacks
    
    Args:
        input_str: The input string to sanitize
        
    Returns:
        str: Sanitized input string
    """
    # Return empty string if input is None
    if not input_str:
        return ""
        
    # Remove potentially dangerous characters
    sanitized = re.sub(r'[<>"\'`;()&|]', '', input_str)
    return sanitized

def format_price(price_value: float) -> str:
    """
    Format price value in thousands to a readable string
    
    Args:
        price_value: Price in thousands of CAD
        
    Returns:
        str: Formatted price string
    """
    if price_value >= 1000:
        # Convert to millions with 2 decimal places
        return f"${price_value/1000:.2f} million CAD"
    else:
        # Keep as thousands
        return f"${int(price_value)} thousand CAD"

def format_location_string(city: str, province: str) -> str:
    """
    Format location string based on province selection
    
    Args:
        city: City name
        province: Province name
        
    Returns:
        str: Formatted location string
    """
    return f"{city}, {province}" if province != "All Canada" else f"{city}, Canada"

def get_province_code(province: str) -> str:
    """
    Get the province code from full province name
    
    Args:
        province: Province name
        
    Returns:
        str: Province code (e.g., "on" for Ontario)
    """
    province_codes = {
        "Ontario": "on",
        "Quebec": "qc",
        "British Columbia": "bc",
        "Alberta": "ab",
        "Nova Scotia": "ns",
        "New Brunswick": "nb",
        "Manitoba": "mb",
        "Prince Edward Island": "pe",
        "Saskatchewan": "sk",
        "Newfoundland and Labrador": "nl",
        "Northwest Territories": "nt",
        "Yukon": "yt",
        "Nunavut": "nu"
    }
    
    return province_codes.get(province, "")

def generate_property_urls(city: str, province: str, max_price: float) -> List[str]:
    """
    Generate search URLs for property listings based on location and price
    
    Args:
        city: City name
        province: Province name
        max_price: Maximum price in thousands CAD
        
    Returns:
        List[str]: List of search URLs
    """
    # Format location for URLs safely
    formatted_location = quote_plus(city.lower())
    
    # Base URLs that work for all provinces
    urls = [
        f"https://www.realtor.ca/map#view=list&Sort=6-D&PropertyTypeGroupID=1&PropertySearchTypeId=1&TransactionTypeId=2&PriceMin=0&PriceMax={int(max_price*1000)}&Currency=CAD"
    ]
    
    # Province-specific URL handling
    if province == "All Canada":
        urls.extend([
            f"https://www.royallepage.ca/en/search/homes/{formatted_location}/",
            f"https://www.zillow.com/homes/for_sale/{formatted_location}_rb/"
        ])
    elif province == "Quebec":
        urls.extend([
            f"https://www.royallepage.ca/en/qc/search/homes/{formatted_location}/",
            f"https://www.centris.ca/en/properties~for-sale~{formatted_location}?view=Thumbnail",
            f"https://www.centris.ca/fr/proprietes-residentielles~a-vendre~{formatted_location}"
        ])
    elif province == "Ontario":
        urls.extend([
            f"https://www.royallepage.ca/en/on/search/homes/{formatted_location}/",
            f"https://condos.ca/search?for=sale&search_by={formatted_location}",
            f"https://www.zoocasa.com/ontario-real-estate?location={formatted_location}"
        ])
    elif province == "British Columbia":
        urls.extend([
            f"https://www.royallepage.ca/en/bc/search/homes/{formatted_location}/",
            f"https://www.rew.ca/properties/areas/{formatted_location}/sort/price/asc",
            f"https://www.zolo.ca/{formatted_location}-real-estate"
        ])
    elif province == "Alberta":
        urls.extend([
            f"https://www.royallepage.ca/en/ab/search/homes/{formatted_location}/",
            f"https://www.honestdoor.com/search?q={formatted_location}"
        ])
    else:
        # Generic URLs for other provinces
        urls.extend([
            f"https://www.royallepage.ca/en/search/homes/{formatted_location}/",
            f"https://www.point2homes.com/CA/Real-Estate-Listings/{formatted_location}.html"
        ])
    
    return urls
    
def create_mock_properties(
    city: str,
    province: str,
    property_type: str,
    max_price: float,
    num_bedrooms: Optional[int] = None,
    num_bathrooms: Optional[int] = None,
    additional_requirements: str = "",
    count: int = 4
) -> List[Dict[str, Any]]:
    """
    Create mock property data for demonstration purposes
    
    Args:
        city: City name
        province: Province name
        property_type: Type of property (Condo/House/etc)
        max_price: Maximum price in thousands CAD
        num_bedrooms: Number of bedrooms required
        num_bathrooms: Number of bathrooms required
        additional_requirements: Free-text additional requirements
        count: Number of mock properties to create
        
    Returns:
        List[Dict[str, Any]]: List of mock property dictionaries
    """
    # Create location string
    location_string = format_location_string(city, province)
    
    # Common street names in Canadian cities
    street_names = [
        "Rue Saint-Laurent", "Boulevard René-Lévesque", "Avenue du Parc", 
        "Rue Sherbrooke", "Rue Sainte-Catherine", "Avenue Mont-Royal",
        "Yonge Street", "Bay Street", "Bloor Street", "King Street",
        "Robson Street", "Granville Street", "Davie Street",
        "Stephen Avenue", "17th Avenue", "Kensington Road"
    ]
    
    # Property name components
    building_prefixes = [
        "The", "Royal", "Grand", "Park", "City", "River", "Sky", "Elite",
        "Mountain", "Lake", "Urban", "Metro", "Century", "Premium"
    ]
    
    building_types = [
        "Residences", "Condominiums", "Tower", "Riverside", "Heights", "Plaza", 
        "Gardens", "Estates", "Apartments", "Village", "Square", "Lofts", 
        "Terraces", "Place", "Pointe", "Court", "Manor"
    ]
    
    # Amenities by property type
    amenities_by_type = {
        "Condo": [
            "24/7 Concierge", "Rooftop Terrace", "Fitness Center", "Swimming Pool", 
            "Party Room", "Pet Spa", "Visitor Parking", "Bicycle Storage",
            "Security System", "EV Charging Stations", "Guest Suites", "Theater Room"
        ],
        "House": [
            "Backyard", "Garage", "Finished Basement", "Deck/Patio", 
            "Private Driveway", "Garden", "Central AC", "Smart Home Features",
            "Fireplace", "Renovated Kitchen", "Hardwood Floors", "Home Office"
        ],
        "Townhouse": [
            "Private Entrance", "Rooftop Terrace", "Attached Garage", "Community Pool",
            "Playground", "Patio", "BBQ Area", "Low Maintenance", 
            "Open Concept", "Modern Kitchen", "Storage Space", "Guest Parking"
        ],
        "Duplex": [
            "Separate Entrances", "Shared Backyard", "Porch", "Basement", 
            "Income Potential", "Updated Appliances", "Modern Finishes", "Parking",
            "Renovated Bathrooms", "Open Floor Plan", "Storage Space", "Dual HVAC Systems"
        ]
    }
    
    # Neighborhood features by province
    neighborhood_features = {
        "Ontario": ["TTC Access", "Nearby Parks", "Shopping at Eaton Centre", "Restaurants"],
        "Quebec": ["Metro Access", "Old Port Proximity", "Cultural Venues", "Cafés"],
        "British Columbia": ["SkyTrain Access", "Ocean Views", "Stanley Park Proximity", "Trails"],
        "Alberta": ["C-Train Access", "River Valley Views", "Downtown Proximity", "Bike Paths"],
        "Nova Scotia": ["Waterfront Access", "Historic Sites", "Harbor Views", "Boardwalk"],
        "All Canada": ["Public Transit", "Parks", "Shopping", "Dining"]
    }
    
    # Create a variety of mock properties using the submitted criteria
    properties = []
    for i in range(count):
        # Vary the prices
        price_adjustment = (i+1) * 50
        
        # Select random building name components
        building_prefix = building_prefixes[i % len(building_prefixes)]
        building_type = building_types[i % len(building_types)]
        
        # Get street name
        street = street_names[i % len(street_names)]
        
        # Get amenities for this property type
        property_amenities = amenities_by_type.get(property_type, amenities_by_type["Condo"])
        # Select 3-5 random amenities
        selected_amenities = property_amenities[i:i+3] if i < len(property_amenities) else property_amenities[:3]
        
        # Get neighborhood features based on province
        features = neighborhood_features.get(province, neighborhood_features["All Canada"])
        
        # Create address with unit number for condos/townhouses
        unit_prefix = ""
        if property_type in ["Condo", "Townhouse"]:
            unit_prefix = f"Unit {i+101}, "
        
        # Create description
        description = (
            f"Beautiful {num_bedrooms if num_bedrooms else '2'}-bedroom {property_type.lower()} "
            f"in {city}. Features include {', '.join(selected_amenities)}. "
            f"Located near {', '.join(features[:2])}. "
            f"{additional_requirements if additional_requirements else 'Modern amenities and convenient location.'}"
        )
        
        # Calculate size based on property type and bedrooms
        base_size = {
            "Condo": 750,
            "House": 1800,
            "Townhouse": 1200,
            "Duplex": 1600
        }.get(property_type, 1000)
        
        # Adjust size for bedrooms
        bedroom_adjustment = 200 * (num_bedrooms - 2 if num_bedrooms else 0)
        size_sqft = base_size + bedroom_adjustment
        
        properties.append({
            "building_name": f"{building_prefix} {city} {building_type}",
            "property_type": property_type,
            "location_address": f"{unit_prefix}{(i+1)*100} {street}, {location_string}",
            "price": f"${max_price - price_adjustment} thousand CAD",
            "description": description,
            "bedrooms": num_bedrooms if num_bedrooms else 2,
            "bathrooms": num_bathrooms if num_bathrooms else 2,
            "size_sqft": size_sqft,
            "year_built": 2000 + i,
            "amenities": selected_amenities
        })
    
    return properties

def get_property_image_url(property_type: str) -> str:
    """
    Get a placeholder image URL based on property type
    
    Args:
        property_type: Type of property (Condo/House/etc)
        
    Returns:
        str: URL to a placeholder image
    """
    # Define multiple fallback images based on property type
    image_options = {
        "condo": "https://images.unsplash.com/photo-1522708323590-d24dbb6b0267?ixlib=rb-4.0.3&auto=format&fit=crop&w=300&h=200&q=80",
        "house": "https://images.unsplash.com/photo-1576941089067-2de3c901e126?ixlib=rb-4.0.3&auto=format&fit=crop&w=300&h=200&q=80", 
        "townhouse": "https://images.unsplash.com/photo-1625602812206-5ec545ca1231?ixlib=rb-4.0.3&auto=format&fit=crop&w=300&h=200&q=80",
        "duplex": "https://images.unsplash.com/photo-1512917774080-9991f1c4c750?ixlib=rb-4.0.3&auto=format&fit=crop&w=300&h=200&q=80",
        "commercial": "https://images.unsplash.com/photo-1497366754035-f200968a6e72?ixlib=rb-4.0.3&auto=format&fit=crop&w=300&h=200&q=80"
    }
    
    # Get appropriate image for property type or use default
    return image_options.get(property_type.lower(), 
                          "https://images.unsplash.com/photo-1560518883-ce09059eeffa?ixlib=rb-4.0.3&auto=format&fit=crop&w=300&h=200&q=80")

def is_valid_api_key(api_key: str) -> bool:
    """
    Validate API key format to prevent security issues
    
    Args:
        api_key: The API key to validate
        
    Returns:
        bool: Whether the API key is valid
    """
    # Simple validation - can be enhanced based on specific API key formats
    if not api_key or len(api_key) < 10:
        return False
    
    # Only allow alphanumeric characters, dashes and underscores
    if not re.match(r'^[a-zA-Z0-9_-]+, api_key):
        return False
        
    return True