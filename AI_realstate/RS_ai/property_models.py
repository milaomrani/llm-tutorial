from typing import Dict, List, Optional
from pydantic import BaseModel, Field

class PropertyData(BaseModel):
    """Schema for property data extraction"""
    building_name: str = Field(description="Name of the building/property", alias="Building_name")
    property_type: str = Field(description="Type of property (commercial, residential, etc)", alias="Property_type")
    location_address: str = Field(description="Complete address of the property")
    price: str = Field(description="Price of the property in CAD", alias="Price")
    description: str = Field(description="Detailed description of the property", alias="Description")
    bedrooms: Optional[int] = Field(description="Number of bedrooms", default=None)
    bathrooms: Optional[int] = Field(description="Number of bathrooms", default=None)
    size_sqft: Optional[int] = Field(description="Size in square feet", default=None)
    year_built: Optional[int] = Field(description="Year the property was built", default=None)
    amenities: Optional[List[str]] = Field(description="List of amenities", default=None)

class PropertiesResponse(BaseModel):
    """Schema for multiple properties response"""
    properties: List[PropertyData] = Field(description="List of property details")

class LocationData(BaseModel):
    """Schema for location price trends"""
    location: str
    price_per_sqft: float
    percent_increase: float
    rental_yield: float

class LocationsResponse(BaseModel):
    """Schema for multiple locations response"""
    locations: List[LocationData] = Field(description="List of location data points")

class FirecrawlResponse(BaseModel):
    """Schema for Firecrawl API response"""
    success: bool
    data: Dict
    status: str
    expiresAt: str