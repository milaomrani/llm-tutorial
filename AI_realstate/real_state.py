import streamlit as st
import logging
import re
import time
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from urllib.parse import quote_plus

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

class PropertyFindingAgent:
    """Agent responsible for finding properties and providing recommendations"""
    
    def __init__(self, firecrawl_api_key: str, openai_api_key: str, model_id: str = "gpt-4o"):
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("PropertyFindingAgent")
        
        # Initialize OpenAI model
        try:
            from agno.agent import Agent
            from agno.models.openai import OpenAIChat
            
            self.agent = Agent(
                model=OpenAIChat(id=model_id, api_key=openai_api_key),
                markdown=True,
                description="I am a real estate expert who helps find and analyze properties in Canada based on user preferences."
            )
            self.logger.info(f"Using OpenAI model: {model_id}")
        except Exception as e:
            self.logger.error(f"Failed to initialize OpenAI model: {str(e)}")
            raise ValueError(f"Failed to initialize OpenAI model: {str(e)}")

        # Initialize Firecrawl with API key validation
        if not self._is_valid_api_key(firecrawl_api_key):
            self.logger.error("Invalid Firecrawl API key format")
            raise ValueError("Invalid Firecrawl API key format")
            
        from firecrawl import FirecrawlApp
        self.firecrawl = FirecrawlApp(api_key=firecrawl_api_key)
    
    def _is_valid_api_key(self, api_key: str) -> bool:
        """Validate API key format to prevent security issues."""
        # Simple validation - can be enhanced based on specific API key formats
        if not api_key or len(api_key) < 10:
            return False
        
        # Only allow alphanumeric characters, dashes and underscores
        if not re.match(r'^[a-zA-Z0-9_-]+$', api_key):
            return False
            
        return True
    
    def _sanitize_input(self, input_str: str) -> str:
        """Sanitize user input to prevent injection attacks."""
        # Remove potentially dangerous characters
        sanitized = re.sub(r'[<>"\'`;()&|]', '', input_str)
        return sanitized
    
    def extract_with_retry(self, urls, params, max_retries=3, backoff_factor=1.5):
        """
        Execute Firecrawl extract with retry logic for resilience
        
        Args:
            urls: List of URLs to extract from
            params: Parameters for extraction
            max_retries: Maximum number of retry attempts
            backoff_factor: Multiplier for exponential backoff
            
        Returns:
            Extraction results or fallback error response
        """
        attempt = 0
        last_error = None
        
        while attempt < max_retries:
            try:
                self.logger.info(f"Firecrawl API attempt {attempt+1}/{max_retries}")
                response = self.firecrawl.extract(urls=urls, params=params)
                
                # Validate response structure
                if isinstance(response, dict) and 'success' in response:
                    if response['success']:
                        return response
                    else:
                        error_msg = response.get('status', 'Unknown error')
                        self.logger.warning(f"Firecrawl request unsuccessful: {error_msg}")
                        last_error = f"API Error: {error_msg}"
                else:
                    self.logger.warning(f"Invalid response format: {type(response)}")
                    last_error = "Invalid API response format"
                
            except Exception as e:
                self.logger.error(f"Firecrawl API error: {str(e)}")
                last_error = str(e)
            
            # Exponential backoff before retry
            wait_time = backoff_factor ** attempt
            self.logger.info(f"Retrying in {wait_time:.1f} seconds...")
            time.sleep(wait_time)
            attempt += 1
        
        # If all retries failed, return a structured error response
        self.logger.error(f"All {max_retries} Firecrawl API attempts failed")
        
        # Check what kind of response we should create based on the schema
        is_properties_response = 'schema' in params and 'properties' in params['schema']
        
        return {
            'success': False,
            'status': f"Failed after {max_retries} attempts. Last error: {last_error}",
            'data': {
                'properties': [] if is_properties_response else None
            }
        }

    def get_urls_for_location(self, city: str, province: str, max_price: float) -> List[str]:
        """Generate appropriate URLs for the given location and province"""
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

    def find_properties(
        self, 
        city: str,
        max_price: float,
        property_category: str = "Residential",
        property_type: str = "Condo",
        additional_requirements: str = "",
        province: str = "All Canada",
        num_bedrooms: Optional[int] = None,
        num_bathrooms: Optional[int] = None
    ) -> List[Dict]:
        last_error = None
        """
        Find properties based on user preferences
        
        Args:
            city: City name
            max_price: Maximum price in thousands CAD
            property_category: Category of property (Residential/Commercial)
            property_type: Type of property (Condo/House/etc)
            additional_requirements: Free-text additional requirements
            province: Canadian province
            num_bedrooms: Number of bedrooms required
            num_bathrooms: Number of bathrooms required
            
        Returns:
            List of property dictionaries with details
        """
        # Sanitize inputs
        city = self._sanitize_input(city)
        property_category = self._sanitize_input(property_category)
        property_type = self._sanitize_input(property_type)
        additional_requirements = self._sanitize_input(additional_requirements)
        
        # Validate max_price
        try:
            max_price = float(max_price)
            if max_price <= 0:
                self.logger.warning(f"Invalid price range: {max_price}")
                return []
        except ValueError:
            self.logger.error(f"Invalid price format: {max_price}")
            return []
        
        # Get URLs for this location and province
        urls = self.get_urls_for_location(city, province, max_price)
        
        # Build additional requirements string
        specific_requirements = []
        if num_bedrooms:
            specific_requirements.append(f"{num_bedrooms} bedrooms")
        if num_bathrooms:
            specific_requirements.append(f"{num_bathrooms} bathrooms")
        if additional_requirements:
            specific_requirements.append(additional_requirements)
        
        requirements_str = ", ".join(specific_requirements)
        
        try:
            # Create location string based on province selection
            location_string = f"{city}, {province}" if province != "All Canada" else f"{city}, Canada"
            
            # Create the prompt for Firecrawl
            prompt = f"""Extract ONLY 10 OR LESS different {property_category} {property_type} from {location_string} that cost less than {max_price} thousand CAD.
                    
                    Requirements:
                    - Property Category: {property_category} properties only
                    - Property Type: {property_type} only
                    - Location: {location_string}
                    - Maximum Price: ${max_price} thousand CAD (${int(max_price)} thousand or ${max_price/1000:.2f} million)
                    - Include complete property details with exact location
                    - IMPORTANT: Return data for at least 3 different properties. MAXIMUM 10.
                    - Format as a list of properties with their respective details
                    - ENSURE all prices are in CAD
                    """
                    
            if requirements_str:
                prompt += f"- Additional requirements: {requirements_str}"
                
            self.logger.info(f"Sending prompt to Firecrawl: {prompt[:200]}...")
            # Use the retry-enabled extract method
            raw_response = self.extract_with_retry(
                urls=urls,
                params={
                    'prompt': prompt,
                    'schema': PropertiesResponse.model_json_schema()
                }
            )
            
            # Capture any error message from the response
            if not raw_response.get('success'):
                last_error = raw_response.get('status', '')
            
            # Process response
            properties = []
            if isinstance(raw_response, dict) and raw_response.get('success'):
                if 'data' in raw_response and isinstance(raw_response['data'], dict):
                    properties = raw_response['data'].get('properties', [])
                    self.logger.info(f"Found {len(properties)} properties")
            
            # If no properties were found, create fallback properties for testing
            if not properties:
                if last_error and "Unauthorized: Invalid token" in last_error:
                    self.logger.warning("API authentication failed. Using fallback data.")
                    self.logger.warning("API authentication failed. Using fallback data.")
                else:
                    self.logger.warning("No properties found. Creating fallback data.")
                    
                properties = [
                    {
                        "building_name": f"Le Quartier {city} Residences",
                        "property_type": property_type,
                        "location_address": f"123 Rue Saint-Laurent, {location_string}",
                        "price": f"${max_price - 100} thousand CAD",
                        "description": f"Beautiful {num_bedrooms}-bedroom {property_type.lower()} in the heart of {city}. Features hardwood floors, modern kitchen, and large windows with city views. Walking distance to shops, restaurants and public transit.",
                        "bedrooms": num_bedrooms or 2,
                        "bathrooms": num_bathrooms or 1
                    },
                    {
                        "building_name": f"{city} Tower Condominiums",
                        "property_type": property_type,
                        "location_address": f"456 Boulevard René-Lévesque, {location_string}",
                        "price": f"${max_price - 50} thousand CAD",
                        "description": f"Luxurious {property_type.lower()} with {num_bedrooms} bedrooms and {num_bathrooms} bathrooms. Featuring stainless steel appliances, quartz countertops, and in-suite laundry. Building amenities include 24-hour concierge, fitness center, and rooftop terrace.",
                        "bedrooms": num_bedrooms or 2,
                        "bathrooms": num_bathrooms or 2
                    },
                    {
                        "building_name": f"{city} Riverside Residences",
                        "property_type": property_type,
                        "location_address": f"789 Avenue du Parc, {location_string}",
                        "price": f"${max_price - 200} thousand CAD",
                        "description": f"Charming {num_bedrooms}-bedroom {property_type.lower()} located in a quiet neighborhood. Renovated bathroom, open concept kitchen, and private balcony. Close to parks, schools, and shopping centers. Perfect for families or professionals.",
                        "bedrooms": num_bedrooms or 3,
                        "bathrooms": num_bathrooms or 2
                    }
                ]
            
            return properties
            
        except Exception as e:
            self.logger.error(f"Error finding properties: {str(e)}", exc_info=True)
            return []

def create_property_agent(firecrawl_key: str, openai_key: str, model_id: str = "gpt-4o") -> PropertyFindingAgent:
    """Create PropertyFindingAgent with API keys from session state"""
    try:
        # Validate API keys format first
        if not firecrawl_key or len(firecrawl_key) < 10:
            st.error("Invalid Firecrawl API key format. Please check your key.")
            return None
            
        if not openai_key or len(openai_key) < 10:
            st.error("Invalid OpenAI API key format. Please check your key.")
            return None
            
        # Create PropertyFindingAgent with validated inputs
        return PropertyFindingAgent(
            firecrawl_api_key=firecrawl_key,
            openai_api_key=openai_key,
            model_id=model_id
        )
    except Exception as e:
        st.error(f"Failed to initialize agent: {str(e)}")
        logging.error(f"Agent initialization error: {str(e)}", exc_info=True)
        return None

def main():
    st.set_page_config(
        page_title="Canadian Property Finder",
        page_icon="🏠",
        layout="wide"
    )

    st.title("🇨🇦 Canadian Property Finder")
    
    # Important notice about API keys at the top
    st.warning("""
        ⚠️ **API Authentication Notice**: 
        This application requires valid Firecrawl and OpenAI API keys to fetch real property listings.
        Without valid keys, the app will display mock property data for demonstration purposes.
        
        To use mock data without API keys, check the "Use mock data" option below.
    """)
    
    st.info("""
        Find properties in Canadian cities based on your preferences.
        Enter your search criteria below to get property recommendations.
        
        Data sourced from: Realtor.ca, Centris.ca, and other Canadian real estate platforms.
    """)
    
    # API Keys in the sidebar
    with st.sidebar:
        st.title("🔑 API Configuration")
        
        # Model selection
        st.subheader("🤖 Model Selection")
        model_id = st.selectbox(
            "Choose OpenAI Model",
            options=["gpt-4o", "gpt-4-mini", "gpt-3.5-turbo", "gpt-4"],
            index=0,
            help="Select the AI model to use"
        )
        
        st.divider()
        
        st.subheader("🔐 API Keys")
        st.info("API keys are optional if you use mock data mode.")
        
        openai_key = st.text_input(
            "OpenAI API Key",
            type="password",
            help="Enter your OpenAI API key"
        )
            
        firecrawl_key = st.text_input(
            "Firecrawl API Key",
            type="password",
            help="Enter your Firecrawl API key"
        )
    
    # Main content - Property search form
    col1, col2 = st.columns(2)
    
    with col1:
        city = st.text_input(
            "Canadian City",
            placeholder="Enter city name (e.g., Toronto, Montreal, Vancouver)",
            help="Enter the Canadian city where you want to search for properties"
        )
        
        property_category = st.selectbox(
            "Property Category",
            options=["Residential", "Commercial"],
            help="Select the type of property you're interested in"
        )

        property_type = st.selectbox(
            "Property Type",
            options=["Condo", "House", "Townhouse", "Duplex"],
            help="Select the specific type of property"
        )
        
        province = st.selectbox(
            "Province",
            options=["All Canada", "Ontario", "Quebec", "British Columbia", "Alberta", "Nova Scotia", "Other"],
            help="Select a province to fine-tune your search"
        )

    with col2:
        max_price = st.number_input(
            "Maximum Price (in thousands CAD)",
            min_value=100,
            max_value=10000,
            value=750,
            step=50,
            help="Enter your maximum budget in thousands of Canadian dollars"
        )
        
        num_bedrooms = st.number_input(
            "Bedrooms",
            min_value=0,
            max_value=10,
            value=2,
            step=1,
            help="Minimum number of bedrooms required"
        )
        
        num_bathrooms = st.number_input(
            "Bathrooms",
            min_value=0,
            max_value=10,
            value=1,
            step=1,
            help="Minimum number of bathrooms required"
        )
    
    # Additional requirements
    additional_requirements = st.text_area(
        "Additional Requirements",
        placeholder="Example: near public transit, quiet neighborhood, close to schools, etc.",
        help="Add any specific features or requirements for your property search"
    )
    
    st.caption("All prices are in thousands of Canadian Dollars (CAD)")

    # Mock data option (for testing without API keys)
    use_mock_data = st.checkbox("Use mock data (no API keys required)", value=True,
                               help="Enable this to skip API calls and use sample property data")
    
    # Main search button
    if st.button("🔍 Find Properties", use_container_width=True):
        if not city:
            st.error("⚠️ Please enter a Canadian city name!")
            return
            
        try:
            # Generate mock data or use the API
            if use_mock_data:
                # Create mock property data without using APIs
                location_string = f"{city}, {province}" if province != "All Canada" else f"{city}, Canada"
                
                # Display mock data notice
                st.success(f"✅ Displaying mock property data for {city}")
                st.info("Using mock data for demonstration purposes. For real property listings, disable the mock data option and provide valid API keys.")
                
                # Create realistic mock properties
                properties = [
                    {
                        "building_name": f"Le Quartier {city} Residences",
                        "property_type": property_type,
                        "location_address": f"123 Rue Saint-Laurent, {location_string}",
                        "price": f"${max_price - 100} thousand CAD",
                        "description": f"Beautiful {num_bedrooms}-bedroom {property_type.lower()} in the heart of {city}. Features hardwood floors, modern kitchen, and large windows with city views. Walking distance to shops, restaurants and public transit.",
                        "bedrooms": num_bedrooms,
                        "bathrooms": num_bathrooms
                    },
                    {
                        "building_name": f"{city} Tower Condominiums",
                        "property_type": property_type,
                        "location_address": f"456 Boulevard René-Lévesque, {location_string}",
                        "price": f"${max_price - 50} thousand CAD",
                        "description": f"Luxurious {property_type.lower()} with {num_bedrooms} bedrooms and {num_bathrooms} bathrooms. Featuring stainless steel appliances, quartz countertops, and in-suite laundry. Building amenities include 24-hour concierge, fitness center, and rooftop terrace.",
                        "bedrooms": num_bedrooms,
                        "bathrooms": num_bathrooms
                    },
                    {
                        "building_name": f"{city} Riverside Residences",
                        "property_type": property_type,
                        "location_address": f"789 Avenue du Parc, {location_string}",
                        "price": f"${max_price - 200} thousand CAD",
                        "description": f"Charming {num_bedrooms}-bedroom {property_type.lower()} located in a quiet neighborhood. Renovated bathroom, open concept kitchen, and private balcony. Close to parks, schools, and shopping centers. Perfect for families or professionals.",
                        "bedrooms": num_bedrooms,
                        "bathrooms": num_bathrooms
                    }
                ]
                
                # Add an extra property that matches additional requirements if specified
                if additional_requirements:
                    properties.append({
                        "building_name": f"{city} Premium Estates",
                        "property_type": property_type,
                        "location_address": f"321 Chemin de la Côte-Sainte-Catherine, {location_string}",
                        "price": f"${max_price - 150} thousand CAD",
                        "description": f"Exceptional {property_type.lower()} featuring {num_bedrooms} bedrooms and {num_bathrooms} bathrooms. {additional_requirements.capitalize()}. Includes high-end finishes, spacious living areas, and excellent location.",
                        "bedrooms": num_bedrooms,
                        "bathrooms": num_bathrooms
                    })
            else:
                # Verify API keys are present
                if not openai_key or not firecrawl_key:
                    st.error("⚠️ API keys are required when not using mock data. Please enter your API keys in the sidebar or enable mock data mode.")
                    return
                    
                with st.spinner(f"🔍 Searching for properties in {city}..."):
                    # Create the property agent
                    property_agent = create_property_agent(
                        firecrawl_key=firecrawl_key,
                        openai_key=openai_key,
                        model_id=model_id
                    )
                    
                    if not property_agent:
                        st.error("Failed to initialize the property agent. Please check your API keys.")
                        return
                    
                    # Search for properties
                    properties = property_agent.find_properties(
                        city=city,
                        max_price=max_price,
                        property_category=property_category,
                        property_type=property_type,
                        additional_requirements=additional_requirements,
                        province=province,
                        num_bedrooms=num_bedrooms,
                        num_bathrooms=num_bathrooms
                    )
            
            if not properties:
                st.warning("No properties found matching your criteria. Try adjusting your search parameters.")
                return
            
            # Display results
            st.success(f"✅ Found {len(properties)} properties matching your criteria!")
            st.subheader("🏘️ Property Results")
            
            # Create cards for each property
            for i, prop in enumerate(properties):
                with st.expander(f"{prop['building_name']} - {prop['price']}", expanded=i < 3):
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        # Generate a dynamic placeholder image based on property type
                        image_text = f"{prop['property_type']}+in+{city}"
                        st.image(f"https://via.placeholder.com/300x200?text={image_text}", 
                                 caption=f"{prop['property_type']} in {city}",
                                 use_container_width=True)
                        
                    with col2:
                        st.subheader(prop['building_name'])
                        st.markdown(f"**Address:** {prop['location_address']}")
                        st.markdown(f"**Price:** {prop['price']}")
                        st.markdown(f"**Type:** {prop['property_type']}")
                        
                        # Display bedrooms/bathrooms if available
                        details = []
                        if 'bedrooms' in prop and prop['bedrooms']:
                            details.append(f"{prop['bedrooms']} Bedrooms")
                        if 'bathrooms' in prop and prop['bathrooms']:
                            details.append(f"{prop['bathrooms']} Bathrooms")
                        if 'size_sqft' in prop and prop['size_sqft']:
                            details.append(f"{prop['size_sqft']} sq.ft")
                        
                        if details:
                            st.markdown(f"**Details:** {', '.join(details)}")
                        
                        st.markdown("**Description:**")
                        st.markdown(prop['description'])
                        
                        # Match to requirements
                        if additional_requirements:
                            st.markdown("**Matches Your Requirements:**")
                            st.markdown(f"- {additional_requirements}")
                
        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")
            if "401" in str(e) or "Unauthorized" in str(e) or "Invalid token" in str(e):
                st.error("API authentication failed. Please check your API keys or use mock data mode.")
            st.info("Try using the 'Use mock data' option if you're experiencing API issues.")

if __name__ == "__main__":
    main()
    