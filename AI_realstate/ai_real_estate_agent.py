from typing import Dict, List, Optional
from pydantic import BaseModel, Field
import streamlit as st
import os
import json
import logging
import subprocess
import requests
import time
from urllib.parse import quote_plus
import re

class PropertyData(BaseModel):
    """Schema for property data extraction"""
    building_name: str = Field(description="Name of the building/property", alias="Building_name")
    property_type: str = Field(description="Type of property (commercial, residential, etc)", alias="Property_type")
    location_address: str = Field(description="Complete address of the property")
    price: str = Field(description="Price of the property in CAD", alias="Price")
    description: str = Field(description="Detailed description of the property", alias="Description")

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

# Add a class to handle Ollama models locally
class LocalOllamaModel:
    """Client for interacting with local Ollama models."""
    
    def __init__(self, model_name: str = "llama3.1:latest", api_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.api_url = api_url
        self.api_key = None  # No API key needed for local Ollama
        
        # Verify Ollama is running and model is available
        self._check_model_availability()
    
    def _check_model_availability(self) -> bool:
        """Check if Ollama is running and the specified model is available."""
        try:
            # Check if Ollama service is running
            response = requests.get(f"{self.api_url}/api/tags")
            response.raise_for_status()
            
            # Check if the requested model is in the list
            models = response.json().get("models", [])
            available_models = []
            
            # Properly extract model names from Ollama API response
            for model in models:
                if isinstance(model, dict) and "name" in model:
                    available_models.append(model["name"])
            
            if self.model_name not in available_models:
                logging.warning(f"Model {self.model_name} not found in available models: {available_models}")
                logging.info(f"You may need to run: ollama pull {self.model_name}")
                return False
                
            return True
        except Exception as e:
            logging.error(f"Error checking Ollama model availability: {str(e)}")
            logging.info("Make sure Ollama is installed and running with 'ollama serve'")
            return False
    
    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 1000) -> str:
        """Generate text using local Ollama model."""
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": False
            }
            
            response = requests.post(f"{self.api_url}/api/generate", json=payload)
            response.raise_for_status()
            
            result = response.json()
            if "response" in result:
                return result["response"]
            else:
                logging.error(f"Unexpected response format from Ollama: {result}")
                return "Error: Unexpected response format"
                
        except Exception as e:
            logging.error(f"Error generating text with Ollama: {str(e)}")
            return f"Error: {str(e)}"

class PropertyFindingAgent:
    """Agent responsible for finding properties and providing recommendations"""
    
    def __init__(self, firecrawl_api_key: str, openai_api_key: str, model_id: str = "gpt-3.5-turbo", use_local_ollama: bool = False):
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("PropertyFindingAgent")
        
        # Initialize the appropriate model
        if use_local_ollama:
            # Use local Ollama model
            self.local_model = LocalOllamaModel(model_name="llama3.1:latest")
            self.use_local_model = True
            self.logger.info(f"Using local Ollama model: llama3.1:latest")
        else:
            # Use Agno/OpenAI model
            try:
                from agno.agent import Agent
                from agno.models.openai import OpenAIChat
                
                self.agent = Agent(
                    model=OpenAIChat(id=model_id, api_key=openai_api_key),
                    markdown=True,
                    description="I am a real estate expert who helps find and analyze properties in Canada based on user preferences."
                )
                self.use_local_model = False
                self.logger.info(f"Using OpenAI model: {model_id}")
            except Exception as e:
                self.logger.error(f"Failed to initialize OpenAI model: {str(e)}")
                self.logger.info("Falling back to local Ollama model")
                self.local_model = LocalOllamaModel(model_name="llama3.1:latest")
                self.use_local_model = True

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
        
    def _run_model(self, prompt: str) -> str:
        """Run the appropriate model based on configuration."""
        if self.use_local_model:
            return self.local_model.generate(prompt)
        else:
            response = self.agent.run(prompt)
            return response.content

    def find_properties(
        self, 
        city: str,
        max_price: float,
        property_category: str = "Residential",
        property_type: str = "Condo",
        additional_requirements: str = "",
        province: str = "All Canada"
    ) -> str:
        """Find and analyze properties based on user preferences"""
        # Sanitize inputs
        city = self._sanitize_input(city)
        property_category = self._sanitize_input(property_category)
        property_type = self._sanitize_input(property_type)
        additional_requirements = self._sanitize_input(additional_requirements)
        
        # Validate max_price to prevent injection
        try:
            max_price = float(max_price)
            if max_price <= 0:  # Cap at reasonable upper limit
                self.logger.warning(f"Invalid price range: {max_price}")
                return "Error: Price must be a positive number"
        except ValueError:
            self.logger.error(f"Invalid price format: {max_price}")
            return "Error: Price must be a valid number"
        
        # Format location for URLs safely
        formatted_location = quote_plus(city.lower())
        formatted_province = quote_plus(province.lower()) if province != "All Canada" else ""
        
        # Create location string based on province selection
        location_string = f"{city}, {province}" if province != "All Canada" else f"{city}, Canada"
        search_location = f"{formatted_location}-{formatted_province}" if province != "All Canada" else formatted_location
        
        # Use safer string formatting for URLs - Make them more generic to increase chances of finding properties
        urls = [
            f"https://www.realtor.ca/map#view=list&Sort=6-D&PropertyTypeGroupID=1&PropertySearchTypeId=1&TransactionTypeId=2&Currency=CAD",
            f"https://www.royallepage.ca/en/search/homes/"
        ]
        
        # Add province-specific URLs but make them more generic
        if province == "Quebec":
            # Quebec-specific URLs (Centris)
            urls.extend([
                f"https://www.centris.ca/en/properties~for-sale",
                f"https://www.centris.ca/fr/proprietes-residentielles~a-vendre"
            ])
        elif province == "Ontario":
            # Ontario-specific URLs
            urls.extend([
                f"https://condos.ca/search?for=sale",
                f"https://www.zoocasa.com/ontario-real-estate"
            ])
        elif province == "British Columbia":
            # BC-specific URLs
            urls.extend([
                f"https://www.rew.ca/properties/areas/{formatted_location}/sort/price/asc",
                f"https://www.zolo.ca/{formatted_location}-real-estate"
            ])
        else:
            # Generic URLs for all other provinces
            urls.extend([
                f"https://www.centris.ca/en/properties~for-sale~{formatted_location}?view=Thumbnail",
                f"https://www.zillow.com/homes/for_sale/{formatted_location}_rb/"
            ])
        
        # Rest of the method with security improvements
        property_type_prompt = property_type
        
        try:
            # Create location string based on province selection
            location_string = f"{city}, {province}" if province != "All Canada" else f"{city}, Canada"
            
            raw_response = self.firecrawl.extract(
                urls=urls,
                params={
                    'prompt': f"""Extract properties that match these criteria as closely as possible:
        
        Requirements:
        - Property Category: {property_category} properties
        - Property Type: {property_type_prompt} or similar types
        - Location: {location_string} or nearby areas
        - Price Range: Up to ${max_price} thousand CAD
        - Include complete property details with location
        - If exact matches aren't available, include similar properties
        - Format as a list of properties with their respective details
        - ENSURE all prices are in CAD
        {f"- Additional requirements to consider: {additional_requirements}" if additional_requirements else ""}
        """,
                    'schema': PropertiesResponse.model_json_schema()
                }
            )
            
            self.logger.info("Successfully retrieved property data")
            
            # Log the raw response for debugging
            self.logger.info(f"Raw response excerpt: {str(raw_response)[:200]}...")

            # Add fallback when no properties found
            if isinstance(raw_response, dict) and raw_response.get('success'):
                properties = raw_response['data'].get('properties', [])
                if not properties:
                    self.logger.warning("No properties found in response, using fallback")
                    # Use the model to generate analysis based on general market knowledge
                    return self._run_model(f"""
        As a Canadian real estate expert, please provide:
        
        1. Analysis of the {property_type} market in {location_string} in the ${max_price}k CAD range
        2. Typical features and prices for {property_type}s in this area
        3. Market trends affecting this property type and location
        4. Investment potential and considerations
        5. Alternative neighborhoods or property types to consider
        
        Format your response with clear headings and helpful insights.
        """)
            else:
                properties = []
                self.logger.warning(f"Invalid response format or no properties found: {raw_response}")
                
            # Limit data size for logging
            self.logger.info(f"Found {len(properties)} properties")
            
            # Create analysis prompt
            location_string = f"{city}, {province}" if province != "All Canada" else f"{city}, Canada"
            
            analysis_prompt = f"""As a Canadian real estate expert, analyze these properties and market trends:

            Properties Found in json format:
            {properties}

            **IMPORTANT INSTRUCTIONS:**
            1. ONLY analyze properties from the above JSON data that match the user's requirements:
               - Property Category: {property_category}
               - Property Type: {property_type}
               - Location: {location_string}
               - Maximum Price: ${max_price} thousand CAD (${int(max_price)} thousand or ${max_price/1000:.2f} million)
               {f"- Additional Requirements: {additional_requirements}" if additional_requirements else ""}
            2. DO NOT create new categories or property types
            3. From the matching properties, select 5-6 properties with prices closest to ${max_price} thousand CAD

            Please provide your analysis in this format:
            
            🏠 SELECTED PROPERTIES
            • List only 5-6 best matching properties with prices closest to ${max_price} thousand CAD
            • For each property include:
              - Name and Location
              - Price (with value analysis)
              - Key Features
              - Pros and Cons

            💰 BEST VALUE ANALYSIS
            • Compare the selected properties based on:
              - Price per sq ft
              - Location advantage
              - Amenities offered

            📍 LOCATION INSIGHTS
            • Specific advantages of the areas where selected properties are located
            • Public transit accessibility
            • School districts
            • Nearby amenities

            💡 RECOMMENDATIONS
            • Top 3 properties from the selection with reasoning
            • Investment potential
            • Points to consider before purchase

            🤝 NEGOTIATION TIPS
            • Property-specific negotiation strategies
            • Current market conditions in Canada
            • Tax considerations for Canadian property buyers

            Format your response in a clear, structured way using the above sections.
            """
            
            # Get analysis from model
            analysis = self._run_model(analysis_prompt)
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error finding properties: {str(e)}", exc_info=True)
            return f"Error: Unable to complete property search. Please try again later. Details: {str(e)}"

    def get_location_trends(self, city: str, province: str = "All Canada") -> str:
        """Get price trends for different localities in the Canadian city"""
        # Sanitize input
        city = self._sanitize_input(city)
        province = self._sanitize_input(province)
        
        # Create location string based on province selection
        location_string = f"{city}, {province}" if province != "All Canada" else f"{city}, Canada"
        
        # Modify URLs based on province
        urls = [
            f"https://www.zolo.ca/insights/{city.lower().replace(' ', '-')}-real-estate-trends",
            f"https://wowa.ca/home-prices-real-estate-trends/{city.lower().replace(' ', '-')}",
            f"https://www.ratehub.ca/mortgage-blog/category/housing-market/",
            f"https://www.realtor.ca/blog/search?query={city.lower().replace(' ', '+')}+market+trends",
            f"https://www.crea.ca/housing-market-stats/quarterly-forecasts/"
        ]
        
        # Add province-specific URLs
        if province == "Quebec":
            urls.append(f"https://www.centris.ca/en/blog?keyword={city.lower().replace(' ', '+')}")
        elif province == "Ontario":
            urls.append(f"https://trreb.ca/index.php/market-stats")
        elif province == "British Columbia":
            urls.append(f"https://www.rebgv.org/market-watch/monthly-market-report.html")
        
        raw_response = self.firecrawl.extract(urls, {
            'prompt': f"""Extract price trends data for ALL major neighborhoods in {location_string}. 
            IMPORTANT: 
            - Return data for at least 5-10 different neighborhoods/localities in {location_string}
            - Include both premium and affordable areas
            - Include price per square foot in CAD
            - Include year-over-year percentage increases
            - Include rental yield data where available
            - Report prices in thousands of CAD
            - Do not skip any locality mentioned in the source
            - Format as a list of locations with their respective data
            - ENSURE all monetary values are in thousands of CAD
            - Pay special attention to data from realtor.ca and CREA stats
            """,
            'schema': LocationsResponse.model_json_schema(),
        })
        
        if isinstance(raw_response, dict) and raw_response.get('success'):
            locations = raw_response['data'].get('locations', [])
    
            analysis = self._run_model(
                f"""As a Canadian real estate expert, analyze these location price trends for {location_string}:

                {locations}

                Please provide:
                1. A bullet-point summary of the price trends for each neighborhood in {location_string}
                2. Identify the top 3 neighborhoods with:
                   - Highest price appreciation
                   - Best rental yields
                   - Best value for money
                3. Investment recommendations:
                   - Best neighborhoods for long-term investment
                   - Best neighborhoods for rental income
                   - Areas showing emerging potential
                4. Specific advice for investors based on these trends
                5. {province if province != "All Canada" else "Canadian"} real estate market contextual factors:
                   - Interest rates impact
                   - {f"{province} provincial" if province != "All Canada" else "Foreign buyer"} regulations
                   - Property tax considerations

                Format the response as follows:
                
                📊 {city.upper()} NEIGHBORHOOD TRENDS SUMMARY
                • [Bullet points for each neighborhood]

                🏆 TOP PERFORMING AREAS IN {city.upper()}
                • [Bullet points for best areas]

                💡 {f"{province.upper()}" if province != "All Canada" else "CANADIAN"} INVESTMENT INSIGHTS
                • [Bullet points with investment advice]

                🎯 RECOMMENDATIONS
                • [Bullet points with specific recommendations]

                {f"🏠 {province.upper()} MARKET CONSIDERATIONS" if province != "All Canada" else "🇨🇦 CANADIAN MARKET CONSIDERATIONS"}
                • [Bullet points about {f"{province}" if province != "All Canada" else "Canadian"}-specific factors]
                """
            )
            
            return analysis
            
        return f"No price trends data available for {city}, Canada"
    
    def answer_question(self, question: str, city: str, property_type: str, max_price: float, province: str = "All Canada") -> str:
        """Answer specific questions about real estate in the given city"""
        # Sanitize input
        question = self._sanitize_input(question)
        city = self._sanitize_input(city)
        province = self._sanitize_input(province)
        
        self.logger.info(f"Answering question: {question} for {city}")
        
        location_string = f"{city}, {province}" if province != "All Canada" else f"{city}, Canada"
        
        prompt = f"""As a Canadian real estate expert, please answer the following question about {location_string} real estate:

        Question: {question}

        Consider:
        - Property type: {property_type}
        - Budget: ${max_price} thousand CAD
        - Location: {location_string}

        Provide a detailed and informative answer based on current Canadian real estate market knowledge.
        Include relevant statistics, market trends, and practical advice where appropriate.
        """
        
        try:
            return self._run_model(prompt)
        except Exception as e:
            self.logger.error(f"Error answering question: {str(e)}")
            return f"Sorry, I couldn't process your question. Error: {str(e)}"

def create_property_agent():
    """Create PropertyFindingAgent with API keys from session state"""
    try:
        # Check for the use of local Ollama model
        use_local_ollama = st.session_state.get('use_local_ollama', False)
        
        if 'property_agent' not in st.session_state:
            # Create PropertyFindingAgent with validated inputs
            st.session_state.property_agent = PropertyFindingAgent(
                firecrawl_api_key=st.session_state.firecrawl_key,
                openai_api_key=st.session_state.openai_key,
                model_id=st.session_state.model_id,
                use_local_ollama=use_local_ollama
            )
    except Exception as e:
        st.error(f"Failed to initialize agent: {str(e)}")
        logging.error(f"Agent initialization error: {str(e)}", exc_info=True)

def main():
    # Set up secure session state
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    
    # Initialize chat history if it doesn't exist
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    st.set_page_config(
        page_title="Canadian AI Real Estate Agent",
        page_icon="🏠",
        layout="wide"
    )

    with st.sidebar:
        st.title("🔑 API Configuration")
        
        # Add option to use local Ollama model
        st.subheader("🤖 Model Selection")
        model_option = st.radio(
            "Choose AI Model Source",
            options=["OpenAI API", "Local Ollama Model"],
            help="Select whether to use OpenAI API or local Ollama model"
        )
        
        use_local_ollama = (model_option == "Local Ollama Model")
        st.session_state.use_local_ollama = use_local_ollama
        
        if use_local_ollama:
            st.info("Using local Ollama model (llama3.1:latest). Make sure Ollama is running with 'ollama serve' and you have downloaded the model with 'ollama pull llama3.1:latest'.")
            st.session_state.model_id = "llama3.1:latest"
            
            # Test Ollama connection
            if st.button("Test Ollama Connection"):
                try:
                    response = requests.get("http://localhost:11434/api/tags")
                    if response.status_code == 200:
                        models = response.json().get("models", [])
                        model_names = [model.get("name") for model in models]
                        if "llama3.1:latest" in model_names:
                            st.success("✅ Ollama is running and llama3.1:latest model is available!")
                        else:
                            st.warning(f"⚠️ Ollama is running but llama3.1:latest model not found. Available models: {', '.join(model_names)}")
                            
                            # Add option to pull model if not found
                            if st.button("Pull llama3.1:latest Model"):
                                try:
                                    st.info("Pulling llama3.1:latest model... This may take a few minutes.")
                                    subprocess.run(["ollama", "pull", "llama3.1:latest"], check=True)
                                    st.success("Successfully pulled llama3.1:latest model!")
                                except Exception as e:
                                    st.error(f"Failed to pull model: {str(e)}")
                    else:
                        st.error("❌ Could not connect to Ollama API")
                        
                        # Add option to start Ollama if not running
                        if st.button("Start Ollama Server"):
                            try:
                                st.info("Starting Ollama server...")
                                subprocess.Popen(["ollama", "serve"], start_new_session=True)
                                st.success("Ollama server starting! Please wait a moment and try connecting again.")
                            except Exception as e:
                                st.error(f"Failed to start Ollama server: {str(e)}")
                except Exception as e:
                    st.error(f"❌ Error connecting to Ollama: {str(e)}")
                    st.info("Is Ollama installed? If not, visit https://ollama.ai/ to install.")
        else:
            model_id = st.selectbox(
                "Choose OpenAI Model",
                options=["gpt-4o-mini", "gpt-3.5-turbo", "gpt-4o", "gpt-4"],
                index=0,  # Set gpt-4o-mini as default
                help="Select the AI model to use"
            )
            st.session_state.model_id = model_id
        
        st.divider()
        
        st.subheader("🔐 API Keys")
        if not use_local_ollama:
            openai_key = st.text_input(
                "OpenAI API Key",
                type="password",
                help="Enter your OpenAI API key"
            )
        else:
            # Placeholder for consistency in the UI
            openai_key = "not-required-for-local-ollama"
            
        firecrawl_key = st.text_input(
            "Firecrawl API Key",
            type="password",
            help="Enter your Firecrawl API key"
        )
        
        if firecrawl_key and (use_local_ollama or openai_key):
            # Store API keys securely in session state
            st.session_state.firecrawl_key = firecrawl_key
            st.session_state.openai_key = openai_key
            create_property_agent()
            
        # Add chat settings in the sidebar
        st.divider()
        st.subheader("💬 Chat Settings")
        
        # Option to clear chat history
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.success("Chat history cleared!")
        
        # Chat model temperature settings (only show if using AI models)
        chat_temperature = st.slider(
            "Chat Response Creativity",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="Lower values give more factual responses, higher values more creative ones"
        )
        st.session_state.chat_temperature = chat_temperature

    # Add tab-based interface for search and chat
    tabs = st.tabs(["🏠 Property Search", "💬 Chat Assistant", "ℹ️ About"])
    
    # Property Search Tab
    with tabs[0]:
        st.title("🇨🇦 Canadian AI Real Estate Agent")
        st.info(
            """
            Welcome to the Canadian AI Real Estate Agent! 
            Enter your search criteria below to get property recommendations 
            and location insights for Canadian real estate.
            
            Data sourced from: Realtor.ca, Centris.ca, and other Canadian real estate platforms.
            """
        )
        
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
    
        with col2:
            max_price = st.number_input(
                "Maximum Price (in thousands CAD)",
                min_value=100,
                max_value=500000,
                value=1000,
                step=50,
                help="Enter your maximum budget in thousands of Canadian dollars"
            )
            
            property_type = st.selectbox(
                "Property Type",
                options=["Condo", "House", "Townhouse", "Duplex"],
                help="Select the specific type of property"
            )
            
        # Add a text area for additional requirements
        st.subheader("Additional Requirements")
        additional_requirements = st.text_area(
            "Specify any additional requirements or preferences",
            placeholder="Example: near public transit, quiet neighborhood, close to schools, minimum 2 bedrooms, etc.",
            help="Add any specific features or requirements for your property search"
        )
        
        # Add region selector that's especially useful for Quebec properties (Centris)
        province = st.selectbox(
            "Province",
            options=["All Canada", "Ontario", "Quebec", "British Columbia", "Alberta", "Nova Scotia", "Other"],
            help="Select a province to fine-tune your search"
        )
    
        st.caption("All prices are in thousands of Canadian Dollars (CAD)")
    
        # Main search button
        if st.button("🔍 Start Property Search", use_container_width=True):
            if 'property_agent' not in st.session_state:
                st.error("⚠️ Please enter your API keys in the sidebar first!")
                return
                
            if not city:
                st.error("⚠️ Please enter a Canadian city name!")
                return
                
            try:
                with st.spinner(f"🔍 Searching for properties in {city}, {province}..."):
                    property_results = st.session_state.property_agent.find_properties(
                        city=city,
                        max_price=max_price,
                        property_category=property_category,
                        property_type=property_type,
                        additional_requirements=additional_requirements,
                        province=province
                    )
                    
                    # Add information about data sources
                    st.success("✅ Canadian property search completed!")
                    st.info("Data sourced from Realtor.ca, Centris.ca, and other Canadian real estate platforms.")
                    
                    st.subheader("🏘️ Property Recommendations")
                    st.markdown(property_results)
                    
                    st.divider()
                    
                    # Create location string based on province selection
                    location_string = f"{city}, {province}" if province != "All Canada" else f"{city}, Canada"
                    
                    with st.spinner(f"📊 Analyzing {location_string} location trends..."):
                        location_trends = st.session_state.property_agent.get_location_trends(city, province)
                        
                        st.success("✅ Location analysis completed!")
                        
                        with st.expander(f"📈 Location Trends Analysis for {location_string}"):
                            st.markdown(location_trends)
                    
            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")
    
    # Chat Assistant Tab
    with tabs[1]:
        st.subheader("💬 Real Estate Chat Assistant")
        st.info("Ask me anything about Canadian real estate, neighborhoods, market trends, or property investment tips!")
        
        # Make sure city and property details are still accessible in this tab
        if "city" not in locals() or not city:
            city_chat = st.text_input("City (for context)", placeholder="Enter a Canadian city name")
            province_chat = "All Canada"
        else:
            city_chat = city
            province_chat = province
            location_string = f"{city}, {province}" if province != "All Canada" else f"{city}, Canada"
            st.success(f"Currently focusing on: {location_string}, {property_type}s up to ${max_price}k CAD")
            
        question = st.text_area(
            "Ask me a question",
            placeholder="Example: What are the best neighborhoods in Toronto for investment?",
            help="Ask any question about Canadian real estate or property investment"
        )


        if st.button("Submit Question"):
            if question:

                st.success("Your question has been submitted!")
            else:
                st.error("⚠️ Please enter a question before submitting.")

    # About Tab
    with tabs[2]:
        st.subheader("ℹ️ About")
        st.markdown(
            """
            This web app is a demo of a Canadian AI Real Estate Agent powered by OpenAI's GPT-3 and Firecrawl APIs.
            
            **Features:**
            - Search for properties in Canadian cities
            - Get location insights and price trends
            - Ask questions about Canadian real estate
            
            **Data Sources:**
            - Realtor.ca, Centris.ca, and other Canadian real estate platforms
            
            **Disclaimer:**
            This app is for demonstration purposes only and does not provide real-time data or financial advice.
            """
        )
        st.markdown(
            """
            **Developed by:** [Milad Omrani](https://www.linkedin.com/in/miladomrani/)
            """
        )
        st.markdown(
            """
            **Contact:** milaomrani@gmail.com
            **Version:** 1.0
            """
        )   
        
if __name__ == "__main__":
    main()