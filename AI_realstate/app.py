import streamlit as st
import logging
from RS_ai.property_agent import PropertyFindingAgent

def create_property_agent():
    """Create PropertyFindingAgent with API keys from session state"""
    try:
        if 'property_agent' not in st.session_state:
            # Create PropertyFindingAgent with validated inputs
            st.session_state.property_agent = PropertyFindingAgent(
                firecrawl_api_key=st.session_state.firecrawl_key,
                openai_api_key=st.session_state.openai_key,
                model_id=st.session_state.model_id
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
        
        # Model selection - OpenAI only now
        st.subheader("🤖 Model Selection")
        model_id = st.selectbox(
            "Choose OpenAI Model",
            options=["gpt-4o", "gpt-4-mini", "gpt-3.5-turbo", "gpt-4"],
            index=0,  # Set gpt-4o as default
            help="Select the AI model to use"
        )
        st.session_state.model_id = model_id
        
        st.divider()
        
        st.subheader("🔐 API Keys")
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
        
        if firecrawl_key and openai_key:
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
        
        # Chat model temperature settings
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
                max_value=10000,
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
            province_chat = st.selectbox(
                "Province (for context)", 
                options=["All Canada", "Ontario", "Quebec", "British Columbia", "Alberta", "Nova Scotia", "Other"],
                index=0
            )
            property_type_chat = st.selectbox(
                "Property Type (for context)",
                options=["Any", "Condo", "House", "Townhouse", "Duplex"],
                index=0
            )
            max_price_chat = st.number_input(
                "Budget in thousands CAD (for context)",
                min_value=100,
                max_value=10000,
                value=1000,
                step=50
            )
        else:
            city_chat = city
            province_chat = province
            property_type_chat = property_type
            max_price_chat = max_price
            location_string = f"{city}, {province}" if province != "All Canada" else f"{city}, Canada"
            st.success(f"Currently focusing on: {location_string}, {property_type}s up to ${max_price}k CAD")
        
        # Display chat history before the input
        if "chat_history" in st.session_state and st.session_state.chat_history:
            st.subheader("Recent Conversation")
            for i, qa in enumerate(st.session_state.chat_history[-3:]):  # Show last 3 exchanges
                st.markdown(f"**You:** {qa['question']}")
                st.markdown(f"**AI:** {qa['answer']}")
            
            if len(st.session_state.chat_history) > 3:
                with st.expander("View Full Conversation History"):
                    for i, qa in enumerate(st.session_state.chat_history):
                        st.markdown(f"**You:** {qa['question']}")
                        st.markdown(f"**AI:** {qa['answer']}")
        
        question = st.text_area(
            "Ask me a question",
            placeholder="Example: What are the best neighborhoods in Toronto for investment?",
            help="Ask any question about Canadian real estate or property investment"
        )

        if st.button("Submit Question"):
            if question:
                if 'property_agent' not in st.session_state:
                    st.error("⚠️ Please enter your API keys in the sidebar first!")
                else:
                    with st.spinner("Processing your question..."):
                        if not city_chat:
                            st.warning("No city specified. Providing general advice.")
                        
                        # Call the answer_question method
                        answer = st.session_state.property_agent.answer_question(
                            question=question,
                            city=city_chat,
                            property_type=property_type_chat,
                            max_price=max_price_chat,
                            province=province_chat
                        )
                        
                        # Add the Q&A to the chat history
                        st.session_state.chat_history.append({"question": question, "answer": answer})
                        
                        # Display the answer
                        st.markdown(f"**Answer:** {answer}")
            else:
                st.error("⚠️ Please enter a question before submitting.")

    # About Tab
    with tabs[2]:
        st.subheader("ℹ️ About")
        st.markdown(
            """
            This web app is a demo of a Canadian AI Real Estate Agent powered by OpenAI's GPT-4o and Firecrawl APIs.
            
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
            **Version:** 1.1
            """
        )   
        
if __name__ == "__main__":
    main()