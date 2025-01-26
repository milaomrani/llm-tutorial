import streamlit as st
import ollama

def get_ollama_response(messages):
    try:
        response = ollama.chat(
            model="deepseek-r1:8b",
            messages=messages
        )
        return response["message"]["content"]
    except Exception as e:
        return f"Error: {str(e)}"

def main():
    st.title("AI Chat Assistant")
    
    # Initialize session state for messages if it doesn't exist
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant. Provide clear and concise answers."
            }
        ]

    # Display chat history
    for message in st.session_state.messages[1:]:  # Skip system message
        if message["role"] == "user":
            with st.container():
                st.markdown("**You:**")
                st.markdown(message["content"])
        else:
            with st.container():
                st.markdown("**Assistant:**")
                st.markdown(message["content"])

    # Chat input
    user_input = st.text_input("Ask me anything:", key="user_input")
    
    if st.button("Send") and user_input:
        # Add user message to history
        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })
        
        # Get AI response
        with st.spinner("Thinking..."):
            ai_response = get_ollama_response(st.session_state.messages)
            
            # Add AI response to history
            st.session_state.messages.append({
                "role": "assistant",
                "content": ai_response
            })
        
        st.rerun()

    # Add a clear chat button
    if st.button("Clear Chat"):
        st.session_state.messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant. Provide clear and concise answers."
            }
        ]
        st.rerun()

if __name__ == "__main__":
    main()
