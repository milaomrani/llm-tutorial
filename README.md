# llm-tutorial

# AI Chat Assistant

A simple Streamlit-based chat interface powered by Ollama's deepseek-r1 model.

## Features
- Real-time chat interface
- Persistent conversation history
- Clear and intuitive UI
- Markdown support for formatted responses
- Chat history clearing functionality

## Prerequisites
Make sure you have the following installed:
- Python 3.8 or higher
- Ollama (with deepseek-r1:8b model installed)

## Installation

1. Clone the repository:
```bash
git clone <[(https://github.com/milaomrani/llm-tutorial.git)]>
```
2. Install required packages:
```bash
pip install streamlit ollama
```
3. Run the application:
```bash
streamlit run stream.py
```

## Usage
- Open your web browser and navigate to the URL shown in the terminal (typically http://localhost:8501)

3. Start chatting:
   - Type your question in the input field
   - Click "Send" or press Enter
   - Wait for the AI's response
   - Use "Clear Chat" to start a new conversation

## Customization
- You can customize the chat interface by editing the `stream.py` file.
- Feel free to add more features or improve the UI as needed.

## Project Structure

## Code Overview

The application consists of two main components:

1. `get_ollama_response()`: Handles communication with the Ollama model
2. `main()`: Contains the Streamlit UI logic and manages the chat state

## Configuration

The application uses Streamlit's session state to maintain conversation history between interactions. The system message can be modified in the code to change the AI's behavior.

## Troubleshooting

Common issues and solutions:

1. If Ollama is not responding:
   - Ensure Ollama service is running
   - Check if the deepseek-r1:8b model is properly installed

2. If the UI is not updating:
   - Clear your browser cache
   - Restart the Streamlit server

## Contributing

Feel free to submit issues and enhancement requests!

## License

[Your chosen license]

## Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Powered by [Ollama](https://ollama.ai/)


