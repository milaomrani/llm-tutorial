import chainlit as cl
import ollama
import asyncio
from PyPDF2 import PdfReader
import os

@cl.on_chat_start
async def start_chat():
    cl.user_session.set(
        "interaction",
        [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            }
        ],
    )
    
    msg = cl.Message(content="Hello, how can I help you today?")
    
    start_message = """Hello, I'm your local assistant.
    How can I help you today?"""
    
    for token in start_message:
        await msg.stream_token(token)
        
    await msg.send()
    
    cl.user_session.set("messages", [msg])
    
    @cl.step(type="tool")
    async def tool(input_message):
        interaction = cl.user_session.get("interaction")
        
        interaction.append({"role": "user",
                            "content": input_message})
        
        response = ollama.chat(model="deepseek-r1:8b",
                              messages=interaction)
        
        interaction.append({"role": "assistant",
                            "content": response["message"]["content"]})
        
        return response["message"]["content"]
    
    
    @cl.on_message
    async def main(message: cl.Message):
        tool_result = await tool(message.content)
        
        msg = cl.Message(content="")
        
        for token in tool_result:
            await msg.stream_token(token)
            
        await msg.send()
        
        # cl.user_session.set("messages", [*cl.user_session.get("messages"), msg])

async def read_pdf(file_path):
    try:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()  # Remove extra whitespace
    except Exception as e:
        return f"Error reading PDF: {str(e)}"

async def chat():
    # Initialize conversation history
    interaction = [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        }
    ]
    
    print("Chat initialized. Commands:")
    print("- Type 'quit' to exit")
    print("- Type 'read pdf' to load a PDF file")
    
    while True:
        user_input = input("\nYou: ")
        
        if user_input.lower() == 'quit':
            print("Goodbye!")
            break
            
        elif user_input.lower() == 'read pdf':
            pdf_path = input("Enter the path to your PDF file: ")
            if os.path.exists(pdf_path):
                print("Reading PDF...")
                pdf_content = await read_pdf(pdf_path)
                
                if pdf_content.startswith("Error reading PDF:"):
                    print(pdf_content)
                    continue
                
                # Format the prompt more clearly
                system_prompt = {
                    "role": "system",
                    "content": "You are now working with the following pdf document. Please help answer questions about it."
                }
                
                document_content = {
                    "role": "user",
                    "content": f"Document content:\n\n{pdf_content}"
                }
                
                # Reset the conversation with the new document
                interaction = [system_prompt, document_content]
                
                print("\nPDF loaded successfully! You can now ask questions about its content.")
            else:
                print("File not found. Please check the path and try again.")
            continue
        
        # Regular chat interaction
        interaction.append({
            "role": "user",
            "content": user_input
        })
        
        try:
            response = ollama.chat(
                model="deepseek-r1:8b",
                messages=interaction
            )
            
            assistant_response = response["message"]["content"]
            print("\nAssistant:", assistant_response)
            
            interaction.append({
                "role": "assistant",
                "content": assistant_response
            })
            
        except Exception as e:
            print(f"\nError: {str(e)}")

if __name__ == "__main__":
    asyncio.run(chat())
    