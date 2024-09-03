from gradio_client import Client

context = []
data = {}

def set_data(new_data): 
    global data
    data = new_data

def answer(question): 
    global context
    

    client = Client("vilarin/Llama-3.1-8B-Instruct")
    context_str = "\n".join(context)
    result = client.predict(
        message=f"Instructions: Be minimally verbose\nContext: {context_str}\n\nQuestion: {question}\n\nData: {str(data)}",
        system_prompt="You are a chatbot assistant tasked to help the user monitor and understand training loss data.",
        temperature=0.8,
        max_new_tokens=1024,
        top_p=1,
        top_k=20,
        penalty=1.2,
        api_name="/chat"
    )
    context.append(f"User: {question}")
    context.append(f"Chatbot: {result}")
    return result
print(answer("What's a transformer model? "))
print(answer("Hmmm... "))