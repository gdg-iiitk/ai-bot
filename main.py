import configparser
import sys
import google.generativeai as genai

config = configparser.ConfigParser()
config.read("cred.cfg")
 
try:
    API_KEY = config["key"]["api"]
except KeyError as e:
    print(e)
    sys.exit(1)
genai.configure(api_key=API_KEY)
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}
model = genai.GenerativeModel(
    model_name="gemini-1.5-pro",
    generation_config=generation_config,
)
chat_session = model.start_chat(
    history=[
        {
            "role": "user",
            "parts": [
                "I will provide you with the standard output  from terminal code executions. Please explain the error, identify its cause, and suggest a solution to fix it",
            ],
        },
        {
            "role": "model",
            "parts": [
                "Okay, I'm ready. Please provide the output.  The more context you can give me (the code you ran, the operating system, etc.), the better I can help.\n",
            ],
        },
    ]
)
stream = input()
response = chat_session.send_message(stream)
print(response.text)
