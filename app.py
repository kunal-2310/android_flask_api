from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_community.llms import OpenAI
from langchain.prompts import ChatPromptTemplate
import os
import re
import json

app = Flask(__name__)
CORS(app)

# Load the OpenAI API Key from environment
api_key = os.getenv("OPENAI_API_KEY")

@app.route('/process/', methods=['POST'])
def process_prompt():
    data = request.get_json()

    llm = OpenAI(api_key=api_key, model="gpt-4o-mini") 
    actual_prompt = data.get('input')
    print(f"Received Prompt: {actual_prompt}")

    # Define a prompt template
    extract_fields = ChatPromptTemplate.from_messages([
        ("system", "You are a good assistant, help me in extracting some data from the prompt that is provided."),
        ("human", """This is just sample data solely meant to understand the working do not copy the data mentioned in this example:
    I will provide you a prompt like 'Assign a high-priority AC maintenance task. The site is a customer named Rajesh Sharma. Schedule it for tomorrow at 11 AM. and duration of the task is 2 hours. location is sector 21,Gurgaon'
    It is your job to analyse the initial prompt and give me the response in JSON format like:
    {{
    "description": "AC Maintenance",
    "location": "Sector 21, Gurgaon",
    "priority": "High",
    "startTime": "16-05-2025 11:00 am",
    "endTime": "16-05-2025 1:00 pm",
    "customerName": "Rajesh Sharma"
    }}"""),
    ("human","important -> do not add extra fields in response other than fields specified by me"),
        ("human", "\n\n\n\n do not generate the output based on the above provided sample that is only for reference how to generate response \
\n\n\nnow the actual prompt is '{prompt}' and if any field data is missing then keep it blank \
\n\n\n if there is nothing provided in quotes '' after actual prompt keyword then keep the structure of the response same and leave the values blank for eg- 'name':''\
\n\n\n assign the priority on the basis of the urgency stated in the initial prompt accordingly to one of these- ['Low','Normal','High','Critical'])
    ])


    # Format the prompt
    messages = extract_fields.format_messages(prompt=actual_prompt)
    final_prompt = "\n\n".join([f"{m.type.upper()}: {m.content}" for m in messages])

    # Call LLM
    result = llm.invoke(final_prompt)
    print(f"Raw LLM Response: {result}")

 # Extract JSON from string using regex
    match = re.search(r'\{.*?\}', result, re.DOTALL)
    if match:
        json_part = match.group()
        result_json = json.loads(json_part)
    else:
        result_json = {
            "error": "Model did not return JSON. Prompt may be incomplete or invalid.",
            "rawResponse": result
        } 
    return jsonify({"answer": result_json})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8000))
    app.run(host='0.0.0.0', port=port)
