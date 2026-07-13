import streamlit as st
import pandas as pd
import json
from groq import Groq
from google.colab import userdata

# Getting Groq API key from Colab Secrets
api_key = userdata.get("Groq_API_Key")

# Creating Groq client
client = Groq(api_key=api_key)

# Loading dataset
df = pd.read_csv("cleaned_data.csv")

# Streamlit page title
st.title("🏠 AI House Analysis Assistant")

st.write(
    "Select a house record and get an AI-generated property analysis."
)

# House selection
house_id = st.selectbox(
    "Select House ID",
    df["Id"].tolist()
)

# Getting selected house
selected_house = df[df["Id"] == house_id].iloc[0]

# Converting row into dictionary
house_data = selected_house.where(
    pd.notnull(selected_house),
    None
).to_dict()

# Button
if st.button("Analyze House"):

    prompt = f"""
You are a real estate assistant.

Analyze the following house details.

Return ONLY valid JSON.

{{
    "overall_quality": "",
    "important_features": [],
    "buyer_advantages": [],
    "summary": ""
}}

House Details:

{json.dumps(house_data, indent=2)}
"""

    # Sending request to Groq
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    # Getting model response
    output = response.choices[0].message.content

    # Cleaning JSON response
    output = output.replace("```json", "")
    output = output.replace("```", "")
    output = output.strip()

    # Converting JSON string into Python dictionary
    result = json.loads(output)

    # Displaying output
    st.subheader("AI Analysis")

    st.write("### Overall Quality")
    st.write(result["overall_quality"])

    st.write("### Important Features")
    for feature in result["important_features"]:
        st.write("✔️", feature)

    st.write("### Buyer Advantages")
    for advantage in result["buyer_advantages"]:
        st.write("✔️", advantage)

    st.write("### Summary")
    st.write(result["summary"])
 
