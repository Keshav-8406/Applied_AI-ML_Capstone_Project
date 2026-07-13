# Part 4 — LLM-Powered Feature: Structured Extraction, Tabular Batch Scoring, or Model Prediction Explanation

## Project Overview

In this project, I created an AI House Analysis Assistant using a Large Language Model (LLM).

The main goal of this project is to take one house record from a tabular dataset and generate a simple explanation for buyers.

Instead of only showing numbers from the dataset, my program uses an LLM to explain the property in easy English and returns the result in structured JSON format.

---

## What I Used

- Python
- Streamlit
- Pandas
- Groq API
- Llama 3.3 70B Versatile Model
- JSON

---

## Dataset

I used the cleaned house dataset (`cleaned_data.csv`).

The user selects one house record from the dataset and my program sends that information to the LLM.

---

## Features

My application can:

- Select any house from the dataset
- Send the house details to the LLM
- Generate structured JSON output
- Show overall house quality
- Show important house features
- Show buyer advantages
- Generate a short summary for buyers

---

## JSON Output Format

The model returns output like this:

```json
{
  "overall_quality": "",
  "important_features": [],
  "buyer_advantages": [],
  "summary": ""
}
```

---

## How My Program Works

1. I load the cleaned dataset.
2. The user selects one house.
3. The selected house data is converted into a Python dictionary.
4. I create a prompt using that data.
5. The prompt is sent to the Groq LLM.
6. The model returns JSON output.
7. My program cleans the response.
8. Finally, the analysis is shown inside the Streamlit app.

---

## Example Output

### Overall Quality

Above Average

### Important Features

- 3 Bedrooms
- 1.5 Bathrooms
- Finished Basement
- Attached Garage
- Wood Deck

### Buyer Advantages

- Good neighborhood
- Public utilities available
- Central air conditioning
- Gas heating
- Private fenced backyard

### Summary

The house is well maintained and has many useful features. It is a good option for buyers looking for a comfortable family home.

---

## Files

```
Part-4/
│
├── app.py
├── cleaned_data.csv
├── requirements.txt
├── README.md
└── house_analysis_output.json
```

---

## How to Run

Install the required libraries.

```bash
pip install -r requirements.txt
```

Run the Streamlit application.

```bash
streamlit run app.py
```

---

## What I Learned

From this project I learned:

- How to use an LLM with Python
- How to write prompts for structured JSON output
- How to connect Groq API with Python
- How to build a simple Streamlit application
- How to explain tabular data using AI

---

## Conclusion

This project helped me understand how Large Language Models can explain tabular data in simple language. Instead of showing only numbers, the model creates meaningful information that can help buyers understand the property more easily.

---

---

## Final Note

This project is completed as Part 4 of my Applied AI & Machine Learning Capstone Project.

In this part, I learned how to use a Large Language Model (LLM) to explain tabular data and generate structured JSON output. I also built a simple Streamlit application to make the project interactive.

Thank you for reviewing my project.

---
Keshav Kaushik
Computer Science Student
