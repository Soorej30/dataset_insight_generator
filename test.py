import json
import requests
import pandas as pd

# def ollama_chat(prompt, schema):
#     r = requests.post(
#         "http://localhost:11434/api/generate",
#         json={
#             "model": "llama3.1:8b",
#             "prompt": prompt,
#             "stream": False,
#             "format": schema
#         }
#     )
#     return r.json()["response"]

# electric_data = pd.read_csv('Electric_Vehicle_Population_Data.csv')
# cols = list(electric_data.columns,)
# data_dict = pd.read_csv('Data Dictionary.csv')
# data_dict

# if len(data_dict) > 0:
#     prompt = f"""
#             You are a data analyst analysing a dataset. You should try and understand the data from the given instructions.
#             Consider this list of columns in the data file I have, and consider this data dictionary explaining the columns in the data
#             columns: {cols}
#             data dictionary: {data_dict}

#             Considering this data description, give me the list of the main 5 columns for which creating a categorical visualization analysis will help me understand the data.
#             Also tell me why you chose these.
#         """
#     print(prompt)
#     schema = {
#         "type": "object",
#         "properties": {
#             "columns": {
#                 "type": "array",
#                 "items": {
#                     "type": "string",
#                     "description": "Unique column name"
#                 },
#             },
#             "reason":{
#                 "type": "string",
#                 "description": "Why these columns where chosen over the others."
#             }
#         },
#         "required": ["columns", "reason"]
#     }

#     optimized_resume_text = ollama_chat(prompt, schema)

#     print(json.loads(optimized_resume_text)['columns'])
# else:
#     prompt = f"""
#             You are a data analyst analysing a dataset. You should try and understand the data from the given instructions.
#             Consider this list of columns in the data file I have
#             columns: {cols}

#             Considering this column names and try to understand what these columns will mean.
#             Then give me the list of 5 categorical columns which will give me the most information about the data.
#             DO NOT CHOOSE FIELDS THAT COULD HAVE MORE THEN 30 CATEGORIES.
#             Also tell me why you chose these.

#             RETURN ONLY THE 5 COLUMN NAMES AND THE REASON
#         """
#     print(prompt)
#     schema = {
#         "type": "object",
#         "properties": {
#             "columns": {
#                 "type": "array",
#                 "items": {
#                     "type": "string",
#                     "description": "Unique column name"
#                 },
#                 "description": "The list of 5 columns that provides most information about the data.",
#                 "max_length": "Maximum length of 5 elements"
#             },
#             "reason":{
#                 "type": "string",
#                 "description": "Why these columns where chosen over the others."
#             }
#         },
#         "required": ["columns", "reason"]
#     }

#     optimized_resume_text = ollama_chat(prompt, schema)

#     print(json.loads(optimized_resume_text))

data_dict = pd.read_csv('Data Dictionary.csv')
num_cols = ['Postal Code', 'Model Year', 'Electric Range', 'Legislative District', 'DOL Vehicle ID', '2020 Census Tract']
cat_cols = ['VIN (1-10)', 'County', 'City', 'State', 'Make', 'Model', 'Electric Vehicle Type', 'Clean Alternative Fuel Vehicle (CAFV) Eligibility', 'Vehicle Location', 'Electric Utility']

filtered_df = data_dict[data_dict['Column_name'].isin(num_cols)]
print(filtered_df)