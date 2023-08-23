import pandas as pd

# Sample training data
data = {
    'Sky': ['Sunny', 'Rainy', 'Sunny', 'Sunny'],
    'AirTemp': ['Warm', 'Cold', 'Warm', 'Warm'],
    'Humidity': ['Normal', 'High', 'High', 'High'],
    'Wind': ['Strong', 'Weak', 'Weak', 'Strong'],
    'Water': ['Warm', 'Warm', 'Warm', 'Cool'],
    'Forecast': ['Same', 'Change', 'Change', 'Same'],
    'EnjoySport': ['Yes', 'No', 'Yes', 'Yes']
}

# Create a DataFrame
df = pd.DataFrame(data)

# Initialize the most specific hypothesis
hypothesis = ['0'] * (len(df.columns) - 1)

# Iterate through the training examples
for index, row in df.iterrows():
    if row['EnjoySport'] == 'Yes':
        for i in range(len(hypothesis)):
            if hypothesis[i] == '0':
                hypothesis[i] = row[i]
            elif hypothesis[i] != row[i]:
                hypothesis[i] = '?'

print("Most Specific Hypothesis:", hypothesis)
