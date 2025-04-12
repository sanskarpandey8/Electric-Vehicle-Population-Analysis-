import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("Electric_Vehicle_Population_Data.csv")

# Display first few rows
print("Dataset Preview:")
print(df.head())

# Shape of the dataset
print("\nDataset Shape:", df.shape)

# Data types and null info
print("\nInfo:")
df.info()

# Summary statistics
print("\nSummary Statistics:")
print(df.describe(include='all'))

# Missing values per column
print("\nMissing Values:")
print(df.isnull().sum())

#Drop Unnecessary Columns (if any) Columns like VIN (1-10) may not be useful for analysis.
df.drop(columns=['VIN (1-10)'], inplace=True)

# Fill missing ZIP codes with mode
df['Postal Code'].fillna(df['Postal Code'].mode()[0], inplace=True)

# Drop rows with missing Electric Range (if necessary for range-based analysis)
df = df.dropna(subset=['Electric Range'])

# Confirm changes
print("\nMissing Values After Handling:")
print(df.isnull().sum())


# Convert Model Year to integer
df['Model Year'] = pd.to_numeric(df['Model Year'], errors='coerce')

# Convert Postal Code to string to preserve leading zeros
df['Postal Code'] = df['Postal Code'].astype(str)

# Convert Base MSRP and Electric Range to numeric
df['Base MSRP'] = pd.to_numeric(df['Base MSRP'], errors='coerce')
df['Electric Range'] = pd.to_numeric(df['Electric Range'], errors='coerce')

#Overview of Numerical Columns
print("\nNumerical Columns Summary:")
print(df[['Model Year', 'Base MSRP', 'Electric Range']].describe())

#Unique Value Counts for Key Categorical Features
print("\nUnique Makes:", df['Make'].nunique())
print("Unique Models:", df['Model'].nunique())
print("Electric Vehicle Types:", df['Electric Vehicle Type'].value_counts())
print("CAFV Eligibility:", df['Clean Alternative Fuel Vehicle (CAFV) Eligibility'].value_counts())

# Check and remove duplicates
print("\nDuplicate Rows:", df.duplicated().sum())
df = df.drop_duplicates()

# Plotting a bar chart to visualize the count of electric vehicles by their make using data from the dataset
make_counts = df['Make'].value_counts()

plt.figure(figsize=(16, 8))
plt.bar(make_counts.index, make_counts.values, color='red')
plt.xlabel('Vehicle Make', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.title('Count of Vehicle by Make', fontsize=14)
plt.xticks(rotation=75)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


#Heatmap - Correlation Matrix
plt.figure(figsize=(10, 6))
numeric_df = df.select_dtypes(include=[np.number])
correlation = numeric_df.corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Heatmap")
plt.show()

#Covariance Matrix
print(" Covariance Matrix:")
print(numeric_df.cov())

#Outlier Detection - Electric Range
plt.figure(figsize=(8, 5))
sns.boxplot(x=df['Electric Range'], color='tomato')
plt.title("Outlier Detection - Electric Range")
plt.xlabel("Range (miles)")
plt.show()

#Replicate Model Year Distribution by Make (Grouped by EV Type)
top_makes = df['Make'].value_counts().head(30).index
df_filtered = df[df['Make'].isin(top_makes)]

palette = {
    'Battery Electric Vehicle (BEV)': 'blue',
    'Plug-in Hybrid Electric Vehicle (PHEV)': 'orangered'
}

plt.figure(figsize=(18, 8))
sns.boxplot(
    data=df_filtered,
    x='Make',
    y='Model Year',
    hue='Electric Vehicle Type',
    palette=palette
)

plt.title("Model Year Distribution by Make", fontsize=14)
plt.xlabel("Make")
plt.ylabel("Model Year")
plt.xticks(rotation=65)
plt.legend(title="Electric Vehicle Type", loc='lower center', bbox_to_anchor=(0.5, -0.35), ncol=2)
plt.tight_layout()
plt.show()


#Objective 1: Most Popular Electric Vehicle Makes
#Which electric vehicle manufacturers are most common?

top_makes = df['Make'].value_counts().head(10)
plt.figure(figsize=(10, 6))
sns.barplot(x=top_makes.values, y=top_makes.index, palette='viridis')
plt.title("Top 10 Electric Vehicle Makes")
plt.xlabel("Number of Vehicles")
plt.ylabel("Make")
plt.show()



#Objective 2: EV Type Distribution
#What are the proportions of different types of electric vehicles?

plt.figure(figsize=(6, 6))
df['Electric Vehicle Type'].value_counts().plot.pie(autopct='%1.1f%%', colors=['skyblue', 'lightgreen', 'salmon'])
plt.title("Electric Vehicle Type Distribution")
plt.ylabel("")
plt.show()



#Objective 3: Electric Range Distribution
#What is the distribution of electric driving range across vehicles?

plt.figure(figsize=(10, 5))
sns.histplot(df['Electric Range'], kde=True, color='purple', bins=40)
plt.title("Electric Range Distribution")
plt.xlabel("Range (in miles)")
plt.ylabel("Count")
plt.show()


#Objective 4: MSRP vs Electric Range
#Is there a relationship between the price (MSRP) and electric range?

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Base MSRP', y='Electric Range', hue='Electric Vehicle Type', alpha=0.7)
plt.title("Base MSRP vs Electric Range")
plt.xlabel("Base MSRP")
plt.ylabel("Electric Range (miles)")
plt.xlim(0, 150000) #remove
plt.show()



#Objective 5: Annual EV Registration Trends
#How has the number of electric vehicle registrations changed over the years?

yearly = df['Model Year'].value_counts().sort_index()
plt.figure(figsize=(10, 5))
sns.lineplot(x=yearly.index, y=yearly.values, marker='o')
plt.title("Electric Vehicle Registrations by Model Year")
plt.xlabel("Model Year")
plt.ylabel("Number of Vehicles")
plt.grid(True)
plt.show()


#Objective 6: CAFV Eligibility Analysis
#How many vehicles are CAFV eligible?

plt.figure(figsize=(7, 5))
sns.countplot(data=df, y='Clean Alternative Fuel Vehicle (CAFV) Eligibility', order=df['Clean Alternative Fuel Vehicle (CAFV) Eligibility'].value_counts().index, palette='Set2')
plt.title("CAFV Eligibility Status")
plt.xlabel("Count")
plt.ylabel("CAFV Eligibility")
plt.show()

#Objective 7: Top Cities with Highest EV Count
#Which cities have the highest number of EVs?

top_cities = df['City'].value_counts().head(10)
plt.figure(figsize=(10, 6))
sns.barplot(x=top_cities.values, y=top_cities.index, palette='coolwarm')
plt.title("Top 10 Cities by Electric Vehicle Count")
plt.xlabel("Number of EVs")
plt.ylabel("City")
plt.show()



# #Objective 8: Distribution by Electric Utility
# #What utilities support the most electric vehicles?

top_utilities = df['Electric Utility'].value_counts().head(10)
plt.figure(figsize=(10, 6))
sns.barplot(x=top_utilities.values, y=top_utilities.index, palette='magma')
plt.title("Top 10 Electric Utilities by EV Support")
plt.xlabel("Vehicle Count")
plt.ylabel("Utility")
plt.show()

# #Objective 9: County-wise EV Spread
# #Which counties have the most electric vehicles?

top_counties = df['County'].value_counts().head(10)
plt.figure(figsize=(10, 6))
sns.barplot(x=top_counties.values, y=top_counties.index, palette='cividis')
plt.title("Top 10 Counties by Electric Vehicle Count")
plt.xlabel("Number of EVs")
plt.ylabel("County")
plt.show()





