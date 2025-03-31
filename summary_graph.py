import pandas as pd
import matplotlib.pyplot as plt
import os

# Load CSV file
df = pd.read_csv("table_occupancy_log.csv")

# Fill NaN values with 0
df.fillna(0, inplace=True)

# Convert timestamp to datetime
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Sum all occupancy values per timestamp
df["Total Occupancy"] = df.iloc[:, 1:].sum(axis=1)  # Sum all columns except timestamp

# Extract date for grouping
df["date"] = df["timestamp"].dt.date

# Create output directory
output_dir = "occupancy_graphs"
os.makedirs(output_dir, exist_ok=True)

# Plot total occupancy for each day
for date, daily_data in df.groupby("date"):
    plt.figure(figsize=(12, 6))
    plt.plot(daily_data["timestamp"], daily_data["Total Occupancy"], linestyle="-", color="b", linewidth=2, label="Total Occupancy")
    plt.scatter(daily_data["timestamp"], daily_data["Total Occupancy"], color="b", s=10)  # Smaller dots
    
    plt.xlabel("Time")
    plt.ylabel("Total Occupancy")
    plt.title(f"Total Table Occupancy on {date}")
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()  # Adjust layout to fit time labels properly
    
    # Save the plot as a PNG file
    filename = os.path.join(output_dir, f"occupancy_{date}.png")
    plt.savefig(filename, dpi=300)  # High resolution
    plt.close()

    print(f"Saved: {filename}")
