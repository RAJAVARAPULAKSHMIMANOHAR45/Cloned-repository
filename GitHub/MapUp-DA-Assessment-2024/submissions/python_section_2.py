#Question 9
import pandas as pd
import numpy as np

def calculate_distance_matrix(file_path):
    # Load the dataset
    df = pd.read_csv(file_path)
    
    # Get unique IDs
    ids = pd.unique(df[['from_id', 'to_id']].values.ravel('K'))
    
    # Initialize the distance matrix
    distance_matrix = pd.DataFrame(np.zeros((len(ids), len(ids))), index=ids, columns=ids)

    # Fill in the known distances
    for _, row in df.iterrows():
        loc_a = row['from_id']
        loc_b = row['to_id']
        distance = row['distance']
        
        distance_matrix.at[loc_a, loc_b] = distance
        distance_matrix.at[loc_b, loc_a] = distance  # Ensure symmetry

    # Calculate cumulative distances
    for k in ids:
        for i in ids:
            for j in ids:
                # Update distance if a shorter path is found
                if distance_matrix.at[i, k] + distance_matrix.at[k, j] < distance_matrix.at[i, j]:
                    distance_matrix.at[i, j] = distance_matrix.at[i, k] + distance_matrix.at[k, j]

    # Set diagonal values to 0
    for loc in ids:
        distance_matrix.at[loc, loc] = 0

    return distance_matrix

#Question 10
import pandas as pd

def unroll_distance_matrix(distance_matrix):
    # Initialize an empty list to hold the results
    unrolled_data = []

    # Iterate over the distance matrix
    for id_start in distance_matrix.index:
        for id_end in distance_matrix.columns:
            if id_start != id_end:  # Exclude pairs where start and end are the same
                distance = distance_matrix.at[id_start, id_end]
                unrolled_data.append({
                    'id_start': id_start,
                    'id_end': id_end,
                    'distance': distance
                })

    # Create a DataFrame from the unrolled data
    unrolled_df = pd.DataFrame(unrolled_data)

    return unrolled_df

#Question 11
import pandas as pd
import numpy as np

def find_ids_within_ten_percentage_threshold(unrolled_df, reference_id):
    # Filter the DataFrame for the given reference_id
    distances = unrolled_df[unrolled_df['id_start'] == reference_id]['distance']
    
    # Check if the reference_id exists in the DataFrame
    if distances.empty:
        raise ValueError(f"Reference id '{reference_id}' not found in the DataFrame.")
    
    # Calculate the average distance
    average_distance = distances.mean()
    
    # Calculate the 10% threshold
    lower_bound = average_distance * 0.9
    upper_bound = average_distance * 1.1
    
    # Find IDs within the threshold
    within_threshold = unrolled_df[
        (unrolled_df['distance'] >= lower_bound) & 
        (unrolled_df['distance'] <= upper_bound)
    ]
    
    # Extract unique id_start values and sort them
    result_ids = sorted(within_threshold['id_start'].unique())
    
    return result_ids

#Question 12
import pandas as pd

def calculate_toll_rate(distance_df):
    # Define the rate coefficients
    rate_coefficients = {
        'moto': 0.8,
        'car': 1.2,
        'rv': 1.5,
        'bus': 2.2,
        'truck': 3.6
    }
    
    # Calculate toll rates for each vehicle type
    for vehicle, coefficient in rate_coefficients.items():
        distance_df[vehicle] = distance_df['distance'] * coefficient
    
    return distance_df

#Question 13
import pandas as pd
import datetime

def calculate_time_based_toll_rates(toll_df):
    # Define days and their corresponding discount factors
    weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    weekends = ['Saturday', 'Sunday']
    
    # Create example days and times for the purposes of this function
    start_days = [day for day in weekdays + weekends for _ in range(5)]
    start_times = [
        datetime.time(0, 0),  # 00:00:00
        datetime.time(10, 0),  # 10:00:00
        datetime.time(18, 0),  # 18:00:00
        datetime.time(23, 59)  # 23:59:59
    ] * 5  # Extend to match number of unique pairs
    end_days = start_days.copy()  # Assuming same day for start and end for simplicity
    end_times = [
        datetime.time(10, 0),  # 10:00:00
        datetime.time(18, 0),  # 18:00:00
        datetime.time(23, 59),  # 23:59:59
        datetime.time(0, 0)     # Next day start at 00:00:00
    ] * 5  # Extend to match number of unique pairs

    # Add new columns to the DataFrame
    toll_df['start_day'] = start_days[:len(toll_df)]
    toll_df['start_time'] = start_times[:len(toll_df)]
    toll_df['end_day'] = end_days[:len(toll_df)]
    toll_df['end_time'] = end_times[:len(toll_df)]
    
    # Apply discount factors based on the day and time
    for index, row in toll_df.iterrows():
        start_day = row['start_day']
        start_time = row['start_time']
        
        if start_day in weekdays:
            if start_time < datetime.time(10, 0):  # 00:00:00 to 10:00:00
                discount_factor = 0.8
            elif datetime.time(10, 0) <= start_time < datetime.time(18, 0):  # 10:00:00 to 18:00
                discount_factor = 1.2
            else:  # 18:00 to 23:59
                discount_factor = 0.8
        else:  # Weekend
            discount_factor = 0.7
        
        # Update vehicle tolls with the discount factor
        for vehicle in ['moto', 'car', 'rv', 'bus', 'truck']:
            toll_df.at[index, vehicle] *= discount_factor
            
    return toll_df






