#Question 1
def reverse_by_n_elements(lst, n):
    result = []
    
    for i in range(0, len(lst), n):
        
        temp = lst[i:i+n]
        
        
        start, end = 0, len(temp) - 1
        while start < end:
            temp[start], temp[end] = temp[end], temp[start]
            start += 1
            end -= 1
        
        
        result.extend(temp)
    
    return result

#Question 2
def group_by_length(strings):
    
    length_dict = {}
    
    
    for string in strings:
        length = len(string)  
        
        
        if length not in length_dict:
            length_dict[length] = []
        
        
        length_dict[length].append(string)
    
   
    return dict(sorted(length_dict.items()))

#Question 3
def flatten_dict(d, parent_key='', sep='.'):
    items = []  
    for key, value in d.items():
        
        new_key = f"{parent_key}{sep}{key}" if parent_key else key
        
        
        if isinstance(value, dict):
            items.extend(flatten_dict(value, new_key, sep=sep).items())
        
        
        elif isinstance(value, list):
            for i, v in enumerate(value):
                list_key = f"{new_key}[{i}]"
                items.extend(flatten_dict({list_key: v}, '', sep=sep).items())
        
        
        else:
            items.append((new_key, value))
    
    return dict(items)

#Question 4
def unique_permutations(nums):
    def backtrack(path, used):
        
        if len(path) == len(nums):
            result.append(path[:])  
            return
        
        for i in range(len(nums)):
            
            if used[i]:
                continue
            if i > 0 and nums[i] == nums[i - 1] and not used[i - 1]:
                continue
            
            
            used[i] = True
            path.append(nums[i])
            
            
            backtrack(path, used)
            
            
            path.pop()
            used[i] = False

    nums.sort() 
    result = []
    used = [False] * len(nums) 
    backtrack([], used)  
    return result

#Question 5
import re

def find_all_dates(text):
    
    patterns = [
        r'\b\d{2}-\d{2}-\d{4}\b',  
        r'\b\d{2}/\d{2}/\d{4}\b',  
        r'\b\d{4}\.\d{2}\.\d{2}\b' 
    ]
    
    
    combined_pattern = '|'.join(patterns)
    
    
    dates = re.findall(combined_pattern, text)
    
    return dates

#Question 6
import polyline
import pandas as pd
import numpy as np

def haversine(coord1, coord2):
    
    R = 6371000  
    lat1, lon1 = np.radians(coord1)  
    lat2, lon2 = np.radians(coord2)

    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1

    a = np.sin(delta_lat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(delta_lon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))

    return R * c  

def decode_polyline_and_create_dataframe(polyline_str):
    
    coordinates = polyline.decode(polyline_str)
    
    
    df = pd.DataFrame(coordinates, columns=['latitude', 'longitude'])
    
    
    distances = [0]  
    for i in range(1, len(df)):
        dist = haversine((df.loc[i-1, 'latitude'], df.loc[i-1, 'longitude']),
                         (df.loc[i, 'latitude'], df.loc[i, 'longitude']))
        distances.append(dist)
    
    df['distance'] = distances  
    return df
#Question 7
def rotate_matrix(matrix):
    n = len(matrix)
    
    
    rotated = [row[::-1] for row in matrix]
    
    rotated = [[rotated[j][i] for j in range(n)] for i in range(n)]
    
    
    transformed = [[0] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            
            row_sum = sum(rotated[i])  
            col_sum = sum(rotated[k][j] for k in range(n))  
            
            
            transformed[i][j] = row_sum + col_sum - rotated[i][j]  

    return transformed
#Question 8
import pandas as pd

def check_time_completeness(df):
    
    df['start_datetime'] = pd.to_datetime(df['startDay'] + ' ' + df['startTime'])
    df['end_datetime'] = pd.to_datetime(df['endDay'] + ' ' + df['endTime'])
    
    
    grouped = df.groupby(['id', 'id_2'])
    
    results = {}
    
    for (id_val, id_2_val), group in grouped:
        
        full_days = pd.date_range(start=group['start_datetime'].min().floor('D'), 
                                   end=group['end_datetime'].max().ceil('D'), 
                                   freq='D')
        
        
        days_present = group['start_datetime'].dt.day_name().unique()
        all_days = {'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'}
        complete_days_check = all_days.issubset(days_present)
        
        
        time_coverage_check = (group['start_datetime'].min().time() <= pd.Timestamp('00:00:00').time() and 
                               group['end_datetime'].max().time() >= pd.Timestamp('23:59:59').time())
        
        
        results[(id_val, id_2_val)] = not (complete_days_check and time_coverage_check)
    
    
    boolean_series = pd.Series(results, dtype=bool)
    
    return boolean_series
