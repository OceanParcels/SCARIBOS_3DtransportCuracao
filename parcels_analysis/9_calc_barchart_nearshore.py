'''
Project: 3D flow and volume transport around Curaçao. 

In this script we calcualte ingredients for plotting of a bar chart of 
total volume transport for each nearshore segment. 

Author: V Bertoncelj
kernel: parcels_dev_local
'''

#%%
# Import libraries
import os
import pickle
import pandas as pd
import numpy as np

# Configuration - LIST ALL TARGET SECTIONS HERE
ALL_TARGET_SECTIONS = [
    ['KC_5D1'],
    ['KC_6D1'],
    ['MP_5D1'],
    ['MP_6D1'],
    ['WP_5D1'],
    ['WP_6D1']
]

DIRECTION = 'backward'  # 'backward' or 'forward'
part_months = ['Y2020M04', 'Y2020M07', 'Y2020M10', 'Y2021M01', 'Y2021M04', 'Y2021M07', 'Y2021M10', 'Y2022M01', 'Y2022M04', 'Y2022M07', 'Y2022M10', 'Y2023M01', 'Y2023M04', 'Y2023M07', 'Y2023M10', 'Y2024M01']

# Directory paths
CACHE_DIR = f'nearshore/cache_allmonths'
VT_BASE_DIR = '/nethome/berto006/transport_in_3D_project/parcels_run/VOLUME_TRANSPORT'
BAR_DATA_DIR = 'nearshore/bar_chart_data'

# Depth thresholds for categorization
SURFACE_DEPTH_THRESHOLD = -162
SUBSURFACE_DEPTH_THRESHOLD = -458.5

# Create directories
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(BAR_DATA_DIR, exist_ok=True)

def load_volume_transport_data(months_list):
    """Load volume transport data from CSV files"""
    vt_cache_file = f"{CACHE_DIR}/vt_data_all_sections_{DIRECTION}.pkl"
    
    if os.path.exists(vt_cache_file):
        print(f"Loading cached VT data from {vt_cache_file}")
        with open(vt_cache_file, 'rb') as f:
            return pickle.load(f)
    
    vt_data = {}
    for month in months_list:
        vt_file_path = f'{VT_BASE_DIR}/SAMPLEVEL_speeds_vt_{month}.csv'
        
        if not os.path.exists(vt_file_path):
            print(f"Warning: VT file not found for {month}: {vt_file_path}")
            continue
        
        try:
            df = pd.read_csv(vt_file_path)
            month_vt_data = dict(zip(df['PARTICLE_ID'], df['VT']))
            vt_data[month] = month_vt_data
            print(f"Loaded VT data for {month}: {len(month_vt_data)} particles")
        except Exception as e:
            print(f"Error loading VT data for {month}: {e}")
    
    # Cache the data
    with open(vt_cache_file, 'wb') as f:
        pickle.dump(vt_data, f)
    
    return vt_data

def load_trajectory_data(target_sections):
    """Load trajectory data and match with VT values"""
    # Check for cached combined data
    combined_cache_file = f"{CACHE_DIR}/combined_trajectory_data_vt_{'_'.join(target_sections)}_{DIRECTION}.pkl"
    
    if os.path.exists(combined_cache_file):
        print(f"Loading cached combined trajectory data from {combined_cache_file}")
        with open(combined_cache_file, 'rb') as f:
            return pickle.load(f)
    
    # Load VT data
    vt_data = load_volume_transport_data(part_months)
    
    # Initialize data structure
    all_data = {
        'particle_depths': [],
        'particle_vt_values': [],
        'particle_ids': []
    }
    
    # Load trajectory data for each section and month
    for section in target_sections:
        section_output_dir = f'nearshore/{section[:2]}/{DIRECTION}'
        
        for month in part_months:
            file_suffix = f"{section}_{DIRECTION}_{month}"
            trajectory_cache_file = f"{section_output_dir}/trajectory_data_{file_suffix}.pkl"
            
            if os.path.exists(trajectory_cache_file):
                print(f"Loading trajectory data from {trajectory_cache_file}")
                with open(trajectory_cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                    
                    # Process each particle
                    for i, (depths, particle_id) in enumerate(zip(cache_data['all_depths'], cache_data['all_particle_ids'])):
                        if not depths:
                            continue
                        
                        # Get max depth for categorization
                        max_depth = min(depths)
                        
                        # Get VT value
                        month_vt_dict = vt_data.get(month, {})
                        try:
                            if '_' in particle_id:
                                numeric_pid = int(particle_id.split('_')[-1])
                            else:
                                numeric_pid = int(particle_id)
                            
                            vt_value = month_vt_dict.get(numeric_pid, 0.0)
                        except (ValueError, TypeError):
                            vt_value = 0.0
                        
                        all_data['particle_depths'].append(max_depth)
                        all_data['particle_vt_values'].append(vt_value)
                        all_data['particle_ids'].append(f"{section}:{particle_id}")
                
                print(f"Loaded data for section {section}, month {month}")
            else:
                print(f"Warning: File {trajectory_cache_file} not found")
    
    # Cache combined data
    with open(combined_cache_file, 'wb') as f:
        pickle.dump(all_data, f)
    
    return all_data

def calculate_vt_distribution(all_data):
    """Calculate volume transport distribution by depth category"""
    categories = ['Surface', 'Mid', 'Deep']
    
    # Initialize totals
    category_totals = {
        'Surface': {'count': 0, 'total_vt': 0.0},
        'Mid': {'count': 0, 'total_vt': 0.0},
        'Deep': {'count': 0, 'total_vt': 0.0}
    }
    
    # Process each particle
    for max_depth, vt_value in zip(all_data['particle_depths'], all_data['particle_vt_values']):
        # Categorize by depth
        if max_depth >= SURFACE_DEPTH_THRESHOLD:
            category = 'Surface'
        elif SUBSURFACE_DEPTH_THRESHOLD <= max_depth < SURFACE_DEPTH_THRESHOLD:
            category = 'Mid'
        else:
            category = 'Deep'
        
        # Update totals
        category_totals[category]['count'] += 1
        category_totals[category]['total_vt'] += vt_value
    
    # Calculate percentages
    total_vt = sum(category_totals[cat]['total_vt'] for cat in categories)
    
    results = []
    for category in categories:
        vt_value = category_totals[category]['total_vt']
        vt_percentage = (vt_value / total_vt * 100) if total_vt > 0 else 0
        results.append({
            'Category': category,
            'VT_Percentage': vt_percentage,
            'VT_Value': vt_value
        })
    
    # Add total row
    results.append({
        'Category': 'Total',
        'VT_Percentage': 100.0,
        'VT_Value': total_vt
    })
    
    return results, category_totals

def save_bar_chart_data(results, target_sections):
    """Save the bar chart data to CSV"""
    sections_filename = "_".join([s.replace("_", "") for s in target_sections])
    vt_data_filename = f"{BAR_DATA_DIR}/BAR_VT_{sections_filename}_{DIRECTION}.csv"
    
    # Create DataFrame and save
    df = pd.DataFrame(results)
    df.to_csv(vt_data_filename, index=False)
    
    print(f"Saved VT distribution data to {vt_data_filename}")
    
    # Print summary
    print("\nVolume Transport Distribution Summary:")
    print(f"Section(s): {', '.join(target_sections)}")
    print(f"Direction: {DIRECTION}")
    print("-" * 40)
    for result in results:
        if result['Category'] != 'Total':
            print(f"{result['Category']}: {result['VT_Value']:.6f} Sv ({result['VT_Percentage']:.1f}%)")
        else:
            print(f"Total: {result['VT_Value']:.6f} Sv")

def main():
    """Main function to calculate and save bar chart data"""
    
    # Loop over all target sections
    for target_sections in ALL_TARGET_SECTIONS:
        print(f"\n{'='*60}")
        print(f"Processing sections: {', '.join(target_sections)}")
        print(f"{'='*60}")
        
        print("Loading trajectory data...")
        all_data = load_trajectory_data(target_sections)
        
        print(f"Processing {len(all_data['particle_depths'])} particles...")
        results, category_totals = calculate_vt_distribution(all_data)
        
        print("Saving bar chart data...")
        save_bar_chart_data(results, target_sections)
    
    print(f"\n{'='*60}")
    print("All sections processed!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()




# %%
