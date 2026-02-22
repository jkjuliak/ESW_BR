import pandas as pd
import numpy as np
from datetime import datetime


class DataPreprocessor:
    
    @staticmethod
    def load_and_clean(filepath: str) -> pd.DataFrame:
        df = pd.read_csv(filepath)
        
        print(f"Loaded {len(df)} rows")
        print(f"Columns: {df.columns.tolist()}")
        df['Date_clean'] = df['Date'].str.replace(r'\s+(EST|EDT)$', '', regex=True)
        df['timestamp'] = pd.to_datetime(df['Date_clean'], errors='coerce')
        
        failed_dates = df['timestamp'].isna().sum()
        if failed_dates > 0:
            print(f"\n dates could not be parsed and will be excluded")
            df = df.dropna(subset=['timestamp'])
        
        df = df.replace('', np.nan)
        
        numeric_cols = [col for col in df.columns if col not in ['Date', 'Excel Time', 'timestamp']]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        print(f"\nAfter cleaning: {df.shape}")
        print(f"Missing values per column:")
        print(df[numeric_cols].isnull().sum())
        
        return df
    
    @staticmethod
    def aggregate_zones(df: pd.DataFrame) -> pd.DataFrame:
        """Example: SUPPLY FLOW (2), (3), (4) -> combined list
        """
        result = df.copy()
        
        # Find all SUPPLY FLOW columns
        supply_flow_cols = [col for col in df.columns if 'SUPPLY FLOW' in col and '(' in col]
        # Find all VAV RHT STEMP columns
        vav_temp_cols = [col for col in df.columns if 'VAV RHT STEMP' in col and '(' in col]
        # Find all Zone Temp columns
        zone_temp_cols = [col for col in df.columns if 'Zone Temp' in col and '(' in col]
        
        print(f"\nFound columns:")
        print(f"  Supply flows: {supply_flow_cols}")
        print(f"  VAV temps: {vav_temp_cols}")
        print(f"  Zone temps: {zone_temp_cols}")
        
        # For each row, collect non-NaN values into lists
        result['vav_flows_list'] = df[supply_flow_cols].values.tolist()
        result['vav_supply_temps_list'] = df[vav_temp_cols].values.tolist()
        result['zone_temps_list'] = df[zone_temp_cols].values.tolist()
        
        # Calculate total supply flow (sum of all zones)
        result['total_supply_flow'] = df[supply_flow_cols].sum(axis=1, skipna=True)
        
        return result
    
    @staticmethod
    def forward_fill_sparse_data(df: pd.DataFrame, columns: list, max_gap_minutes: int = 15) -> pd.DataFrame:
        """
        Forward fill sparse data (like SAV 1 Supply Air Temp that updates every 15 min)

        """
        result = df.copy()
        result = result.set_index('timestamp')
        
        for col in columns:
            if col in result.columns:
                # Forward fill with limit
                result[col] = result[col].ffill(limit=max_gap_minutes // 5)  # 5-min intervals
        
        return result.reset_index()
    
    @staticmethod
    def add_missing_ahu_data(df: pd.DataFrame, 
                            outdoor_temp_default: float = 32.0,
                            ahu_supply_temp_col: str = 'SAV 1 Supply Air Temp') -> pd.DataFrame:
        """
        Add missing AHU-level data required by the pipeline
        
        """
        result = df.copy()

        if 'OA TEMP' not in result.columns:
            print(f"\nWARNING: No outdoor temperature data found.")
            print(f"Using default value: {outdoor_temp_default}°F")
            print("For accurate results, add actual outdoor temperature data!")
            result['OA TEMP'] = outdoor_temp_default
        
        # Use total supply flow as AHU supply flow
        if 'total_supply_flow' in result.columns:
            result['AHU HRU ENT SFLOW'] = result['total_supply_flow']
        
        # Use SAV 1 Supply Air Temp as AHU supply temperature
        if ahu_supply_temp_col in result.columns:
            result['AHU HRU Coil LAT'] = result[ahu_supply_temp_col]
        
        return result
    
    @staticmethod
    def prepare_for_pipeline(filepath: str, 
                            output_filepath: str = None,
                            outdoor_temp_source: str = None) -> pd.DataFrame:

        # Cleaned DataFrame ready for HVAC energy calculations
        
        df.load_and_clean(filepath)
        
        df.aggregate_zones(df)
        
        sparse_cols = ['SAV 1 Supply Air Temp']
        df.forward_fill_sparse_data(df, sparse_cols)
        
        df.add_missing_ahu_data(df)
        
        if outdoor_temp_source:
            print(f"\n5. Loading outdoor temperature from {outdoor_temp_source}...")
            outdoor_df = pd.read_csv(outdoor_temp_source)
            outdoor_df['timestamp'] = pd.to_datetime(outdoor_df['timestamp'])

            df = df.merge(outdoor_df[['timestamp', 'outdoor_temp']], 
                         on='timestamp', how='left')
            df['OA TEMP'] = df['outdoor_temp']
        
        ######edit if needed for new lab spaces.
        required_cols = ['timestamp', 'AHU HRU ENT SFLOW', 'AHU HRU Coil LAT', 'total_supply_flow']
        df_clean = df.dropna(subset=required_cols)
        
        if output_filepath:
            df_clean.to_csv(output_filepath, index=False)
        
        print(f"\nFinal dataset: {len(df_clean)} rows")
        print(f"Date range: {df_clean['timestamp'].min()} to {df_clean['timestamp'].max()}")

        print(f"  - timestamp: Combined date/time")
        print(f"  - AHU HRU ENT SFLOW: Total supply flow")
        print(f"  - AHU HRU Coil LAT: Supply air temperature")
        print(f"  - OA TEMP: Outdoor temperature")
        print(f"  - vav_flows_list: List of VAV flows per row")
        print(f"  - zone_temps_list: List of zone temps per row")
        
        return df_clean

