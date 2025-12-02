"""
HVAC Energy Consumption Calculator :))))

LOOK AT THIS TODAY, and let me know if you guys have any questions, divide this up amongst yourselves or run them together
on Jimmy's extracted data, and then talk about how we could graph this data. 

this is gonna calculate
- Mixed air temperature
- Return air temperature
- AHU thermal power consumption
- Fan power consumption
- VAV box reheat energy consumption IS IN THE WORKS not in this file.
- Total energy consumption
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class HVACConstants:
    C_P_AIR = 1005  # Specific heat for air in J/(kg·C)
    CFM_TO_KG_S = 0.0004719 * 1.2  # Conversion factor from CFM to kg/s (includes density)
    BOILER_EFFICIENCY = 0.85  # Coefficient of boiler


class HVACDataLoader:
    
    @staticmethod
    def load_io_csv(filepath: str, datetime_col: str = 'timestamp') -> pd.DataFrame:
        """
        Load CSV file with I/O point data
        
        arguments --> filepath: Path to CSV file
        returns --> datetime_col: Name of datetime column
            
        returns
            DataFrame with datetime index
        """
        df = pd.DataFrame(pd.read_csv(filepath))
        if datetime_col in df.columns:
            df[datetime_col] = pd.to_datetime(df[datetime_col])
            df.set_index(datetime_col, inplace=True)
        return df
    
    @staticmethod
    def load_multiple_io_files(filepaths: Dict[str, str]) -> Dict[str, pd.DataFrame]:
        # Load multiple CSV files
        return {name: HVACDataLoader.load_io_csv(path) 
                for name, path in filepaths.items()}


class Equation1:
    # Mixed Air Temperature and Return Air Temperature 
    
    @staticmethod
    def calculate_mixed_air_temp(T_out: float, T_ra: float, r: float) -> float:
        return r * T_out + (1 - r) * T_ra
    
    @staticmethod
    def calculate_return_air_temp(m_vav_list: List[float], T_vav_list: List[float]) -> float:
        # m_vav_list --> List of mass flow rates for each VAV box (kg/s)
        # T_vav_list --> the List of temperatures for each VAV box (°C)
            
        numerator = sum(m * T for m, T in zip(m_vav_list, T_vav_list))
        denominator = sum(m_vav_list)
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator # this is return air temperature
    
    @staticmethod
    def calculate_fresh_air_ratio(T_mix: float, T_out: float, T_ra: float) -> float:
        # Calculate fresh air ratio from temperatures, MAKE SURE ALL OF THEM ARE IN CELCIUS
        # T_mix --> Mixed air temperature 
        # T_out --> Outdoor air temperature 
        # T_ra -->  Return air temperature 
            
       
        if abs(T_out - T_ra) < 0.01:  # div by zero checker
            return 0.0
        
        r = (T_mix - T_ra) / (T_out - T_ra)
        return np.clip(r, 0, 1)  # Ensure r is between 0 and 1, fresh air ratio


class Equation2:
    # AHU Thermal Power Consumption
    
    @staticmethod
    def calculate_ahu_thermal_power(m_s_ahu: float, c_p: float, 
                                   T_r_ahu: float, T_mix: float) -> float:
        # Calculate AHU thermal power consumption
        # m_s_ahu: AHU supply air mass flowrate (kg/s)
        # c_p: Air specific heat (J/(kg·°C))
        # T_r_ahu: Return air temperature (°C)
        # T_mix: Mixed air temperature (°C)
            
        return m_s_ahu * c_p * (T_r_ahu - T_mix) # AHU thermal power consumption (watts)
    
    @staticmethod
    def sum_vav_mass_flow(m_vav_list: List[float]) -> float:
        # Sum mass flow rates from all VAV boxes
        # m_vav_list --> list of all flow rates
       
        return sum(m_vav_list) #Total mass flow rate (kg/s)


class Equation3:
    # Fan Power Consumption (verify if this is true Pranati i chatted the fan energy info tbh)
    
    @staticmethod
    def calculate_fan_power(a1: float, a2: float, a3: float, a4: float,
                          m_s_ahu: float, m_rated: float, P_rated: float) -> float:
        
        #  Calculate fan power consumption using polynomial curve
        
        # a1, a2, a3, a4: Coefficients of fan power consumption curve
        # m_s_ahu --> AHU supply air mass flowrate (kg/s)
        # m_rated -->  Rated mass flowrate (kg/s)
        # P_rated --> Rated power (W)
            
        m_ratio = m_s_ahu / m_rated if m_rated > 0 else 0
        
        return (a1 * (m_ratio**3) + a2 * (m_ratio**2) + 
                a3 * m_ratio + a4) # Fan power consumption (W)
    
    @staticmethod
    def calculate_ashrae_fan_power(m_s_ahu: float, m_rated: float, P_rated: float) -> float:
        """
        # Calculate fan power using ASHRAE 90.1 Appendix G curve: 
        P* = 0.82(m*)³ + 0.18(m*)
        """
        
        # m_s_ahu --> Current mass flowrate (kg/s)
        # m_rated --> Rated mass flowrate (kg/s)
        # P_rated --> Rated power (W)
            
        if m_rated == 0:
            return 0
        
        m_star = m_s_ahu / m_rated
        P_star = 0.82 * (m_star**3) + 0.18 * m_star
        
        return P_star * P_rated # Fan power consumption (W)
    
    @staticmethod
    def get_ashrae_coefficients(P_rated: float, m_rated: float) -> Dict[str, float]:
        """
        Get ASHRAE curve coefficients
        
        Returns:
            Dictionary with a1, a2, a3, a4 coefficients
        """
        return {
            'a1': 0.82 * P_rated / (m_rated**3),
            'a2': 0,
            'a3': 0.18 * P_rated / m_rated,
            'a4': 0
        }

