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




class HVACPipeline:
    """Complete pipeline for HVAC energy calculation"""
    
    def __init__(self, constants: Optional[HVACConstants] = None):
        self.constants = constants or HVACConstants()
        
    def cfm_to_kg_s(self, cfm: float) -> float:
        """Convert CFM to kg/s"""
        return cfm * self.constants.CFM_TO_KG_S
    
    def process_timestep(self, 
                        T_out: float,
                        supply_flow_ahu: float,  # CFM
                        supply_temp_ahu: float,  # °F
                        vav_flows: List[float],  # List of CFM values
                        vav_temps: List[float],  # List of °F values (zone temps)
                        vav_supply_temps: List[float],  # Supply temps at VAV boxes
                        m_rated: float,  # Rated flow in kg/s
                        P_rated: float,  # Rated power in W
                        use_ashrae: bool = True) -> Dict[str, float]:
        """
        Process a single timestep and calculate all energy values
        
        Args:
            T_out: Outdoor temperature (°F)
            supply_flow_ahu: AHU supply flow (CFM)
            supply_temp_ahu: AHU supply temperature (°F)
            vav_flows: List of VAV box flows (CFM)
            vav_temps: List of VAV box/zone temperatures (°F)
            vav_supply_temps: List of supply temps at VAV boxes (°F)
            m_rated: Rated mass flow (kg/s)
            P_rated: Rated power (W)
            use_ashrae: Use ASHRAE fan curve (default True)
            
        Returns:
            Dictionary with all calculated values
        """
        # Convert temperatures to Celsius
        T_out_c = (T_out - 32) * 5/9
        T_s_ahu_c = (supply_temp_ahu - 32) * 5/9
        vav_temps_c = [(t - 32) * 5/9 for t in vav_temps]
        vav_supply_temps_c = [(t - 32) * 5/9 for t in vav_supply_temps]
        
        # Convert flows to kg/s
        m_s_ahu = self.cfm_to_kg_s(supply_flow_ahu)
        m_vav_list = [self.cfm_to_kg_s(flow) for flow in vav_flows]
        
        # Equation 1: Calculate return air temperature
        T_ra = Equation1.calculate_return_air_temp(m_vav_list, vav_temps_c)
        
        # Calculate mixed air temperature (need to determine r or T_mix from data)
        # For now, we'll estimate fresh air ratio or use available data
        # If you have T_mix from PHT-EAT, use that directly
        # Otherwise estimate: r = 0.2 (typical 20% outdoor air)
        r = 0.2  # This should come from system data or be calculated
        T_mix = Equation1.calculate_mixed_air_temp(T_out_c, T_ra, r)
        
        # Equation 2: AHU thermal power
        Q_ahu = Equation2.calculate_ahu_thermal_power(
            m_s_ahu, self.constants.C_P_AIR, T_s_ahu_c, T_mix
        )
        
        # Equation 3: Fan power
        if use_ashrae:
            P_fan = Equation3.calculate_ashrae_fan_power(m_s_ahu, m_rated, P_rated)
        else:
            coeffs = Equation3.get_ashrae_coefficients(P_rated, m_rated)
            P_fan = Equation3.calculate_fan_power(
                coeffs['a1'], coeffs['a2'], coeffs['a3'], coeffs['a4'],
                m_s_ahu, m_rated, P_rated
            )
        
        # Equation 4: VAV reheat power
        vav_data = list(zip(m_vav_list, vav_supply_temps_c, 
                           [f"VAV_{i}" for i in range(len(m_vav_list))]))
        Q_reheat_sum = Equation4.calculate_total_reheat_power(
            vav_data, self.constants.C_P_AIR, T_s_ahu_c
        )
        
        # Total energy
        energy_breakdown = TotalEnergyCalculator.calculate_energy_by_type(
            Q_ahu, P_fan, Q_reheat_sum, self.constants.BOILER_EFFICIENCY
        )
        
        return {
            'timestamp_values': {
                'T_out_c': T_out_c,
                'T_ra_c': T_ra,
                'T_mix_c': T_mix,
                'T_s_ahu_c': T_s_ahu_c,
                'fresh_air_ratio': r,
                'm_s_ahu_kg_s': m_s_ahu,
            },
            'power_values': {
                'Q_ahu_W': Q_ahu,
                'P_fan_W': P_fan,
                'Q_reheat_sum_W': Q_reheat_sum,
            },
            'energy_breakdown': energy_breakdown
        }
    
    def process_timeseries(self, df: pd.DataFrame,
                          column_mapping: Dict[str, str],
                          m_rated: float,
                          P_rated: float) -> pd.DataFrame:
        """
        Process entire time series
        
        Args:
            df: DataFrame with all I/O points
            column_mapping: Maps required fields to column names
            m_rated: Rated mass flow
            P_rated: Rated power
            
        Returns:
            DataFrame with calculated energy values
        """
        results = []
        
        for idx, row in df.iterrows():
            # Extract values using column mapping
            T_out = row[column_mapping['outdoor_temp']]
            supply_flow = row[column_mapping['supply_flow']]
            supply_temp = row[column_mapping['supply_temp']]
            
            # Get VAV data (this depends on your data structure)
            # You'll need to specify which columns are VAV flows and temps
            vav_flow_cols = column_mapping.get('vav_flows', [])
            vav_temp_cols = column_mapping.get('vav_temps', [])
            vav_supply_temp_cols = column_mapping.get('vav_supply_temps', [])
            
            vav_flows = [row[col] for col in vav_flow_cols if col in row.index]
            vav_temps = [row[col] for col in vav_temp_cols if col in row.index]
            vav_supply_temps = [row[col] for col in vav_supply_temp_cols if col in row.index]
            
            result = self.process_timestep(
                T_out, supply_flow, supply_temp,
                vav_flows, vav_temps, vav_supply_temps,
                m_rated, P_rated
            )
            
            # Flatten result dictionary
            flat_result = {'timestamp': idx}
            flat_result.update(result['timestamp_values'])
            flat_result.update(result['power_values'])
            flat_result.update(result['energy_breakdown'])
            
            results.append(flat_result)
        
        return pd.DataFrame(results)

####### BY CHATGPT
def main_example():

    """Example usage of the HVAC pipeline 
    DO NOT USE THIS this is a template that you can structure our analysis code around"""
    
    # Initialize pipeline
    pipeline = HVACPipeline()
    
    # Example single timestep calculation
    result = pipeline.process_timestep(
        T_out=31.0,  # °F
        supply_flow_ahu=16271.0,  # CFM
        supply_temp_ahu=65.0,  # °F
        vav_flows=[197.0, 203.396],  # CFM for each VAV
        vav_temps=[67.8, 55.1],  # °F zone temps
        vav_supply_temps=[65.0, 55.1],  # °F supply temps at VAV
        m_rated=16271.0 * 0.0004719 * 1.2,  # Convert rated CFM to kg/s
        P_rated=7500.0,  # Estimated 7.5-10 HP -> ~7500W
        use_ashrae=True
    )
    
    print("Single Timestep Results:")
    print("-" * 50)
    print("\nTemperatures and Flows:")
    for key, value in result['timestamp_values'].items():
        print(f"  {key}: {value:.3f}")
    
    print("\nPower Consumption:")
    for key, value in result['power_values'].items():
        print(f"  {key}: {value:.1f} W")
    
    print("\nEnergy Breakdown:")
    for key, value in result['energy_breakdown'].items():
        print(f"  {key}: {value:.1f} W")
    
    print("\n" + "=" * 50)
    print(f"Total Energy Consumption: {result['energy_breakdown']['total']:.1f} W")
    print(f"  Thermal (Heating): {result['energy_breakdown']['thermal_heating']:.1f} W")
    print(f"  Electrical (Fan): {result['energy_breakdown']['electrical_fan']:.1f} W")


if __name__ == "__main__":
    main_example()