import requests
import numpy as np
import pandas as pd
import folium
from folium import plugins
import plotly.graph_objects as go
from datetime import datetime
from geopy.distance import geodesic
import json
import math
import argparse
import sys
import os
from typing import Dict, List, Tuple, Optional
from tabulate import tabulate
import matplotlib.pyplot as plt

class LithuanianEnergyCLI:
    def __init__(self):
        self.city_coordinates = {
            "Vilnius": (54.6872, 25.2797),
            "Kaunas": (54.8985, 23.9036),
            "Klaipėda": (55.7033, 21.1443),
            "Šiauliai": (55.9349, 23.3137),
            "Panevėžys": (55.7372, 24.3685),
            "Alytus": (54.3959, 24.0414),
            "Marijampolė": (54.5569, 23.3548),
            "Mažeikiai": (56.3100, 22.3333),
            "Jonava": (55.0833, 24.2833),
            "Utena": (55.5000, 25.6000)
        }
        
        self.energy_constants = {
            'solar_panel_efficiency': 0.18,
            'wind_turbine_efficiency': 0.45,
            'biomass_energy_density': 4.5,
            'roof_utilization_factor': 0.7,
            'grid_loss_factor': 0.05,
            'lithuania_avg_wind_speed': 5.2,
            'lithuania_avg_solar_hours': 1690,
            'biomass_yield_per_ha': 8,
        }
        
        self.existing_facilities = {
            'wind_farms': [
                {'name': 'Akmene Wind Farm', 'capacity_mw': 216, 'location': (56.25, 22.75)},
                {'name': 'Rokiskis Wind Farm', 'capacity_mw': 72, 'location': (55.95, 25.58)},
            ],
            'solar_farms': [
                {'name': 'Kazlu Ruda Solar Park', 'capacity_mw': 13, 'location': (54.76, 23.53)},
            ],
            'biomass_plants': [
                {'name': 'Kaunas CHP', 'capacity_mw': 90, 'location': (54.8985, 23.9036)},
            ]
        }
        
        self.weather_data = {
            'monthly_solar': [20, 40, 80, 120, 180, 200, 210, 180, 120, 70, 30, 15],
            'monthly_wind_speeds': [6.2, 6.0, 5.8, 5.5, 5.2, 5.0, 4.8, 4.9, 5.3, 5.8, 6.0, 6.1],
        }

    def display_cities(self):
        print("\nAvailable Lithuanian Cities:")
        print("=" * 50)
        for i, (city, coords) in enumerate(self.city_coordinates.items(), 1):
            print(f"{i:2}. {city:15} Coordinates: {coords[0]:.4f}, {coords[1]:.4f}")
        print()

    def calculate_solar(self, city, area_m2=100, panel_type='monocrystalline'):
        location = self.city_coordinates.get(city)
        if not location:
            return None
        
        lat, lon = location
        latitude_factor = abs(55 - lat) / 10
        coastal_boost = 1.1 if city in ['Klaipėda', 'Palanga'] else 1.0
        
        panel_efficiencies = {
            'monocrystalline': 0.22,
            'polycrystalline': 0.18,
            'thin_film': 0.15
        }
        
        efficiency = panel_efficiencies.get(panel_type, 0.18)
        optimal_tilt = lat * 0.9 + 15
        
        monthly_production = []
        for month_idx, monthly_irrad in enumerate(self.weather_data['monthly_solar']):
            tilt_factor = math.sin(math.radians(optimal_tilt + 15 - abs(55 - lat)))
            temp_factor = 1 - 0.0045 * 15
            
            monthly_energy = (
                area_m2 * 
                monthly_irrad * 
                efficiency * 
                self.energy_constants['roof_utilization_factor'] * 
                tilt_factor * 
                temp_factor * 
                coastal_boost * 
                (1 - self.energy_constants['grid_loss_factor'])
            )
            monthly_production.append(round(monthly_energy))
        
        annual_production = sum(monthly_production)
        
        cost_per_kw = {
            'monocrystalline': 1200,
            'polycrystalline': 1000,
            'thin_film': 900
        }
        
        installed_capacity_kw = area_m2 * efficiency
        installation_cost = installed_capacity_kw * cost_per_kw.get(panel_type, 1000)
        feed_in_tariff = 0.10
        annual_revenue = annual_production * feed_in_tariff
        payback_years = installation_cost / annual_revenue
        
        return {
            'city': city,
            'annual_production_kwh': annual_production,
            'monthly_production_kwh': monthly_production,
            'optimal_tilt_degrees': round(optimal_tilt, 1),
            'capacity_factor': round(annual_production / (area_m2 * 1000 * efficiency * 8760) * 100, 1),
            'co2_savings_kg': round(annual_production * 0.233),
            'installation_cost_eur': round(installation_cost),
            'annual_revenue_eur': round(annual_revenue),
            'payback_years': round(payback_years, 1)
        }

    def calculate_wind(self, city, turbine_height_m=100, turbine_capacity_kw=3000):
        location = self.city_coordinates.get(city)
        if not location:
            return None
        
        base_speeds = {
            'Klaipėda': 6.0,
            'Vilnius': 4.8,
            'Kaunas': 5.0,
            'Šiauliai': 5.5,
            'Panevėžys': 5.2
        }
        
        base_wind_speed = base_speeds.get(city, self.energy_constants['lithuania_avg_wind_speed'])
        alpha = 0.14
        height_correction = (turbine_height_m / 10) ** alpha
        adjusted_wind_speed = base_wind_speed * height_correction
        
        cut_in_speed = 3.0
        rated_speed = 12.0
        cut_out_speed = 25.0
        
        def power_at_speed(v):
            if v < cut_in_speed or v > cut_out_speed:
                return 0
            elif v < rated_speed:
                return turbine_capacity_kw * ((v ** 3 - cut_in_speed ** 3) / (rated_speed ** 3 - cut_in_speed ** 3))
            else:
                return turbine_capacity_kw
        
        np.random.seed(42)
        n_samples = 10000
        weibull_c = adjusted_wind_speed * 1.12
        simulated_speeds = weibull_c * np.random.weibull(2.0, n_samples)
        simulated_power = np.array([power_at_speed(v) for v in simulated_speeds])
        
        capacity_factor = np.mean(simulated_power) / turbine_capacity_kw
        annual_production_kwh = turbine_capacity_kw * capacity_factor * 8760
        
        installation_cost = turbine_capacity_kw * 1500
        maintenance_cost_annual = installation_cost * 0.02
        feed_in_tariff = 0.08
        annual_revenue = annual_production_kwh * feed_in_tariff
        payback_years = installation_cost / (annual_revenue - maintenance_cost_annual)
        
        return {
            'city': city,
            'annual_production_kwh': round(annual_production_kwh),
            'capacity_factor_percent': round(capacity_factor * 100, 1),
            'average_wind_speed_ms': round(adjusted_wind_speed, 2),
            'optimal_turbine_height': round(turbine_height_m * (1.2 if city in ['Klaipėda', 'Palanga'] else 1.0)),
            'co2_savings_kg': round(annual_production_kwh * 0.233),
            'installation_cost_eur': round(installation_cost),
            'annual_revenue_eur': round(annual_revenue),
            'payback_years': round(payback_years, 1)
        }

    def calculate_biomass(self, region, available_land_ha=100):
        biomass_sources = {
            'agricultural_residues': {
                'straw': {'yield_tha': 2.5, 'energy_content_gj_t': 15},
                'manure': {'yield_tha': 5.0, 'energy_content_gj_t': 10},
            },
            'energy_crops': {
                'willow': {'yield_tha': 8.0, 'energy_content_gj_t': 19},
                'miscanthus': {'yield_tha': 10.0, 'energy_content_gj_t': 17},
            }
        }
        
        region_factors = {
            'Žemaitija': 1.1,
            'Aukštaitija': 0.9,
            'Dzūkija': 1.0,
            'Suvalkija': 1.2,
        }
        
        region_factor = region_factors.get(region, 1.0)
        potential_by_source = {}
        total_annual_energy_gj = 0
        
        for source_type, sources in biomass_sources.items():
            for source, data in sources.items():
                annual_yield_t = available_land_ha * data['yield_tha'] * region_factor
                annual_energy_gj = annual_yield_t * data['energy_content_gj_t']
                electricity_efficiency = 0.25
                annual_electricity_kwh = annual_energy_gj * 277.78 * electricity_efficiency
                
                potential_by_source[f"{source}"] = {
                    'annual_yield_t': round(annual_yield_t, 1),
                    'annual_electricity_kwh': round(annual_electricity_kwh),
                    'co2_savings_kg': round(annual_electricity_kwh * 0.233)
                }
                
                total_annual_energy_gj += annual_energy_gj
        
        return {
            'region': region,
            'biomass_potentials': potential_by_source,
            'total_annual_energy_gj': round(total_annual_energy_gj, 1),
            'recommended_technology': 'CHP plant' if region in ['Žemaitija', 'Aukštaitija'] else 'Biogas plant'
        }

    def compare_cities(self, cities, energy_type='solar'):
        results = []
        for city in cities:
            if energy_type == 'solar':
                result = self.calculate_solar(city)
                if result:
                    results.append({
                        'City': city,
                        'Annual Production (kWh)': result['annual_production_kwh'],
                        'Capacity Factor (%)': result['capacity_factor'],
                        'CO2 Savings (kg)': result['co2_savings_kg'],
                        'Payback (years)': result['payback_years']
                    })
            elif energy_type == 'wind':
                result = self.calculate_wind(city)
                if result:
                    results.append({
                        'City': city,
                        'Annual Production (kWh)': result['annual_production_kwh'],
                        'Capacity Factor (%)': result['capacity_factor_percent'],
                        'Avg Wind Speed (m/s)': result['average_wind_speed_ms'],
                        'Payback (years)': result['payback_years']
                    })
        
        return pd.DataFrame(results)

    def create_scenario(self, city, target_percent=50):
        population_factors = {
            'Vilnius': 540000,
            'Kaunas': 300000,
            'Klaipėda': 150000,
            'Šiauliai': 100000,
            'Panevėžys': 90000
        }
        
        annual_consumption_kwh = population_factors.get(city, 100000) * 2500
        target_energy = annual_consumption_kwh * (target_percent / 100)
        
        solar_potential = self.calculate_solar(city, area_m2=1000000)
        wind_potential = self.calculate_wind(city, turbine_capacity_kw=5000)
        
        available_sources = {
            'solar': solar_potential['annual_production_kwh'] if solar_potential else 0,
            'wind': wind_potential['annual_production_kwh'] if wind_potential else 0,
        }
        
        costs = {'solar': 800, 'wind': 1200}
        mix = {}
        remaining_target = target_energy
        
        for source_name, potential in sorted(available_sources.items(), key=lambda x: costs[x[0]]):
            allocation = min(potential, remaining_target)
            if allocation > 0:
                mix[source_name] = {
                    'allocation_kwh': allocation,
                    'percentage': round(allocation / target_energy * 100, 1),
                    'required_capacity_kw': round(allocation / 8760 * 1.2),
                    'investment_eur': round(allocation / 8760 * 1.2 * costs[source_name])
                }
                remaining_target -= allocation
        
        total_investment = sum(v['investment_eur'] for v in mix.values())
        annual_savings = annual_consumption_kwh * 0.15
        payback_period = total_investment / annual_savings
        
        return {
            'city': city,
            'target_percent': target_percent,
            'current_consumption_kwh': annual_consumption_kwh,
            'target_renewable_kwh': target_energy,
            'optimal_mix': mix,
            'total_investment_eur': total_investment,
            'co2_reduction_kg': round(target_energy * 0.233),
            'payback_years': round(payback_period, 1)
        }

    def export_results(self, data, filename, format='csv'):
        if format == 'csv':
            if isinstance(data, pd.DataFrame):
                data.to_csv(f"{filename}.csv", index=False)
            else:
                pd.DataFrame([data]).to_csv(f"{filename}.csv", index=False)
        elif format == 'json':
            with open(f"{filename}.json", 'w') as f:
                json.dump(data, f, indent=2)
        print(f"Results exported to {filename}.{format}")

    def print_table(self, data, title):
        print(f"\n{title}")
        print("=" * 80)
        if isinstance(data, pd.DataFrame):
            print(tabulate(data, headers='keys', tablefmt='grid', showindex=False))
        elif isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, dict):
                    print(f"\n{key}:")
                    for k, v in value.items():
                        print(f"  {k}: {v}")
                else:
                    print(f"{key:30}: {value}")
        print()

def main():
    parser = argparse.ArgumentParser(description='Lithuanian Renewable Energy Potential Calculator')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    parser_list = subparsers.add_parser('list', help='List available cities')
    
    parser_solar = subparsers.add_parser('solar', help='Calculate solar potential')
    parser_solar.add_argument('city', help='City name')
    parser_solar.add_argument('--area', type=float, default=100, help='Area in m² (default: 100)')
    parser_solar.add_argument('--panel', choices=['monocrystalline', 'polycrystalline', 'thin_film'], 
                            default='monocrystalline', help='Panel type')
    
    parser_wind = subparsers.add_parser('wind', help='Calculate wind potential')
    parser_wind.add_argument('city', help='City name')
    parser_wind.add_argument('--height', type=float, default=100, help='Turbine height in meters')
    parser_wind.add_argument('--capacity', type=float, default=3000, help='Turbine capacity in kW')
    
    parser_biomass = subparsers.add_parser('biomass', help='Calculate biomass potential')
    parser_biomass.add_argument('region', help='Region name')
    parser_biomass.add_argument('--area', type=float, default=100, help='Area in hectares')
    
    parser_compare = subparsers.add_parser('compare', help='Compare cities')
    parser_compare.add_argument('cities', nargs='+', help='City names to compare')
    parser_compare.add_argument('--type', choices=['solar', 'wind'], default='solar', help='Energy type')
    
    parser_scenario = subparsers.add_parser('scenario', help='Create energy mix scenario')
    parser_scenario.add_argument('city', help='City name')
    parser_scenario.add_argument('--target', type=float, default=50, help='Target coverage percentage')
    
    parser_export = subparsers.add_parser('export', help='Export results')
    parser_export.add_argument('command', choices=['solar', 'wind', 'scenario'], help='Command to export')
    parser_export.add_argument('city', help='City name')
    parser_export.add_argument('--format', choices=['csv', 'json'], default='csv', help='Export format')
    parser_export.add_argument('--filename', help='Output filename')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    calculator = LithuanianEnergyCLI()
    
    if args.command == 'list':
        calculator.display_cities()
    
    elif args.command == 'solar':
        result = calculator.calculate_solar(args.city, args.area, args.panel)
        if result:
            calculator.print_table(result, f"SOLAR POTENTIAL FOR {args.city.upper()}")
        else:
            print(f"City '{args.city}' not found. Use 'list' to see available cities.")
    
    elif args.command == 'wind':
        result = calculator.calculate_wind(args.city, args.height, args.capacity)
        if result:
            calculator.print_table(result, f"WIND POTENTIAL FOR {args.city.upper()}")
        else:
            print(f"City '{args.city}' not found. Use 'list' to see available cities.")
    
    elif args.command == 'biomass':
        result = calculator.calculate_biomass(args.region, args.area)
        calculator.print_table(result, f"BIOMASS POTENTIAL FOR {args.region.upper()}")
    
    elif args.command == 'compare':
        result = calculator.compare_cities(args.cities, args.type)
        calculator.print_table(result, f"{args.type.upper()} POTENTIAL COMPARISON")
    
    elif args.command == 'scenario':
        result = calculator.create_scenario(args.city, args.target)
        calculator.print_table(result, f"ENERGY MIX SCENARIO FOR {args.city.upper()}")
    
    elif args.command == 'export':
        if args.command == 'solar':
            data = calculator.calculate_solar(args.city)
        elif args.command == 'wind':
            data = calculator.calculate_wind(args.city)
        elif args.command == 'scenario':
            data = calculator.create_scenario(args.city)
        
        filename = args.filename or f"{args.command}_{args.city}"
        calculator.export_results(data, filename, args.format)

if __name__ == "__main__":
    main()
