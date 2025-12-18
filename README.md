[# Lithuanian Renewable Energy Calculator CLI]

[## Installation]
[bash] [git clone https://github.com/AScotM/energy_calculator.git] [cd energy_calculator] [pip install -r requirements.txt] []

[## Usage Examples]

[To list available Lithuanian cities:]
[bash] [python energy_cli.py list] []

[To calculate solar potential for Vilnius:]
[bash] [python energy_cli.py solar Vilnius --area 150 --panel monocrystalline] []

[To calculate wind potential for Klaipėda:]
[bash] [python energy_cli.py wind Klaipėda --height 120 --capacity 5000] []

[To calculate biomass potential for Žemaitija region:]
[bash] [python energy_cli.py biomass Žemaitija --area 200] []

[To compare multiple cities for solar potential:]
[bash] [python energy_cli.py compare Vilnius Kaunas Klaipėda --type solar] []

[To create energy mix scenario for Kaunas:]
[bash] [python energy_cli.py scenario Kaunas --target 60] []

[To export solar results to JSON:]
[bash] [python energy_cli.py export solar Vilnius --format json --filename vilnius_solar] []

[## Available Commands]

[list - List all supported Lithuanian cities]
[solar - Calculate solar energy potential]
[wind - Calculate wind energy potential]
[biomass - Calculate biomass energy potential]
[compare - Compare multiple cities]
[scenario - Create energy mix scenario]
[export - Export results to file]

[## Options Summary]

[Solar command options: --area (m²), --panel (monocrystalline|polycrystalline|thin_film)]
[Wind command options: --height (m), --capacity (kW)]
[Biomass command options: --area (hectares)]
[Compare command options: --type (solar|wind)]
[Scenario command options: --target (percentage)]
[Export command options: --format (csv|json), --filename (output name)]

[## Supported Cities]
[Vilnius, Kaunas, Klaipėda, Šiauliai, Panevėžys, Alytus, Marijampolė, Mažeikiai, Jonava, Utena]

[## Dependencies]
[pandas, numpy, folium, plotly, geopy, tabulate, matplotlib]
