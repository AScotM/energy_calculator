[Usage Examples]

[To list available Lithuanian cities:]
[python energy_cli.py list]

[To calculate solar potential for Vilnius:]
[python energy_cli.py solar Vilnius --area 150 --panel monocrystalline]

[To calculate wind potential for Klaipėda:]
[python energy_cli.py wind Klaipėda --height 120 --capacity 5000]

[To calculate biomass potential for Žemaitija region:]
[python energy_cli.py biomass Žemaitija --area 200]

[To compare multiple cities for solar potential:]
[python energy_cli.py compare Vilnius Kaunas Klaipėda --type solar]

[To create energy mix scenario for Kaunas:]
[python energy_cli.py scenario Kaunas --target 60]

[To export solar results to JSON:]
[python energy_cli.py export solar Vilnius --format json --filename vilnius_solar]
