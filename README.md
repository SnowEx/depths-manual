# depths-manual repo
This repository has scripts to process manually collected snow depths and depth profiles collected during the NASA SnowEx field campaigns (2017-2023)

# data types
- __Manual depths__
Manual snow depths are point measurements collected using a probe, such as an avalanche probe or a SnowMetrics snow depth probe. The probe is inserted vertically into the snowpack until it meets the ground surface. The resulting depth is recorded as a single value, usually to the nearest half centimeter.
Measurement uncertainty can result from overprobing (pushing the probe into the substrate) or underprobing (failing to penetrate a dense or icy layer near the base). 

- __Depth profiles__
Depth profiles record both the top and bottom boundaries of the snowpack at a given point, typically using a ruler or probe in combination with excavation. The top of the snowpack is measured from the snow surface, while the bottom is determined by visually identifying the interface between snow and the underlying surface (e.g., soil or vegetation) after excavating the snow.
In some environments, especially vegetated areas, the snowpack may not rest directly on the ground, and air gaps or vegetation mats may create ambiguity in identifying the true base. Depth profiles help account for these cases, improving accuracy by distinguishing between the actual snowpack and void space. The vertical distance between the top and bottom measurements represents the total snow depth or snowpack thickness.

# organization 
There are up to four main directories organized by SnowEx field campaign year: SnowEx2017, SnowEx2020, SnowEx2021, and SnowEx2023, hereinafter referred to as S17, S20, S21, and S23.

# current repo status
- S23
  - depth-profiles
    - depth_profiles_Oct22.py
    - depth_profiles_Mar23.py
    - depth_profiles_Oct23.py


# SnowEx (brief) background: 
| Year | Campaign Type | Measurement Focus |
|------|---------------|--------------------|
| 2017 | IOP           | Colorado, focused on multiple instruments in a forest gradient. |
| 2020 | IOP, TS       | Western U.S focused on Time Series of L-band InSAR, active/passive microwave for SWE and thermal IR for snow surface temp. |
| 2021 | TS            | Western U.S, continued Time Series of L-band InSAR, also addressed prairie & snow albedo questions. |
| 2023 | IOP           | Alaska Tundra & Boreal forest, focused on addressing SWE/snow depth and albedo objectives. |

*IOP = Intense Observation Period (~2-3 week, daily observations); TS = Time Series (~3-5 month winter, weekly observations)*
