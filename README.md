# This is a project for animating COVID-19 cases across the country

## Obtain county level shapefiles

From: https://www.census.gov/geographies/mapping-files/time-series/geo/carto-boundary-file.html
```
wget https://www2.census.gov/geo/tiger/GENZ2018/shp/cb_2018_us_county_500k.zip
unzip cb_2018_us_county_500k.zip
mkdir cb_2018_us_county_500k
mv cb_2018_us_county_500k* cb_2018_us_county_500k/
```

Investgate these, and rasterize them.  See `main1.py`.

## Obtain COVID-19 data at the county level

For example from Johns Hopkins University:
```
git clone https://github.com/CSSEGISandData/COVID-19.git
```

## Animate

This code creates individual frames, which can be assembled into a movie.

One option is imagemagick, to combine into an animated gif.
For example:
```
convert -delay 40 *.png animation.gif
```
It may be necessary to reduce the size of some images before doing this.
For details see: https://imagemagick.org/index.php

Another option is to open an image sequence in QuickTime and save as a movie, if
using a mac.





