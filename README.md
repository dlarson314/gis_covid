# Obtain county level shapefiles

From: https://www.census.gov/geographies/mapping-files/time-series/geo/carto-boundary-file.html
```
wget https://www2.census.gov/geo/tiger/GENZ2018/shp/cb_2018_us_county_500k.zip
unzip cb_2018_us_county_500k.zip
mkdir cb_2018_us_county_500k
mv cb_2018_us_county_500k* cb_2018_us_county_500k/
```

Investgate these, and rasterize them.  See `main1.py`.

# Obtain COVID-19 data at the county level

For example from Johns Hopkins University:
```
git clone https://github.com/CSSEGISandData/COVID-19.git
```

# Decide that animated gifs are a reasonable approach to animation

This requires creating individual frames, and using imagemagick to combine them
into an animated gif.  For example:
```
convert -delay 40 *.png animation.gif
```

