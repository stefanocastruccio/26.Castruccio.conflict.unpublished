#!/bin/bash
# Malaria Atlas Project PfPR 2-10 (coverage Malaria__202608_Global_Pf_Parasite_Rate, annual 2000-2025, public WCS, no login),
# clipped to 40S-72N, one GeoTIFF per year (~430 MB, 2.5 arcmin). Network: run on the login node.
D=${WORK:-/groups/scastruc/conflict_spatial_work}/covariates/raw/map; mkdir -p $D
C=Malaria__202608_Global_Pf_Parasite_Rate
for y in $(seq 2000 2025); do
  f=$D/pfpr_$y.tif; [ -s $f ] && file $f | grep -q TIFF && { echo "$y present"; continue; }
  curl -s --retry 5 --max-time 1800 -o $f.part \
    "https://data.malariaatlas.org/geoserver/Malaria/wcs?service=WCS&version=2.0.1&request=GetCoverage&coverageId=$C&format=image/geotiff&subset=Lat(-40,72)&subset=time(%22${y}-01-01T00:00:00.000Z%22)"
  file $f.part | grep -q TIFF && mv $f.part $f && echo "$y $(du -m $f | cut -f1) MB" || { echo "$y FAILED: $(head -c 200 $f.part)"; rm -f $f.part; }
done
