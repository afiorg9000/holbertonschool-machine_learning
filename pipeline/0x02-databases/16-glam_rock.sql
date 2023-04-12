-- selfjoin grlam_rock
SELECT band_name, IF(split IS NULL, 2020 - formed, split - formed)
AS lifespan
FROM metal_bands
WHERE style LIKE '%Glam rock%'
ORDER BY lifespan DESC;