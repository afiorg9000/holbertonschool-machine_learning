-- selfjoin to table
SELECT origin, SUM(fans)
AS nb_fans
FROM metal_bands
GROUP BY origin
ORDER by nb_fans DESC;