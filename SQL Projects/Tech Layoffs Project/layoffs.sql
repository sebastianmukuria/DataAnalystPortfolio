SELECT current_schema();  -- should show project_world_layoffs

SELECT table_schema, table_name
FROM information_schema.tables
WHERE table_name = 'layoffs';



SET search_path TO project_world_layoffs, public;
-- Data Cleaning

SELECT *
FROM layoffs;

-- 1. Remove Duplicates
-- 2. Standardize the Data
-- 3. Null Values or blank values
-- 4. Remove Any Columns

CREATE TABLE project_world_layoffs.layoffs_staging
(
  LIKE project_world_layoffs.layoffs INCLUDING ALL
);

INSERT INTO layoffs_staging
SELECT *
FROM layoffs;

SELECT *
FROM layoffs_staging;

SELECT *,
ROW_NUMBER() OVER
(PARTITION BY company, industry, total_laid_off, percentage_laid_off, 'date') AS row_num
FROM layoffs_staging;

WITH duplicate_cte AS
(
SELECT *,
ROW_NUMBER() OVER
(PARTITION BY company, location, total_laid_off, date, percentage_laid_off,
                   industry, source, stage, funds_raised, country, date_added) AS row_num
FROM layoffs_staging
)

SELECT *
FROM duplicate_cte
WHERE row_num > 1;

-- no duplicates found :)
-- Standardizing the data

SELECT DISTINCT(company)
FROM layoffs_staging;

SELECT DISTINCT industry
FROM layoffs_staging;

SELECT DISTINCT country
FROM layoffs_staging;

--seeing UAE and Unitaed Arab Emirates, will combine those

UPDATE layoffs_staging
SET country = 'United Arab Emirates'
WHERE country = 'UAE';

SELECT DISTINCT country
FROM layoffs_staging;

-- check date column
SELECT date
FROM layoffs_staging
LIMIT 1;

-- Noticed percentage laid off is a text field (has percentage sign, lets clean
UPDATE layoffs_staging
SET percentage_laid_off = REPLACE(percentage_laid_off, '%', '');

ALTER TABLE layoffs_staging
ALTER COLUMN percentage_laid_off
TYPE numeric
USING NULLIF(REPLACE(percentage_laid_off, '')::numeric;

SELECT * 
FROM layoffs_staging
LIMIT 10;

-- now it's numeric, but I actually want a decimal

UPDATE layoffs_staging
SET percentage_laid_off = ROUND (percentage_laid_off, 2);

SELECT * 
FROM layoffs_staging
WHERE percentage_laid_off IS NOT NULL;

-- check NULLs

SELECT *
FROM layoffs_staging
WHERE industry IS NULL
OR industry = '';

SELECT *
FROM layoffs_staging
WHERE company = 'Appsmith';

SELECT *
FROM layoffs_staging
WHERE percentage_laid_off IS NULL AND total_laid_off IS NULL

-- can probably delete this rows? since we would need these fields and I'm seeking to explore layoff data


DELETE
FROM layoffs_staging
WHERE total_laid_off IS NULL
AND percentage_laid_off IS NULL;

SELECT *
FROM layoffs_staging;

-- Looks good, moving to Exploratory Data Analysis.




-- EXPLORATORY DATA ANALYSIS 




-- highest total laid off, highest percentage laid off
SELECT MAX(total_laid_off), MAX(percentage_laid_off)
FROM layoffs_staging;

-- companies that laid off everybody 
SELECT *
FROM layoffs_staging
WHERE percentage_laid_off = 1;


-- companies that laid off everybody sorted by funds_raised
SELECT *
FROM layoffs_staging
WHERE percentage_laid_off = 1
ORDER BY funds_raised DESC;

-- companies with the most layoffs
SELECT company, SUM(total_laid_off)
FROM layoffs_staging
GROUP BY company
ORDER BY 2 DESC;

SELECT MIN (date), MAX(date)
from layoffs_staging;

-- industries with the most layoffs
SELECT industry, SUM(total_laid_off) as "Total Laid Off"
FROM layoffs_staging
GROUP BY industry
ORDER BY 2 DESC;

-- countries with the most layoffs, average percentage laid off per company (not as useful)
SELECT country, SUM(total_laid_off), AVG(percentage_laid_off)
FROM layoffs_staging
GROUP BY country
ORDER BY 2 DESC;

-- layoffs by year
SELECT DATE_PART('year',date), SUM(total_laid_off)
FROM layoffs_staging
GROUP BY DATE_PART('year',date)
ORDER  BY 1 DESC;

SELECT stage, SUM(total_laid_off)
FROM layoffs_staging
GROUP BY stage
ORDER  BY 1 DESC;

-- layoffs by month, but let's get the year as well
SELECT DATE_PART('month',date) AS Month, SUM(total_laid_off)
FROM layoffs_staging
WHERE DATE_PART('month',date) IS NOT NULL
GROUP BY Month
ORDER BY 1 ASC

-- layoffs by month/year
SELECT DATE_PART('year',date) as Year,DATE_PART('month',date) AS Month, SUM(total_laid_off)
FROM layoffs_staging
WHERE date IS NOT NULL
GROUP BY Year, Month
ORDER BY 1,2 ASC

-- rolling total by month/year with the amount added each month
WITH Rolling_Total AS
(
SELECT DATE_PART('year',date) as Year,DATE_PART('month',date) AS Month, SUM(total_laid_off) AS total_off
FROM layoffs_staging
WHERE date IS NOT NULL
GROUP BY Year, Month
ORDER BY 1,2 ASC
)
SELECT Year, Month, SUM(total_off) OVER(ORDER BY Year, Month) AS Rolling_total, total_off
FROM Rolling_total;


-- total layoffs by year for each company, sorted by amount laid off
SELECT company, DATE_PART('year',date) AS Year, SUM(total_laid_off)
FROM layoffs_staging
GROUP BY company, DATE_PART('year',date)
ORDER BY 3 DESC;


-- ranking the companies with the top 5 highest total layoffs per year 
WITH company_year AS
(
SELECT company, DATE_PART('year',date) AS Year, SUM(total_laid_off) as total_off
FROM layoffs_staging
GROUP BY company, DATE_PART('year',date)
), company_year_rank AS
(
SELECT *, DENSE_RANK() OVER (PARTITION BY Year ORDER BY total_off DESC) as Ranking
FROM company_year
WHERE total_off IS NOT NULL
)
SELECT *
FROM company_year_rank
WHERE Ranking <= 5;








