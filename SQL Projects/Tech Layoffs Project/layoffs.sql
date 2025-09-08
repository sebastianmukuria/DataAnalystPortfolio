-- Show the current schema
SELECT current_schema();  -- Expected: project_world_layoffs

-- List tables named "layoffs"
SELECT table_schema, table_name
FROM information_schema.tables
WHERE table_name = 'layoffs';

-- Set the search path
SET search_path TO project_world_layoffs, public;

--------------------------------------------------------
-- STEP 1: DATA CLEANING
--------------------------------------------------------

-- Inspect raw data
SELECT *
FROM layoffs;

-- Create a staging table for cleaning
CREATE TABLE project_world_layoffs.layoffs_staging
(
  LIKE project_world_layoffs.layoffs INCLUDING ALL
);

-- Copy data into staging table
INSERT INTO layoffs_staging
SELECT *
FROM layoffs;

-- Verify data copied
SELECT *
FROM layoffs_staging;

-- Check for duplicates using ROW_NUMBER
SELECT *,
ROW_NUMBER() OVER (
  PARTITION BY company, location, total_laid_off, date, percentage_laid_off,
               industry, source, stage, funds_raised, country, date_added
) AS row_num
FROM layoffs_staging;

-- Identify duplicate rows
WITH duplicate_cte AS (
  SELECT *,
         ROW_NUMBER() OVER (
           PARTITION BY company, location, total_laid_off, date, percentage_laid_off,
                        industry, source, stage, funds_raised, country, date_added
         ) AS row_num
  FROM layoffs_staging
)
SELECT *
FROM duplicate_cte
WHERE row_num > 1;

-- No duplicates found

-- Standardize categorical data
SELECT DISTINCT company FROM layoffs_staging;
SELECT DISTINCT industry FROM layoffs_staging;
SELECT DISTINCT country FROM layoffs_staging;

-- Example: fix inconsistent country names
UPDATE layoffs_staging
SET country = 'United Arab Emirates'
WHERE country = 'UAE';

-- Check result
SELECT DISTINCT country
FROM layoffs_staging;

-- Inspect date field
SELECT date
FROM layoffs_staging
LIMIT 1;

-- Clean percentage_laid_off (stored as text with % symbol)
UPDATE layoffs_staging
SET percentage_laid_off = REPLACE(percentage_laid_off, '%', '');

-- Convert percentage_laid_off to numeric
ALTER TABLE layoffs_staging
ALTER COLUMN percentage_laid_off
TYPE numeric
USING NULLIF(REPLACE(percentage_laid_off, '')::numeric);

-- Preview cleaned data
SELECT * 
FROM layoffs_staging
LIMIT 10;

-- Round percentages to 2 decimals
UPDATE layoffs_staging
SET percentage_laid_off = ROUND(percentage_laid_off, 2);

-- Check non-null percentages
SELECT * 
FROM layoffs_staging
WHERE percentage_laid_off IS NOT NULL;

-- Check for NULLs or blanks in key fields
SELECT *
FROM layoffs_staging
WHERE industry IS NULL OR industry = '';

SELECT *
FROM layoffs_staging
WHERE company = 'Appsmith';

-- Identify rows missing both total and percentage laid off
SELECT *
FROM layoffs_staging
WHERE percentage_laid_off IS NULL
  AND total_laid_off IS NULL;

-- Delete unusable rows (missing both key metrics)
DELETE
FROM layoffs_staging
WHERE total_laid_off IS NULL
  AND percentage_laid_off IS NULL;

-- Final cleaned dataset
SELECT *
FROM layoffs_staging;

--------------------------------------------------------
-- STEP 2: EXPLORATORY DATA ANALYSIS (EDA)
--------------------------------------------------------

-- Max values for total and percentage laid off
SELECT MAX(total_laid_off), MAX(percentage_laid_off)
FROM layoffs_staging;

-- Companies with 100% layoffs
SELECT *
FROM layoffs_staging
WHERE percentage_laid_off = 1;

-- Companies with 100% layoffs sorted by funds raised
SELECT *
FROM layoffs_staging
WHERE percentage_laid_off = 1
ORDER BY funds_raised DESC;

-- Companies with the most layoffs
SELECT company, SUM(total_laid_off)
FROM layoffs_staging
GROUP BY company
ORDER BY 2 DESC;

-- Date range of layoffs
SELECT MIN(date), MAX(date)
FROM layoffs_staging;

-- Industries with the most layoffs
SELECT industry, SUM(total_laid_off) AS Total_Laid_Off
FROM layoffs_staging
GROUP BY industry
ORDER BY 2 DESC;

-- Countries with the most layoffs + avg percentage per company
SELECT country, SUM(total_laid_off), AVG(percentage_laid_off)
FROM layoffs_staging
GROUP BY country
ORDER BY 2 DESC;

-- Total layoffs by year
SELECT DATE_PART('year', date), SUM(total_laid_off)
FROM layoffs_staging
GROUP BY DATE_PART('year', date)
ORDER BY 1 DESC;

-- Layoffs by funding stage
SELECT stage, SUM(total_laid_off)
FROM layoffs_staging
GROUP BY stage
ORDER BY 2 DESC;

-- Layoffs by month
SELECT DATE_PART('month', date) AS Month, SUM(total_laid_off)
FROM layoffs_staging
WHERE DATE_PART('month', date) IS NOT NULL
GROUP BY Month
ORDER BY 1 ASC;

-- Layoffs by year and month
SELECT DATE_PART('year', date) AS Year,
       DATE_PART('month', date) AS Month,
       SUM(total_laid_off)
FROM layoffs_staging
WHERE date IS NOT NULL
GROUP BY Year, Month
ORDER BY 1, 2 ASC;

-- Rolling total of layoffs by month/year
WITH Rolling_Total AS (
  SELECT DATE_PART('year', date) AS Year,
         DATE_PART('month', date) AS Month,
         SUM(total_laid_off) AS total_off
  FROM layoffs_staging
  WHERE date IS NOT NULL
  GROUP BY Year, Month
)
SELECT Year, Month,
       SUM(total_off) OVER (ORDER BY Year, Month) AS Rolling_total,
       total_off
FROM Rolling_Total;

-- Total layoffs per company by year
SELECT company,
       DATE_PART('year', date) AS Year,
       SUM(total_laid_off)
FROM layoffs_staging
GROUP BY company, DATE_PART('year', date)
ORDER BY 3 DESC;

-- Top 5 companies with the most layoffs per year
WITH company_year AS (
  SELECT company,
         DATE_PART('year', date) AS Year,
         SUM(total_laid_off) AS total_off
  FROM layoffs_staging
  GROUP BY company, DATE_PART('year', date)
),
company_year_rank AS (
  SELECT *,
         DENSE_RANK() OVER (PARTITION BY Year ORDER BY total_off DESC) AS Ranking
  FROM company_year
  WHERE total_off IS NOT NULL
)
SELECT *
FROM company_year_rank
WHERE Ranking <= 5;
