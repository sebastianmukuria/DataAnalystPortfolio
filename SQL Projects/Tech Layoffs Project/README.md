# 📊 Tech Layoffs Data Analysis (SQL Project)

This project explores global tech layoffs using SQL for data cleaning and exploratory data analysis (EDA). The dataset contains company-level layoff records, including fields like company, industry, country, date, total laid off, and percentage laid off.

The goal is to demonstrate data analyst skills in:

- Data cleaning & transformation

- Handling duplicates, nulls, and inconsistent values

- Standardizing fields (dates, percentages, country names)

- Exploratory analysis of layoffs across companies, industries, countries, and time

## 📂 Dataset

This project uses this [Tech Layoffs dataset](https://www.kaggle.com/datasets/swaptr/layoffs-2022), containing data from 2019 to Sep. 2025 (credit to the original contributors).

## 🛠️ Tech Stack

- PostgreSQL (SQL queries and analysis)

- PgAdmin 4 (query execution and inspection)

## 📂 Project Workflow
1. Data Cleaning

- Removed duplicates using ROW_NUMBER() and CTEs

- Standardized fields (e.g., merging "UAE" and "United Arab Emirates")

- Converted percentage values from text with % symbols into numeric decimals

- Handled null/blank values in key fields

- Deleted rows missing both total_laid_off and percentage_laid_off

2. Exploratory Data Analysis (EDA)

- Identified companies with 100% layoffs (shutdowns)

- Ranked companies with the highest layoffs overall

- Aggregated layoffs by industry, country, funding stage, and year/month

- Created rolling totals to show layoff trends over time

- Ranked top 5 companies with the most layoffs per year using DENSE_RANK()

## 📈 Key Insights

- Certain industries (e.g., consumer internet, fintech) saw the largest layoffs.

- Layoffs peaked in specific years/months, aligning with global market shifts.

- Several companies completely shut down operations (100% layoffs).

- High-growth companies with significant funding were not immune to layoffs.

## 📜 Example Queries
Top 5 companies with the most layoffs per year:
```sql
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
```

Rolling total layoffs by month/year:
```sql
WITH Rolling_Total AS (
  SELECT DATE_PART('year', date) AS Year,
         DATE_PART('month', date) AS Month,
         SUM(total_laid_off) AS total_off
  FROM layoffs_staging
  WHERE date IS NOT NULL
  GROUP BY Year, Month
)
SELECT Year, Month,
       SUM(total_off) OVER(ORDER BY Year, Month) AS Rolling_total,
       total_off
FROM Rolling_total;
```
