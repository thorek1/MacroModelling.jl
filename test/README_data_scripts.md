# Euro Area Data Download Scripts

This directory contains scripts for downloading and processing Euro Area macroeconomic data.

## Files

### R Script (Original)
- `getEAdata.R` - Original R script that downloads and processes Euro Area data from:
  - AWM (Area-Wide Model) database
  - Conference Board TED database (hours worked)
  - Eurostat API (various macroeconomic series)

### Julia Script (New)
- `getEAdata.jl` - Julia translation of the R script with equivalent functionality

## Julia Script Usage

### Prerequisites

Install required Julia packages:
```julia
using Pkg
Pkg.add(["CSV", "DataFrames", "XLSX", "HTTP", "JSON3", "Statistics", "Interpolations"])
```

### Required Data Files

Before running the script, download these data files to the `data/` directory:

1. **AWM Database** (`awm19up15.csv`)
   - Source: https://eabcn.org/data/area-wide-model
   - Description: Area-Wide Model database for the Euro Area (ECB)

2. **Conference Board TED Database** (`TED---Output-Labor-and-Labor-Productivity-1950-2015.xlsx`)
   - Description: Hours worked data from The Conference Board

### Running the Script

```bash
cd test
julia getEAdata.jl
```

Or from Julia REPL:
```julia
include("getEAdata.jl")
```

### Key Functions

The Julia script provides these main functions (equivalent to R counterparts):

#### Data Download
- `download_eurostat_bulk(dataset_code)` - Downloads Eurostat data in TSV format

#### Data Processing  
- `chain(to_rebase, basis, date_chain)` - Chains two time series at a specific date
- `parse_eurostat_time(s)` - Parses Eurostat time strings (quarterly, monthly, annual)

#### Interpolation and Smoothing
- `na_approx(x)` - Linear interpolation for missing values (≈ `zoo::na.approx`)
- `na_spline(x)` - Cubic spline interpolation (≈ `zoo::na.spline`)
- `structts_smooth(x)` - Structural time series smoothing (≈ `StructTS` + `tsSmooth`)
- `hpfilter(x; λ=1600)` - Hodrick-Prescott filter (≈ `mFilter::hpfilter`)

## Package Equivalents

| R Package | Julia Equivalent | Notes |
|-----------|------------------|-------|
| `tidyverse` (dplyr, tidyr) | `DataFrames.jl` | Data manipulation |
| `lubridate` | `Dates` (stdlib) | Date handling |
| `zoo` | `Interpolations.jl` | Time series interpolation |
| `mFilter` | Custom implementation | HP filter |
| `rio` | `CSV.jl`, `XLSX.jl` | File I/O |
| `eurostat` | `HTTP.jl` + custom | Eurostat API access |
| StructTS | Custom/`StateSpaceModels.jl` | Time series smoothing |

## Data Sources

### Eurostat API
- Base URL: `https://ec.europa.eu/eurostat/api/dissemination/`
- Bulk Download: `https://ec.europa.eu/eurostat/estat-navtree-portlet-prod/BulkDownloadListing`
- Documentation: https://ec.europa.eu/eurostat/web/user-guides/data-browser/api-data-access

### Datasets Used
- `namq_10_gdp` - GDP, consumption, investment (volumes and deflators)
- `namq_10_a10` - Compensation of employees (wages)
- `namq_10_a10_e` - Employment and hours worked
- `irt_st_q` - Interest rates (3-month)
- `lfsq_pganws` - Population (working age)
- `demo_pjanbroad` - Annual population by country

## Output Files

The scripts generate:
- `EA_SW_rawdata.csv` - Raw quarterly data for all series
- `EA_SW_data.csv` - Transformed data (per capita, growth rates, etc.)

## Differences Between R and Julia Versions

1. **API Access**: 
   - R uses the `eurostat` package which wraps the Eurostat API
   - Julia uses direct HTTP calls to Eurostat's bulk download endpoint
   
2. **Time Series Libraries**:
   - R has mature `zoo` and `mFilter` packages
   - Julia uses `Interpolations.jl` with custom implementations for HP filter and structural smoothing

3. **Data Wrangling**:
   - R uses `tidyverse` (dplyr, tidyr)
   - Julia uses `DataFrames.jl` with similar but not identical syntax

4. **Kalman Smoothing**:
   - R: `StructTS()` + `tsSmooth()`
   - Julia: Custom double exponential smoothing (simplified)
   - For production: Consider `StateSpaceModels.jl` in Julia

## Notes

- The Julia version provides the same core functionality but some implementation details differ
- Eurostat API access in Julia is less mature than in R, so the script uses bulk downloads where possible
- Some smoothing algorithms are simplified; for research use, consider specialized packages
- The script is designed to be extended with the full data processing pipeline from the R version

## Testing

To verify the Julia script produces equivalent results to the R script:

1. Run both scripts with the same input data
2. Compare the output CSV files
3. Check that key series match within reasonable tolerance (allowing for minor numerical differences)

## Future Enhancements

- [ ] Implement full Kalman filter using `StateSpaceModels.jl`
- [ ] Create a proper Eurostat.jl package for easier API access
- [ ] Add automated testing to compare R and Julia outputs
- [ ] Implement parallel processing for bulk downloads
- [ ] Add visualization functions (using `Plots.jl` or `Makie.jl`)

## References

- AWM Database: https://eabcn.org/data/area-wide-model
- Eurostat API: https://ec.europa.eu/eurostat/web/user-guides/data-browser/api-data-access
- Conference Board: https://www.conference-board.org/data/economydatabase/
