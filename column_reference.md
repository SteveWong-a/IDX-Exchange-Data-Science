# CRMLS 6-Month-Focus — Column Reference Guide

> [!NOTE]
> **Source**: CRMLS (California Regional MLS) via Trestle/CoreLogic OData API  
> **Period**: Jan 2025 – Jun 2025 &nbsp;|&nbsp; **Total rows**: ~128 K &nbsp;|&nbsp; **Columns**: 80  
> **Files**: `CRMLSSold202501_filled.csv` → `CRMLSSold202506.csv`

---

## 1 — Pricing & Financial (6 cols)

| # | Column | Type | Description |
|---|--------|------|-------------|
| 1 | `OriginalListPrice` | float | First published asking price when the listing initially hit the MLS. |
| 2 | `ListPrice` | float | Most recent asking price at the time the property went under contract (may differ from original if price was adjusted). |
| 3 | `ClosePrice` | float | Final recorded sale price at close of escrow. **Primary target variable for valuation models.** |
| 4 | `TaxAnnualAmount` | float | Annual property tax amount on record. ⚠️ 99.6 % missing — almost unusable. |
| 5 | `AssociationFee` | float | Monthly HOA / association dues (0 = none). |
| 6 | `AssociationFeeFrequency` | object | How often the HOA fee is billed (e.g. Monthly, Quarterly). |

## 2 — Dates & Market Timing (6 cols)

| # | Column | Type | Description |
|---|--------|------|-------------|
| 7 | `CloseDate` | date str | Date escrow closed (YYYY-MM-DD). Defines which monthly CSV the record appears in. |
| 8 | `ListingContractDate` | date str | Date the listing agreement was signed between seller and listing agent. |
| 9 | `PurchaseContractDate` | date str | Date the buyer's purchase offer was accepted by the seller. |
| 10 | `ContractStatusChangeDate` | date str | Date the MLS status last changed (usually mirrors `CloseDate` for sold records). |
| 11 | `DaysOnMarket` | int | CDOM (Cumulative Days on Market) — calendar days from initial listing to contract. **Key market velocity metric.** Negative values exist (data quality issue). |
| 12 | `TaxYear` | float | Assessment tax year. ⚠️ ~100 % missing. |

## 3 — Property Physical Attributes (16 cols)

| # | Column | Type | Description |
|---|--------|------|-------------|
| 13 | `PropertyType` | object | High-level category: `Residential`, `ResidentialLease`, `Land`, `ManufacturedInPark`, `ResidentialIncome`, `CommercialSale`, `CommercialLease`, `BusinessOpportunity`. |
| 14 | `PropertySubType` | object | Granular type (34 values): `SingleFamilyResidence`, `Condominium`, `Townhouse`, `Apartment`, `Duplex`, etc. |
| 15 | `LivingArea` | float | Interior livable square footage (excl. garage, unfinished areas). |
| 16 | `BuildingAreaTotal` | float | Total enclosed building area including all finished/unfinished space. ⚠️ 86.5 % missing. |
| 17 | `AboveGradeFinishedArea` | float | Finished sq ft above ground level. ⚠️ 100 % missing. |
| 18 | `BelowGradeFinishedArea` | float | Finished sq ft below ground (basement). ⚠️ 99.6 % missing. |
| 19 | `BedroomsTotal` | int | Total bedroom count. |
| 20 | `BathroomsTotalInteger` | int | Total bathrooms (integer, may round half-baths). |
| 21 | `MainLevelBedrooms` | int | Bedrooms on the main/ground floor. |
| 22 | `Stories` | float | Number of stories (1 or 2 in this data). |
| 23 | `Levels` | object | Text description of levels (e.g. `MultiSplit`, `One`, `Two`). |
| 24 | `Flooring` | object | Comma-separated flooring materials (e.g. `Carpet,Tile,Wood`). |
| 25 | `FireplaceYN` | bool | Whether the property has a fireplace. |
| 26 | `FireplacesTotal` | float | Number of fireplaces. ⚠️ 100 % missing. |
| 27 | `YearBuilt` | int | Year the primary structure was originally constructed. |
| 28 | `NewConstructionYN` | bool | Whether this is a newly constructed property (never occupied). |

## 4 — Lot & Land (5 cols)

| # | Column | Type | Description |
|---|--------|------|-------------|
| 29 | `LotSizeAcres` | float | Lot size in acres. |
| 30 | `LotSizeSquareFeet` | float | Lot size in sq ft. |
| 31 | `LotSizeArea` | float | Lot size in the native unit reported (usually sq ft). |
| 32 | `LotSizeDimensions` | object | Free-text lot dimensions (e.g. "50x100"). ⚠️ 94.6 % missing. |
| 33 | `SubdivisionName` | object | Named subdivision or tract. |

## 5 — Parking & Garage (4 cols)

| # | Column | Type | Description |
|---|--------|------|-------------|
| 34 | `AttachedGarageYN` | bool | Whether the garage is attached to the dwelling. |
| 35 | `GarageSpaces` | float | Number of garage parking spaces. |
| 36 | `ParkingTotal` | float | Total parking spaces (garage + driveway + uncovered). |
| 37 | `CoveredSpaces` | float | Covered (but not garaged) parking spaces. ⚠️ 100 % missing. |

## 6 — Boolean Feature Flags (4 cols)

| # | Column | Type | Description |
|---|--------|------|-------------|
| 38 | `ViewYN` | bool | Whether the property has a notable view. |
| 39 | `WaterfrontYN` | bool/float | Whether the property is waterfront. ⚠️ 99.9 % missing. |
| 40 | `BasementYN` | bool/float | Whether the property has a basement. ⚠️ 98.3 % missing. |
| 41 | `PoolPrivateYN` | bool | Whether the property has a private pool. |

## 7 — Location & Geography (8 cols)

| # | Column | Type | Description |
|---|--------|------|-------------|
| 42 | `UnparsedAddress` | object | Full street address as a single string. |
| 43 | `City` | object | City name. |
| 44 | `StateOrProvince` | object | State (virtually all `CA`). |
| 45 | `PostalCode` | object | ZIP code. |
| 46 | `CountyOrParish` | object | County name (e.g. `Los Angeles`, `Orange`, `San Bernardino`). |
| 47 | `MLSAreaMajor` | object | MLS-defined geographic sub-area code/name. |
| 48 | `Latitude` | float | Property latitude (WGS-84). |
| 49 | `Longitude` | float | Property longitude (WGS-84). |

## 8 — Listing Agent / Office (9 cols)

| # | Column | Type | Description |
|---|--------|------|-------------|
| 50 | `ListAgentFullName` | object | Listing agent's full name. |
| 51 | `ListAgentFirstName` | object | Listing agent's first name. |
| 52 | `ListAgentLastName` | object | Listing agent's last name. |
| 53 | `ListAgentEmail` | object | Listing agent's email. |
| 54 | `ListAgentAOR` | object | Listing agent's Association of Realtors board. |
| 55 | `ListOfficeName` | object | Listing brokerage office name. |
| 56 | `CoListOfficeName` | object | Co-listing brokerage office (if any). |
| 57 | `CoListAgentFirstName` | object | Co-listing agent first name. |
| 58 | `CoListAgentLastName` | object | Co-listing agent last name. |

## 9 — Buyer Agent / Office (6 cols)

| # | Column | Type | Description |
|---|--------|------|-------------|
| 59 | `BuyerAgentMlsId` | object | Buyer's agent MLS ID. |
| 60 | `BuyerAgentFirstName` | object | Buyer's agent first name. |
| 61 | `BuyerAgentLastName` | object | Buyer's agent last name. |
| 62 | `BuyerAgentAOR` | object | Buyer's agent AOR board. |
| 63 | `BuyerOfficeName` | object | Buyer's brokerage office name. |
| 64 | `CoBuyerAgentFirstName` | object | Co-buyer agent first name. ⚠️ 92 % missing. |

## 10 — School Information (6 cols)

| # | Column | Type | Description |
|---|--------|------|-------------|
| 65 | `ElementarySchool` | object | Assigned elementary school. ⚠️ 89.2 % missing. |
| 66 | `MiddleOrJuniorSchool` | object | Assigned middle school. ⚠️ 89.2 % missing. |
| 67 | `HighSchool` | object | Assigned high school. ⚠️ 85.9 % missing. |
| 68 | `ElementarySchoolDistrict` | object | Elementary school district. ⚠️ 100 % missing. |
| 69 | `MiddleOrJuniorSchoolDistrict` | float | Middle school district. ⚠️ 100 % missing. |
| 70 | `HighSchoolDistrict` | object | High school district. |

## 11 — MLS Identifiers & System (5 cols)

| # | Column | Type | Description |
|---|--------|------|-------------|
| 71 | `ListingKey` | int | Unique primary key for the listing in the MLS system. |
| 72 | `ListingKeyNumeric` | int | Numeric version of ListingKey (identical in this dataset). |
| 73 | `ListingId` | object | Human-readable MLS listing ID (e.g. `PF21138545`). |
| 74 | `MlsStatus` | object | MLS status — all `Closed` in this sold dataset. |
| 75 | `StreetNumberNumeric` | float | Numeric portion of the street address. |

## 12 — Other / Miscellaneous (5 cols)

| # | Column | Type | Description |
|---|--------|------|-------------|
| 76 | `BuilderName` | object | Builder/developer name (for new construction). ⚠️ 96.1 % missing. |
| 77 | `BusinessType` | object | Business type for commercial/business-opportunity listings. ⚠️ 99.7 % missing. |
| 78 | `BuyerOfficeAOR` | object | Buyer office's AOR board. |
| 79 | `latfilled` | bool | Flag: was Latitude imputed/geocoded (True) or natively supplied by MLS (False)? Only in the `_filled` Jan file. |
| 80 | `lonfilled` | bool | Flag: was Longitude imputed/geocoded? Only in the `_filled` Jan file. |

---

## Quick Missingness Summary

> [!WARNING]
> **Columns that are ≥ 85 % null** and likely unusable without heavy imputation:
> `ElementarySchoolDistrict`, `MiddleOrJuniorSchoolDistrict`, `FireplacesTotal`, `AboveGradeFinishedArea`, `CoveredSpaces`, `WaterfrontYN`, `TaxYear`, `BusinessType`, `TaxAnnualAmount`, `BelowGradeFinishedArea`, `BasementYN`, `BuilderName`, `LotSizeDimensions`, `CoBuyerAgentFirstName`, `ElementarySchool`, `MiddleOrJuniorSchool`, `BuildingAreaTotal`, `HighSchool`, `latfilled`, `lonfilled`

> [!TIP]
> **Cleanest numeric columns for modeling** (< 10 % missing):  
> `ClosePrice`, `ListPrice`, `OriginalListPrice`, `DaysOnMarket`, `LivingArea`, `BedroomsTotal`, `BathroomsTotalInteger`, `YearBuilt`, `LotSizeSquareFeet`, `GarageSpaces`, `Stories`
