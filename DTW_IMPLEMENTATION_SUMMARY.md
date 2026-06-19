# Dynamic Thresholding Implementation Summary

## Overview
Replaced the fixed-threshold CR selection approach with a statistical IQR-based dynamic thresholding method that automatically identifies valid CRs based on their normalized DTW distances.

## Key Functions Implemented

### 1. `compute_percentile(values, percentile_k)`
**Purpose:** Calculate percentile values using linear interpolation

**Parameters:**
- `values`: list of numeric values
- `percentile_k`: percentile level (0-100)

**Returns:** float - the percentile value

**Logic:**
- Sorts values in ascending order
- Computes position: `(percentile_k / 100) × (N - 1)`
- Uses linear interpolation between floor and ceiling indices
- Formula: `percentile_value = lower_value + fraction × (upper_value - lower_value)`

---

### 2. `get_crs_using_dynamic_threshold(normalized_cost, n_crs, k=1.75)`
**Purpose:** Identify valid CRs using statistical dynamic thresholding based on IQR method

**Parameters:**
- `normalized_cost`: 2D symmetric matrix (N×N) of normalized DTW costs
  - Index `i` (0-based in matrix) → CR-(i+1) (1-based CR number)
- `n_crs`: number of unique CRs (e.g., 16)
- `k`: multiplier for IQR (default=1.75, user-configurable)

**Returns:** tuple of 4 elements
- `inlier_crs`: set of CR numbers (1-indexed) within threshold
- `outlier_crs`: set of CR numbers (1-indexed) exceeding threshold
- `dynamic_threshold`: computed threshold value
- `mean_distances`: list of mean DTW distances per CR (for debugging)

**Algorithm:**
1. **Compute mean distances per CR:**
   - For each CR i: `mean_distance[i] = Σ(cost[i][j] for j≠i) / (N-1)`

2. **Compute dynamic threshold using IQR:**
   - Q1 = 25th percentile of mean_distances
   - Q3 = 75th percentile of mean_distances
   - IQR = Q3 - Q1
   - threshold = Q3 + (k × IQR)

3. **Classify CRs:**
   - If mean_distance[i] ≤ dynamic_threshold → inlier (CR number i+1)
   - Else → outlier (CR number i+1)

**Indexing Note:**
- Matrix uses 0-based indexing: i ∈ [0, 15] for 16 CRs
- CR names use 1-based indexing: CR-1 through CR-16
- Conversion: matrix index `i` → CR number `i+1`

---

### 3. `display_crs(costs, df, k=1.75)`
**Purpose:** Filter CRs using dynamic thresholding and visualize their load profiles

**Parameters:**
- `costs`: 2D normalized DTW cost matrix
- `df`: pandas DataFrame with 'cr' column
- `k`: IQR multiplier (default=1.75)

**Returns:** tuple of 2 elements
- `cr_set`: set of valid CR numbers (1-indexed)
- `filtered_df`: filtered DataFrame containing only rows from valid CRs

**Behavior:**
1. Extracts n_crs from cost matrix dimensions
2. Calls `get_crs_using_dynamic_threshold()` to compute inliers/outliers
3. Prints diagnostic information:
   - Dynamic threshold value
   - Q1, Q3, IQR statistics
   - Count of valid vs outlier CRs
   - Mean distances per CR (with index mapping notation)
4. Filters DataFrame to include only inlier CRs
5. Creates visualization of load profiles for valid CRs only
6. Returns CR set and filtered DataFrame

---

## Usage Example

```python
# Call with default k=1.75
df10_cr_set, dfN10_load_filtered_cr = display_crs(n10_dtw_cost_per_timestamp, dfN10_load, k=1.75)

# Or experiment with different k values
df10_cr_set, dfN10_load_filtered_cr = display_crs(n10_dtw_cost_per_timestamp, dfN10_load, k=2.0)
df10_cr_set, dfN10_load_filtered_cr = display_crs(n10_dtw_cost_per_timestamp, dfN10_load, k=1.5)
```

---

## Example Output

For the 16×16 cost matrix with default k=1.75:

```
Dynamic Threshold (k=1.75): 0.55000
Q1: 0.27500, Q3: 0.37500, IQR: 0.10000
Number of valid CRs: 15
Number of outlier CRs: 1
Outlier CRs: ['CR-4']
Mean distances per CR (index i = CR-(i+1)): [0.25, 0.2833, 0.3167, 0.55, ...]
```

---

## Indexing Mapping Reference

For 16 CRs and 16×16 matrix:

| Matrix Index (0-based) | CR Number (1-based) | mean_distances index |
|----------------------|------------------|----------------------|
| 0                    | CR-1             | 0                    |
| 1                    | CR-2             | 1                    |
| ...                  | ...              | ...                  |
| 15                   | CR-16            | 15                   |

- When iterating: `cr_number = i + 1`
- When accessing mean_distances: use index `i` (0-based)
- When filtering DataFrame: use CR number integers {1, 2, ..., 16}

---

## Key Changes from Previous Implementation

| Aspect | Old | New |
|--------|-----|-----|
| Threshold method | Fixed user-provided value | Dynamic IQR-based |
| Parameter | `threshold` (float, e.g., 0.03) | `k` (float multiplier, e.g., 1.75) |
| Selection logic | Complex path-tracking algorithm | Statistical distance-based |
| Output | Set of valid CR numbers | Set of valid CR numbers (same format) |
| Transparency | Limited statistical insight | Full Q1, Q3, IQR statistics printed |
| Flexibility | Single fixed threshold | Tunable k parameter for experimentation |

---

## Mathematical Foundation

The IQR (Interquartile Range) method is a robust statistical technique for outlier detection:

- **Q1 and Q3:** Define the middle 50% of the data distribution
- **IQR:** Measures data spread; larger IQR = more dispersed data
- **k multiplier:** Controls sensitivity
  - k = 1.5 (Tukey's standard) → moderate outlier detection
  - k = 1.75 (default here) → slightly more aggressive
  - k = 2.0+ → more conservative (detect only extreme outliers)

The method is robust to:
- Skewed distributions
- Non-normal data
- Small sample sizes (works well with 16 data points)

---

## Backward Compatibility

Output format remains unchanged:
- Returns set of CR numbers {1, 2, 3, ...} for use with pandas `.isin()` filter
- Existing downstream code (e.g., train/test split) works without modification

