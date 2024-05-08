# SMA-vs-EWMA
# Title: Validation of Simple Moving Averages and Exponential Weighted Moving Averages for analysing Acute Chronic Workload Ratios using correlation and similarity measures in professional football


To determine the most effective method for analyzing time series data concerning the Acute Chronic Workload Ratio (ACWR), this study explored correlations between the Simple Moving Average (SMA) and the Exponentially Weighted Moving Average (EWMA) over a 7/28-day period using a decay factor (λ). Our analysis included five GPS metrics: Total Distance, Distance in Speed Zones 3+4+5 (>19,9 km/h), High Metabolic Load Distance, Accelerations , and Decelerations, gathered from 22 players across 47 days—excluding the initial 28 days, totaling 596 data points per pair [SMA/EWMA].


> This project is designed for statistical analysis and paper submission, written in **Python** **Anaconda’s Jupyter Notebook**  utilizing libraries such as *Pandas*, *NumPy*, *Matplotlib*, *SciPy*, *Scikit-Learn*,  *Statsmodels*, *OpenPyXL*, *Dcor*, and *IPython.display*.
> 
## Team Structure

#### 
Vladimir Vuksanovikj - University of St Cyril and Methodius1 (Faculty of Physical Education Sport and Health, Skopje, Macedonia);
>
Mihailo Sejkeroski - FC Slavija- Sofija (Football Club, Sofia, Bulgaria);
>
Nuno André Nunes - Solent University (Southampton, England, United Kingdom); 
>
Elena Soklevska Ilievski - European University (Skopje, Macedonia);
>
Aleksandar Aceski - University of St Cyril and Methodius (Faculty of Physical Education Sport and Health, Skopje, Macedonia);
>
Vlatko Nedelkovski - University of St Cyril and Methodius (Faculty of Physical Education Sport and Health, Skopje, Macedonia);
>
Kostadin Kodzoman - Saba High School (PhD Student, Skopje Macedonia);


---

## Project Specification
- Descriptive
  - basic statistics

```
# Prepare a DataFrame to store the descriptive statistics
desc_stats = pd.DataFrame(columns=['Metric', 'Min', 'Max', 'Mean', 'SD', 'Skew', 'Kurtosis', 'Shapiro-Wilk', 'Kolmogorov-Smirnov'])

# Calculate descriptive statistics and normality tests
for metric in metrics:
    series = data[metric].dropna()
    desc_stats = desc_stats.append({
        'Metric': metric,
        'Min': series.min(),
        'Max': series.max(),
        'Mean': series.mean(),
        'SD': series.std(),
        'Skew': skew(series),
        'Kurtosis': kurtosis(series),
        'Shapiro-Wilk': shapiro(series)[1],
        'Kolmogorov-Smirnov': kstest(series, 'norm', args=(series.mean(), series.std()))[1]
    }, ignore_index=True)

# Define a function to apply the color formatting
def highlight_non_normal(val):
    color = 'orange' if val < 0.05 else None
    return f'background-color: {color}'

# Apply the color formatting
styled_desc_stats = desc_stats.style.applymap(
    highlight_non_normal, subset=['Shapiro-Wilk', 'Kolmogorov-Smirnov']
)
```

- #Kruskal Wallis H test
  - Non- parametric- for all sistem of variables

```
# Prepare a DataFrame to store the Kruskal-Wallis test results
kw_results = pd.DataFrame(columns=['Metric', 'H-Statistic', 'P-Value', 'Is Significant'])

# Perform the Kruskal-Wallis H test for each metric
for metric in metrics:
    # Check if the metric exists in the DataFrame to prevent KeyError
    if metric in data.columns:
        # Prepare the list of data arrays for the test, one array per player
        grouped_data = [group.dropna().values for name, group in data.groupby('Player')[metric] if len(group.dropna()) > 0]

        # Perform the Kruskal-Wallis H test if there is enough data
        if len(grouped_data) > 1:
            h_statistic, p_value = kruskal(*grouped_data)
            kw_results = kw_results.append({
                'Metric': metric,
                'H-Statistic': h_statistic,
                'P-Value': p_value,
                'Is Significant': p_value < 0.05  # Assuming a significance level of 0.05
            }, ignore_index=True)
        else:
            kw_results = kw_results.append({
                'Metric': metric,
                'H-Statistic': 'N/A',
                'P-Value': 'N/A',
                'Is Significant': 'N/A'
            }, ignore_index=True)
    else:
        print(f"Metric {metric} not found in the dataset.")

# Display the results table directly in the Jupyter Notebook
display(kw_results)
```

- Dunn posthoc test
  - non-parametric

```
for metric in metrics:
    print(f"Processing metric: {metric}")
    groups = data.groupby('Player')[metric].apply(list).to_dict()
    grouped_data = [g for g in groups.values() if len(g) > 0]

    if len(grouped_data) > 1:
        k_stat, p_val = kruskal(*grouped_data)
        if p_val < 0.05:
            flat_data = [item for sublist in grouped_data for item in sublist]
            group_labels = [k for k, v in groups.items() for _ in v]

            posthoc_data = pd.DataFrame({'Value': flat_data, 'Group': group_labels})
            p_matrix = sp.posthoc_dunn(posthoc_data, val_col='Value', group_col='Group', p_adjust='bonferroni')

            # Apply styling for significant p-values for notebook display
            styled_p_matrix = p_matrix.style.applymap(lambda x: 'background-color: #FFA07A' if x < 0.05 else '')
            display(styled_p_matrix)

            # Save to Excel with conditional formatting
            output_excel_path = f'/Users/vladimirvuksanovikj/Downloads/Shani Anova/Dunn_Post_Hoc_{metric.replace(" ", "_")}.xlsx'
            writer = pd.ExcelWriter(output_excel_path, engine='xlsxwriter')
            p_matrix.to_excel(writer, sheet_name='Post_Hoc_Results')
            workbook = writer.book
            worksheet = writer.sheets['Post_Hoc_Results']
            format_red = workbook.add_format({'bg_color': '#FFA07A'})

            # Excel conditional formatting
            worksheet.conditional_format('B2:Z1000', {  # Adjust the range according to your actual data range
                'type': 'cell',
                'criteria': '<',
                'value': 0.05,
                'format': format_red
            })
            writer.save()
            print(f"Results for {metric} saved to {output_excel_path}")
        else:
            print(f"No significant differences found for {metric} via Kruskal-Wallis (p={p_val}).")
    else:
        print(f"Not enough data to perform Kruskal-Wallis test on {metric}.")
```

- Scatter plots

```
# Create scatter plots for each pair of columns
for index, (col1, col2) in enumerate(pairs, 1):
    plt.figure(figsize=(8, 6))
    
    # Adjusting the y-values of col2 for better visibility
    adjusted_y = df[col2] + 0.5  # Shift y-values of col2 up by 0.5

    # Scatter plot settings for each variable
    plt.scatter(df[col1], df[col2], color='blue', marker='o', label=f'{col1} (Blue Circles)')
    plt.scatter(df[col1], adjusted_y, color='red', marker='s', label=f'{col2} (Red Squares)')

    # Calculate Spearman correlation
    correlation, _ = spearmanr(df[col1].dropna(), df[col2].dropna())

    # Fit and plot a non-linear trend line for each pair
    x_new = np.linspace(min(df[col1]), max(df[col1]), 100)
    params = curve_fit(quadratic_fit, df[col1].dropna(), df[col2].dropna())[0]  # Fit parameters for the quadratic fit
    y_fit = quadratic_fit(x_new, *params)  # Apply the quadratic fit function
    plt.plot(x_new, y_fit, 'g--', label='Quadratic Fit')

    plt.title(f'Scatter Plot: {col1} vs. {col2}\nSpearman\'s rho={correlation:.2f}')
    plt.xlabel(col1)
    plt.ylabel(col2)
    plt.legend()
    plt.grid(True)

    # Save the plot to a bytes object and then to an image file
    img_data = BytesIO()
    plt.savefig(img_data, format='png')
    img_data.seek(0)
    img = Image(img_data)
    # Add image to the Excel sheet
    cell_location = f'A{1 + (index - 1) * 40}'  # Adjust cell location for each image to avoid overlap
    ws.add_image(img, cell_location)
    plt.close()

# Save the workbook with the images
wb.save(destination_file)
print(f"All scatter plots have been saved to '{destination_file}'.")
```
- SMA i EWMA normality check

```
# Define the columns to test for normality
columns_to_test = [
    "EWMA_ACWR DZ_3+4+5", "DZ 3+4+5_ACWR",
    "EWMA_ACWR HMLD", "HMLD_ACWR",
    "EWMA_ACWR TD", "Total D_ACWR",
    "EWMA_ACWR ACC", "ACC_ACWR",
    "EWMA_ACWR DEC", "DEC_ACWR"
]

# Initialize a new Excel workbook
wb = Workbook()
ws = wb.active
ws.title = "Normality Test Results"

# Header for the Excel table
header = ["Column Name", "Shapiro Stat", "Shapiro P-value", "KS Stat", "KS P-value", "Skewness", "Kurtosis", "Normality"]
ws.append(header)

# Perform normality tests and measure skewness and kurtosis for each column
results = []
for column in columns_to_test:
    if column in df.columns:  # Check if the column exists in the DataFrame
        data = df[column].dropna()  # Drop NA values for the test
        shapiro_stat, shapiro_p = shapiro(data)
        ks_stat, ks_p = kstest(data, 'norm', args=(data.mean(), data.std()))
        data_skew = skew(data)
        data_kurtosis = kurtosis(data)
        is_normal = "Yes" if shapiro_p > 0.05 and ks_p > 0.05 else "No"

        # Append results
        results.append((column, shapiro_stat, shapiro_p, ks_stat, ks_p, data_skew, data_kurtosis, is_normal))

        # Write each row to the Excel sheet and apply formatting
        row = ws.max_row + 1
        ws.append([column, shapiro_stat, shapiro_p, ks_stat, ks_p, data_skew, data_kurtosis, is_normal])
        if is_normal == "No":
            for col in range(1, 9):
                ws.cell(row=row, column=col).font = Font(color="FF0000")
```
- Spearman's Rank Correlation

```
# Define the column pairs for correlation calculation
pairs = [
    ("EWMA_ACWR DZ_3+4+5", "DZ 3+4+5_ACWR"),
    ("EWMA_ACWR HMLD", "HMLD_ACWR"),
    ("EWMA_ACWR TD", "Total D_ACWR"),
    ("EWMA_ACWR ACC", "ACC_ACWR"),
    ("EWMA_ACWR DEC", "DEC_ACWR")
]

results = []
for col1, col2 in pairs:
    data1, data2 = df[col1].dropna(), df[col2].dropna()
    correlation, p_value = spearmanr(data1, data2)
    r_squared = correlation**2
    sample_size = len(data1)
    
    # Confidence interval using Fisher's z transformation approximation
    fisher_z = np.arctanh(correlation)
    se = 1 / np.sqrt(sample_size - 3)
    z_critical = norm.ppf(0.975)  # 95% confidence
    ci_lower = np.tanh(fisher_z - z_critical * se)
    ci_upper = np.tanh(fisher_z + z_critical * se)

    results.append({
        'Column Pair': f'{col1} & {col2}',
        'Correlation': correlation,
        'P-value': p_value,
        '95% Confidence Interval Lower': ci_lower,
        '95% Confidence Interval Upper': ci_upper,
        'Sample Size': sample_size
    })

# Convert results to a DataFrame
result_df = pd.DataFrame(results)
```

- Kendall's Tau Correlation

```
# Define the column pairs for correlation calculation
pairs = [
    ("EWMA_ACWR DZ_3+4+5", "DZ 3+4+5_ACWR"),
    ("EWMA_ACWR HMLD", "HMLD_ACWR"),
    ("EWMA_ACWR TD", "Total D_ACWR"),
    ("EWMA_ACWR ACC", "ACC_ACWR"),
    ("EWMA_ACWR DEC", "DEC_ACWR")
]

results = []
for col1, col2 in pairs:
    data1, data2 = df[col1].dropna(), df[col2].dropna()
    correlation, p_value = kendalltau(data1, data2)
    results.append({
        'Column Pair': f'{col1} & {col2}',
        'Correlation': correlation,
        'P-value': p_value,
        'Sample Size': len(data1)
    })

# Convert results to a DataFrame
result_df = pd.DataFrame(results)
```

- Distance Correlation

```
# Define the column pairs for correlation calculation
pairs = [
    ("EWMA_ACWR DZ_3+4+5", "DZ 3+4+5_ACWR"),
    ("EWMA_ACWR HMLD", "HMLD_ACWR"),
    ("EWMA_ACWR TD", "Total D_ACWR"),
    ("EWMA_ACWR ACC", "ACC_ACWR"),
    ("EWMA_ACWR DEC", "DEC_ACWR")
]

results = []
for col1, col2 in pairs:
    data1, data2 = df[col1].dropna(), df[col2].dropna()
    distance_corr = dcor.distance_correlation(data1, data2)
    energy_distance = dcor.energy_distance(data1, data2)
    
    # Conduct the permutation test to calculate the p-value
    # It is computationally expensive, adjust `num_resamples` if needed for faster performance
    test_result = dcor.independence.distance_covariance_test(data1, data2, num_resamples=1000)
    p_value = test_result.p_value

    results.append({
        'Column Pair': f'{col1} & {col2}',
        'Distance Correlation': distance_corr,
        'P-value': p_value,
        'Energy Distance': energy_distance,
        'Sample Size': len(data1)
    })

# Convert results to a DataFrame
result_df = pd.DataFrame(results)
```

- T-test SMA / EWMA pairs

```
# Define the pairs of variables
pairs = [
    ("DZ345_SMA", "DZ345_EWMA"),
    ("HMLD_SMA", "HMLD_EWMA"),
    ("TD_SMA", "TD_EWMA"),
    ("ACC_SMA", "ACC_EWMA"),
    ("DEC_SMA", "DEC_EWMA")
]

# Create an empty DataFrame to store results
results = pd.DataFrame(columns=['Variable Pair', 't-statistic', 'p-value'])

# Perform a paired t-test for each pair
for sma, ewma in pairs:
    # Ensure no missing data in pairs
    pair_data = data[[sma, ewma]].dropna()
    
    # Calculate the t-test on TWO RELATED samples
    t_stat, p_val = ttest_rel(pair_data[sma], pair_data[ewma])
    
    # Append the results
    results = results.append({
        'Variable Pair': f'{sma} / {ewma}',
        't-statistic': t_stat,
        'p-value': p_val
    }, ignore_index=True)

# Save the results to an Excel file
```

To download the dataset please click [HERE](documention/dataset_bank.csv)

## 

## May the Force be with you
