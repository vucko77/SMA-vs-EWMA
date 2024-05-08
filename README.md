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
Nuno André Nunes - Solent University3 (Southampton, England, United Kingdom); 
>
Elena Soklevska Ilievski - European University (Skopje, Macedonia);
>
Aleksandar Aceski - University of St Cyril and Methodius1 (Faculty of Physical Education Sport and Health, Skopje, Macedonia);
>
Vlatko Nedelkovski - University of St Cyril and Methodius1 (Faculty of Physical Education Sport and Health, Skopje, Macedonia);
>
Kostadin Kodzoman - Saba High School5 (PhD Student, Skopje Macedonia);


---

## Project Specification
#Descriptive
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

## Dataset

To download the dataset please click [HERE](documention/dataset_bank.csv)

## 

## May the Force be with you
