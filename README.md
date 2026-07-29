# Categorical Data Encoding Using Pandas and Scikit-learn

## Overview

This project demonstrates how categorical data is converted into numerical representations using various encoding techniques. Since Machine Learning algorithms work primarily with numerical data, categorical values must be transformed before model training.

The project implements:

* One Hot Encoding using Pandas.
* Label Encoding using Pandas.
* Label Encoding using Scikit-learn.
* One Hot Encoding using Scikit-learn's `ColumnTransformer`.
* DataFrame manipulation techniques.
* Feature reordering for Machine Learning datasets.

These preprocessing techniques are fundamental components of Data Preprocessing and Feature Engineering in Machine Learning workflows.

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn

---

## Skills Demonstrated

This project demonstrates practical experience with:

* Data Preprocessing
* One Hot Encoding
* Label Encoding
* Feature Engineering
* Working with Categorical Data
* Column Transformation
* DataFrame Manipulation
* Preparing Datasets for Machine Learning Models

---

## Project Workflow

```text
Load Dataset
      ↓
Identify Categorical Features
      ↓
Perform One Hot Encoding
      ↓
Merge Encoded Columns
      ↓
Remove Original Columns
      ↓
Perform Label Encoding
      ↓
Convert Categories into Numerical Values
      ↓
Apply ColumnTransformer
      ↓
Generate Machine Learning Ready Dataset
```

---

## Understanding the Dataset

The dataset contains the following features:

```text
Country
YearsExperience
Salary
Purchased
```

Example:

| Country   | YearsExperience | Salary | Purchased |
| --------- | --------------- | ------ | --------- |
| USA       | 5               | 65000  | Yes       |
| Dubai     | 3               | 45000  | No        |
| Canada    | 8               | 85000  | Yes       |
| Australia | 6               | 75000  | No        |

Machine Learning algorithms cannot directly understand:

```text
USA
Canada
Dubai
Australia

or

Yes
No
```

Therefore, these categorical values must be converted into numerical representations.

---

# Part 1: One Hot Encoding Using Pandas

## Step 1: Load the Dataset

```python
dataset = pd.read_csv("Salary_Dataset.csv")
```

Pandas loads the CSV file into a DataFrame for preprocessing.

---

## Step 2: Create Dummy Variables

```python
country_dummies = pd.get_dummies(
    dataset["country"]
)
```

### What Does `get_dummies()` Do?

It converts every category into a separate binary column.

Example:

```text
Before

Country

USA
Canada
Dubai
Australia


↓

After


Australia  Canada  Dubai  USA

0            0       0     1
0            1       0     0
0            0       1     0
1            0       0     0
```

### Why Use One Hot Encoding?

Suppose we use:

```text
USA = 1
Canada = 2
Dubai = 3
Australia = 4
```

The model may incorrectly assume:

```text
4 > 3 > 2 > 1
```

which introduces an artificial relationship between categories.

One Hot Encoding avoids this problem by treating every category independently.

---

## Step 3: Merge the Encoded Columns

```python
dataset = pd.concat(
    [dataset, country_dummies],
    axis=1
)
```

### What Does `concat()` Do?

Before:

```text
Country
Salary
Purchased
```

After:

```text
Country
Salary
Purchased
Australia
Canada
Dubai
USA
```

The newly created encoded columns are appended to the original dataset.

---

## Step 4: Remove the Original Column

```python
dataset.drop(
    "country",
    axis=1,
    inplace=True
)
```

Output:

```text
Australia
Canada
Dubai
USA
YearsExperience
Salary
Purchased
```

The original categorical feature is no longer required because its information is now represented numerically.

---

## Step 5: Rearrange Columns

```python
dataset = dataset[
    [
        "Australia",
        "Canada",
        "Dubai",
        "USA",
        "YearsExperience",
        "Salary",
        "Purchased"
    ]
]
```

Reordering columns improves readability and creates a cleaner dataset structure.

---

# Part 2: Label Encoding Using Pandas

The following code performs Label Encoding.

```python
dataset["Purchased"] = (
    dataset["Purchased"]
    .astype("category")
    .cat.codes
)
```

### Example

Before:

```text
Purchased

Yes
No
Yes
No
```

After:

```text
Purchased

1
0
1
0
```

### How Does It Work?

```python
.astype("category")
```

converts the column into a categorical data type.

```python
.cat.codes
```

assigns numerical codes to each category automatically.

Example:

```text
No  → 0

Yes → 1
```

### Why Use Label Encoding?

Label Encoding is useful when:

* Only two categories exist.
* Binary classification problems are involved.
* The categories do not require independent columns.

Examples include:

```text
Purchased

Yes
No


Gender

Male
Female


Pass

True
False
```

---

# Part 3: Label Encoding Using Scikit-learn

Scikit-learn provides an alternative implementation.

```python
from sklearn.preprocessing import LabelEncoder
```

An encoder object is created using:

```python
le = LabelEncoder()
```

The transformation is performed using:

```python
dataset["Purchased"] = (
    le.fit_transform(
        dataset["Purchased"]
    )
)
```

### What Does `fit_transform()` Do?

```text
fit()
   ↓
Learns the Categories
   ↓
transform()
   ↓
Converts Categories into Numbers
```

Example:

```text
Before

Purchased

Yes
No
Yes
No


↓

After

1
0
1
0
```

### Why Use Scikit-learn?

Scikit-learn provides:

* Cleaner Machine Learning pipelines.
* Better scalability.
* Industry-standard preprocessing techniques.
* Easier integration with Machine Learning models.

---

# Part 4: One Hot Encoding Using ColumnTransformer

Scikit-learn also provides automated column transformations.

```python
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OneHotEncoder
```

The transformer is created using:

```python
ct = ColumnTransformer(

    transformers=[
        (
            "encoding",
            OneHotEncoder(),
            [0]
        )
    ],

    remainder="passthrough"
)
```

### Understanding the Parameters

```python
("encoding", OneHotEncoder(), [0])
```

means:

```text
encoding
     ↓
Transformation Name


OneHotEncoder()
     ↓
Encoding Technique


[0]
     ↓
First Column

(country)
```

### What Does `remainder="passthrough"` Mean?

```text
Country
     ↓
Encoded

YearsExperience
     ↓
Remain Unchanged

Salary
     ↓
Remain Unchanged

Purchased
     ↓
Remain Unchanged
```

Without:

```python
remainder="passthrough"
```

all remaining columns would be discarded.

---

## Applying the Transformation

```python
dataset_1 = pd.DataFrame(

    ct.fit_transform(
        dataset_1
    )

)
```

The transformer:

1. Learns the categorical values.
2. Creates dummy variables.
3. Preserves the remaining columns.
4. Returns the transformed dataset.

---

## Renaming the Columns

```python
dataset_1.columns = [

    "Australia",
    "Canada",
    "Dubai",
    "USA",
    "YearsExperience",
    "Salary",
    "Purchased"

]
```

Output:

```text
Australia
Canada
Dubai
USA
YearsExperience
Salary
Purchased
```

The resulting dataset is now suitable for Machine Learning applications.

---

## One Hot Encoding vs Label Encoding

| Technique         | Output             | Best Used For                |
| ----------------- | ------------------ | ---------------------------- |
| Label Encoding    | 0,1,2,3            | Binary or ordered categories |
| One Hot Encoding  | Separate Columns   | Independent categories       |
| ColumnTransformer | Automated Encoding | Machine Learning Pipelines   |
| get_dummies()     | Separate Columns   | Pandas-based preprocessing   |

---

## Learning Outcomes

By completing this project, I learned:

* How categorical data is represented in Machine Learning datasets.
* How One Hot Encoding prevents artificial relationships between categories.
* How Label Encoding converts categories into numerical values.
* How `pd.get_dummies()` performs encoding in Pandas.
* How `LabelEncoder()` automates categorical transformations.
* How `ColumnTransformer()` builds scalable preprocessing pipelines.
* How Scikit-learn integrates preprocessing techniques with Machine Learning models.
* How to prepare categorical features for real-world Machine Learning applications.

---

## Conclusion

This project demonstrates several industry-standard techniques for encoding categorical data. One Hot Encoding is used when categories are independent, while Label Encoding is suitable for binary or ordered categorical variables. Scikit-learn's `ColumnTransformer` further simplifies preprocessing by combining multiple transformations into a reusable Machine Learning pipeline. These techniques form the foundation of feature engineering and are essential for preparing datasets before training Machine Learning models.
