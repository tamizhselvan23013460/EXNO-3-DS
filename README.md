## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
```
import pandas as pd
df=pd.read_csv('/content/Encoding Data.csv')
df
```
![image](https://github.com/user-attachments/assets/f06ea673-fe59-435c-b43d-96709dee12a7)

```
df.shape
```
![image](https://github.com/user-attachments/assets/46104b5b-81f0-42a9-8fca-790aa15ffa50)

```
df.info()
```

![image](https://github.com/user-attachments/assets/4ba94d5d-0331-4088-89d6-72045b21c968)

```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[['ord_2']])
```

![image](https://github.com/user-attachments/assets/100731a5-aa5a-4476-a89a-3328b2500712)

```
df['bo_2']=e1.fit_transform(df[['ord_2']])
df
```
![image](https://github.com/user-attachments/assets/d6c8a66c-ed58-49a9-b098-00a1aedb5c6d)

```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(df['ord_2'])
dfc
```

![image](https://github.com/user-attachments/assets/fba9fec9-4402-4c27-bd12-56c04cebe0eb)

```
dfc=df.copy()
```
```
dfc['con_2']=le.fit_transform(df['ord_2'])
dfc
```
![image](https://github.com/user-attachments/assets/a97c03c2-1ece-4c7b-b06b-5c236ed7eee7)

```
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse_output=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[['nom_0']]))
df2=pd.concat([df2,enc],axis=1)
df2
```
![image](https://github.com/user-attachments/assets/766058a5-4d93-442d-8590-7b07c597cd82)

```
pd.get_dummies(df2,columns=['nom_0'])
```
![image](https://github.com/user-attachments/assets/38058120-1bf8-4b88-bab2-138cc7a30f26)


```
pip install --upgrade category_encoders
```
![image](https://github.com/user-attachments/assets/83ca22a4-3ade-42a5-9714-155648dae17b)

```
from category_encoders import BinaryEncoder
```
```
df=pd.read_csv('/content/data.csv')
df
```
![image](https://github.com/user-attachments/assets/c0ee897a-eb8a-47eb-8bc0-784252a2deea)

```
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
dfb=pd.concat([df,nd],axis=1)
sfb1=df.copy()
dfb
```
![image](https://github.com/user-attachments/assets/ab40f9d0-28c0-4a0d-8977-77f3f758a336)

```
from category_encoders import TargetEncoder
```
```
te=TargetEncoder()
cc=df.copy()
new = te.fit_transform(X=cc["City"],y=cc["Target"])
```
```
cc=pd.concat([cc,new],axis=1)
cc
```
![image](https://github.com/user-attachments/assets/b1aa6a40-4e42-45e8-9796-d3b9301e60bf)

```
import pandas as pd
from scipy import stats
import numpy as np
```
```
df=pd.read_csv('/content/Data_to_Transform (1).csv')
df
```
![image](https://github.com/user-attachments/assets/b249c85c-f5c7-477c-8d09-448257209b0f)

```
df.shape
```
![image](https://github.com/user-attachments/assets/a474f579-c157-4bd7-9d2e-73de8b7d0043)

```
df.skew()
```
![image](https://github.com/user-attachments/assets/986213d9-bd13-4460-9ff6-c9e523b90b03)

```
np.log(df['Highly Positive Skew'])
```
![image](https://github.com/user-attachments/assets/afb2ece3-3007-4171-96f3-2e12dd7aaf6e)

```
np.reciprocal(df['Moderate Negative Skew'])
```
![image](https://github.com/user-attachments/assets/8d85e3f2-c215-4c65-8719-22d6391555c6)

```
np.sqrt(df['Highly Positive Skew'])
```
![image](https://github.com/user-attachments/assets/6edf6572-dcfc-4b2b-894e-2b304abc4785)

```
df.skew()
```
![image](https://github.com/user-attachments/assets/e255047c-0ccc-4700-9c65-3b29639bcde5)


```
df['Highly Positive Skew']=np.sqrt(df['Highly Positive Skew'])
df
```
![image](https://github.com/user-attachments/assets/9da07f05-e2b0-4753-a692-e0fe56056bd9)

```
df.skew()
```
![image](https://github.com/user-attachments/assets/9e368a1a-b65e-4522-bc4e-29d53dc6b9d3)

```
df['Highly Positive Skew_boxcox'],parameters=stats.boxcox(df['Highly Positive Skew'])
df
```
![image](https://github.com/user-attachments/assets/76286e75-41d3-46a9-886e-895c27e3ff54)

```
df['Moderate Negative Skew_yeojohnson'],parameters=stats.yeojohnson(df['Moderate Negative Skew'])
df
```
![image](https://github.com/user-attachments/assets/b04615f5-3400-4656-b73c-65374fe226a4)

```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
```
```
sm.qqplot(df['Moderate Negative Skew'],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/9f1cd58c-5402-49f0-a478-667b3bea2611)

```
sm.qqplot(np.reciprocal(df['Moderate Negative Skew']),line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/c00ec446-47cf-4854-8df6-c62a67f6ddb1)

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
```
```
df['Moderate Negative Skew']=qt.fit_transform(df[['Moderate Negative Skew']])

sm.qqplot(df['Moderate Negative Skew'],line='45')
plt.show()
```

![image](https://github.com/user-attachments/assets/71bd5663-3200-4734-b2df-ea416278e204)

# RESULT:
Thus Feature encodind and transformation process is performed on the given data.



       
