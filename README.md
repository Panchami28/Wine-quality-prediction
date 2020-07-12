## Introduction


"Wine has been a part of civilized life for some seven thousand years. It is the only beverage that feeds the body, soul and spirit of man and at the same time stimulates the mind..." 
This is a quote by Robert Mondavi

Having read that , I was wondering if we could predict the quality of wine when the values of its features are given to us. While doing so manually takes many years of experience,expertise and deep knowledge in the field of wine, its a seconds job with machine learning. Using suitable model and training the model using suitable dataset yields us the answer.

Wine quality prediction can be solved as classification or regression problem. Here we are solving it as a classification problem. The main objective of this project is to experiment with the various classifier models and pick out the model yielding highest accuracy. We have made use of Kaggle's red wine dataset in this project. Click [here](https://github.com/Panchami28/Wine-quality-prediction/blob/master/datasets_4458_8204_winequality-red.csv) to download the dataset. In this dataset , quality of red wine is determined by 11 input variables namely 
fixed acidity,
volatile acidity,
citric acid,
residual sugar,
chlorides,
free sulfur dioxide,
total sulfur dioxide,
density,
pH,
sulfates,
alcohol.


## Setup

We start out by importing the standard libraries

```markdown
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
%matplotlib inline
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/Panchami28/Wine-quality-prediction/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://help.github.com/categories/github-pages-basics/) or [contact support](https://github.com/contact) and we’ll help you sort it out.
