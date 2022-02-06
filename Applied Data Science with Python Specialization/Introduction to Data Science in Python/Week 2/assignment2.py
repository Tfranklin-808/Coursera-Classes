#!/usr/bin/env python
# coding: utf-8

# # Assignment 2
# For this assignment you'll be looking at 2017 data on immunizations from the CDC. Your datafile for this assignment is in [assets/NISPUF17.csv](assets/NISPUF17.csv). A data users guide for this, which you'll need to map the variables in the data to the questions being asked, is available at [assets/NIS-PUF17-DUG.pdf](assets/NIS-PUF17-DUG.pdf). **Note: you may have to go to your Jupyter tree (click on the Coursera image) and navigate to the assignment 2 assets folder to see this PDF file).**

# ## Question 1
# Write a function called `proportion_of_education` which returns the proportion of children in the dataset who had a mother with the education levels equal to less than high school (<12), high school (12), more than high school but not a college graduate (>12) and college degree.
# 
# *This function should return a dictionary in the form of (use the correct numbers, do not round numbers):* 
# ```
#     {"less than high school":0.2,
#     "high school":0.4,
#     "more than high school but not college":0.2,
#     "college":0.2}
# ```
# 

# In[1]:


def proportion_of_education():
    # your code goes here
    # YOUR CODE HERE
    import pandas as pd
    mb = pd.read_csv("assets/NISPUF17.csv")
    n = mb['EDUC1'].value_counts()/mb['EDUC1'].shape[0]
    li = ['less than high school', 'high school', 'more than high school but not college', 'college']
    di = {}
    i = 0
    for l in li:
        di[l] = n[i+1] 
        i += 1
    
    return di
    #raise NotImplementedError()


# In[2]:


assert type(proportion_of_education())==type({}), "You must return a dictionary."
assert len(proportion_of_education()) == 4, "You have not returned a dictionary with four items in it."
assert "less than high school" in proportion_of_education().keys(), "You have not returned a dictionary with the correct keys."
assert "high school" in proportion_of_education().keys(), "You have not returned a dictionary with the correct keys."
assert "more than high school but not college" in proportion_of_education().keys(), "You have not returned a dictionary with the correct keys."
assert "college" in proportion_of_education().keys(), "You have not returned a dictionary with the correct keys."


# ## Question 2
# 
# Let's explore the relationship between being fed breastmilk as a child and getting a seasonal influenza vaccine from a healthcare provider. Return a tuple of the average number of influenza vaccines for those children we know received breastmilk as a child and those who know did not.
# 
# *This function should return a tuple in the form (use the correct numbers:*
# ```
# (2.5, 0.1)
# ```

# In[3]:


def average_influenza_doses():
    # YOUR CODE HERE
    import pandas as pd
    import numpy as np
    mb = pd.read_csv("assets/NISPUF17.csv")
    gp1 = mb[mb['CBF_01'] == 1]
    gp2 = mb[mb['CBF_01'] == 2]
    
    return np.mean(gp1['P_NUMFLU']), np.mean(gp2['P_NUMFLU'])
    #raise NotImplementedError()


# In[4]:


assert len(average_influenza_doses())==2, "Return two values in a tuple, the first for yes and the second for no."


# ## Question 3
# It would be interesting to see if there is any evidence of a link between vaccine effectiveness and sex of the child. Calculate the ratio of the number of children who contracted chickenpox but were vaccinated against it (at least one varicella dose) versus those who were vaccinated but did not contract chicken pox. Return results by sex. 
# 
# *This function should return a dictionary in the form of (use the correct numbers):* 
# ```
#     {"male":0.2,
#     "female":0.4}
# ```
# 
# Note: To aid in verification, the `chickenpox_by_sex()['female']` value the autograder is looking for starts with the digits `0.0077`.

# In[11]:



def chickenpox_by_sex():
    # YOUR CODE HERE
    import pandas as pd
    mb = pd.read_csv("assets/NISPUF17.csv")
    
    v1m = mb[(mb['P_NUMVRC'] >=1) & (mb['HAD_CPOX'] == 1) & (mb['SEX'] == 1)]
    v1f = mb[(mb['P_NUMVRC'] >=1) & (mb['HAD_CPOX'] == 1) & (mb['SEX'] == 2)]
    v2m = mb[(mb['P_NUMVRC'] >=1) & (mb['HAD_CPOX'] == 2) & (mb['SEX'] == 1)]
    v2f = mb[(mb['P_NUMVRC'] >=1) & (mb['HAD_CPOX'] == 2) & (mb['SEX'] == 2)]
    
    rm = v1m.shape[0]/v2m.shape[0]
    rf = v1f.shape[0]/v2f.shape[0]
    r = [rm, rf]
    
    sex = ['male', 'female']
    d = {}
    i = 0
    for s in sex:
        d[s] = r[i]
        i+=1
    return d  
    #raise NotImplementedError()


# In[12]:


assert len(chickenpox_by_sex())==2, "Return a dictionary with two items, the first for males and the second for females."


# ## Question 4
# A correlation is a statistical relationship between two variables. If we wanted to know if vaccines work, we might look at the correlation between the use of the vaccine and whether it results in prevention of the infection or disease [1]. In this question, you are to see if there is a correlation between having had the chicken pox and the number of chickenpox vaccine doses given (varicella).
# 
# Some notes on interpreting the answer. The `had_chickenpox_column` is either `1` (for yes) or `2` (for no), and the `num_chickenpox_vaccine_column` is the number of doses a child has been given of the varicella vaccine. A positive correlation (e.g., `corr > 0`) means that an increase in `had_chickenpox_column` (which means more no’s) would also increase the values of `num_chickenpox_vaccine_column` (which means more doses of vaccine). If there is a negative correlation (e.g., `corr < 0`), it indicates that having had chickenpox is related to an increase in the number of vaccine doses.
# 
# Also, `pval` is the probability that we observe a correlation between `had_chickenpox_column` and `num_chickenpox_vaccine_column` which is greater than or equal to a particular value occurred by chance. A small `pval` means that the observed correlation is highly unlikely to occur by chance. In this case, `pval` should be very small (will end in `e-18` indicating a very small number).
# 
# [1] This isn’t really the full picture, since we are not looking at when the dose was given. It’s possible that children had chickenpox and then their parents went to get them the vaccine. Does this dataset have the data we would need to investigate the timing of the dose?

# In[15]:


def corr_chickenpox():
    import scipy.stats as stats
    import numpy as np
    import pandas as pd
    
    mb = pd.read_csv("assets/NISPUF17.csv")
    
    v1 = mb[(mb['P_NUMVRC'] >=0) & (mb['HAD_CPOX'] <= 2)]
    
    no_yes = v1['HAD_CPOX']
    
    est_vaccine = v1['P_NUMVRC']
    
    # this is just an example dataframe
    df=pd.DataFrame({"had_chickenpox_column":no_yes,
                   "num_chickenpox_vaccine_column":est_vaccine})

    # here is some stub code to actually run the correlation
    corr, pval=stats.pearsonr(df["had_chickenpox_column"],df["num_chickenpox_vaccine_column"])
    
    # just return the correlation
    return corr

    # YOUR CODE HERE
    #raise NotImplementedError()


# In[16]:


assert -1<=corr_chickenpox()<=1, "You must return a float number between -1.0 and 1.0."

