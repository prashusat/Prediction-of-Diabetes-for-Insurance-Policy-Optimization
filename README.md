<h1>Task</h1>
Diabetes is a highly prevalent and expensive chronic condition, costing about $330 billion to
Americans annually. Most of the cost is attributed to the ‘type-2’ version of diabetes, which is
typically diagnosed in middle age.
Today is December 31, 2016. A commercial health insurance company has contracted you to
predict which of their members are most likely to be newly-diagnosed with type-2 diabetes in

Your goal is to featurize their data, train and optimize a predictive model, and explain your
results and approach. (Note: “newly-diagnosed” means members who were ​ NOT previously
coded ​with diabetes ​prior​ to 2016-12-31, inclusive).
<h1>Details about the dataset</h1>
The data provided are real patient claims records from a large insurance company,
appropriately de-identified. The data sets have already been split into training and test sets
(‘train.txt’ & ‘test.txt’). The proportions of members are about 70% train and 30% test.
Each line in both text files is a patient record, represented as a json string. The health record is
parameterized by a set of encounter dates in a YYYY-MM-DD format. The structure of each
patient json is as follows:
- ‘bday’ - patient date of birth in YYYY-MM-DD format
- ‘is_male’ - True = Male, False = Female
- ‘patient_id’ - de-identified patient id (each patient is given a unique value)
- ‘resources’ - dictionary of encounter_date → list of codes (described below)
- ‘observations’ - dictionary of encounter_date → list of dictionaries (described below)
- ‘tag_dm2’ - indicates date of first type-2 diabetes diagnosis - will either have a
   YYYY-MM-DD date or be an empty ‘’ string; this information will be censored from the
   holdout set. (described above)
- ‘split’ - indicates a member is in the ‘train’ or ‘test’ set; ​information beyond 2017-01-
   has been ​ **removed** ​ from test.txt​.
Each patient record has a key ‘tag_dm2’, whose value is ​ either ​ a ‘YYYY-MM-DD’ date string
indicating the date of first code of a diagnosis of diabetes, ​ or ​ an empty string ‘’ (indicating no
diabetes in their record).
Your task is to predict each test set member’s probability of being ​newly-diagnosed​ with
diabetes in 2017. Information for ​each test set member’s​ health record beyond 2017-01-01 has
been removed; the true diabetes status of the member is hidden from you and will be used by
Lumiata’s data science team to evaluate your solution.
You should cohort your data (i.e construct the response variable) in the training set according to
the following definitions (check your work with the training set counts given below for each
definition):

A ‘​claim​’ is someone whose ‘tag_dm2’ date is between 2017-01-01 and 2017-12-31,
inclusive (training set count of ‘claim’ = 3410) - the response for these members is a ‘1’
A ‘​never-claim​’ is someone whose ‘tag_dm2’ date is ​ either ​ after 2017-12-31, exclusive,
or ​ is an empty string ‘’ (training set count of ‘never-claim’ = 70110) - the response for
these members is a ‘0’
A ‘​prior​’ is someone whose ‘tag_dm2’ date is ​ before ​ 2017-01-01, exclusive - typically
‘priors’ are filtered out of the matrix before training. You may include these people in
training, but keep in mind they will be filtered out of ‘test’ when we evaluate your solution.
Each patient record also has two keys describing their health history - ‘resources’ &
‘observations’.
The ‘resources’ key specifies the diagnoses, medications, and procedures that were
noted/prescribed/performed at each doctor’s visit - these are represented by different coding
systems (icd9/10, rxnorm, cpt, respectively.) Each encounter date in the ‘resources’ key is
mapped to the corresponding list of codes issued at that doctor’s visit.
The codes have the format _. For instance, ‘icd9_272.0’, which corresponds to
high cholesterol:
http://www.icd9data.com/2015/Volume1/240-279/270-279/272/272.0.htm
Note ​ - encounter dates in ‘resources’ can sometimes have no codes in the code list!
The ‘observations’ key specifies the lab tests that were completed - each encounter date is
mapped to a list of dictionaries, each of which has the following keys:
‘code’ - the ‘loinc’ code corresponding to the lab test
‘interpretation’ - whether the lab was ‘H’ for high, ‘L’ for low, ‘N’ for normal, or ‘A’ for
abnormal
‘value’ - the value extracted from the lab
For instance, the lab could have been a blood glucose test ‘loinc_2345-7’, whose value may
have been 130, and hence whose interpretation would be ‘H’ (a cut-off for high blood glucose is
106:
https://s.details.loinc.org/LOINC/2345-7.html?sections=Comprehensive​ )
Note ​ - the values in the ‘interpretation’ and ‘value’ keys can sometimes be ‘None’!
The keys in the ‘resources’ and ‘observation’ dictionary correspond to the encounter date with
the doctor. All dates are formatted as string in YYYY-MM-DD format, e.g. “2016-04-30”.
The format of the file you submit to us should be a csv file, formatted as
‘<your_name_here>_dm2_solution.csv’. We should be able to read in your solution using
pandas as follows:
for each test set patient_id.

<h1>Conclusions</h1>

1) The most important feature I could observe after deep exploration of the data was the feature called icd10_e11 . This feature was extracted from the resources key under the record of each patient.The reason why I think this feature is the most important is because it acted as a direct signal to let me know if a person is affected by type-II diabetes or not. This is because icd10_e11 is actually the code which indicates type-II diabetes in medical literature hence acting sort of like a label by itself.<br/>

The above fact can also be proven by looking at the coefficients of the line which seperates the classes in logistic regression. We can interpret this fact by looking at the coefficients. We can very well observe below that the fourth number below corresponds to the feature icd10_e11 and this is the largest number of all thereby proving that its the most important feature.
<br/>

2) The 2nd most indicative feature to me was the feature which was obtained from the observations key of each patients record. This was the interpreatation of the lab test 'loinc_4548-4' .This was encoded into bag of words and this being high turned out to be a very important feature. Only the fact that it was high or not was retained in a single column of vector. The scientific reason why this is so important is because this indicates the level of haemogloibin. Haemoglobin is a compound which directly affects the blood glucose level because it carries glucose molecules in it and therefore blood glucose level increase leads to the condition of diabetes. This again can be observed from the output of coefficients above where the second number is the second largest.

3) The next most important feature that I could see from exploring the data was the age of a person.This feature was designed using the bithday dates of each patient. The age of a person had a good correlation with the fact that a person was affected by diabetes or not. This is because a person above the age 60 is more likely to get affected by type-II diabetes and a person below the age 20 was less likely to be affected by this. This was observed by observing the distribution pertaining to both the classes.
<br/>
4)Another very important feature that I found from the dataset was the code icd10_i10 which relates to hypertension in medical literature. This again influences the fact that a person will have diabetes or not by a large factor. A paper states that there is a very large relation between hypertension and diabetes. https://care.diabetesjournals.org/content/40/9/1273
<br/>
5)Another important feature that I personally engineered was a combination of multiple codes indicating various health conditions in medical literature ('cpt_99232','icd10_i25','icd10_j96','icd10_e78','icd10_j44','icd10_i50','icd10_n18','cpt_83036'). This was again obtained from the features key in the dataset. These were the most imporatant features indicated by SelectKBest but they did not have any significant effect when individually added to the model but they ended up having a very significant effect when added up together. The sum was taken after bow encoding. The distribution of the sum of these features was observed and it indicated a significant difference in the way it was distributed for patients with diabetes and patients without diabetes.
<br/>
<h3>Model used for training:</h3>
Why was logistic regression chosen?<br/>

The model that the data was finally trained on was logistic regression model. A decision tree was also tried out and a decision tree with a depth of 4 gave the same exact performance of logistic regression. Since Logistic Regression gave results on par with decision tree it was chosen to model the data rather than decision trees. The reason for this being logistic regression is a simpler model and if it gives such a good performance it is sort of indicative that the data is Linearly seperable.<br/>

<h3>Specifications of the model:</h3>

Logistic regression is a simple classification algorithm to seperate linearly seperable data. The data that it was trained on had about 5 features hence we had to train 5 parameters for the model. Also L2 REGULARISATION was used.
<br/>
<h3>Optimisation of the Model:</h3>
I personally believe that one of the most important thing for a data scientist is exploring at the data and gathering the best features from it. Hence I spent most of my initial time engineering the features and making sure to get the best out of them. Most of the important features was explored using various plots and the best were retained.

The model was later optimized for best performance using a single fold cross-validation. We just had one parameter to tune that is the coefficient of the regularisation term. A linear search across different values of the regularisation parameter was done and we obtained the best regularisation parameter value to be 10^-4. The metric used to evaluate this was PRAUC(same as average precision score). The precision-recall curve for different values of the hyperparameter was also plotted and then the best model was choosen by looking at the metrics and the curve and taking into consideration the bias-variance tradeoff. The threshold value of the model was chosen to be 0.4. This was done because with a standard threshold of 0.5 we were getting more false negatives than false positives. In a medical application we are always okay with a person being falsely clssified to have a disease because in that case he can always take a second opinion but he/she cannot be falsely diagnosed with having no disease since that could be really dangerous to the person. By changing threshold we had slightly more false positives but we reduced the false negatives by a lot.</b><br/>

How overfitting was prevented?<br/>
Overfitting here was prevented using L2-regularisation. I searched linearly across various values of the regularisation parameter and the best value was obtained using cross-validation. The best value was found to be 10^-4.
