# DIQ-assignment
This repository contains our assignment for the Data & information quality course held at Politecnico di Milano (accademic year: 2022/2023).

In this assignment we try different imputation approaches to deal with missing values in a dataset. We are provided with two complete [datasets](https://github.com/VladMarianCimpeanu/DIQ-assignment/tree/main/datasets), and we inject several missing values though [this script](https://github.com/VladMarianCimpeanu/DIQ-assignment/blob/main/code/utility/dirty_completeness.py), generating for each original dataset 5 new versions (50%, 60%, 70%, 80% and 90% of completeness).

For the evaluation we consider how accurate is the reconstruction of a given imputation method. First we try to compute the accuracy using the exact matching approach, then, since there are some features for which there exist a similarity measure, we try to compute the reconstruction accuracy by using a similarity-based approach.

As final evaluation we select two Machine learning algorithms and perform a classification task using the different datasets. In this scenario we are able to understand how an imputation method, and the completeness level can impact machine learning performances.
