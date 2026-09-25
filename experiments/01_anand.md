


Baseline model and thought for now ::>>>>
1. Make some preprocessing : data cleaning 
2. Normalise the data : remove commas , extra spaces
3. Convert the name to the n-grams for n=3 or 5; //// To match the maximum char match
4. For Address matching : Convert the address value also to the n-gram for n=3 to 4 // char match strategy
### Feature Engineering 
5. Create new Feature with these datas
   - normalized city + first few characters of name
   - address number + city
   - name prefix + city
   - normalized address components

6. Now we have the basic columns for each row
7. Create an array of this features for each row -->[nor_name, nor_address, ngrma_name , ngram_address , feature_creating_featuress....]



### Now we are required to create the labels for it .

1. We use train_ground_truth.tsv which contains :>> record of mathched row
S1-965667	S2-681193310,S2-743505751,S3-775321672,S3-11291185,S3-860443364

- so we map each id's_array to each one with label of 1 ::>>> Matched 
  and for the rest 0 ::>>> unmatched
  Ex:-
  S1-965667_array    S2-681193310_array    1
  S1-965667_array    S3-775321672_array    1
  .
  .
  .
  by that we can create about n^2 label with 1 ::>> matched 

- and for rest of the rows we map it with label 0 ::>>>> unmatched 
  S1-965667_array    S2-746193310_array    0
  S1-965667_array    S3-385321672_array    0
  .
  .
  .


After that we have X , Y and Label now we can train our model ...... 



Lets first do it have some prediction and understanding than we will start optimising it...........>>>>>>

One updation, for crating Unmatched data (label 0), we are currently using to map a row with all the unmatched row which becames very costly and unbalanced labeling data 
:::::>>>>>>>

# New Strategy, 
- Only map those rows which look similar but they do not (instead of maping to all). 
- For that we can move for just clustering the unlabeled rows that we have created using 
  k-mean clustering.
  
  ### Now we start labelling from here 
  - For each row in the cluster, we start mapping to rest of the rows in cluster which actully not_matched with it as per the train_ground_truth.tsv. 
  So by doing so we got more efficent and clean data that train on those datas which look similar but actully they are not. #Label 0
  - and we also have those datas which actully matched:  #Label 1



  