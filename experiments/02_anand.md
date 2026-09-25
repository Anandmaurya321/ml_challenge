
## Instead of using  'K-mean Clustering' I am shifting to blocking strategy :: > 

- Where we are creating block by taking multiple features 
   - Block 1: Normalized city + first few characters of name
   - Block 2: address number + city
   - Block 3: name prefix + city
   - Block 4: normalized address components

- Merge all this block into a single unit : 
  - Final Block: B1 U B2 U B3 U B4 
  - Now this Block contain list of those rows which are looking HARD_NEGATIVE
  - So by doing so  WE CREATE HARD NEGATIVE FEATURES 
  - And since we have to increase the precision we can have 10% yes_labaled_data and
    90% No_labeled_data ::::>>> benificial for us


    


