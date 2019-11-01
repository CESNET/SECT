#Backlog

### Clustering of events into ip groups - using minimal info
* preprocess into pandas
* time series construction
* n x n distance matrix
* dtw + umap + dbscan clustering
* then classify
* further processing - maybe not necessary
    * divide events into clusters
    * run correlation inside clusters

    #### Fastest path to proof of concept solution
    * current pre-processing
    * construct time series through aggregation
    * data frame with numpy arrays
    * all sq or random sample sq ? with fastdtw 
    * do dtw as in example 
    * classify
    * label the clusters and write extension to exact correlation algorithm

    * Collect research that suggest that this is good idea to do (better than feature extraction i did - maybe compare dtw and selected features)
 
### Better time series with outlook on aggregation window 
* merge info when event does not cease during one window (better for dtw?

### Use CSV writer in preprocessing
It is also possible to just pickle whole pandas dataframe
```
import csv

with open("out.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(a)
```
Instead i use dataframe pickle method

### What to do now ?

* week of clustering
* longer term clustering
* cluster verification
   * nfdump connection for flows filtering or
   * ?? 

Opytaj sa Martina na nejake dobre zdroje k ML
Komunikuj
Opytaj sa kto dalsi este robi Python ML

Zapajaj sa do skupiny
