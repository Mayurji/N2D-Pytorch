## N2D: (Not Too) Deep Clustering via Clustering the Local Manifold of an Autoencoded Embedding

## Abstract

Deep clustering has increasingly been demonstrating superiority over conventional shallow clustering algorithms. Deep clustering algorithms usually combine representation learning with deep neural networks to achieve this performance, typically optimizing a clustering and non-clustering loss. In such cases, an autoencoder is typically connected with a clustering network, and the final clustering is jointly learned by both the autoencoder and clustering network. Instead, we propose to learn an autoencoded embedding and then search this further for the underlying manifold. For simplicity, we then cluster this with a shallow clustering algorithm, rather than a deeper network. We study a number of local and global manifold learning methods on both the raw data and autoencoded embedding, concluding that UMAP in our framework is able to find the best clusterable manifold of the embedding. This suggests that local manifold learning on an autoencoded embedding is effective for discovering higher quality clusters. We quantitatively show across a range of image and time-series datasets that our method has competitive performance against the latest deep clustering algorithms, including out-performing current state-of-the-art on several. We postulate that these results show a promising research direction for deep clustering.

## Keypoint

* Autoencoder represents the high dimensional data to lower dimensional feature space without explicitly learning the local structure.
* The result of autoencoder's lower dimensional feature space is projected using 
manifold learning, which represents the local structure of the data helping in creating quality clusters.

## Manifold Learning Techniques

* Isomap
* t-SNE
* UMAP

## Visualization 

### Clustering 2D Space
![Ground Truth](/mnist-n2d-viz/mnist-n2d.png)

## Result

**Accuracy: 97.832**\
**NMI(Normalized Mutual Information): 94.221**\
**ARS(Adjusted Random Score): 95.261**

![Predicted](/mnist-n2d-viz/result.png)

**Comparsion over other dataset and clustering techniques**

![N2D](/mnist-n2d-viz/N2D_Result.png)

## Packages Installation

```python
pip install -r requirements.txt
```

## Run

    python n2d.py mnist 0 --umap_dim=2 --umap_neighbors=20 --manifold_learner=UMAP --save_dir=mnist-n2d-viz --umap_min_dist=0.00 --visualize

## [**Keras Implementation**](https://github.com/rymc/n2d#abstract)

## Citation

    @inproceedings{McConville2020,
      author = {Ryan McConville and Raul Santos-Rodriguez and Robert J Piechocki and Ian Craddock},
      title = {N2D:(Not Too) Deep Clustering via Clustering the Local Manifold of an Autoencoded Embedding},
      booktitle = {25th International Conference on Pattern Recognition, {ICPR} 2020},
      publisher = {{IEEE} Computer Society},
      year = {2020},
    }
