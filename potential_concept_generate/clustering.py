import faiss
import time
import numpy as np

def run_kmeans(x, nmb_clusters, verbose=False):
    """Runs kmeans on 1 GPU.
    Args:
        x: data
        nmb_clusters (int): number of clusters
    Returns:
        list: ids of data in each cluster
    """
    n_data, d = x.shape

    # faiss implementation of k-means
    clus = faiss.Clustering(d, nmb_clusters)

    # Change faiss seed at each k-means so that the randomly picked
    # initialization centroids do not correspond to the same feature ids
    # from an epoch to another.
    clus.seed = np.random.randint(1234)

    clus.niter = 20
    clus.max_points_per_centroid = 10000000
    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.useFloat16 = False
    flat_config.device = 0
    index = faiss.GpuIndexFlatL2(res, d, flat_config)


    # perform the training

    clus.train(x, index)
    _, I = index.search(x, 1)
    # losses = faiss.vector_to_array(clus.obj)
    stats = clus.iteration_stats
    losses = np.array([
        stats.at(i).obj for i in range(stats.size())
    ])
    if verbose:
        print('k-means loss evolution: {0}'.format(losses))

    return [int(n[0]) for n in I], losses[-1], index


def preprocess_features(npdata, pca=256):
    """Preprocess an array of features.
    Args:
        npdata (np.array N * ndim): features to preprocess
        pca (int): dim of output
    Returns:
        np.array of dim N * pca: data PCA-reduced, whitened and L2-normalized
    """
    _, ndim = npdata.shape
    npdata =  npdata.astype('float32')

    # Apply PCA-whitening with Faiss
    mat = faiss.PCAMatrix (ndim, pca, eigen_power=-0.5)

    mat.train(npdata)
    assert mat.is_trained
    npdata = mat.apply_py(npdata)

    # L2 normalization
    row_sums = np.linalg.norm(npdata, axis=1)
    npdata = npdata / row_sums[:, np.newaxis]

    return npdata,mat

class Kmeans(object):
    def __init__(self, k, preprocess):
        self.k = k
        self.preprocess = preprocess
        self.index = None
        self.mat = None
        self.images_lists = None

    def saveindex(self,path):
        print(f"saving{path}")
        t = faiss.index_gpu_to_cpu(self.index)
        faiss.write_index(t,path)

    def savemat(self, path):
        faiss.write_VectorTransform(self.mat, path)

    def loadmat(self,path):
        self.mat = faiss.read_VectorTransform(path)

    def loadindex(self,path):
        print(f"Loading index: {path}")
        nindex = faiss.read_index(path)
        res = faiss.StandardGpuResources()
        self.index = faiss.index_cpu_to_gpu(res, 0, nindex)

    def getimagelist(self,feature):
        assert self.index is not None
        if self.preprocess:
            assert self.mat is not None
            feature = self.mat.apply_py(feature)

        # L2 normalization
        row_sums = np.linalg.norm(feature, axis=1)
        feature = feature / row_sums[:, np.newaxis]

        D, I = self.index.search(feature, 1)
        self.images_lists = [[] for i in range(self.k)]
        for i in range(len(feature)):
            self.images_lists[int(I[i])].append(i)

    def cluster(self, data:np.ndarray, verbose=False):
        """Performs k-means clustering.
            Args:
                x_data (np.array N * dim): data to cluster
        """
        end = time.time()
        data = data.astype('float32')
        if self.preprocess:
            # PCA-reducing, whitening and L2-normalization
            data,mat = preprocess_features(data)
            self.mat = mat

        # cluster the data
        I, loss, self.index = run_kmeans(data, self.k, verbose)
        self.images_lists = [[] for i in range(self.k)]
        for i in range(len(data)):
            self.images_lists[I[i]].append(i)

        if verbose:
            print('k-means time: {0:.0f} s'.format(time.time() - end))

        return loss

    def search(self,x:np.ndarray,k):
        assert self.index is not None
        if self.preprocess:
            assert self.mat is not None
            x = self.mat.apply_py(x)

        x = x.astype('float32')
        # L2 normalization
        row_sums = np.linalg.norm(x, axis=1)
        x = x / row_sums[:, np.newaxis]

        D,I = self.index.search(x,k)

        return D,I


