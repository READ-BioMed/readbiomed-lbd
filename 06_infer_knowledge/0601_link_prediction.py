#!/usr/bin/env python
# coding: utf-8


import sys

import pandas as pd
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import precision_recall_curve,roc_curve,precision_score,confusion_matrix
from sklearn import metrics
import numpy as np
from numpy import argmax
sys.path.insert(1, '/data/projects/punim0478/gracie/jbi_ne/code')
from create_edge import create_yr_link
from create_train_test_data import fix_nodes,data_pipeline
import json
# import networkx as nx
from collections import Counter
import time

import matplotlib.pyplot as plt
from matplotlib import pyplot
from math import isclose
import math
from sklearn.decomposition import PCA
import os
import multiprocessing
from IPython.display import display, HTML
from sklearn.model_selection import train_test_split

import torch
from cogdl.data import Graph
from cogdl.models.emb.sdne import SDNE
from cogdl.models.emb.hope import HOPE
from cogdl.models.emb.grarep import GraRep
from cogdl.models.emb.line import LINE 
from cogdl.models.emb.deepwalk import DeepWalk
from cogdl.models.emb.node2vec import Node2vec

from stellargraph import StellarGraph, datasets
from stellargraph.data import EdgeSplitter

import networkx as nx

# # import GraphEmbedding libraries
# sys.path.insert(1, '/content/drive/MyDrive/GraphEmbedding')
sys.path.insert(1, '/data/projects/punim0478/gracie/jbi_ne/code/GraphEmbedding')
from ge.classify import read_node_label, Classifier
from ge import Struc2Vec

## arguments,

## arg 02: path for reading the corpus
# path = "/Users/yidesdo21/Projects/outputs/18_dgl_sample_csv/"
path = "/data/projects/punim0478/gracie/jbi_ne/data/16_dgl_csv/"

## arg 04: t_range is the range of train/test datasets
# t_range = (2002,2022)  # for testing
t_range = (2002,2003)

## arg 05: ne_model is the name of the NE model,
ne_models = ["sdne"] 

## arg 06: k is the proportion of top ranked results for the precision @ k evaluation metrics
k=10

# ## arg 01: save_path for saving the plots
# save_path = "/data/projects/punim0478/gracie/jbi_ne/output/plots/"+ne_model
# if not os.path.exists(save_path):
#     os.makedirs(save_path)
    
# ## arg 03: save_raw_path for saving the {year:(auprc,auroc)} dictionary
# save_raw_path = "/data/projects/punim0478/gracie/jbi_ne/output/auprc/"+ne_model
# if not os.path.exists(save_raw_path):
#     os.makedirs(save_raw_path)


# ## arg 04: save_pr_path for saving the raw precision,recall value
# save_pr_path = "/data/projects/punim0478/gracie/jbi_ne/output/pr_raw/"+ne_model
# if not os.path.exists(save_pr_path):
#     os.makedirs(save_pr_path)

## replace the save path argument to each model in the for loop
## arg 05: save_pak_path for saving the precision at k results, k == 30% of positives
# save_pak_path = "/data/projects/punim0478/gracie/jbi_ne/output/prec_at_"+str(k)+"/"+ne_model
# if not os.path.exists(save_pak_path):
#     os.makedirs(save_pak_path)

# ### Load the dataset, and build a Stellargraph with the dataset to train the node embeddings
# * The dataset to generate the node embeddings.
# * The node embeddings for the training data are generated with the feature network of the training data.
# * The node emebddings for the testing data are generated with the feature network of the testing data.

# In[15]:

## for test, delete when running the experiment
# g_f0_tr,g_lt_tt = 1977,2002

def node_edge_data(g_f0_tr,g_lt_tt,ne_model,k,path=path):
    """obtain the node and edge data for training the node embeddings,
           g_f0_tr: start of the training data,
           g_lt_tt: end of the testing data,
       return idx_dict: a dictionary to index node uid and node index, node index starts from 0,
              nodes_data: a dataframe with node uid and node index,
              edges_data: a dataframe with src_uid, dst_uid, labels, train_fea_mask, test_fea_mask, induce_mask,
                          the induce mask column is used to train node embeddings"""
    nodes_data = pd.read_csv(path+str(g_f0_tr)+"_"+str(g_lt_tt)+"/ne/"+'nodes.csv')
    nodes_data['node_idx'] = range(len(nodes_data))
    idx_dict = dict(zip(nodes_data.node_id, nodes_data.node_idx))

    edges_data = pd.read_csv(path+str(g_f0_tr)+"_"+str(g_lt_tt)+"/ne/"+'edges.csv')
    edges_data['src_idx'] = edges_data.apply(lambda x: idx_dict.get(x['src_id']), axis=1)
    edges_data['dst_idx'] = edges_data.apply(lambda x: idx_dict.get(x['dst_id']), axis=1)
    
    ## save the node index to do error analysis
    # if g_lt_tt == 2021:
    save_raw_path = "/data/projects/punim0478/gracie/jbi_ne/output/prec_at_"+str(k)+"_raw/"+ne_model+"/"
    if not os.path.exists(save_raw_path):
        os.makedirs(save_raw_path)

    with open(save_raw_path+ne_model+"_"+str(g_lt_tt)+'_node_idx.json', 'w') as fout:
        json.dump(idx_dict, fout, cls=NpEncoder) 

    return nodes_data,edges_data,idx_dict

# nodes_data,edges_data,idx_dict = node_edge_data(g_f0_tr,g_lt_tt,path=path)
# print(nodes_data.head())
# print(edges_data.head())
# print(idx_dict)
# print("----------------------------")

def graph_to_ne(nodes_data,edges_data,mask="induce_mask",g_library="cogdl"):
    """generate a graph to learn the node embeddings,
        the feature network of the training data or the testing data,
        either return a graph for stellar, or return a graph for networkx
        input -- nodes_data: a dataframe with node uid and node index,
                 edges_data: a dataframe with src_uid, dst_uid, labels, train_fea_mask, test_fea_mask, induce_mask,
                                the induce mask column is used to train node embeddings
                 mask: mask of whether the feature network from the training data or the testing data
                       changed to the label network of the training data
        output -- return a graph used to train the node embeddings
                 
    """
    graph_df = edges_data[edges_data[mask] == True][["src_id","dst_id","src_idx","dst_idx"]]
    
    ns = nodes_data[["node_idx"]]
    graph_es = graph_df[["src_idx","dst_idx"]]
    
    if g_library == "stellar":
        graph = StellarGraph(nodes=ns, edges=graph_es,
                                source_column='src_idx',target_column='dst_idx',)
        print(graph.info())
    
    elif g_library == "nx":
        graph = nx.from_pandas_edgelist(graph_es, "src_idx","dst_idx")
    
    elif g_library == "cogdl":
        edges = torch.tensor(graph_es.values.tolist()).t()
        x = torch.tensor(ns.values.tolist())
        graph = Graph(edge_index=edges,x=x) 

    return graph

# cog_graph = graph_to_ne(nodes_data,edges_data,mask="induce_mask",g_library="cogdl")
# print(cog_graph.num_nodes)
# print(cog_graph.num_edges)


def get_clf_data(file_path,idx_dict,mask="train_mask"):
    """retrieve the training and testing data for training a binary classifier,
        input -- file_path: path to the edge.csv file,
                 idx_dict: map from node uid to node index,
                 mask: training data or the testing data,
        return -- examples: 2d array for node pairs, [[src,idx],[src,idx]...]
                  labels: 1d array, true labels for node pairs """
    train_test_data = pd.read_csv(file_path)
    train_test_data['src_idx'] = train_test_data.apply(lambda x: idx_dict.get(x['src_id']), axis=1)
    train_test_data['dst_idx'] = train_test_data.apply(lambda x: idx_dict.get(x['dst_id']), axis=1)

    data = train_test_data[train_test_data[mask] == True]

#     pos,neg = data[data["label"]==1],data[data["label"]==0]  
    
    examples = data[["src_idx","dst_idx"]].to_numpy()
    labels = data["label"].to_numpy()
    
    return examples,labels


### Train the node embeddings

def sdne_embedding(graph):
    """train the node embeddings with sdne,
        return sdne_emb, an numpy.ndarray, the shape is [num_of_nodes,num_of_embeddings_for_each_node]"""
    sdne = SDNE(hidden_size1=1000, hidden_size2=128, droput=0.5,   # default
                alpha=0.1, beta=0,   # Yue et al., 2020
                nu1=1e-4, nu2=1e-3,  # default
                epochs=6, lr=1e-3,  # random set, need to fine tune, consistent with gcn and graphsage
                cpu=False)            #  need to change to gpu when running on spartan
    sdne_emb = sdne(graph)
    # print(sdne_emb.shape)
    # print(graph.nodes())

    # def get_embedding(u):
    #     u_index = graph.nodes().index(u)
    #     return sdne_emb[u_index]

    # return get_embedding

    return sdne_emb

# sdne_ne = sdne_embedding(cog_graph)
# print(sdne_ne[0])
# print(sdne_ne[0].shape)

def hope_embedding(graph):
    """train the node embeddings with sdne,
        return sdne_emb, an numpy.ndarray, the shape is [num_of_nodes,num_of_embeddings_for_each_node]"""
    hope = HOPE(128,0.01)  # use default, hidden_size = 128, beta = 0.01
    hope_emb = hope(graph)
    return hope_emb

def grarep_embedding(graph):
    grarep = GraRep(128,    # hidden_size=, consistent with all othe ne models
                    4)             # step, Yue et al., 2020
    grarep_emb = grarep(graph)
    return grarep_emb

def line_embedding(graph):
    line = LINE(128,80,20,5,1000,0.025,3)  # hidden_size, walk_length, walk_num, negative, batch_size, alpha, order, all defaults
    line_emb = line(graph)
    return line_emb

def deepwalk_embedding(graph):
    deepwalk = DeepWalk(128,128,128,5,10,10) # hidden_size, walk_length (128 in Yue et.al), walk_num (128 in Yue et. al), window_size, worker, iteration. Use defaults
    deepwalk_emb = deepwalk(graph)
    return deepwalk_emb

def node2vec_embedding(graph):
    node2vec = Node2vec(128,80,40,5,10,10,0.25,0.25) # hidden_size, walk_length, walk_num, window_size, worder, iteration, p (0.25, Yue et al.), q (0.25, Yue et al.). Use defaults
    node2vec_emb = node2vec(graph)
    return node2vec_emb

def struc2vec_embedding(graph):
    struc2vec = Struc2Vec(graph, walk_length=128, num_walks=128, workers=4, verbose=40, ) #init model
    struc2vec.train(window_size = 5, iter = 3)# train model
    struc2vec_emb = struc2vec.get_embeddings()# get embedding vectors
    return struc2vec_emb


from stellargraph.data import BiasedRandomWalk


def create_biased_random_walker(graph, walk_num, walk_length):
    # parameter settings for "p" and "q":
    p = 1.0
    q = 1.0
    return BiasedRandomWalk(graph, n=walk_num, length=walk_length, p=p, q=q)


# In[51]:


from stellargraph.mapper import FullBatchLinkGenerator, FullBatchNodeGenerator, Node2VecLinkGenerator, Node2VecNodeGenerator
from stellargraph.layer import GCN, LinkEmbedding, Node2Vec, link_classification
from stellargraph.data import UnsupervisedSampler
from tensorflow import keras

def gcn_embedding(graph, name):
    
    walk_length = 5
    epochs = 6
    batch_size = 50
    
    # Set the embedding dimensions and walk number:
    dimensions = [128, 128]
    walk_number = 1

    print(f"Training GCN for '{name}':")

    graph_node_list = list(graph.nodes())

    # Create the biased random walker to generate random walks
    walker = create_biased_random_walker(graph, walk_number, walk_length)

    # Create the unsupervised sampler to sample (target, context) pairs from random walks
    unsupervised_samples = UnsupervisedSampler(
        graph, nodes=graph_node_list, walker=walker
    )

    # Define a GCN training generator, which generates the full batch of training pairs
    generator = FullBatchLinkGenerator(graph, method="gcn")

    # Create the GCN model
    gcn = GCN(
        layer_sizes=dimensions,
        activations=["relu", "relu"],
        generator=generator,
        dropout=0.3,
    )

    # Build the model and expose input and output sockets of GCN, for node pair inputs
    x_inp, x_out = gcn.in_out_tensors()

    # Use the dot product of node embeddings to make node pairs co-occurring in short random walks represented closely
    ## the binary operator here is for the NE model, while the binary operator list in the link prediction section
    ##. is for the link prediction classifier
    prediction = LinkEmbedding(activation="sigmoid", method="ip")(x_out)
    prediction = keras.layers.Reshape((-1,))(prediction)

    # Stack the GCN encoder and prediction layer into a Keras model, and specify the loss
    model = keras.Model(inputs=x_inp, outputs=prediction)
    model.compile(
        optimizer=keras.optimizers.Adam(lr=1e-3),
        loss=keras.losses.binary_crossentropy,
        metrics=[keras.metrics.binary_accuracy],
    )

    # Train the model
    batches = unsupervised_samples.run(batch_size)
    for epoch in range(epochs):
        print(f"Epoch: {epoch+1}/{epochs}")
        batch_iter = 1
        for batch in batches:
            samples = generator.flow(batch[0], targets=batch[1], use_ilocs=True)[0]
            [loss, accuracy] = model.train_on_batch(x=samples[0], y=samples[1])
            output = (
                f"{batch_iter}/{len(batches)} - loss:"
                + " {:6.4f}".format(loss)
                + " - binary_accuracy:"
                + " {:6.4f}".format(accuracy)
            )
            if batch_iter == len(batches):
                print(output)
            else:
                print(output, end="\r")
            batch_iter = batch_iter + 1

    # Get representations for all nodes in ``graph``
    embedding_model = keras.Model(inputs=x_inp, outputs=x_out)
    node_embeddings = embedding_model.predict(
        generator.flow(list(zip(graph_node_list, graph_node_list)))
    )
    node_embeddings = node_embeddings[0][:, 0, :]

    def get_embedding(u):
        u_index = graph_node_list.index(u)
        return node_embeddings[u_index]

    return get_embedding


from stellargraph.mapper import GraphSAGELinkGenerator, GraphSAGENodeGenerator
from stellargraph.layer import GraphSAGE


def graphsage_embedding(graph, name):
    
    walk_length = 5
    epochs = 6
    batch_size = 50    
    
    # Set the embedding dimensions, the numbers of sampled neighboring nodes and walk number:
    dimensions = [128, 128]
    num_samples = [10, 5]
    walk_number = 1

    print(f"Training GraphSAGE for '{name}':")

    graph_node_list = list(graph.nodes())

    # Create the biased random walker to generate random walks
    walker = create_biased_random_walker(graph, walk_number, walk_length)

    # Create the unsupervised sampler to sample (target, context) pairs from random walks
    unsupervised_samples = UnsupervisedSampler(
        graph, nodes=graph_node_list, walker=walker
    )

    # Define a GraphSAGE training generator, which generates batches of training pairs
    generator = GraphSAGELinkGenerator(graph, batch_size, num_samples)

    # Create the GraphSAGE model
    graphsage = GraphSAGE(
        layer_sizes=dimensions,
        generator=generator,
        bias=True,
        dropout=0.0,
        normalize="l2",
    )

    # Build the model and expose input and output sockets of GraphSAGE, for node pair inputs
    x_inp, x_out = graphsage.in_out_tensors()

    # Use the link_classification function to generate the output of the GraphSAGE model
    prediction = link_classification(
        output_dim=1, output_act="sigmoid", edge_embedding_method="ip"
    )(x_out)

    # Stack the GraphSAGE encoder and prediction layer into a Keras model, and specify the loss
    model = keras.Model(inputs=x_inp, outputs=prediction)
    model.compile(
        optimizer=keras.optimizers.Adam(lr=1e-3),
        loss=keras.losses.binary_crossentropy,
        metrics=[keras.metrics.binary_accuracy],
    )

    # Train the model
    model.fit(
        generator.flow(unsupervised_samples),
        epochs=epochs,
        verbose=2,
        use_multiprocessing=False,
        workers=4,
        shuffle=True,
    )

    # Build the model to predict node representations from node features with the learned GraphSAGE model parameters
    x_inp_src = x_inp[0::2]
    x_out_src = x_out[0]
    embedding_model = keras.Model(inputs=x_inp_src, outputs=x_out_src)

    # Get representations for all nodes in ``graph``
    node_gen = GraphSAGENodeGenerator(graph, batch_size, num_samples).flow(
        graph_node_list
    )
    node_embeddings = embedding_model.predict(node_gen, workers=1, verbose=0)

    def get_embedding(u):
        u_index = graph_node_list.index(u)
        return node_embeddings[u_index]

    return get_embedding



# ### Train and evaluate the link prediction model
# 



from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


# 1. link embeddings
def link_examples_to_features(link_examples, transform_node, binary_operator,g_library):
    """node embeddings from stellar graph and networkx graph are in different format,
        we need to ensure the binary operator gets the embeddings,"""
    if g_library == "nx":
        return [
            binary_operator(transform_node.get(src), transform_node.get(dst))
            for src, dst in link_examples
        ]  
    
    elif g_library == "stellar":  
        return [
            binary_operator(transform_node(src), transform_node(dst))
            for src, dst in link_examples
        ]

    elif g_library == "cogdl":
        return [
            binary_operator(transform_node[src], transform_node[dst])
            for src, dst in link_examples
        ]


# 2. training classifier
def train_link_prediction_model(
    link_examples, link_labels, get_embedding, binary_operator, g_library
):
    """get_embedding: node embeddings. We can use get_embedding[src] to get the src node embedding.
       link_examples: node pairs for each edge, (src,dst). The edges are the potential links.
       link_labels: true labels for the potential edges.
       binary_operator: methods to generate edge embeddings with node emebddings.
       g_library: either the graph comes from networkx or the stellar, the way of retrieving node 
                embeddings are different, either "stellar" or "nx"
    """
    clf = link_prediction_classifier()
    link_features = link_examples_to_features(
        link_examples, get_embedding, binary_operator,g_library
    )
    
    clf.fit(link_features, link_labels)
    return clf

class NpEncoder(json.JSONEncoder):
    """for saving dictionary as json"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def evaluate_prec_k(clf,link_features,link_labels,link_examples_test,yr,ne_model,save_raw,k=k):
    """calculate precision at k -- TP among the top k ranked outputs,
        Precision@k = (# of recommended items @k that are relevant) / (# of recommended items @k)
                    = TP_among_top_k/top_k_returned_outputs
        input: k -- the proportion of positives that can be returned,
               link_labels -- y_true,
               link_features -- a list of lists, each list is the embedding for each link
               link_examples_test -- node pairs, 

        """
    predicted = clf.predict_proba(link_features)

    # check which class corresponds to positive links
    #  positive_column is a number, e.g. 1 
    positive_column = list(clf.classes_).index(1)    

    y_pred = predicted[:, positive_column] 
    # print("20 y_pred:")
    # print(y_pred[:20])

    n_pos = np.sum(link_labels == positive_column)
    # print("first 20 true labels:")
    # print(link_labels[:20])
    # print("The number of positives in y_true:")
    # print(n_pos)

    # the ranked indices for prediction prob. from the highest to the lowest
    order = np.argsort(y_pred)[::-1]
    # print("ranked indices for top 20 prob.:")
    # print(order[:20])

    # take the true labels of (top ranked k indices of predicted labels)
    # round up 
    k_trans = math.ceil(n_pos*k*0.01)
    # print("number of top k:")
    # print(k_trans)

    ## y_pred_k is the prob. of the node pair being labeled as "true"
    y_pred_k = np.take(y_pred, order[:k_trans])
    # print("predicted prob. for top 20 predictions:")
    # print(y_pred_k[:20])

    y_true_k = np.take(link_labels, order[:k_trans])
    # print("true labels for top 20 predictions:")
    # print(y_true_k[:20])

    ## calcualte precision at k -- TPs among top k ranked cases
    ### the biggest problem to calculate the precision score is that the predicted values are probabilities
    # prec_ak_micro = precision_score(y_true_k, y_pred_k, average='micro')
    # prec_ak_macro = precision_score(y_true_k, y_pred_k, average='macro')
    # print("precision at k, micro, macro:", prec_ak_micro, prec_ak_macro)

    node_pairs_k = link_examples_test[order[:k_trans]].tolist()
    # print("node pairs for top 20 predictions:")
    # print(node_pairs_k[:20])

    n_relevant = np.sum(y_true_k == positive_column)
    # print("n_relevant:")
    # print(n_relevant)
    # print("----------")

    ## raw_dict is used to save the top k node pairs, and the true labels for these node pairs,
    ## . the predicted labels are unnecessary to use in calculating precision@k
    ## . with the top k true labels and k value, we can calculate precision@k 
    raw_dict = dict()
    raw_dict["node_pairs"] = node_pairs_k
    raw_dict["true_labels"] = y_true_k
    raw_dict["predicted_labels"] = y_pred_k

    if save_raw == True:
        save_raw_path = "/data/projects/punim0478/gracie/jbi_ne/output/prec_at_"+str(k)+"_raw_fns/"+ne_model+"/"
        if not os.path.exists(save_raw_path):
            os.makedirs(save_raw_path)

        with open(save_raw_path+ne_model+"_"+str(yr)+'_raw_dict.json', 'w') as fout:
            json.dump(raw_dict, fout, cls=NpEncoder) 

    return float(n_relevant) / min(n_pos, k_trans)


def scorer_prec_k(clf,link_features,link_labels,k=k):
    """calculate precision at k -- TP among the top k ranked outputs,
        Precision@k = (# of recommended items @k that are relevant) / (# of recommended items @k)
                    = TP_among_top_k/top_k_returned_outputs
        input: k -- the proportion of positives that can be returned,
               link_labels -- y_true,

        """
    ### need revision
    # pass
    predicted = clf.predict_proba(link_features)

    # check which class corresponds to positive links
    #  positive_column is a number, e.g. 1 
    positive_column = list(clf.classes_).index(1)    

    y_pred = predicted[:, positive_column] 

    n_pos = np.sum(link_labels == positive_column)

    # the ranked indices for prediction prob. from the highest to the lowest
    order = np.argsort(y_pred)[::-1]

    # take the true labels of (top ranked k indices of predicted labels)
    # round up 
    k_trans = math.ceil(n_pos*k*0.01)
    y_true = np.take(link_labels, order[:k_trans])

    n_relevant = np.sum(y_true == positive_column)

    return float(n_relevant) / min(n_pos, k_trans)   


def achieve_raw(test_prec,test_rec,threshold,test_proba,instances,
                y_test,save_pr_path,yr):
    """get the raw FN and FP
        test_prec: precision of the testing data,
        test_rec: recall of the testing data,
        threshold: the threshold that gets the precision and recall,
        test_proba: prob. for the true class,
        instances: i.e,,examples_test, (src,dst) indices,
        y_test: true labels for the instances,

    """
    optimal_dict = dict()

    # find the raw precision,recall,threshold with the the harmonic mean of precision and recall
    ## the last point is when precision is 1.0,
    # print(test_prec[-100:])
    # print(test_rec[-100:])
    # ix = argmax(test_prec[:-10])
    ix = argmax(test_prec[:-1])
    print('Best Threshold=%f' % (threshold[ix]))
    print("Corresponding precision and recall:", test_prec[ix], test_rec[ix])

    ## calculate the confusion matrix for the optimal point
    # print(test_proba)
    # X_test are for the features, instances are instances
    # node pairs that are predicted to be true and false with the optimal point
    instances = np.array(instances)

    # threshold to get the right index for the instances, predicted probabilities, predicted labels, and true labels
    thre_true,thre_false = test_proba>=threshold[ix], test_proba<threshold[ix]

    # predicted probabilities that corresponds to true and false node pairs 
    test_proba_true,test_proba_false = test_proba[thre_true],test_proba[thre_false]

    # predicted nodes pairs with true and false
    optimal_true = instances[thre_true].tolist()
    optimal_false = instances[thre_false].tolist()

    # predicted label
    optimal_y_pred = [1]*len(optimal_true)+[0]*len(optimal_false)


    # true labels for the node pairs
    optimal_true_label = y_test[thre_true]
    optimal_false_label = y_test[thre_false]
    optimal_y_true = optimal_true_label.tolist()+optimal_false_label.tolist()

    tn, fp, fn, tp = confusion_matrix(optimal_y_true,optimal_y_pred).ravel()

    print("tn, fp, fn, tp:", (tn, fp, fn, tp))
    print("--------------------")

    ## get top 100 fp and top 100 fn:
    # find true positive and false positives
    pred_pos = len(optimal_true) 
    tp,fp = 0,0
    tp_pairs,fp_pairs = list(),list()

    for i in range(pred_pos):
        if optimal_true_label[i] == 1:
            tp += 1 
            tp_pairs.append((optimal_true[i],test_proba_true[i]))

        elif optimal_true_label[i] == 0:
            fp += 1
            fp_pairs.append((optimal_true[i],test_proba_true[i]))

    # sorted_tp_pairs = sorted(tp_pairs, key=lambda x: x[1], reverse=True)
    sorted_fp_pairs = sorted(fp_pairs, key=lambda x: x[1], reverse=True)

    print("tp,fp:",(tp,fp))
    # print("tp_pairs:", sorted_tp_pairs[:20])
    print("fp_pairs:", sorted_fp_pairs[:20])

    # find true negative and false negative
    pred_neg = len(optimal_false) 
    tn,fn = 0,0
    tn_pairs,fn_pairs = list(),list()

    for i in range(pred_neg):
        if optimal_false_label[i] == 1:
            fn += 1 
            fn_pairs.append((optimal_false[i],test_proba_false[i]))

        elif optimal_false_label[i] == 0:
            tn += 1
            tn_pairs.append((optimal_false[i],test_proba_false[i]))

    # sorted_tn_pairs = sorted(tn_pairs, key=lambda x: x[1])
    sorted_fn_pairs = sorted(fn_pairs, key=lambda x: x[1])

    print("tn,fn:",(tn,fn))
    print("fn_pairs:", sorted_fn_pairs[:20])


    optimal_dict["thre_prec_rec"] = (threshold[ix], test_prec[ix], test_rec[ix])
    optimal_dict["confusion_matrix"] = (tn, fp, fn, tp)
    optimal_dict["all_fp"] = sorted_fp_pairs
    optimal_dict["all_fn"] = sorted_fn_pairs

    # print(optimal_dict)

    with open(save_pr_path+"_"+str(yr)+""+'_optimal_dict.json', 'w') as fout:
        json.dump(optimal_dict, fout, cls=NpEncoder) 

    # print("Data range:", str(g_f0_tr)+"_"+str(g_lt_tt))
    # print("Supervised leanring with [cn,pa,jc,aa] as features")
    # print("Training AUPRC is:",train_score)
    # print("Testing AUPRC is:",test_score)
    print("--------------------------------")


def evaluate_pr_auc(clf, link_features, instances, link_labels,plot_auc,yr,save_path,save_pr_path):
    predicted = clf.predict_proba(link_features)

    # check which class corresponds to positive links
    positive_column = list(clf.classes_).index(1)

    prec, rec, threshold = precision_recall_curve(link_labels, predicted[:, positive_column])
    
    auprc = metrics.auc(rec, prec) 
    
    if plot_auc == True:
      # plot the PR curve, for the testing phase
      plot_auprc(save_path,auprc,rec,prec,yr)

      ## get the raw FNs for error analysis
      achieve_raw(prec,rec,threshold,predicted[:, positive_column],instances,
                link_labels,save_pr_path,yr)  
    #   pass

    # if yr == 2021 or yr == 2020:
    if yr == 2002:
        pr_record = list()
        pr_record.append((prec,rec))

        with open(save_pr_path+'/pr_record_'+str(yr)+".json", 'w') as fout:
            json.dump(pr_record, fout, cls=NpEncoder)


    return auprc

def scorer_pr_auc(clf, link_features, link_labels):
    predicted = clf.predict_proba(link_features)

    # check which class corresponds to positive links
    positive_column = list(clf.classes_).index(1)

    prec, rec, _ = precision_recall_curve(link_labels, predicted[:, positive_column])

    return metrics.auc(rec, prec) 

def evaluate_roc_auc(clf, link_features, link_labels):
    predicted = clf.predict_proba(link_features)

    # check which class corresponds to positive links
    positive_column = list(clf.classes_).index(1)
    
    return roc_auc_score(link_labels, predicted[:, positive_column])


def link_prediction_classifier(max_iter=2000):
    """a binary classifier to perform link prediciton,
        scoring: a custom scorer """
#     lr_clf = LogisticRegressionCV(Cs=10, cv=10, scoring="roc_auc", max_iter=max_iter)
    lr_clf = LogisticRegressionCV(Cs=10, cv=10, scoring=scorer_pr_auc, max_iter=max_iter)
    # lr_clf = LogisticRegressionCV(Cs=10, cv=10, scoring=scorer_prec_k, max_iter=max_iter)
    
#     lr_clf = LogisticRegressionCV(Cs=10, cv=10, scoring=evaluate_roc_auc, max_iter=max_iter)
    return Pipeline(steps=[("sc", StandardScaler()), ("clf", lr_clf)])


# 3. and 4. evaluate classifier
def evaluate_link_prediction_model(
    clf, link_examples_test, link_labels_test, get_embedding, 
    binary_operator, g_library, plot_auc,yr,ne_model,save_raw,k,save_path,save_pr_path,
):
    """if plot_auc is set to False, then the auprc curve won't be plotted"""
    link_features_test = link_examples_to_features(
        link_examples_test, get_embedding, binary_operator,g_library
    )
#     score = evaluate_roc_auc(clf, link_features_test, link_labels_test)
    auprc_score = evaluate_pr_auc(clf, link_features_test, link_examples_test, link_labels_test, plot_auc,yr,save_path,save_pr_path)
    auroc_score = evaluate_roc_auc(clf, link_features_test, link_labels_test)
    # print(link_examples_test)
    prec_k = evaluate_prec_k(clf, link_features_test, link_labels_test,link_examples_test,yr,ne_model,save_raw,k)

    return auprc_score,auroc_score,prec_k


## plot the PR curve, why the testing results are so bad?
def plot_auprc(save_path,auprc,rec,prec,yr):
    # pass
    # print(clf)
    pyplot.figure(figsize=(6, 4), dpi=100)

    ## precision-recall curve
    ## the precision-recall [1.0,0.0] point has no point, removing them
    
    # print(rec[-1])
    # print(prec[-1])
#     "\t{}: {:0.4f}".format(name, val))
    ## auprc: numpy.float64 -> float,
    print(auprc)
    print(auprc.item())
    pyplot.plot(rec[:-1], prec[:-1], marker=',', label='AUC = {0:.4g}'.format(auprc.item()))   #  if t == 2019, the label network is 2020-2021

    # also set the range of the y axis
    pyplot.axis([0, 1, 0, 1])

    # axis labels
    pyplot.xlabel('Recall')
    pyplot.ylabel('Precision')
    # show the legend
    pyplot.legend()
    # show the plot
    pyplot.title("Precision-recall curve")

    # save plot
    pyplot.savefig(save_path+"/PR curve at year {}".format(str(yr)))



def operator_hadamard(u, v):
    return u * v

def operator_l1(u, v):
    return np.abs(u - v)

def operator_l2(u, v):
    return (u - v) ** 2

def operator_avg(u, v):
    return (u + v) / 2.0

def operator_cat(u, v):
    return np.concatenate([u, v], axis=0)

def run_link_prediction(binary_operator,g_library,embedding_train,yr,k,save_path,save_pr_path,save_raw=False,plot_auc=False):
    """plot_auc sets to False, don't plot the auprc when training and validating the classifier"""

    clf = train_link_prediction_model(
        examples_train, labels_train, embedding_train, binary_operator,g_library 
    )

    auprc_score,auroc_score,prec_k = evaluate_link_prediction_model(
        clf,
        examples_model_selection,
        labels_model_selection,
        embedding_train,
        binary_operator,
        g_library,
        plot_auc,
        yr,
        ne_model,
        save_raw,
        k,
        save_path,
        save_pr_path
    )

    return {
        "classifier": clf,
        "binary_operator": binary_operator,
        "auprc_score": auprc_score,
        "auroc_score": auroc_score,
        "precision_at_k": prec_k,
    }




# # Build a model pipeline


def data_processing(g_f0_tr,g_lt_tt,ne_model,k,path):
    """constructing training data and testing data for the classifier is the same 
      for all node embedding models."""

    ## load the dataset to build a feature network and train the node embeddings
    nodes_data,edges_data,idx_dict = node_edge_data(g_f0_tr=g_f0_tr,g_lt_tt=g_lt_tt,ne_model=ne_model,k=k,path=path)

    ## construct test data to test the link prediction model
    examples_test,labels_test = get_clf_data(file_path=path+str(g_f0_tr)+"_"+str(g_lt_tt)+"/clf/"+'edges.csv',
                                 idx_dict=idx_dict,mask="test_mask")
  
    ## construct training data to train the link prediction model
    # training data includes both the training and validation data
    # examples that include both the training and validation data
    # labels that include both the training and validation data
    examples,labels = get_clf_data(file_path=path+str(g_f0_tr)+"_"+str(g_lt_tt)+"/clf/"+'edges.csv',
                                       idx_dict=idx_dict,mask="train_mask")

    # split the training data into training data and validation data
    (
          examples_train,
          examples_model_selection,
          labels_train,
          labels_model_selection,
    ) = train_test_split(examples, labels, train_size=0.7, 
                          test_size=0.3, shuffle=True, stratify=labels)   # stratify makes the numer of instances in each label the same

      ## nodes_data: a dataframe with node uid and node index,
      ## edges_data: a dataframe with src_uid, dst_uid, labels, train_fea_mask, test_fea_mask, induce_mask
      ## examples_test: (src,dst) node pairs for the testing data; labels_test: true labels for the node pairs for the testing data
      ## examples_train: (src,dst) node pairs for the training data; labels_train: true labels
      ## examples_model_selection: (src,dst) node pairs for the validation data, used to select the best node combination method; labels_model_selection: true labels

    return nodes_data,edges_data,examples_test,labels_test,\
             examples_train,examples_model_selection,labels_train,labels_model_selection



def model_pipeline(nodes_data,edges_data,examples_test,labels_test,examples_train,
                   labels_train,examples_model_selection,labels_model_selection,yr,k,save_path,save_pr_path,
                   g_library="cogdl", ne_model="sdne"):
    """ graph_train: graph to generate the node embeddings,
        m_name: the ne model name, 
        g_library: where the graph library comes from, networkx 
        or stellargraph, the retrieval of node embeddings are different,
        the pipeline to train the node embedding and train the binary classifier, 
        and evaluate the binary classifier  """

    ## train node embedding for the feature network of the training data
    graph_train = graph_to_ne(nodes_data=nodes_data,edges_data=edges_data,
                              mask="induce_mask",g_library=g_library)    
    
    ## train NE with different NE models

    if g_library == "cogdl":
        if ne_model == "sdne":
            embedding_train = sdne_embedding(graph_train)
        
        elif ne_model == "grarep":
            embedding_train = grarep_embedding(graph_train)    

        elif ne_model == "hope":
            embedding_train = hope_embedding(graph_train)    

        elif ne_model == "line":
            embedding_train = line_embedding(graph_train)
        
        elif ne_model == "node2vec":
            embedding_train = node2vec_embedding(graph_train)

        elif ne_model == "deepwalk":
            embedding_train = deepwalk_embedding(graph_train)

    if g_library == "stellar":
        if ne_model == "gcn":
            embedding_train = gcn_embedding(graph_train, "Train Graph")

        elif ne_model == "graphsage":
            embedding_train = graphsage_embedding(graph_train, "Train Graph")
            
    if g_library == "nx":
        if ne_model == "struc2vec":
            embedding_train = struc2vec_embedding(graph_train)

    ## use the node embedding to train a classifier
    # need to change the embedding_train parameter with different node embeddings
    # binary_operators = [operator_hadamard, operator_l1, operator_l2, operator_avg,operator_cat]
    binary_operators = [operator_hadamard, operator_l1, operator_l2, operator_avg]

    # embedding_train = embedding_train_line
    results = [run_link_prediction(op,g_library=g_library,
                                 embedding_train=embedding_train,
                                 plot_auc=False, yr=yr, k=k, save_path=save_path, save_pr_path=save_pr_path) for op in binary_operators]
    best_result = max(results, key=lambda result: result["auprc_score"])
    # best_result = max(results, key=lambda result: result["precision_at_k"])

    print(f"Best result from '{best_result['binary_operator'].__name__}'")

    result_df = pd.DataFrame(
        [(result["binary_operator"].__name__, result["auprc_score"], result["auroc_score"], result["precision_at_k"]) for result in results],
        columns=("name", "PR AUC score", "ROC AUC score", "Precision at k"),
    #     columns=("name", "ROC AUC score"),
    ).set_index("name")

    print(result_df)

    ## evaluate the classifier with the test data and the node embeddings from the feature network of the testing data
    # embedding_test = embedding_test_line

    ## use the ne trained from the label networks of the training data for testing data
    test_auprc_score,test_auroc_score,test_prec_k = evaluate_link_prediction_model(
        best_result["classifier"],
        examples_test,
        labels_test,
        embedding_train,   
        best_result["binary_operator"],
        g_library=g_library,
        plot_auc=True,
        yr=yr,
        ne_model=ne_model,
        save_raw=True,
        k=k,
        save_path=save_path,
        save_pr_path=save_pr_path,
    )
    print(
        f"PR AUC score on test set using '{best_result['binary_operator'].__name__}': {test_auprc_score}",
        f"ROC AUC score on test set using '{best_result['binary_operator'].__name__}': {test_auroc_score}",
        f"Precision at k on test set using '{best_result['binary_operator'].__name__}': {test_prec_k}"
    )

    return test_auprc_score,test_auroc_score,test_prec_k



# run LINE on more years, i.e., more train/test pairs
# run struc2vec 

g_f0_tr = 1977

## iteratre different ne models 
for ne_model in ne_models:
    yr_results = dict()

    # save_pak_path = "/data/projects/punim0478/gracie/jbi_ne/output/prec_at_"+str(k)+"/"+ne_model
    # if not os.path.exists(save_pak_path):
    #     os.makedirs(save_pak_path)

    ## arg 01: save_path for saving the plots
    save_path = "/data/projects/punim0478/gracie/jbi_ne/output/plots/"+ne_model
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    ## arg 03: save_raw_path for saving the {year:(auprc,auroc)} dictionary
    save_raw_path = "/data/projects/punim0478/gracie/jbi_ne/output/auprc/"+ne_model
    if not os.path.exists(save_raw_path):
        os.makedirs(save_raw_path)

    ## arg 04: save_pr_path for saving the raw precision,recall value
    save_pr_path = "/home/yiyuanp1/jbi_ne/output/pr_raw/"+ne_model
    if not os.path.exists(save_pr_path):
        os.makedirs(save_pr_path)

    if ne_model in ["gcn","graphsage"]:
        g_library = "stellar"
    elif ne_model in ["sdne","grarep","hope","line","node2vec","deepwalk"]:
        g_library = "cogdl"
    elif ne_model == "struc2vec":
        g_library = "nx"

    for t in range(t_range[0],t_range[1]):

        nodes_data,edges_data,examples_test,labels_test,\
            examples_train,examples_model_selection,\
            labels_train,labels_model_selection = data_processing(g_f0_tr=g_f0_tr,g_lt_tt=t,ne_model=ne_model,k=k,path=path)

        test_score_line = model_pipeline(nodes_data,edges_data,examples_test,labels_test,examples_train,
                    labels_train,examples_model_selection,labels_model_selection,save_path=save_path,save_pr_path=save_pr_path,
                    ne_model=ne_model,g_library=g_library,yr=t,k=k)

        ## placeholder, add more ne models ... test_score_... ##

        yr_results[str(g_f0_tr)+"_"+str(t)] = test_score_line
            
        print(yr_results)
        
        # change the save path for different ne models 
        with open(save_raw_path+'/yr_results.json', 'w') as fout:
            json.dump(yr_results, fout)


# %%
