import argparse
import time
from src.NLMs import NLM
from src.dhgak import dhgak
from src.utils import crossvalidation_kernel, get_label2id, set_seeds, load_data_save_slices
# ignore the convergence warning
import warnings
warnings.filterwarnings("ignore")


def arg_parser():
    arg_parser = argparse.ArgumentParser()
    
    # data parameters
    arg_parser.add_argument('--dataset', type=str, default='MUTAG')
    arg_parser.add_argument('--data_path', type=str, default='./datasets')
    arg_parser.add_argument('--seed', type=int, default=0)
    arg_parser.add_argument('--gridsearch', type=bool, default=True)    # search for the best C
    arg_parser.add_argument('--crossvalidation', type=bool, default=True)   # 10-fold cross-validation
    
    # load from slices encoding
    arg_parser.add_argument('--load_slices_encoding', type=bool, default=True)    # load slice encoding from file (H, b)
    arg_parser.add_argument('--save_slices_encoding', type=bool, default=True)    # save slice encoding to file (H, b)
    arg_parser.add_argument('--node_centrality_path', type=str, default='node_centrality')  # path to save node centralities
    
    # kernel parameters
    arg_parser.add_argument('--alpha', type=float, default=0.6) # decay factor
    arg_parser.add_argument('--H', type=int, default=5) # H hop
    arg_parser.add_argument('--b', type=int, default=2) # b width
    
    # clustering parameters
    arg_parser.add_argument('--cluster_factor', type=float, default=0.1) # cluster factor * num_graphs = num_clusters
    arg_parser.add_argument('--cluster_method', type=str, default='k-means') # k-means, GMM, DBSCAN  separate by comma, e.g. 'k-means,GMM' for two clustering methods
    arg_parser.add_argument('--kmeans_init', type=str, default='random') # k-means++, ramdom,
    arg_parser.add_argument('--T', type=int, default=3)
    arg_parser.add_argument('--cluster_jobs', type=int, default=5) # -1: use all cpu cores, 1: use one core
     
    # NLM parameters
    arg_parser.add_argument('--model_name', type=str, default='bert')    # bert or w2v or label(onehot embedding)
    arg_parser.add_argument('--load_path', type=str, default='nlms/bert_model/bert-base-uncased')   # path of the pretrained bert model
    arg_parser.add_argument('--out_batch_size', type=int, default=450)
    arg_parser.add_argument('--max_length', type=int, default=250)
    
    arg_parser.add_argument('--device', type=str, default='0') # e.g. single gpu:'5' or 'cpu'
    
    return arg_parser.parse_args()

if __name__ == '__main__':
    args = arg_parser()
    set_seeds(args.seed)

    # load data
    nodes_slices, y, num_nodes = load_data_save_slices(args.dataset, args.b, args.H, args.data_path, args.load_slices_encoding,
                                                 args.save_slices_encoding, args.node_centrality_path)
    
    # Deep embedding of slices
    slices_embs = []
    device = args.device if args.device == 'cpu' else 'cuda:'+args.device
    label2id = get_label2id(nodes_slices) if args.model_name == 'label' else None
    nlm = NLM(args.model_name, random_state=args.seed, bert_path=args.load_path, device=device, usage='fit', label2id=label2id)
    start = time.time()
    for hth in range(args.H+1):
        hth_sentences = [s[hth] for s in nodes_slices]
        embeddings = nlm.fit(hth_sentences, y, bert_max_length=args.max_length, bert_out_batch_size=args.out_batch_size)
        slices_embs.append(embeddings)
    
    for h in range(1, args.H+1):
        slices_embs[h] = args.alpha * slices_embs[h-1] + slices_embs[h]
        
    # DGAK and DHGAK
    clustering_methods = args.cluster_method.split(',')
    
    kernel_matrix = dhgak(slices_embs, num_nodes, clustering_methods, args.T, args.cluster_factor,
                    args.cluster_jobs, args.seed, kmeans_init=args.kmeans_init)
    end = time.time()
    print('compute kernel matrix done!')
    
    # cross-validation
    acc, _ = crossvalidation_kernel(kernel_matrix, y, 10, args.seed, args.gridsearch, args.crossvalidation, )
    print(f'time cost: {round(end-start, 2)}s')