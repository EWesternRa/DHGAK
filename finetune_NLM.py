import argparse
import os

def arg_parser():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--dataset', type=str, default='MUTAG')
    arg_parser.add_argument('--data_path', type=str, default='./datasets')
    arg_parser.add_argument('--seed', type=int, default=0)
    
    arg_parser.add_argument('--H', type=int, default=10) # H hop
    arg_parser.add_argument('--b', type=int, default=2) # b width

    arg_parser.add_argument('--load_slices_encoding', type=bool, default=True)    # load slice encoding from file (H, b)
    arg_parser.add_argument('--save_slices_encoding', type=bool, default=True)    # save slice encoding to file (H, b)
    arg_parser.add_argument('--node_centrality_path', type=str, default='node_centrality')  # path to save node centralities
    
    arg_parser.add_argument('--model', type=str, default='bert')
    arg_parser.add_argument('--load_path', type=str, default='nlms/bert_model/bert-base-uncased')   # path of the pretrained model
    arg_parser.add_argument('--save_path', type=str, default='nlms/bert_model/finetuned-bert')  # path to save the finetuned model
    arg_parser.add_argument('--epochs', type=int, default=3)
    arg_parser.add_argument('--batch_size', type=int, default=24)
    arg_parser.add_argument('--max_length', type=int, default=250)
    arg_parser.add_argument('--lr', type=float, default=5e-5)
    arg_parser.add_argument('--device', type=str, default='1,')  # separate by comma, e.g. '0,1,2,3'
    
    return arg_parser.parse_args()
    

if __name__ == '__main__':
    args = arg_parser()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    from src.NLMs import NLM
    print(args)

    # load data
    from src.utils import load_data_save_slices
    nodes_slices, y, num_nodes = load_data_save_slices(args.dataset, args.b, args.H, args.data_path, args.load_slices_encoding,
                                                 args.save_slices_encoding, args.node_centrality_path)
        
    # finetune NLM model
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    nlm = NLM(args.model, 0, args.load_path, args.device, usage='finetune')
    sentences = []
    for hth in range(args.H):
        hth_sentences = [s[hth] for s in nodes_slices]
        # limit the length of sentences
        hth_sentences = [s[:args.max_length] for s in hth_sentences]
        sentences += [' '.join(map(str, s)) for s in hth_sentences]
    runs_path = f'./bert_runs'  # checkpoint path
    if not os.path.exists(runs_path):
        os.makedirs(runs_path)
    try:
        nlm.finetune(sentences,
                    epochs=args.epochs,
                    random_state=args.seed,
                    batch_size=args.batch_size,
                    max_length=args.max_length,
                    lr=args.lr,
                    bert_output_dir=runs_path)
    except Exception as e:
        print(e)
        print('finetune bert model failed.')
        exit()
    print('finetune done.')
    
    # save model
    detail_save_path = args.save_path + f'/model-{args.dataset}-finetuned-H{str(args.H)}b{str(args.b)}-e{str(args.epochs)}'
    nlm.save_model(detail_save_path)