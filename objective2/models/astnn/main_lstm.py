import torch
import argparse
import numpy as np

from models.astnn.load_data_astnn import DATA_astnn
from models.astnn.run import train, test
from models.astnn.model_lstm import MODEL
import random
import os
from gensim.models.word2vec import Word2Vec

# set random seed
def seed_torch(seed=0):

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def main(fold):

    parser = argparse.ArgumentParser()
    # 可设置参数
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--EPOCH', type=int, default=50)
    parser.add_argument('--fold', type=int, default=fold)
    parser.add_argument('--dataset', type=str, default='pdkt')
    parser.add_argument('--set_seed', type=bool, default=True)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--init_lr', type=float, default=0.005)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--concept_embed_dim', type=int, default=64)
    parser.add_argument('--max_len', type=int, default=10)
    parser.add_argument('--ast_dim', type=int, default=128)  # <= 128

    root = 'data/' + parser.parse_args().dataset + '/astnn/'
    train_path = root + 'astnn_train' + str(parser.parse_args().fold)
    val_path = root + 'astnn_valid' + str(parser.parse_args().fold)
    test_path = root + 'astnn_test'

    if parser.parse_args().dataset == 'codeforces':

        # PKT parameters
        parser.add_argument('--num_concepts', type=int, default=37)  # 实际数量
        parser.add_argument('--num_problems', type=int, default=7152)
        parser.add_argument('--seqlen', type=int, default=200)
        parser.add_argument('--hidden_dim', type=int, default=64)
        parser.add_argument('--hidden_layers', type=int, default=2)

        ast_dim = parser.parse_args().ast_dim

        word2vec = Word2Vec.load(root + "node_w2v_128").wv
        ast_embeddings = np.zeros((word2vec.vectors.shape[0] + 1, ast_dim), dtype="float32")
        ast_embeddings[:word2vec.vectors.shape[0], :ast_dim] = word2vec.vectors[:, :ast_dim]

        parser.add_argument('--max_tokens', type=int, default=word2vec.vectors.shape[0] + 1)
        parser.add_argument('--ast_embed_dim', type=int, default=ast_dim)
        parser.add_argument('--ast_encode_dim', type=int, default=ast_dim)


    if parser.parse_args().dataset == 'pdkt':

        parser.add_argument('--num_concepts', type=int, default=106)
        parser.add_argument('--num_problems', type=int, default=600)
        parser.add_argument('--seqlen', type=int, default=200)
        parser.add_argument('--hidden_dim', type=int, default=64)
        parser.add_argument('--hidden_layers', type=int, default=2)

        ast_dim = parser.parse_args().ast_dim

        word2vec = Word2Vec.load(
            root + "node_w2v_128"
        ).wv

        ast_embeddings = np.zeros(
            (
                word2vec.vectors.shape[0] + 1,
                ast_dim
            ),
            dtype="float32"
        )

        ast_embeddings[
            :word2vec.vectors.shape[0],
            :ast_dim
        ] = word2vec.vectors[:, :ast_dim]

        parser.add_argument(
            '--max_tokens',
            type=int,
            default=word2vec.vectors.shape[0] + 1
        )

        parser.add_argument(
            '--ast_embed_dim',
            type=int,
            default=ast_dim
        )

        parser.add_argument(
            '--ast_encode_dim',
            type=int,
            default=ast_dim
        )

    #  ________________________________________________________________________________________________________

    params = parser.parse_args()
    print(params)
    results_dir = "results/pdkt_lstm"
    os.makedirs(results_dir, exist_ok=True)

    print("Results directory:", results_dir)
    # set random seed
    if params.set_seed is True:
        seed_torch(0)

    # load data
    data = DATA_astnn(num_concepts=params.num_concepts, seqlen=params.seqlen)

    print(train_path)
    train_p_id, train_c_id, train_ast, train_target_c, train_result, train_c_embed, \
    train_x_result = data.load_data(train_path)
    print(val_path)
    val_p_id, val_c_id, val_ast, val_target_c, val_result, val_c_embed, \
    val_x_result = data.load_data(val_path)

    model = MODEL(num_concepts=params.num_concepts,
                  num_problems=params.num_problems,
                  hidden_dim=params.hidden_dim,
                  hidden_layers=params.hidden_layers,
                  concept_embed_dim=params.concept_embed_dim,
                  ast_embed_dim=params.ast_embed_dim,
                  max_tokens=params.max_tokens,
                  ast_encode_dim=params.ast_encode_dim,
                  ast_pretrained_weight=ast_embeddings,
                  max_len=params.max_len,
                  batch_size=params.batch_size,
                  gpu=params.gpu)

    model.init_params()
    model.init_embeddings()

    optimizer = torch.optim.Adam(params=model.parameters(), lr=params.init_lr, weight_decay=params.weight_decay)

    if params.gpu >= 0:
        print('device: ' + str(params.gpu))
        torch.cuda.set_device(params.gpu)
        model.cuda()

    best_valid_auc = 0
    count = 0
    import csv

    # csv_file = open(
    #     os.path.join(results_dir, "epoch_metrics_lstm.csv"),
    #     "w",
    #     newline=""
    # )

    # csv_writer = csv.writer(csv_file)

    # csv_writer.writerow([
    #     "epoch",
    #     "train_auc",
    #     "train_acc",
    #     "valid_auc",
    #     "valid_acc",
    #     "valid_error"
    # ])
    for idx in range(0):
        train_loss, train_metrics, _, _ = train(
            model,
            params,
            optimizer,
            train_p_id,
            train_c_id,
            train_ast,
            train_target_c,
            train_result,
            train_c_embed,
            train_x_result
        )

        train_accuracy = train_metrics["accuracy"]
        train_auc = train_metrics["auc"]    
        print('Epoch %d/%d, loss : %3.5f, auc : %3.5f, accuracy : %3.5f' % (
            idx + 1, params.EPOCH, train_loss, train_auc, train_accuracy))

        valid_loss, valid_metrics, _, _ = test(
            model,
            params,
            val_p_id,
            val_c_id,
            val_ast,
            val_target_c,
            val_result,
            val_c_embed,
            val_x_result
        )

        valid_accuracy = valid_metrics["accuracy"]
        valid_auc = valid_metrics["auc"]
        valid_error = 1 - valid_accuracy

        csv_writer.writerow([
            idx + 1,
            train_auc,
            train_accuracy,
            valid_auc,
            valid_accuracy,
            valid_error
        ])

        csv_file.flush()
        print('Epoch %d/%d, valid auc : %3.5f, valid accuracy : %3.5f' % (
            idx + 1, params.EPOCH, valid_auc, valid_accuracy))

        # if valid_auc > best_valid_auc and valid_auc - best_valid_auc > 0.0003:
        if valid_auc > best_valid_auc:
            torch.save(
                model.state_dict(),
                os.path.join(results_dir, "best_lstm_model.pt")
            )
            count = 0
            print('%3.4f to %3.4f' % (best_valid_auc, valid_auc))
            best_valid_auc = valid_auc
            best_train_auc = train_auc
            best_model = MODEL(
                num_concepts=params.num_concepts,
                num_problems=params.num_problems,
                hidden_dim=params.hidden_dim,
                hidden_layers=params.hidden_layers,
                concept_embed_dim=params.concept_embed_dim,
                ast_embed_dim=params.ast_embed_dim,
                max_tokens=params.max_tokens,
                ast_encode_dim=params.ast_encode_dim,
                ast_pretrained_weight=ast_embeddings,
                max_len=params.max_len,
                batch_size=params.batch_size,
                gpu=params.gpu
            )

            best_model.load_state_dict(model.state_dict())
            if params.gpu >= 0:
                best_model.cuda()

        else:
           pass

    best_model = MODEL(
        num_concepts=params.num_concepts,
        num_problems=params.num_problems,
        hidden_dim=params.hidden_dim,
        hidden_layers=params.hidden_layers,
        concept_embed_dim=params.concept_embed_dim,
        ast_embed_dim=params.ast_embed_dim,
        max_tokens=params.max_tokens,
        ast_encode_dim=params.ast_encode_dim,
        ast_pretrained_weight=ast_embeddings,
        max_len=params.max_len,
        batch_size=params.batch_size,
        gpu=params.gpu
    )

    best_model.load_state_dict(
        torch.load(
            os.path.join(results_dir, "best_lstm_model.pt"),
            map_location="cuda" if params.gpu >= 0 else "cpu"
        )
    )

    if params.gpu >= 0:
        best_model.cuda()

    print("Loaded PDKT LSTM checkpoint.")
    # test
    test_p_id, test_c_id, test_ast, test_target_c, test_result, test_c_embed, \
    test_x_result = data.load_data(test_path)

    test_loss, metrics, y_true, y_score = test(
        best_model,
        params,
        test_p_id,
        test_c_id,
        test_ast,
        test_target_c,
        test_result,
        test_c_embed,
        test_x_result
    )

    test_auc = metrics["auc"]
    test_accuracy = metrics["accuracy"]

    from sklearn.metrics import f1_score, mean_squared_error

    f1 = f1_score(
        y_true,
        (y_score >= 0.5).astype(int)
    )

    mse = mean_squared_error(
        y_true,
        y_score
    )

    print("\n" + "=" * 50)
    print("FINAL PDKT LSTM METRICS")
    print("=" * 50)

    print(f"AUC        : {test_auc:.6f}")
    print(f"ACC        : {test_accuracy:.6f}")
    print(f"ERROR RATE : {1 - test_accuracy:.6f}")
    print(f"F1 SCORE   : {f1:.6f}")
    print(f"MSE        : {mse:.6f}")

    csv_file.close()

    return 0.0, 0.0, test_auc   
  


if __name__ == '__main__':

    best_train_auc, best_valid_auc, test_auc = main(1)

    print(
        'Best Train AUC:',
        best_train_auc,
        '| Best Valid AUC:',
        best_valid_auc,
        '| Test AUC:',
        test_auc
    )