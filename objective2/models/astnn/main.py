from sched import scheduler

from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import torch
import argparse
import numpy as np
import copy
import random
import os
import csv
from gensim.models.word2vec import Word2Vec

from models.astnn.load_data_astnn import DATA_astnn
from models.astnn.run import train, test

# IMPORTANT
from models.astnn.model_longformer import MODEL


# =========================================================
# SEED
# =========================================================

def seed_torch(seed=0):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False


# =========================================================
# MAIN
# =========================================================

def main(fold, params_override=None):

    parser = argparse.ArgumentParser()

    # =====================================
    # BASIC PARAMS
    # =====================================

    parser.add_argument('--gpu', type=int, default=0)

    parser.add_argument('--EPOCH', type=int, default=50)

    parser.add_argument('--fold', type=int, default=fold)

    parser.add_argument('--dataset', type=str, default='codeforces')

    parser.add_argument('--set_seed', type=bool, default=True)

    parser.add_argument('--batch_size', type=int, default=1)

    parser.add_argument('--init_lr', type=float, default=0.0002)

    parser.add_argument('--weight_decay', type=float, default=0.0002)

    parser.add_argument('--concept_embed_dim', type=int, default=64)

    parser.add_argument('--max_len', type=int, default=20)

    parser.add_argument('--ast_dim', type=int, default=128)

    params = parser.parse_args()

    # =====================================
    # HYPERPARAM OVERRIDE
    # =====================================

    if params_override is not None:

        for key, value in params_override.items():

            setattr(params, key, value)

    print("Params:", params)

    # =====================================
    # PATHS
    # =====================================

    base_dir = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "../../"
        )
    )

    root = os.path.join(
        base_dir,
        "data",
        params.dataset,
        "astnn"
    ) + "/"

    train_path = root + 'astnn_train' + str(params.fold)

    val_path = root + 'astnn_valid' + str(params.fold)

    test_path = root + 'astnn_test'

    # =====================================
    # DATASET CONFIG
    # =====================================

    if params.dataset == 'codeforces':

        params.num_concepts = 37

        params.num_problems = 7152

        params.seqlen = 200

        params.hidden_dim = getattr(
            params,
            "hidden_dim",
            64
        )

        params.hidden_layers = getattr(
            params,
            "hidden_layers",
            2
        )

        ast_dim = params.ast_dim

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

        params.max_tokens = (
            word2vec.vectors.shape[0] + 1
        )

        params.ast_embed_dim = ast_dim

        params.ast_encode_dim = ast_dim

    # =====================================
    # SEED
    # =====================================

    if params.set_seed:

        seed_torch(0)

    # =====================================
    # DATA LOADING
    # =====================================

    data = DATA_astnn(
        num_concepts=params.num_concepts,
        seqlen=params.seqlen
    )

    print("Train path:", train_path)

    train_p_id, train_c_id, train_ast, \
    train_target_c, train_result, \
    train_c_embed, train_x_result = data.load_data(train_path)

    print("Val path:", val_path)

    val_p_id, val_c_id, val_ast, \
    val_target_c, val_result, \
    val_c_embed, val_x_result = data.load_data(val_path)

    # =====================================
    # MODEL
    # =====================================

    model = MODEL(

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

        gpu=params.gpu,

        ff_dropout=getattr(params, "ff_dropout", 0.25),

        attn_dropout=getattr(params, "attn_dropout", 0.25),

        gru_dropout=getattr(params, "gru_dropout", 0.2),

        # dim_head=getattr(params, "dim_head", 48),

        # nb_features=getattr(params, "nb_features", 64),
        embedding_dropout=getattr(
            params,
            "embedding_dropout",
            0.3
        ),
        attention_window=getattr(
            params,
            "attention_window",
            32
        )
    )

    model.init_params()

    model.init_embeddings()

    # =====================================
    # OPTIMIZER
    # =====================================

    optimizer = torch.optim.AdamW(

        model.parameters(),

        lr=params.init_lr,

        weight_decay=params.weight_decay
    )

    # =====================================
    # SCHEDULER
    # =====================================

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(

        optimizer,

        mode='max',

        factor=0.5,

        patience=2
    )

    # =====================================
    # GPU
    # =====================================

    if params.gpu >= 0:

        torch.cuda.set_device(params.gpu)

        model.cuda()

    csv_file = open("epoch_metrics.csv", "w", newline="")

    csv_writer = csv.writer(csv_file)

    csv_writer.writerow([
        "epoch",
        "train_auc",
        "train_acc",
        "valid_auc",
        "valid_acc",
        "valid_error"
    ])
    # =====================================
    # TRAINING
    # =====================================

    best_valid_auc = 0

    best_train_auc = 0

    best_model = None

    count = 0

    for idx in range(params.EPOCH):

        train_loss, train_accuracy, train_auc = train(

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

        print(

            f'Epoch {idx+1}/{params.EPOCH} | '

            f'loss={train_loss:.4f} | '

            f'auc={train_auc:.4f} | '

            f'acc={train_accuracy:.4f} | '

            f'lr={optimizer.param_groups[0]["lr"]:.6f}'
        )

        valid_loss, valid_accuracy, valid_auc, y_true, y_score = test(

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

        print(
            f'Valid AUC={valid_auc:.4f}, '
            f'ACC={valid_accuracy:.4f}'
        )
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
        # fpr, tpr, _ = roc_curve(y_true, y_score)

        # plt.figure()

        # plt.plot(fpr, tpr)

        # plt.xlabel("False Positive Rate")
        # plt.ylabel("True Positive Rate")

        # plt.title(
        #     f"ROC Curve Epoch {idx+1} | AUC={valid_auc:.4f}"
        # )

        # plt.savefig(
        #     f"roc_epoch_{idx+1}.png"
        # )

        # plt.close()
        scheduler.step(valid_auc)

# =================================
# SAVE BEST
# =================================

        if valid_auc > best_valid_auc:

            save_path = os.path.join(
                    base_dir,
                    "results",
                    "best_longformer_model.pt"
            )

            torch.save(
                model.state_dict(),
                save_path
            )

            best_valid_auc = valid_auc

            best_train_auc = train_auc

            best_model = copy.deepcopy(model)

            count = 0

        else:

            count += 1

            if count == 50:

                print("Early stopping triggered.")

                break

    # =====================================
    # TEST
    # =====================================

    test_p_id, test_c_id, test_ast, \
    test_target_c, test_result, \
    test_c_embed, test_x_result = data.load_data(test_path)

    test_loss, test_accuracy, test_auc,  y_true, y_score = test(

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

    print(
        f'Test AUC={test_auc:.4f}, '
        f'ACC={test_accuracy:.4f}'
    )
    fpr, tpr, _ = roc_curve(y_true, y_score)

    plt.figure()

    plt.plot(
        fpr,
        tpr,
        label=f"AUC = {test_auc:.4f}"
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--"
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    plt.title("Final Test ROC Curve")

    plt.legend()

    plt.savefig("roc_final.png")

    plt.close()
    csv_file.close()
    return {
    "train_auc": best_train_auc,
    "valid_auc": best_valid_auc,
    "test_auc": test_auc
    }


# =========================================================
# RUN
# =========================================================

if __name__ == '__main__':

    result = main(
    fold=1,
    params_override={
        "init_lr": 0.00017986266166303768,
        "ff_dropout": 0.22469715844352778,
        "attn_dropout": 0.25004466156010574,
        "embedding_dropout": 0.3383136695767353,
        "attention_window": 8,
        "max_len": 55
    }
)

    print(
        'Best Train AUC:',
        result["train_auc"],

        '| Best Valid AUC:',
        result["valid_auc"],

        '| Test AUC:',
        result["test_auc"]
    )