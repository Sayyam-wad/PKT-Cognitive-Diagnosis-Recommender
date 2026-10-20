import math
import numpy as np
import torch
from sklearn import metrics
from tqdm import tqdm

from models.astnn import utils


def calculate_metrics(y_true, y_score):

    y_pred = (y_score >= 0.5).astype(int)

    if len(np.unique(y_true)) > 1:
        auc = metrics.roc_auc_score(
            y_true,
            y_score
        )
    else:
        auc = 0.5

    if len(np.unique(y_true)) > 1:
        pr_auc = metrics.average_precision_score(
            y_true,
            y_score
        )
    else:
        pr_auc = 0.0

    accuracy = metrics.accuracy_score(
        y_true,
        y_pred
    )

    precision = metrics.precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = metrics.recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = metrics.f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    return {
        "auc": auc,
        "pr_auc": pr_auc,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def train(
    model,
    params,
    optimizer,
    p_id,
    c_id,
    ast,
    target_c,
    result,
    c_embed,
    x_result
):

    N = int(
        math.floor(
            len(p_id) / params.batch_size
        )
    )

    shuffle_index = np.random.permutation(
        p_id.shape[0]
    )

    p_id = p_id[shuffle_index]
    c_id = c_id[shuffle_index]

    ast = [
        ast[i]
        for i in shuffle_index
    ]

    target_c = target_c[shuffle_index]
    result = result[shuffle_index]
    c_embed = c_embed[shuffle_index]
    x_result = x_result[shuffle_index]

    pred_list = []
    target_list = []

    model.train()

    epoch_loss = 0

    for idx in tqdm(range(N)):

        p_id_seq = p_id[
            idx * params.batch_size:
            (idx + 1) * params.batch_size,
            :
        ]

        c_id_seq = c_id[
            idx * params.batch_size:
            (idx + 1) * params.batch_size,
            :
        ]

        ast_seq = ast[
            idx * params.batch_size:
            (idx + 1) * params.batch_size
        ]

        target_c_seq = target_c[
            idx * params.batch_size:
            (idx + 1) * params.batch_size,
            :
        ]

        result_seq = result[
            idx * params.batch_size:
            (idx + 1) * params.batch_size,
            :
        ]

        c_embed_seq = c_embed[
            idx * params.batch_size:
            (idx + 1) * params.batch_size,
            :
        ]

        x_result_seq = x_result[
            idx * params.batch_size:
            (idx + 1) * params.batch_size,
            :
        ]

        input_p_id = utils.variable(
            torch.LongTensor(p_id_seq),
            params.gpu
        )

        input_c_id = utils.variable(
            torch.FloatTensor(c_id_seq),
            params.gpu
        )

        input_target_c = utils.variable(
            torch.FloatTensor(target_c_seq),
            params.gpu
        )

        input_result = utils.variable(
            torch.FloatTensor(result_seq),
            params.gpu
        ).view(-1, 1)

        input_c_embed = utils.variable(
            torch.FloatTensor(c_embed_seq),
            params.gpu
        )

        input_x_result = utils.variable(
            torch.FloatTensor(x_result_seq),
            params.gpu
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        loss, filtered_pred, filtered_target = model(
            input_p_id,
            input_c_id,
            ast_seq,
            input_target_c,
            input_result,
            input_c_embed,
            input_x_result
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        epoch_loss += utils.to_scalar(
            loss
        )

        pred_list.append(
            filtered_pred.detach()
            .cpu()
            .numpy()
        )

        target_list.append(
            filtered_target.detach()
            .cpu()
            .numpy()
        )

    all_pred = np.concatenate(
        pred_list,
        axis=0
    )

    all_target = np.concatenate(
        target_list,
        axis=0
    )

    metric_values = calculate_metrics(
        all_target,
        all_pred
    )

    return (
        epoch_loss / N,
        metric_values,
        all_target,
        all_pred
    )


def test(
    model,
    params,
    p_id,
    c_id,
    ast,
    target_c,
    result,
    c_embed,
    x_result
):

    N = int(
        math.floor(
            len(p_id) / params.batch_size
        )
    )

    pred_list = []
    target_list = []

    model.eval()

    epoch_loss = 0

    with torch.no_grad():

        for idx in tqdm(range(N)):

            p_id_seq = p_id[
                idx * params.batch_size:
                (idx + 1) * params.batch_size,
                :
            ]

            c_id_seq = c_id[
                idx * params.batch_size:
                (idx + 1) * params.batch_size,
                :
            ]

            ast_seq = ast[
                idx * params.batch_size:
                (idx + 1) * params.batch_size
            ]

            target_c_seq = target_c[
                idx * params.batch_size:
                (idx + 1) * params.batch_size,
                :
            ]

            result_seq = result[
                idx * params.batch_size:
                (idx + 1) * params.batch_size,
                :
            ]

            c_embed_seq = c_embed[
                idx * params.batch_size:
                (idx + 1) * params.batch_size,
                :
            ]

            x_result_seq = x_result[
                idx * params.batch_size:
                (idx + 1) * params.batch_size,
                :
            ]

            input_p_id = utils.variable(
                torch.LongTensor(p_id_seq),
                params.gpu
            )

            input_c_id = utils.variable(
                torch.FloatTensor(c_id_seq),
                params.gpu
            )

            input_target_c = utils.variable(
                torch.FloatTensor(target_c_seq),
                params.gpu
            )

            input_result = utils.variable(
                torch.FloatTensor(result_seq),
                params.gpu
            ).view(-1, 1)

            input_c_embed = utils.variable(
                torch.FloatTensor(c_embed_seq),
                params.gpu
            )

            input_x_result = utils.variable(
                torch.FloatTensor(x_result_seq),
                params.gpu
            )

            loss, filtered_pred, filtered_target = model(
                input_p_id,
                input_c_id,
                ast_seq,
                input_target_c,
                input_result,
                input_c_embed,
                input_x_result
            )

            epoch_loss += utils.to_scalar(
                loss
            )

            pred_list.append(
                filtered_pred.detach()
                .cpu()
                .numpy()
            )

            target_list.append(
                filtered_target.detach()
                .cpu()
                .numpy()
            )

    all_pred = np.concatenate(
        pred_list,
        axis=0
    )

    all_target = np.concatenate(
        target_list,
        axis=0
    )

    metric_values = calculate_metrics(
        all_target,
        all_pred
    )

    return (
        epoch_loss / N,
        metric_values,
        all_target,
        all_pred
    )