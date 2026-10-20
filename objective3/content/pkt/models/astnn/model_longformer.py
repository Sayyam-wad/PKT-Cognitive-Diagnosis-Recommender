import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from transformers import LongformerConfig, LongformerModel

from models.astnn.utils import *


class BatchTreeEncoder(nn.Module):

    def __init__(
            self,
            vocab_size,
            embedding_dim,
            encode_dim,
            batch_size,
            use_gpu,
            pretrained_weight=None):

        super(BatchTreeEncoder, self).__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim
        )

        self.encode_dim = encode_dim

        self.W_c = nn.Linear(
            embedding_dim,
            encode_dim
        )

        self.W_l = nn.Linear(
            encode_dim,
            encode_dim
        )

        self.W_r = nn.Linear(
            encode_dim,
            encode_dim
        )

        self.activation = F.relu

        self.stop = -1

        self.batch_size = batch_size

        self.use_gpu = use_gpu

        if pretrained_weight is not None:

            self.embedding.weight.data.copy_(

                torch.from_numpy(
                    pretrained_weight
                )
            )

    def forward(self, x, bs):
        if not x:
            return self.embedding.weight.new_zeros((0, self.encode_dim))

        levels, root_ids, parent_ids = [list(x)], [list(range(bs))], []
        while True:
            children, parents, child_roots = [], [], []
            for parent_index, node in enumerate(levels[-1]):
                for child in node[1:]:
                    if child[0] != -1:
                        children.append(child)
                        parents.append(parent_index)
                        child_roots.append(root_ids[-1][parent_index])
            if not children:
                break
            levels.append(children)
            root_ids.append(child_roots)
            parent_ids.append(torch.tensor(parents, dtype=torch.long, device=self.embedding.weight.device))

        offsets = []
        tokens = []
        root_indices = []
        start = 0
        for nodes, roots in zip(levels, root_ids):
            offsets.append((start, start + len(nodes)))
            start += len(nodes)
            tokens.extend(node[0] for node in nodes)
            root_indices.extend(roots)
        own_values = self.W_c(self.embedding(torch.tensor(
            tokens, dtype=torch.long, device=self.embedding.weight.device
        )))

        encoded_levels = [None] * len(levels)
        for depth in range(len(levels) - 1, -1, -1):
            begin, end = offsets[depth]
            values = own_values[begin:end]
            if depth < len(levels) - 1:
                child_sums = values.new_zeros(values.shape).index_add(
                    0, parent_ids[depth], encoded_levels[depth + 1]
                )
                values = values + child_sums
            encoded_levels[depth] = values

        node_values = torch.cat(encoded_levels)
        root_indices = torch.tensor(root_indices, dtype=torch.long, device=node_values.device)
        pooled = node_values.new_full((bs, self.encode_dim), float('-inf'))
        return pooled.scatter_reduce(
            0, root_indices[:, None].expand_as(node_values), node_values,
            reduce="amax", include_self=True
        )


class MODEL(nn.Module):

    def __init__(
            self,
            num_concepts,
            num_problems,
            hidden_dim,
            hidden_layers,
            concept_embed_dim,
            ast_embed_dim,
            max_tokens,
            ast_encode_dim,
            ast_pretrained_weight,
            max_len,
            batch_size,
            gpu,
            ff_dropout=0.25,
            attn_dropout=0.25,
            gru_dropout=0.2,
            embedding_dropout=0.3,
            attention_window=32):

        super(MODEL, self).__init__()

        self.hidden_dim = hidden_dim

        self.hidden_layers = hidden_layers

        self.gpu = gpu

        self.num_concepts = num_concepts

        self.ast_encode_dim = ast_encode_dim

        self.batch_size = batch_size

        self.max_len = max_len

        self.concept_embed_dim = concept_embed_dim

        self.embedding_dropout = embedding_dropout

        ###################################################################
        # DKT Embedding
        ###################################################################

        self.p_id_embed = nn.Embedding(
            num_problems + 1,
            concept_embed_dim
        )

        ###################################################################
        # ASTNN Encoder
        ###################################################################

        self.encoder = BatchTreeEncoder(
            max_tokens,
            ast_embed_dim,
            ast_encode_dim,
            batch_size,
            gpu,
            ast_pretrained_weight
        )

        ###################################################################
        # BiGRU
        ###################################################################

        self.bigru = nn.GRU(
            ast_encode_dim,
            hidden_size=32,
            num_layers=2,
            dropout=gru_dropout,
            bidirectional=True,
            batch_first=True
        )

        ###################################################################
        # Feature Dimension
        ###################################################################

        self.LSTM_feature_dim = (
                2
                + 64
                + num_concepts
                + 1
        )

        ###################################################################
        # Longformer
        ###################################################################

        self.longformer_config = LongformerConfig(

            hidden_size=self.LSTM_feature_dim,

            num_attention_heads=4,

            num_hidden_layers=1,

            intermediate_size=256,

            attention_window=attention_window,

            hidden_dropout_prob=ff_dropout,

            attention_probs_dropout_prob=attn_dropout
        )

        self.longformer = LongformerModel(
            self.longformer_config
        )

        ###################################################################
        # Prediction Layer
        ###################################################################

        self.predict_Linear = nn.Linear(
            self.LSTM_feature_dim,
            num_concepts + 1,
            bias=True
        )

    def init_embeddings(self):

        nn.init.kaiming_normal_(
            self.p_id_embed.weight
        )

    def init_params(self):

        nn.init.kaiming_normal_(
            self.predict_Linear.weight
        )

        nn.init.constant_(
            self.predict_Linear.bias,
            0
        )

    def get_zeros(self, num):

        zeros = variable(

            Variable(
                torch.zeros(
                    num,
                    self.ast_encode_dim
                )
            ),

            self.gpu
        )

        return zeros

    def _encode_ast_batch(self, ast, batch_size, seqlen):
        """Pool only real AST roots; empty ASTs and padded steps stay zero."""
        device = self.encoder.embedding.weight.device
        item_count = batch_size * seqlen
        sequences = torch.zeros(
            item_count, self.max_len, self.ast_encode_dim, device=device
        )
        lengths = torch.zeros(item_count, dtype=torch.long, device=device)
        roots = []
        spans = []
        for row in range(batch_size):
            row_asts = ast[row][:seqlen]
            for step in range(seqlen):
                trees = row_asts[step][:self.max_len] if step < len(row_asts) else []
                start = len(roots)
                roots.extend(trees)
                spans.append((row * seqlen + step, start, len(roots)))

        if roots:
            encoded_roots = self.encoder(roots, len(roots))
            for item, start, end in spans:
                length = end - start
                if length:
                    sequences[item, :length] = encoded_roots[start:end]
                    lengths[item] = length

        pooled = sequences.new_zeros((item_count, 64))
        nonempty = lengths.gt(0).nonzero(as_tuple=True)[0]
        if nonempty.numel():
            packed = pack_padded_sequence(
                sequences[nonempty], lengths[nonempty].cpu(), batch_first=True,
                enforce_sorted=False
            )
            packed_output, _ = self.bigru(packed)
            output, _ = pad_packed_sequence(
                packed_output, batch_first=True, total_length=self.max_len
            )
            mask = torch.arange(self.max_len, device=device)[None, :] < lengths[nonempty, None].to(device)
            pooled = pooled.index_copy(
                0, nonempty,
                output.masked_fill(~mask[:, :, None], float('-inf')).max(dim=1).values
            )
        return pooled.view(batch_size, seqlen, 64)

    def concept_logits(self, p_id, c_id, ast, c_embed, cur_result):
        """Return per-concept logits for every timestep, using history prefixes."""
        batch_size, seqlen = p_id.shape
        active = p_id.gt(0)
        if (active & (~active).cummax(dim=1).values).any():
            raise ValueError("p_id sequences must be right padded")
        code_embedding = self._encode_ast_batch(ast, batch_size, seqlen)
        code_embedding = F.dropout(
            code_embedding, p=self.embedding_dropout, training=self.training
        )
        inputs = torch.cat([c_embed, code_embedding, cur_result], dim=2)
        endpoints = active.nonzero(as_tuple=False)
        history = inputs.new_zeros((batch_size * seqlen, self.LSTM_feature_dim))
        if endpoints.numel():
            rows, positions = endpoints.unbind(dim=1)
            prefix_inputs = inputs.index_select(0, rows)
            columns = torch.arange(seqlen, device=p_id.device)
            prefix_mask = (active.index_select(0, rows) & (columns[None, :] <= positions[:, None]))
            prefix_states = self.longformer(
                inputs_embeds=prefix_inputs,
                attention_mask=prefix_mask.to(dtype=torch.long)
            ).last_hidden_state
            endpoint_states = prefix_states[torch.arange(len(positions), device=p_id.device), positions]
            flat_positions = rows * seqlen + positions
            history = history.index_copy(0, flat_positions, endpoint_states)
        history = history.view(batch_size, seqlen, self.LSTM_feature_dim)

        logits = self.predict_Linear(history.reshape(batch_size * seqlen, -1))
        logits = logits.view(batch_size, seqlen, -1)
        logits = logits * active[:, :, None]
        self.last_history_states = history.detach()
        self.last_concept_logits = logits.detach()
        return logits

    def predict_mastery(self, p_id, c_id, ast, c_embed, cur_result):
        """Return uncalibrated final mastery probabilities; no history yields 0.5."""
        if p_id.shape[1] == 0:
            return self.predict_Linear.weight.new_full((p_id.shape[0], self.num_concepts), 0.5)
        logits = self.concept_logits(p_id, c_id, ast, c_embed, cur_result)
        valid = p_id.gt(0)
        positions = torch.arange(1, p_id.shape[1] + 1, device=p_id.device)
        last = (valid * positions).max(dim=1).values - 1
        rows = torch.arange(p_id.shape[0], device=p_id.device)
        final_logits = logits[rows, last.clamp_min(0)]
        final_logits = final_logits * valid.any(dim=1, keepdim=True)
        return torch.sigmoid(final_logits[:, 1:])

    def forward(
            self,
            p_id,
            c_id,
            ast,
            target_c,
            result,
            c_embed,
            cur_result):

        bs = p_id.shape[0]

        seqlen = p_id.shape[1]

        ###################################################################
        # Prediction
        ###################################################################

        num_concepts = torch.sum(
            target_c,
            2
        ).view(-1)

        prediction = self.concept_logits(p_id, c_id, ast, c_embed, cur_result)
        prediction = prediction.reshape(bs * seqlen, -1)

        prediction_1d = torch.bmm(

            prediction.unsqueeze(1),

            target_c.view(
                bs * seqlen,
                -1
            ).unsqueeze(2)

        ).squeeze(2)

        mask = num_concepts.gt(0)

        num_concepts = torch.masked_select(
            num_concepts,
            mask
        )

        filtered_pred = torch.masked_select(
            prediction_1d.squeeze(1),
            mask
        )

        filtered_pred = torch.div(
            filtered_pred,
            num_concepts
        )

        filtered_target = torch.masked_select(
            result.squeeze(1),
            mask
        )

        loss = F.binary_cross_entropy_with_logits(filtered_pred, filtered_target) if mask.any() else prediction.sum() * 0

        return (
            loss,
            torch.sigmoid(filtered_pred),
            filtered_target
        )
