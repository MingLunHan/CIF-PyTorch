import torch
import torch.nn as nn


class CifMiddleware(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        # Load configurations
        self.cif_threshold = cfg.cif_threshold
        self.cif_output_dim = cfg.cif_embedding_dim
        self.encoder_embed_dim = cfg.encoder_embed_dim
        self.produce_weight_type = cfg.produce_weight_type
        self.conv_cif_width = cfg.conv_cif_width
        self.conv_cif_dropout = cfg.conv_cif_dropout
        self.apply_scaling = cfg.apply_scaling
        self.apply_tail_handling = cfg.apply_tail_handling
        self.tail_handling_firing_threshold = cfg.tail_handling_firing_threshold

        # Build weight generator
        if self.produce_weight_type == "dense":
            self.dense_proj = Linear(
                self.encoder_embed_dim, self.encoder_embed_dim).cuda()
            self.weight_proj = Linear(
                self.encoder_embed_dim, 1).cuda()
        elif self.produce_weight_type == "conv":
            self.conv = torch.nn.Conv1d(
                self.encoder_embed_dim,
                self.encoder_embed_dim,
                self.conv_cif_width,
                stride=1, padding=int(self.conv_cif_width / 2),
                dilation=1, groups=1,
                bias=True, padding_mode='zeros'
            ).cuda()
            self.conv_dropout = torch.nn.Dropout(
                p=self.conv_cif_dropout).cuda()
            self.weight_proj = Linear(
                self.encoder_embed_dim, 1).cuda()
        else:
            self.weight_proj = Linear(
                self.encoder_embed_dim, 1).cuda()

        # Build the final projection layer (if encoder_embed_dim is not equal to cif_output_dim)
        if self.cif_output_dim != self.encoder_embed_dim:
            self.cif_output_proj = Linear(
                self.encoder_embed_dim, self.cif_output_dim, bias=False).cuda()

    def forward(self, encoder_outputs, target_lengths):
        """
        Args:
            encoder_outputs: a dictionary that includes
                encoder_raw_out:
                    the raw outputs of acoustic encoder, with shape B x T x C
                encoder_padding_mask:
                    the padding mask (whose padded regions are filled with ones) of encoder outputs, with shape B x T
            target_lengths: the length of targets (necessary when training), with shape B
        Return:
            A dictionary:
                cif_out:
                    the cif outputs
                cif_out_padding_mask:
                    the padding infomation for cif outputs (whose padded regions are filled with zeros)
                quantity_out:
                    the sum of weights for the calculation of quantity loss
        """

        # Collect inputs
        encoder_raw_outputs = encoder_outputs["encoder_raw_out"]        # B x T x C
        encoder_padding_mask = encoder_outputs["encoder_padding_mask"]  # B x T

        # Produce weights for integration (accumulation)
        if self.produce_weight_type == "dense":
            proj_out = self.dense_proj(encoder_raw_outputs)
            act_proj_out = torch.relu(proj_out)
            sig_input = self.weight_proj(act_proj_out)
            weight = torch.sigmoid(sig_input)
        elif self.produce_weight_type == "conv":
            conv_input = encoder_raw_outputs.permute(0, 2, 1)
            conv_out = self.conv(conv_input)
            proj_input = conv_out.permute(0, 2, 1)
            proj_input = self.conv_dropout(proj_input)
            sig_input = self.weight_proj(proj_input)
            weight = torch.sigmoid(sig_input)
        else:
            sig_input = self.weight_proj(encoder_raw_outputs)
            weight = torch.sigmoid(sig_input)
        # weight has shape B x T x 1

        not_padding_mask = ~encoder_padding_mask
        weight = torch.squeeze(weight, dim=-1) * not_padding_mask.int()  # weight has shape B x T
        org_weight = weight

        # Apply scaling strategies
        if self.training and self.apply_scaling and target_lengths is not None:
            # Conduct scaling when training
            weight_sum = weight.sum(-1)             # weight_sum has shape B
            normalize_scalar = torch.unsqueeze(
                target_lengths / weight_sum, -1)    # normalize_scalar has shape B x 1
            weight = weight * normalize_scalar

        # Prepare for Integrate and fire
        batch_size = encoder_raw_outputs.size(0)
        max_length = encoder_raw_outputs.size(1)
        encoder_embed_dim = encoder_raw_outputs.size(2)
        padding_start_id = not_padding_mask.sum(-1)  # shape B

        accumulated_weights = torch.zeros(batch_size, 0).cuda()
        accumulated_states = torch.zeros(batch_size, 0, encoder_embed_dim).cuda()
        fired_states = torch.zeros(batch_size, 0, encoder_embed_dim).cuda()

        # Begin integrate and fire
        for i in range(max_length):
            # Get previous states from the recorded tensor
            prev_accumulated_weight = torch.zeros([batch_size]).cuda() if i == 0 else accumulated_weights[:, i - 1]
            prev_accumulated_state = \
                torch.zeros([batch_size, encoder_embed_dim]).cuda() if i == 0 else accumulated_states[:, i - 1, :]

            # Decide whether to fire a boundary
            cur_is_fired = ((prev_accumulated_weight + weight[:, i]) >= self.cif_threshold).unsqueeze(dim=-1)
            # cur_is_fired with shape B x 1

            # Update the accumulated weights
            cur_weight = torch.unsqueeze(weight[:, i], -1)
            # cur_weight has shape B x 1
            prev_accumulated_weight = torch.unsqueeze(prev_accumulated_weight, -1)
            # prev_accumulated_weight also has shape B x 1
            remained_weight = torch.ones_like(prev_accumulated_weight).cuda() - prev_accumulated_weight
            # remained_weight with shape B x 1

            # Obtain the accumulated weight of current step
            cur_accumulated_weight = torch.where(
                cur_is_fired,
                cur_weight - remained_weight,
                cur_weight + prev_accumulated_weight)  # B x 1

            # Obtain accumulated state of current step
            cur_accumulated_state = torch.where(
                cur_is_fired.repeat(1, encoder_embed_dim),
                (cur_weight - remained_weight) * encoder_raw_outputs[:, i, :],
                prev_accumulated_state + cur_weight * encoder_raw_outputs[:, i, :])  # B x C

            # Obtain fired state of current step:
            # firing locations has meaningful representations, while non-firing locations is all-zero embeddings
            cur_fired_state = torch.where(
                cur_is_fired.repeat(1, encoder_embed_dim),
                prev_accumulated_state + remained_weight * encoder_raw_outputs[:, i, :],
                torch.zeros([batch_size, encoder_embed_dim]).cuda())  # B x C

            # Handle the tail
            if (not self.training) and self.apply_tail_handling:
                # When encoder output position exceeds the max valid position,
                # if accumulated weights is greater than tail_handling_firing_threshold,
                # current state should be reserved, otherwise it is discarded.
                cur_fired_state = torch.where(
                    i == padding_start_id.unsqueeze(dim=-1).repeat([1, encoder_embed_dim]),
                    # shape B x C
                    torch.where(
                        cur_accumulated_weight.repeat([1, encoder_embed_dim]) <= self.tail_handling_firing_threshold,
                        # shape B x C
                        torch.zeros([batch_size, encoder_embed_dim]).cuda(),
                        # less equal than tail_handling_firing_threshold, discarded.
                        cur_accumulated_state / (cur_accumulated_weight + 1e-10)
                        # bigger than tail_handling_firing_threshold, normalized and kept.
                    ), cur_fired_state)
                # shape B x T

            # For normal condition, including both training and evaluation
            # Mask padded locations with all-zero vectors
            cur_fired_state = torch.where(
                torch.full([batch_size, encoder_embed_dim], i).cuda() >
                padding_start_id.unsqueeze(dim=-1).repeat([1, encoder_embed_dim]),
                torch.zeros([batch_size, encoder_embed_dim]).cuda(), cur_fired_state)

            # Update accumulation-related values: T_c stands for the length of integrated features
            accumulated_weights = torch.cat(
                (accumulated_weights, cur_accumulated_weight), 1)                       # B x T_c
            accumulated_states = torch.cat(
                (accumulated_states, torch.unsqueeze(cur_accumulated_state, 1)), 1)     # B x T_c x C
            fired_states = torch.cat(
                (fired_states, torch.unsqueeze(cur_fired_state, 1)), 1)                 # B x T_c x C

        # Extract cif_outputs for each utterance
        fired_marks = (torch.abs(fired_states).sum(-1) != 0.0).int()    # B x T_c
        fired_utt_length = fired_marks.sum(-1)                          # B
        fired_max_length = fired_utt_length.max().int()                 # The maximum of fired times in current batch
        cif_outputs = torch.zeros([0, fired_max_length, encoder_embed_dim]).cuda()

        def dynamic_partition(data: torch.Tensor, partitions: torch.Tensor, num_partitions=None):
            assert len(partitions.shape) == 1, "Only one dimensional partitions supported"
            assert (data.shape[0] == partitions.shape[0]), "Partitions requires the same size as data"
            if num_partitions is None:
                num_partitions = max(torch.unique(partitions))
            return [data[partitions == index] for index in range(num_partitions)]

        # Loop over all samples
        for j in range(batch_size):
            # Get information of j-th sample
            cur_utt_fired_mark = fired_marks[j, :]
            cur_utt_fired_state = fired_states[j, :, :]
            cur_utt_outputs = dynamic_partition(cur_utt_fired_state, cur_utt_fired_mark, 2)
            cur_utt_output = cur_utt_outputs[1]             # Get integrated representations
            cur_utt_length = cur_utt_output.size(0)         # The total number of firing
            pad_length = fired_max_length - cur_utt_length  # Get padded length
            cur_utt_output = torch.cat(
                (cur_utt_output, torch.full([pad_length, encoder_embed_dim], 0.0).cuda()), dim=0
            )  # Pad current utterance cif outputs to fired_max_length
            cur_utt_output = torch.unsqueeze(cur_utt_output, 0)
            # Reshape to 1 x T_c x C

            # Concatenate cur_utt_output and cif_outputs along batch axis
            cif_outputs = torch.cat([cif_outputs, cur_utt_output], 0)

        cif_out_padding_mask = (torch.abs(cif_outputs).sum(-1) != 0.0).int()
        # cif_out_padding_mask has shape B x T_c, where locations with value 0 are the padded locations.

        if self.training:
            quantity_out = org_weight.sum(-1)
        else:
            quantity_out = weight.sum(-1)

        if self.cif_output_dim != encoder_embed_dim:
            cif_outputs = self.cif_output_proj(cif_outputs)

        return {
            "cif_out": cif_outputs,                         # B x T_c x C
            "cif_out_padding_mask": cif_out_padding_mask,   # B x T_c
            "quantity_out": quantity_out                    # B
        }


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m
