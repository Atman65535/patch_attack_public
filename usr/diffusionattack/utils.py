import torch


def aggregate_attention(prompts, attention_store, res: int, from_where, is_cross: bool, select: int, is_cpu=True):
    """

    Args:
        prompts:
        attention_store: 'controller': AttentionControlEdit
        res:
        from_where:
        is_cross:
        select:
        is_cpu:

    Returns:

    """
    out = []
    attention_maps = attention_store.get_average_attention()
    num_pixels = res ** 2
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if item.shape[1] == num_pixels:
                cross_maps = item.reshape(len(prompts), -1, res, res, item.shape[-1])[select]
                out.append(cross_maps)
    out = torch.cat(out, dim=0)
    out = out.sum(0) / out.shape[0]
    return out.cpu() if is_cpu else out

def build_label_embeddings(model,
                           label_dict,
                           diffusion_steps,
                           optimize_iterations,
                           label_cls=19):
    """

    Args:
        model: stable diffusion model
        label_dict: true label dict describes label
        diffusion_steps:
        optimize_iterations: the iteration for optimize embed
        label_cls: from 0 to label_cls, find all labels.

    Returns: new dict of word embeddings. which is used for later steps

    """
    embedding_dict = dict()
    for label in range(label_cls):
        description_word = label_dict[label]
        token = model.tokenizer(description_word)
        embedding = model.text_encoder(token)
        for ind in range(optimize_iterations):
            pass

        embedding_dict[label] = embedding


    return embedding_dict
    pass