import torch

# TODO fine here
def preprocess(image, res=256):
    """
    this is a checker of image type and shape
    Args:
        image: [B, C, res, res]
        res: maybe 256
    """
    # image = image.resize((res, res), resample=Image.LANCZOS)
    # image = np.array(image).astype(np.float32) / 255.0
    # image = image[None].transpose(0, 3, 1, 2)
    # image = torch.from_numpy(image)[:, :3, :, :].cuda()
    # return 2.0 * image - 1.0
    if ((image.shape[-1] == image.shape[-2]) and image.shape[-1] == res):
        raise TypeError(f"expected image shape is same as {res}, but get {image.shape}")
    if image.shape[2] != 3:
        raise TypeError("wrong image shape after preprocess in main function!")

def encoder(image, model, res=256):
    """
    Args:
        image: [B,C,H,W], H=W=res
        model: model
        res: res
    Returns: latent after vae encoder.
    """
    generator = torch.Generator().manual_seed(8888) # cpu random seed
    image = preprocess(image, res)
    gpu_generator = torch.Generator(device=image.device) # gpu random seed
    gpu_generator.manual_seed(generator.initial_seed())
    return 0.18215 * model.vae.encode(image).latent_dist.sample(generator=gpu_generator)


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