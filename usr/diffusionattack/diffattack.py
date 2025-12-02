import torch

from diffusers import StableDiffusionPipeline, DDIMScheduler
from mmengine.config import Config, ConfigDict

from utils import aggregate_attention

# TODO fine here
def preprocess(image, res=512):
    # image = image.resize((res, res), resample=Image.LANCZOS)
    # image = np.array(image).astype(np.float32) / 255.0
    # image = image[None].transpose(0, 3, 1, 2)
    # image = torch.from_numpy(image)[:, :3, :, :].cuda()
    # return 2.0 * image - 1.0
    print(f"preprocess in diffattack.py have not finish!")
    pass

@torch.no_grad()
def ddim_reverse_sample(image, prompt, model, num_inference_steps: int = 20, guidance_scale: float = 2.5,
                        res=256):
    """
            ==========================================
            ============ DDIM Inversion ==============
            ==========================================
    """
    batch_size = 1

    max_length = 77
    uncond_input = model.tokenizer(
        [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    )
    uncond_embeddings = model.text_encoder(uncond_input.input_ids.to(model.device))[0]

    text_input = model.tokenizer(
        prompt[0],
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]

    context = [uncond_embeddings, text_embeddings]
    context = torch.cat(context)

    model.scheduler.set_timesteps(num_inference_steps)

    latents = encoder(image, model, res=res)
    timesteps = model.scheduler.timesteps.flip(0)

    all_latents = [latents]

    #  Not inverse the last step, as the alpha_bar_next will be set to 0 which is not aligned to its real value (~0.003)
    #  and this will lead to a bad result.
    for t in tqdm(timesteps[:-1], desc="DDIM_inverse"):
        latents_input = torch.cat([latents] * 2)
        noise_pred = model.unet(latents_input, t, encoder_hidden_states=context)["sample"]

        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)

        next_timestep = t + model.scheduler.config.num_train_timesteps // model.scheduler.num_inference_steps
        alpha_bar_next = model.scheduler.alphas_cumprod[next_timestep] \
            if next_timestep <= model.scheduler.config.num_train_timesteps else torch.tensor(0.0)

        "leverage reversed_x0"
        reverse_x0 = (1 / torch.sqrt(model.scheduler.alphas_cumprod[t]) * (
                latents - noise_pred * torch.sqrt(1 - model.scheduler.alphas_cumprod[t])))

        latents = reverse_x0 * torch.sqrt(alpha_bar_next) + torch.sqrt(1 - alpha_bar_next) * noise_pred

        all_latents.append(latents)

    #  all_latents[N] -> N: DDIM steps  (X_{T-1} ~ X_0)
    return latents, all_latents

def init_latent(latent, model, height, width, batch_size):
    latents = latent.expand(batch_size, model.unet.in_channels, height // 8, width // 8).to(model.device)
    return latent, latents

def diffusion_step(model, latents, context, t, guidance_scale):
    latents_input = torch.cat([latents] * 2)
    noise_pred = model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
    noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
    latents = model.scheduler.step(noise_pred, t, latents)["prev_sample"]
    return latents

def register_attention_control(model, controller):
    def ca_forward(self, place_in_unet):
        def forward(
                hidden_states: torch.FloatTensor,
                encoder_hidden_states: Optional[torch.FloatTensor] = None,
                attention_mask: Optional[torch.FloatTensor] = None,
                temb: Optional[torch.FloatTensor] = None,
                # scale: float = 1.0,
        ):
            if self.spatial_norm is not None:
                hidden_states = self.spatial_norm(hidden_states, temb)

            batch_size, sequence_length, _ = (
                hidden_states.shape
                if encoder_hidden_states is None
                else encoder_hidden_states.shape
            )

            if attention_mask is not None:
                attention_mask = self.prepare_attention_mask(
                    attention_mask, sequence_length, batch_size
                )
                # scaled_dot_product_attention expects attention_mask shape to be
                # (batch, heads, source_length, target_length)
                attention_mask = attention_mask.view(
                    batch_size, self.heads, -1, attention_mask.shape[-1]
                )  # type: ignore

            if self.group_norm is not None:
                hidden_states = self.group_norm(
                    hidden_states.transpose(1, 2)
                ).transpose(1, 2)

            query = self.to_q(hidden_states)

            is_cross = encoder_hidden_states is not None
            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            elif self.norm_cross:
                encoder_hidden_states = self.norm_encoder_hidden_states(
                    encoder_hidden_states
                )
            key = self.to_k(encoder_hidden_states)
            value = self.to_v(encoder_hidden_states)

            def reshape_heads_to_batch_dim(tensor):
                batch_size, seq_len, dim = tensor.shape
                head_size = self.heads
                tensor = tensor.reshape(
                    batch_size, seq_len, head_size, dim // head_size
                )
                tensor = tensor.permute(0, 2, 1, 3).reshape(
                    batch_size * head_size, seq_len, dim // head_size
                )
                return tensor

            query = reshape_heads_to_batch_dim(query)
            key = reshape_heads_to_batch_dim(key)
            value = reshape_heads_to_batch_dim(value)

            sim = torch.einsum("b i d, b j d -> b i j", query, key) * self.scale
            attn = sim.softmax(dim=-1)
            attn = controller(attn, is_cross, place_in_unet)
            out = torch.einsum("b i j, b j d -> b i d", attn, value)

            def reshape_batch_dim_to_heads(tensor):
                batch_size, seq_len, dim = tensor.shape
                head_size = self.heads
                tensor = tensor.reshape(
                    batch_size // head_size, head_size, seq_len, dim
                )
                tensor = tensor.permute(0, 2, 1, 3).reshape(
                    batch_size // head_size, seq_len, dim * head_size
                )
                return tensor

            out = reshape_batch_dim_to_heads(out)
            out = self.to_out[0](out)
            out = self.to_out[1](out)

            out = out / self.rescale_output_factor
            return out

        return forward

class DiffAttack:
    def __init__(self,
                 cfg: ConfigDict,
                 pretrained_path="Manojb/stable-diffusion-2-base"):
        self.ldm_stable = StableDiffusionPipeline.from_pretrained(pretrained_path).to("cuda:0")
        self.ldm_stable.vae.requires_grad_(False)
        self.ldm_stable.text_encoder.requires_grad_(False)
        self.ldm_stable.unet.requires_grad_(False)

        self.ldm_stable.scheduler = DDIMScheduler.from_config(ldm_stable.scheduler.config)
        self.label_dict = cfg.label_dict

    def diffattack(self,
                   model: StableDiffusionPipeline,
                   controller,
                   num_inference_steps: int = 20,
                   guidance_scale: float = 2.5,
                   image=None,  # image tensor, 0~1, RGB
                   gt_label: torch.Tensor=None,
                   resolution=256,
                   start_step=15,
                   iterations=30,
                   topN=1
                   ):
        """

        Args:
            model: self.ldm_model
            controller:
            num_inference_steps:
            guidance_scale:
            image:
            gt_label:
            resolution:
            start_step:
            iterations:
            topN: maybe no use, just inherit from author

        Returns:

        """
        # here we just cosider the diffuison model
        # classifier just not at here.
        # here are some component.requires_grad_(False), move to init
        height, width = resolution
        assert (image.shape[-1] == image.shape[-2] and
                image.shape[-1] == resolution), \
            "invalid input images or batch!"

        #TODO prompt here
        cond_label_prompt = self._get_label_from_gt(gt_label)
        uncond_prompt = ''
        # ["label description", '']
        prompt = [cond_label_prompt, uncond_prompt]

        true_label_token = model.tokenizer.encode(cond_label_prompt)
        uncond_label_token = model.tokenizer.encode(uncond_prompt)
        print(f"encode token: true{true_label_token}, uncond{uncond_label_token}")

        # list of latent space variables
        latent, inversion_latents = ddim_reverse_sample(image, prompt, model,
                                                        num_inference_steps, guidance_scale, resolution)
        inversion_latents = inversion_latents[::-1]

        init_prompt = [prompt[0]]
        batch_size = len(init_prompt)
        latent = inversion_latents[start_step - 1]

        """
         ===========Init undond embeddings=============
        """
        max_length = 77
        uncond_input = model.tokenizer(
            [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )

        uncond_embeddings = model.text_encoder(uncond_input.input_ids.to(model.device))[0]
        uncond_embeddings.requires_grad_(True)

        text_input = model.tokenizer(
            init_prompt,
            padding="max_length",
            max_length=model.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]

        all_uncond_embeddings = []
        # height // 8, width // 8
        latent, latents = init_latent(latent, model, height, width, batch_size)

        # TODO check LR here. Original one is 1e-1
        optimizer = torch.optim.AdamW([uncond_embeddings],lr=1e-2)
        loss_func = torch.nn.MSELoss()

        context = torch.cat([uncond_embeddings, text_embeddings])

        for ind, t in enumerate(tqdm(model.scheduler.timesteps[1 + start_step - 1:], desc="Optimize_uncond_embed")):
            for _ in range(10 + 2 * ind):
                out_latents = diffusion_step(model, latents, context, t, guidance_scale)
                optimizer.zero_grad()
                loss = loss_func(out_latents, inversion_latents[start_step - 1 + ind + 1])
                loss.backward()
                optimizer.step()

                context = torch.cat([uncond_embeddings, text_embeddings])
            with torch.no_grad():
                latents = diffusion_step(model, latents, context, t, guidance_scale).detach()
                all_uncond_embeddings.append(uncond_embeddings.detatch().clone())

        """
        ==========Latents Attack==============
        """
        uncond_embeddings.requires_grad_(False)

        register_attention_control(model, controller)

        batch_size = len(prompt)

        text_input = model.tokenizer(
            prompt,
            padding="max_length",
            max_length=model.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
        context = [[torch.cat([all_uncond_emb[i]] * batch_size), text_embeddings] for i in range(len(all_uncond_emb))]
        context = [torch.cat(i) for i in context]

        ########## optim latent
        original_latent = latent.clone()
        latent.require_grad_(True)

        optimizer = torch.optim.AdamW([latent], lr=1e-2)
        crossentropy_loss = torch.nn.CrossEntropyLoss()
        init_image = preprocess(image)\

        #TODO Pseudo mask ? REFER TO APPENDIX D

        pbar = tqdm(range(iterations), desc="Iterations")

        for _, _ in enumerate(pbar):
            controller.loss = 0

            #  The DDIM should begin from 1, as the inversion cannot access X_T but only X_{T-1}
        controller.reset()
        latents = torch.cat([original_latent, latent])

        for ind, t in enumerate(model.scheduler.timesteps[1 + start_step - 1:]):
            latent = diffusion_step(model, latents, context[ind], t, guidance_scale)

        before_attention_map = aggregate_attention(prompt, controller, args.res // 32, ("up", "down"), True, 0, is_cpu=False)
        after_attention_map = aggregate_attention(prompt, controller, args.res // 32, ("up", "down"), True, 1, is_cpu=False)

        before_true_label_attention_map = before_attention_map[:, :, 1:len(true_label_token) -1]
        after_true_label_attention_map = after_attention_map[:, :, 1:len(true_label_token) - 1]

        # TODO INIT mask ?

        ## end of todo
        init_out_image = model.vae.decode(1 / 0.18215 * latents)['sample'][1:] * init_mask + (1 - init_mask) * init_image

        # “Deceive” Strong Diffusion Model. Details please refer to Section 3.3
        variance_cross_attn_loss = after_true_label_attention_map.var() * args.cross_attn_loss_weight

        # Preserve Content Structure. Details please refer to Section 3.4
        self_attn_loss = controller.loss * args.self_attn_loss_weight

        loss = self_attn_loss + attack_loss + variance_cross_attn_loss

        if verbose:
            pbar.set_postfix_str(
                f"attack_loss: {attack_loss.item():.5f} "
                f"variance_cross_attn_loss: {variance_cross_attn_loss.item():.5f} "
                f"self_attn_loss: {self_attn_loss.item():.5f} "
                f"loss: {loss.item():.5f}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    def _get_label_from_gt(gt):
        """
        gt: input gt map of one image
        Returns:a string describing the image

        """
        top1, top2 = _get_top2_labels(gt)
        if top2:
            return self.label_dict[top1] + "and" + self.label_dict[top2]
        return self.label_dict[top1]





def _get_top2_labels(gt_map:torch.Tensor,
                     ignore_label=255,
                     thres=0.5):
    """
    Args:
        gt_map: gt_map of current image, which should be valid
        through all pixels.
        NO IGNORE LABEL! FOR VALID DIFFUSING
        ignore_label: you shouldn't use this
        thres: threshold for return 1 or 2 label

    Returns:the label of top1 or 2 type.
    usage : label_dict[top1] + "and" label_dict[top2]

    """
    labels = gt_map.view(-1)
    unique_labels, counts = lables.unique(return_counts=True)
    if len(unique_labels) > 1:
        top2 = torch.topk(counts, k=2).indices
        ind1 = unique_labels[top2[0]]
        ind2 = unique_labels[top2[1]]
        if counts[ind2] > counts[ind1] * thres:
            return ind1, ind2
        else:
            return ind1
    else:
        # this part should put outside.
        # if unique_labels[0] == ignore_label:
        #
        #     return None, None
        # else:
        return unique_labels[0]