import torch

from diffusers import StableDiffusionPipeline, DDIMScheduler
from mmengine.config import Config, ConfigDict

from diffusion_utils import aggregate_attention, preprocess, encoder
from monkeypatch import register_attention_control, reset_attention_control

from ddim_reverse import ddim_reverse_get_transition_steps, ddim_reverse_to_attack_start_latent

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
        latent, inversion_latents = ddim_reverse_get_transition_steps(image, prompt, model,
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

        reset_attention_control(model)

    def _batch_optimize_uncond_embed(self,
                                     model,
                                     num_inference_steps=100,
                                     image:torch.Tensor=None,
                                     resolution=256,
                                     transition_steps=5):

        return best_uncond_embeddings
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