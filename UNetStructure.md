# Experiment: UNet structure
Date: 2025.12.16
Status:  ~~Done~~ / Suspended / Failed

>[!info] Scope & Limitation
>This report focuses on structural and tensor-level observation of the SD UNet.
>Does NOT attempt to assign semantic meaning to internal features or skip connectons
---
## Goal
Figure out the IO tensor shape of each attention layer. 
Count the attention layers.

## Setup
- Lib: diffusers 0.30.0, python 3.8
- Model: Stable-Diffusion Pipeline 
- Dataset: Just one image(RGB, 128x128)
- Parameters: resolution = 128, batch_size = 2 (Classifier Free Guidance)
- Baseline: [DiffAttack](https://github.com/WindVChen/DiffAttack)

## Procedure
1. Add break point in UNet class `BasicTransformerBlock`, and `Attention` module in diffusers source code.
2. Debug mode, find invokes.
3. Break Point: unet_2d_condition: preprocess: self.conv_in, down_blocks
---
## Observation
### IO in UNet Stages
- Image to Latent (VAE encoder): down sample 8 times.
- Zero: Input sample : \[2, 4, 16, 16]
- `UNet2DConditionModel`:    
  1. Input: sample \[2, 4, 16, 16], cross attention emb\[2, 77, 1024]
  2. Preprocess: `sample = self.conv_in(sample)` -> \[2, 320, 16, 16]

  3. Down sample for 4 times.
     1. input (sample): \[2, 320, 16, 16] & \[2, 77, 1024] -> \[2, 320, 8, 8]
     2. \[2, 320, 8, 8] -> \[2, 640, 4, 4]
     3. \[2, 640, 4, 4] -> \[2, 1280, 2, 2]
     4. \[2, 1280, 2, 2] -> \[2, 1280, 2, 2]
> 12 Attention operations in total.(4, 4, 4, 0)
  
  4. Mid-Block
     \[2, 1280, 2, 2] -> \[2, 1280, 2, 2]  
     this stage only have one block.
> 2 Attention operations in total.
  
  5. Up Sample
     1. (no attention) \[2, 1280, 2, 2] -> \[2, 1280, 4, 4]
     2. \[2, 1280, 4, 4] -> \[2, 1280, 8, 8]
     3. \[2, 1280, 8, 8] -> \[2, 640, 16, 16]
     4. \[2, 640, 16, 16] -> \[2, 320, 16, 16]
> 18 Attention operations in total (0, 6, 6, 6)     

  6. Conv Out
     Norm -> `self.conv_out` 
> this is `Conv2d(320, 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))`
     
  Then you get a noise prediction.
### Skip Connection Tracking
#### Channel Tracking

- Fixed layer channel in UNet blocks:
```
down: 320 -> 640 -> 1280 -> 1280
mid : 1280
up  : 1280 -> 1280 -> 640 -> 320
```

- Skip connection in UNet structure.
	- 1. ResNet residual connection
	$$x⟶F(x)+x$$
	- 2.UNet skip connection
	$$y = Conv([x_{down}, x_{up}])$$
	skip different depth layer. Must project to target channel by convolution.

| Generate Stage | Initialize       | Down Block 1          | Down Block 1          | Down Block 1   | Down Block 2          | Down Block 2          | Down Block 2    | Down Block 3          | Down Block 3           | Down Block 3    | Down Block 4    | Down Block 4    |
| -------------- | ---------------- | --------------------- | --------------------- | -------------- | --------------------- | --------------------- | --------------- | --------------------- | ---------------------- | --------------- | --------------- | --------------- |
| Generate Step  | Input            | ResNet->Transformer 1 | ResNet->Transformer 2 | Down sample    | ResNet->Transformer 1 | ResNet->Transformer 2 | Down sample     | ResNet->Transformer 1 | ResNet->Transformer 2  | Down sample     | ResNet 1        | ResNet 2        |
| Shape          | (2, 320, 16, 16) | (2, 320, 16, 16)      | (2, 320, 16, 16)      | (2, 320, 8, 8) | (2, 640, 8, 8)        | (2, 640, 8, 8)        | (2, 640, 4, 4)  | (2, 1280, 4, 4)       | (2, 1280, 4, 4)        | (2, 1280, 2, 2) | (2, 1280, 2, 2) | (2, 1280, 2, 2) |
| Usage Stage    | Up Block 4       | Up Block 4            | Up Block 4            | Up Block 3     | Up Block 3            | Up Block 3            | Up Block 2      | Up Block 2            | Up Block 2             | Up Block 1      | Up Block 1      | Up Block 1      |
| Usage Step     | Res->Attn 3      | Res->Attn 2           | Res->Attn 1           | Res->Attn 3    | Res->Attn 2           | Res->Attn 1           | Res->Attn 3     | Res->Attn 2           | after up , Res->Attn 1 | ResNet 3        | ResNet 2        | ResNet 1        |
| Input Latent   | (2, 320, 16, 16) | (2, 320, 16, 16)      | (2, 640, 16, 16)      | (2, 640, 8, 8) | (2, 640, 8, 8)        | (2, 1280, 8, 8)       | (2, 1280, 4, 4) | (2, 1280, 4, 4)       | (2, 1280, 4, 4)        | (2, 1280, 2, 2) | (2, 1280, 2, 2) | (2, 1280, 2, 2) |
| prj Chann      | 320              | 320                   | 320                   | 640            | 640                   | 640                   | 1280            | 1280                  | 1280                   | 1280            | 1280            | 1280            |

---
**These skip connections are generated as below:**
```python
# class: CrossAttnDownBlock2D: Please refer to appendix to read structure
for i, (resnet, attn) in enumerate(blocks):
	hidden_states = resnet(hidden_states, temb) # hidden_state is the input tensor
	hiddenstates = attn(hidden_states, encoder_hidden_states)[0]
	output_staes = output_states + (hidden_states,) # here is the connection
	
if self.downsamplers is not None:
	hidden_states = downsampler(hidden_states)
	output_states = output_states + (hidden_states,)
```

So we can see clearly that the skip connections come from every ResNet+Attn feature extraction and downsampling.

The skip connections are used as below:
```python
for resnet, attn in zip(self.resnets, self.attentions):
	hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
	hidden_states =  resnet(hidden_states, temb)
	hidden_states = attn(hidden_states, encoder_hidden_states)

if upsampler is not None:
	# up sample and back to concatenate.
```

---
>[!example] Channel and Resolution Changes
> - When down stage, the resolution is divided by 2 by down sample module in the tail of first 3 down blocks. And the channel is doubled by `ResNet` at the begining of the down block after down sample.
> <br>
> - When up stage, the resolution is doubled by up sample module in the tail of first 3 up blocks. While the channel is reduced by **skip connection ResNet**

> [!question] Why we use Skip Connections?
> - Spatial Alignment
> - Reserve the DOF discarded in down sampling
> - stable gradient path.

### **Attention Map Tracking**

**Only 3 Resolution: 320, 640, 1280**
**Only related to Channel**
- Input (`Transformer2DModel`) eg: (2, 320, 16, 16), (2, 77, 1024)
	$$x \in \mathbb{R}^{B \times C \times H \times W} $$
- Preprocess (`Transformer2DModel`) -> (2, 256, 320)
	$$x \to x' \in \mathbb{R}^{B \times(HW) \times C} $$
- Norm and other configs (`BasicTransformerBlock`)
	- attn1 in: (2, 256, 320), None -> (2, 256, 320)
		if only cross attn, this block is a cross attn(UNet here is a self attn).
	```	
	hidden_states = attn_output + hidden_states
	```
	- attn2 in: (2, 256, 320), (2, 77, 1024) -> (2, 256, 320)
		this is cross attention for sure.

- Pure Attention (`Attention.forward`)
	$$Q = x' W_Q, \ \ \ \  K =x'W_K, \ \ \ \ V = x'W_V $$
	then multi head: 
	$$\begin{aligned}
	 Q & \in \mathbb{R}^{(B\times head)\times (HW)\times C} \\
	 K & \in \mathbb{R}^{(B\times head)\times (token\ length)\times C} \\
	 V & \in \mathbb{R}^{(B\times head)\times (token\ length)\times C} \end{aligned}
	$$
	then
	$$\begin{aligned}
	map = \frac{QK^T}{\sqrt{d}} \in \mathbb{R}^{(B \times head)\times (HW) \times token\ length}
	\end{aligned}$$
	$$
	Attention = softmax(map)V \in \mathbb{R}^{(H \times head) \times(HW) \times C}
	$$

- Finally after each attention:
	`hidden_states = hidden_states + attention`
	**So the Attention block is absolutely a feature extractor, not some converter like convolution. This mechanism boost some features while diminish some.**

So in this example, if head is 10, we have Q(20, 256, 32), K(20, 256, 32) and V(20, 256, 32) in self attention, map is (20, 256, 256)

If cross attention: K(20, 77, 32) V(20, 77, 32), map(20, 256, 77).

>[!abstract] Map Usage
>We have self attention map: (head\*batch, res\* res, res\*res), and cross attention map: (head\*batch, res\*res, token). When we use them, we need permute the map and get true meaning. 
> - Cross attention map: we should use \[1:len(token)-1] map and calc variance.
> - Self attention map: we need calc average, and rearrange them into a picture. 

---
## Sum
1. 32 attention invoking
2. 1280, 640, 320 channels
3. **The channel and resolution aren't matched. They are independent.**

## Issue
- I cannot understand the meaning of UNet skip connection at all! Fully cannot. 

## Next Step
- write Attention Control Module and UNet Hijack Module.
- Find out UNet skip mechanism later
- Take a research on ResNet connection later

## Appendix

### 1. UNet2DConditionModel.down_blocks
---
#### (0): CrossAttnDownBlock2D
```
(0): CrossAttnDownBlock2D(
  (attentions): ModuleList(
    (0): Transformer2DModel(
      (norm): GroupNorm(32, 320, eps=1e-06, affine=True)
      (proj_in): Linear(in_features=320, out_features=320, bias=True)
      (transformer_blocks): ModuleList(
        (0): BasicTransformerBlock(
          (norm1): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
          (attn1): Attention(
            (to_q): Linear(in_features=320, out_features=320, bias=False)
            (to_k): Linear(in_features=320, out_features=320, bias=False)
            (to_v): Linear(in_features=320, out_features=320, bias=False)
            (to_out): ModuleList(
              (0): Linear(in_features=320, out_features=320, bias=True)
              (1): Dropout(p=0.0, inplace=False)
            )
          )
          (norm2): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
          (attn2): Attention(
            (to_q): Linear(in_features=320, out_features=320, bias=False)
            (to_k): Linear(in_features=1024, out_features=320, bias=False)
            (to_v): Linear(in_features=1024, out_features=320, bias=False)
            (to_out): ModuleList(
              (0): Linear(in_features=320, out_features=320, bias=True)
              (1): Dropout(p=0.0, inplace=False)
            )
          )
          (norm3): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
          (ff): FeedForward(
            (net): ModuleList(
              (0): GEGLU(
                (proj): Linear(in_features=320, out_features=2560, bias=True)
              )
              (1): Dropout(p=0.0, inplace=False)
              (2): Linear(in_features=1280, out_features=320, bias=True)
            )
          )
        )
      )
      (proj_out): Linear(in_features=320, out_features=320, bias=True)
    )
    
    
    (1): Transformer2DModel(
      (norm): GroupNorm(32, 320, eps=1e-06, affine=True)
      (proj_in): Linear(in_features=320, out_features=320, bias=True)
      (transformer_blocks): ModuleList(
        (0): BasicTransformerBlock(
          (norm1): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
          (attn1): Attention(
            (to_q): Linear(in_features=320, out_features=320, bias=False)
            (to_k): Linear(in_features=320, out_features=320, bias=False)
            (to_v): Linear(in_features=320, out_features=320, bias=False)
            (to_out): ModuleList(
              (0): Linear(in_features=320, out_features=320, bias=True)
              (1): Dropout(p=0.0, inplace=False)
            )
          )
          (norm2): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
          (attn2): Attention(
            (to_q): Linear(in_features=320, out_features=320, bias=False)
            (to_k): Linear(in_features=1024, out_features=320, bias=False)
            (to_v): Linear(in_features=1024, out_features=320, bias=False)
            (to_out): ModuleList(
              (0): Linear(in_features=320, out_features=320, bias=True)
              (1): Dropout(p=0.0, inplace=False)
            )
          )
          (norm3): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
          (ff): FeedForward(
            (net): ModuleList(
              (0): GEGLU(
                (proj): Linear(in_features=320, out_features=2560, bias=True)
              )
              (1): Dropout(p=0.0, inplace=False)
              (2): Linear(in_features=1280, out_features=320, bias=True)
            )
          )
        )
      )
      (proj_out): Linear(in_features=320, out_features=320, bias=True)
    )
    
  )
  (resnets): ModuleList(
    (0): ResnetBlock2D(
      (norm1): GroupNorm(32, 320, eps=1e-05, affine=True)
      (conv1): Conv2d(320, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (time_emb_proj): Linear(in_features=1280, out_features=320, bias=True)
      (norm2): GroupNorm(32, 320, eps=1e-05, affine=True)
      (dropout): Dropout(p=0.0, inplace=False)
      (conv2): Conv2d(320, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (nonlinearity): SiLU()
    )
    (1): ResnetBlock2D(
      (norm1): GroupNorm(32, 320, eps=1e-05, affine=True)
      (conv1): Conv2d(320, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (time_emb_proj): Linear(in_features=1280, out_features=320, bias=True)
      (norm2): GroupNorm(32, 320, eps=1e-05, affine=True)
      (dropout): Dropout(p=0.0, inplace=False)
      (conv2): Conv2d(320, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (nonlinearity): SiLU()
    )
  )
  (downsamplers): ModuleList(
    (0): Downsample2D(
      (conv): Conv2d(320, 320, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    )
  )
)
```
---
#### (1): CrossAttnDownBlock2D
``` 
(1): CrossAttnDownBlock2D(
    (attentions): ModuleList(
      (0): Transformer2DModel(
        (norm): GroupNorm(32, 640, eps=1e-06, affine=True)
        (proj_in): Linear(in_features=640, out_features=640, bias=True)
        (transformer_blocks): ModuleList(
          (0): BasicTransformerBlock(
            (norm1): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
            (attn1): Attention(
              (to_q): Linear(in_features=640, out_features=640, bias=False)
              (to_k): Linear(in_features=640, out_features=640, bias=False)
              (to_v): Linear(in_features=640, out_features=640, bias=False)
              (to_out): ModuleList(
                (0): Linear(in_features=640, out_features=640, bias=True)
                (1): Dropout(p=0.0, inplace=False)
              )
            )
            (norm2): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
            (attn2): Attention(
              (to_q): Linear(in_features=640, out_features=640, bias=False)
              (to_k): Linear(in_features=1024, out_features=640, bias=False)
              (to_v): Linear(in_features=1024, out_features=640, bias=False)
              (to_out): ModuleList(
                (0): Linear(in_features=640, out_features=640, bias=True)
                (1): Dropout(p=0.0, inplace=False)
              )
            )
            (norm3): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
            (ff): FeedForward(
              (net): ModuleList(
                (0): GEGLU(
                  (proj): Linear(in_features=640, out_features=5120, bias=True)
                )
                (1): Dropout(p=0.0, inplace=False)
                (2): Linear(in_features=2560, out_features=640, bias=True)
              )
            )
          )
        )
        (proj_out): Linear(in_features=640, out_features=640, bias=True)
      )
      (1): Transformer2DModel(
        (norm): GroupNorm(32, 640, eps=1e-06, affine=True)
        (proj_in): Linear(in_features=640, out_features=640, bias=True)
        (transformer_blocks): ModuleList(
          (0): BasicTransformerBlock(
            (norm1): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
            (attn1): Attention(
              (to_q): Linear(in_features=640, out_features=640, bias=False)
              (to_k): Linear(in_features=640, out_features=640, bias=False)
              (to_v): Linear(in_features=640, out_features=640, bias=False)
              (to_out): ModuleList(
                (0): Linear(in_features=640, out_features=640, bias=True)
                (1): Dropout(p=0.0, inplace=False)
              )
            )
            (norm2): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
            (attn2): Attention(
              (to_q): Linear(in_features=640, out_features=640, bias=False)
              (to_k): Linear(in_features=1024, out_features=640, bias=False)
              (to_v): Linear(in_features=1024, out_features=640, bias=False)
              (to_out): ModuleList(
                (0): Linear(in_features=640, out_features=640, bias=True)
                (1): Dropout(p=0.0, inplace=False)
              )
            )
            (norm3): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
            (ff): FeedForward(
              (net): ModuleList(
                (0): GEGLU(
                  (proj): Linear(in_features=640, out_features=5120, bias=True)
                )
                (1): Dropout(p=0.0, inplace=False)
                (2): Linear(in_features=2560, out_features=640, bias=True)
              )
            )
          )
        )
        (proj_out): Linear(in_features=640, out_features=640, bias=True)
      )
    )
    (resnets): ModuleList(
      (0): ResnetBlock2D(
        (norm1): GroupNorm(32, 320, eps=1e-05, affine=True)
        (conv1): Conv2d(320, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (time_emb_proj): Linear(in_features=1280, out_features=640, bias=True)
        (norm2): GroupNorm(32, 640, eps=1e-05, affine=True)
        (dropout): Dropout(p=0.0, inplace=False)
        (conv2): Conv2d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (nonlinearity): SiLU()
        (conv_shortcut): Conv2d(320, 640, kernel_size=(1, 1), stride=(1, 1))
      )
      (1): ResnetBlock2D(
        (norm1): GroupNorm(32, 640, eps=1e-05, affine=True)
        (conv1): Conv2d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (time_emb_proj): Linear(in_features=1280, out_features=640, bias=True)
        (norm2): GroupNorm(32, 640, eps=1e-05, affine=True)
        (dropout): Dropout(p=0.0, inplace=False)
        (conv2): Conv2d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (nonlinearity): SiLU()
      )
    )
    (downsamplers): ModuleList(
      (0): Downsample2D(
        (conv): Conv2d(640, 640, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
      )
    )
  )
```
---
#### (2): CrossAttnDownBlock2D
```
(2): CrossAttnDownBlock2D(
    (attentions): ModuleList(
      (0): Transformer2DModel(
        (norm): GroupNorm(32, 1280, eps=1e-06, affine=True)
        (proj_in): Linear(in_features=1280, out_features=1280, bias=True)
        (transformer_blocks): ModuleList(
          (0): BasicTransformerBlock(
            (norm1): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
            (attn1): Attention(
              (to_q): Linear(in_features=1280, out_features=1280, bias=False)
              (to_k): Linear(in_features=1280, out_features=1280, bias=False)
              (to_v): Linear(in_features=1280, out_features=1280, bias=False)
              (to_out): ModuleList(
                (0): Linear(in_features=1280, out_features=1280, bias=True)
                (1): Dropout(p=0.0, inplace=False)
              )
            )
            (norm2): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
            (attn2): Attention(
              (to_q): Linear(in_features=1280, out_features=1280, bias=False)
              (to_k): Linear(in_features=1024, out_features=1280, bias=False)
              (to_v): Linear(in_features=1024, out_features=1280, bias=False)
              (to_out): ModuleList(
                (0): Linear(in_features=1280, out_features=1280, bias=True)
                (1): Dropout(p=0.0, inplace=False)
              )
            )
            (norm3): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
            (ff): FeedForward(
              (net): ModuleList(
                (0): GEGLU(
                  (proj): Linear(in_features=1280, out_features=10240, bias=True)
                )
                (1): Dropout(p=0.0, inplace=False)
                (2): Linear(in_features=5120, out_features=1280, bias=True)
              )
            )
          )
        )
        (proj_out): Linear(in_features=1280, out_features=1280, bias=True)
      )
      (1): Transformer2DModel(
        (norm): GroupNorm(32, 1280, eps=1e-06, affine=True)
        (proj_in): Linear(in_features=1280, out_features=1280, bias=True)
        (transformer_blocks): ModuleList(
          (0): BasicTransformerBlock(
            (norm1): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
            (attn1): Attention(
              (to_q): Linear(in_features=1280, out_features=1280, bias=False)
              (to_k): Linear(in_features=1280, out_features=1280, bias=False)
              (to_v): Linear(in_features=1280, out_features=1280, bias=False)
              (to_out): ModuleList(
                (0): Linear(in_features=1280, out_features=1280, bias=True)
                (1): Dropout(p=0.0, inplace=False)
              )
            )
            (norm2): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
            (attn2): Attention(
              (to_q): Linear(in_features=1280, out_features=1280, bias=False)
              (to_k): Linear(in_features=1024, out_features=1280, bias=False)
              (to_v): Linear(in_features=1024, out_features=1280, bias=False)
              (to_out): ModuleList(
                (0): Linear(in_features=1280, out_features=1280, bias=True)
                (1): Dropout(p=0.0, inplace=False)
              )
            )
            (norm3): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
            (ff): FeedForward(
              (net): ModuleList(
                (0): GEGLU(
                  (proj): Linear(in_features=1280, out_features=10240, bias=True)
                )
                (1): Dropout(p=0.0, inplace=False)
                (2): Linear(in_features=5120, out_features=1280, bias=True)
              )
            )
          )
        )
        (proj_out): Linear(in_features=1280, out_features=1280, bias=True)
      )
    )
    (resnets): ModuleList(
      (0): ResnetBlock2D(
        (norm1): GroupNorm(32, 640, eps=1e-05, affine=True)
        (conv1): Conv2d(640, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (time_emb_proj): Linear(in_features=1280, out_features=1280, bias=True)
        (norm2): GroupNorm(32, 1280, eps=1e-05, affine=True)
        (dropout): Dropout(p=0.0, inplace=False)
        (conv2): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (nonlinearity): SiLU()
        (conv_shortcut): Conv2d(640, 1280, kernel_size=(1, 1), stride=(1, 1))
      )
      (1): ResnetBlock2D(
        (norm1): GroupNorm(32, 1280, eps=1e-05, affine=True)
        (conv1): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (time_emb_proj): Linear(in_features=1280, out_features=1280, bias=True)
        (norm2): GroupNorm(32, 1280, eps=1e-05, affine=True)
        (dropout): Dropout(p=0.0, inplace=False)
        (conv2): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (nonlinearity): SiLU()
      )
    )
    (downsamplers): ModuleList(
      (0): Downsample2D(
        (conv): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
      )
    )
  )
```
---
#### (3): DownBlock2D
``` 
  (3): DownBlock2D(
    (resnets): ModuleList(
      (0): ResnetBlock2D(
        (norm1): GroupNorm(32, 1280, eps=1e-05, affine=True)
        (conv1): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (time_emb_proj): Linear(in_features=1280, out_features=1280, bias=True)
        (norm2): GroupNorm(32, 1280, eps=1e-05, affine=True)
        (dropout): Dropout(p=0.0, inplace=False)
        (conv2): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (nonlinearity): SiLU()
      )
      (1): ResnetBlock2D(
        (norm1): GroupNorm(32, 1280, eps=1e-05, affine=True)
        (conv1): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (time_emb_proj): Linear(in_features=1280, out_features=1280, bias=True)
        (norm2): GroupNorm(32, 1280, eps=1e-05, affine=True)
        (dropout): Dropout(p=0.0, inplace=False)
        (conv2): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (nonlinearity): SiLU()
      )
    )
  )
)
```
---
We can find that only first three layers contains down sample (div 2), last layer don't have.  

And all are `TransformerBlock` at bottom.

---
### Mid Block
---
#### UNetMidBlock2DCrossAttn
```
UNetMidBlock2DCrossAttn(
  (attentions): ModuleList(
    (0): Transformer2DModel(
      (norm): GroupNorm(32, 1280, eps=1e-06, affine=True)
      (proj_in): Linear(in_features=1280, out_features=1280, bias=True)
      (transformer_blocks): ModuleList(
        (0): BasicTransformerBlock(
          (norm1): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
          (attn1): Attention(
            (to_q): Linear(in_features=1280, out_features=1280, bias=False)
            (to_k): Linear(in_features=1280, out_features=1280, bias=False)
            (to_v): Linear(in_features=1280, out_features=1280, bias=False)
            (to_out): ModuleList(
              (0): Linear(in_features=1280, out_features=1280, bias=True)
              (1): Dropout(p=0.0, inplace=False)
            )
          )
          (norm2): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
          (attn2): Attention(
            (to_q): Linear(in_features=1280, out_features=1280, bias=False)
            (to_k): Linear(in_features=1024, out_features=1280, bias=False)
            (to_v): Linear(in_features=1024, out_features=1280, bias=False)
            (to_out): ModuleList(
              (0): Linear(in_features=1280, out_features=1280, bias=True)
              (1): Dropout(p=0.0, inplace=False)
            )
          )
          (norm3): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
          (ff): FeedForward(
            (net): ModuleList(
              (0): GEGLU(
                (proj): Linear(in_features=1280, out_features=10240, bias=True)
              )
              (1): Dropout(p=0.0, inplace=False)
              (2): Linear(in_features=5120, out_features=1280, bias=True)
            )
          )
        )
      )
      (proj_out): Linear(in_features=1280, out_features=1280, bias=True)
    )
  )
  (resnets): ModuleList(
    (0): ResnetBlock2D(
      (norm1): GroupNorm(32, 1280, eps=1e-05, affine=True)
      (conv1): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (time_emb_proj): Linear(in_features=1280, out_features=1280, bias=True)
      (norm2): GroupNorm(32, 1280, eps=1e-05, affine=True)
      (dropout): Dropout(p=0.0, inplace=False)
      (conv2): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (nonlinearity): SiLU()
    )
    (1): ResnetBlock2D(
      (norm1): GroupNorm(32, 1280, eps=1e-05, affine=True)
      (conv1): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (time_emb_proj): Linear(in_features=1280, out_features=1280, bias=True)
      (norm2): GroupNorm(32, 1280, eps=1e-05, affine=True)
      (dropout): Dropout(p=0.0, inplace=False)
      (conv2): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (nonlinearity): SiLU()
    )
  )
)

```

---
---

### Up Blocks

#### (0): UpBlock2D
```
(0): UpBlock2D(
    (resnets): ModuleList(
      (0): ResnetBlock2D(
        (norm1): GroupNorm(32, 2560, eps=1e-05, affine=True)
        (conv1): Conv2d(2560, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (time_emb_proj): Linear(in_features=1280, out_features=1280, bias=True)
        (norm2): GroupNorm(32, 1280, eps=1e-05, affine=True)
        (dropout): Dropout(p=0.0, inplace=False)
        (conv2): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (nonlinearity): SiLU()
        (conv_shortcut): Conv2d(2560, 1280, kernel_size=(1, 1), stride=(1, 1))
      )
      (1): ResnetBlock2D(
        (norm1): GroupNorm(32, 2560, eps=1e-05, affine=True)
        (conv1): Conv2d(2560, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (time_emb_proj): Linear(in_features=1280, out_features=1280, bias=True)
        (norm2): GroupNorm(32, 1280, eps=1e-05, affine=True)
        (dropout): Dropout(p=0.0, inplace=False)
        (conv2): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (nonlinearity): SiLU()
        (conv_shortcut): Conv2d(2560, 1280, kernel_size=(1, 1), stride=(1, 1))
      )
      (2): ResnetBlock2D(
        (norm1): GroupNorm(32, 2560, eps=1e-05, affine=True)
        (conv1): Conv2d(2560, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (time_emb_proj): Linear(in_features=1280, out_features=1280, bias=True)
        (norm2): GroupNorm(32, 1280, eps=1e-05, affine=True)
        (dropout): Dropout(p=0.0, inplace=False)
        (conv2): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (nonlinearity): SiLU()
        (conv_shortcut): Conv2d(2560, 1280, kernel_size=(1, 1), stride=(1, 1))
      )
    )
    (upsamplers): ModuleList(
      (0): Upsample2D(
        (conv): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
    )
  )
```

No attention here.

---

#### (1): CrossAttnUpBlock2D
``` 
  (1): CrossAttnUpBlock2D(
    (attentions): ModuleList(
      (0): Transformer2DModel(
        (norm): GroupNorm(32, 1280, eps=1e-06, affine=True)
        (proj_in): Linear(in_features=1280, out_features=1280, bias=True)
        (transformer_blocks): ModuleList(
          (0): BasicTransformerBlock(
            (norm1): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
            (attn1): Attention(
              (to_q): Linear(in_features=1280, out_features=1280, bias=False)
              (to_k): Linear(in_features=1280, out_features=1280, bias=False)
              (to_v): Linear(in_features=1280, out_features=1280, bias=False)
              (to_out): ModuleList(
                (0): Linear(in_features=1280, out_features=1280, bias=True)
                (1): Dropout(p=0.0, inplace=False)
              )
            )
            (norm2): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
            (attn2): Attention(
              (to_q): Linear(in_features=1280, out_features=1280, bias=False)
              (to_k): Linear(in_features=1024, out_features=1280, bias=False)
              (to_v): Linear(in_features=1024, out_features=1280, bias=False)
              (to_out): ModuleList(
                (0): Linear(in_features=1280, out_features=1280, bias=True)
                (1): Dropout(p=0.0, inplace=False)
              )
            )
            (norm3): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
            (ff): FeedForward(
              (net): ModuleList(
                (0): GEGLU(
                  (proj): Linear(in_features=1280, out_features=10240, bias=True)
                )
                (1): Dropout(p=0.0, inplace=False)
                (2): Linear(in_features=5120, out_features=1280, bias=True)
              )
            )
          )
        )
        (proj_out): Linear(in_features=1280, out_features=1280, bias=True)
      )
      (1): Transformer2DModel(
        (norm): GroupNorm(32, 1280, eps=1e-06, affine=True)
        (proj_in): Linear(in_features=1280, out_features=1280, bias=True)
        (transformer_blocks): ModuleList(
          (0): BasicTransformerBlock(
            (norm1): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
            (attn1): Attention(
              (to_q): Linear(in_features=1280, out_features=1280, bias=False)
              (to_k): Linear(in_features=1280, out_features=1280, bias=False)
              (to_v): Linear(in_features=1280, out_features=1280, bias=False)
              (to_out): ModuleList(
                (0): Linear(in_features=1280, out_features=1280, bias=True)
                (1): Dropout(p=0.0, inplace=False)
              )
            )
            (norm2): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
            (attn2): Attention(
              (to_q): Linear(in_features=1280, out_features=1280, bias=False)
              (to_k): Linear(in_features=1024, out_features=1280, bias=False)
              (to_v): Linear(in_features=1024, out_features=1280, bias=False)
              (to_out): ModuleList(
                (0): Linear(in_features=1280, out_features=1280, bias=True)
                (1): Dropout(p=0.0, inplace=False)
              )
            )
            (norm3): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
            (ff): FeedForward(
              (net): ModuleList(
                (0): GEGLU(
                  (proj): Linear(in_features=1280, out_features=10240, bias=True)
                )
                (1): Dropout(p=0.0, inplace=False)
                (2): Linear(in_features=5120, out_features=1280, bias=True)
              )
            )
          )
        )
        (proj_out): Linear(in_features=1280, out_features=1280, bias=True)
      )
      (2): Transformer2DModel(
        (norm): GroupNorm(32, 1280, eps=1e-06, affine=True)
        (proj_in): Linear(in_features=1280, out_features=1280, bias=True)
        (transformer_blocks): ModuleList(
          (0): BasicTransformerBlock(
            (norm1): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
            (attn1): Attention(
              (to_q): Linear(in_features=1280, out_features=1280, bias=False)
              (to_k): Linear(in_features=1280, out_features=1280, bias=False)
              (to_v): Linear(in_features=1280, out_features=1280, bias=False)
              (to_out): ModuleList(
                (0): Linear(in_features=1280, out_features=1280, bias=True)
                (1): Dropout(p=0.0, inplace=False)
              )
            )
            (norm2): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
            (attn2): Attention(
              (to_q): Linear(in_features=1280, out_features=1280, bias=False)
              (to_k): Linear(in_features=1024, out_features=1280, bias=False)
              (to_v): Linear(in_features=1024, out_features=1280, bias=False)
              (to_out): ModuleList(
                (0): Linear(in_features=1280, out_features=1280, bias=True)
                (1): Dropout(p=0.0, inplace=False)
              )
            )
            (norm3): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
            (ff): FeedForward(
              (net): ModuleList(
                (0): GEGLU(
                  (proj): Linear(in_features=1280, out_features=10240, bias=True)
                )
                (1): Dropout(p=0.0, inplace=False)
                (2): Linear(in_features=5120, out_features=1280, bias=True)
              )
            )
          )
        )
        (proj_out): Linear(in_features=1280, out_features=1280, bias=True)
      )
    )
    (resnets): ModuleList(
      (0): ResnetBlock2D(
        (norm1): GroupNorm(32, 2560, eps=1e-05, affine=True)
        (conv1): Conv2d(2560, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (time_emb_proj): Linear(in_features=1280, out_features=1280, bias=True)
        (norm2): GroupNorm(32, 1280, eps=1e-05, affine=True)
        (dropout): Dropout(p=0.0, inplace=False)
        (conv2): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (nonlinearity): SiLU()
        (conv_shortcut): Conv2d(2560, 1280, kernel_size=(1, 1), stride=(1, 1))
      )
      (1): ResnetBlock2D(
        (norm1): GroupNorm(32, 2560, eps=1e-05, affine=True)
        (conv1): Conv2d(2560, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (time_emb_proj): Linear(in_features=1280, out_features=1280, bias=True)
        (norm2): GroupNorm(32, 1280, eps=1e-05, affine=True)
        (dropout): Dropout(p=0.0, inplace=False)
        (conv2): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (nonlinearity): SiLU()
        (conv_shortcut): Conv2d(2560, 1280, kernel_size=(1, 1), stride=(1, 1))
      )
      (2): ResnetBlock2D(
        (norm1): GroupNorm(32, 1920, eps=1e-05, affine=True)
        (conv1): Conv2d(1920, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (time_emb_proj): Linear(in_features=1280, out_features=1280, bias=True)
        (norm2): GroupNorm(32, 1280, eps=1e-05, affine=True)
        (dropout): Dropout(p=0.0, inplace=False)
        (conv2): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (nonlinearity): SiLU()
        (conv_shortcut): Conv2d(1920, 1280, kernel_size=(1, 1), stride=(1, 1))
      )
    )
    (upsamplers): ModuleList(
      (0): Upsample2D(
        (conv): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
    )
  )
```
Usage:
here we see `for resnet, attn in zip(self.resnets, self.attentions):`, we call the `Transformer2DModel` for 3 times in one up sampling step. In each `Transformer2DModel`, we call `Attention` for 2 times.

The `Transformer2DModel` doesn't change the shape of tensor. Just extract features and return one tensor has the same shape as input.

#### (2): CrossAttnUpBlock2D(

```
  (2): CrossAttnUpBlock2D(
    (attentions): ModuleList(
      (0): Transformer2DModel(
        (norm): GroupNorm(32, 640, eps=1e-06, affine=True)
        (proj_in): Linear(in_features=640, out_features=640, bias=True)
        (transformer_blocks): ModuleList(
          (0): BasicTransformerBlock(
            (norm1): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
            (attn1): Attention(
              (to_q): Linear(in_features=640, out_features=640, bias=False)
              (to_k): Linear(in_features=640, out_features=640, bias=False)
              (to_v): Linear(in_features=640, out_features=640, bias=False)
              (to_out): ModuleList(
                (0): Linear(in_features=640, out_features=640, bias=True)
                (1): Dropout(p=0.0, inplace=False)
              )
            )
            (norm2): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
            (attn2): Attention(
              (to_q): Linear(in_features=640, out_features=640, bias=False)
              (to_k): Linear(in_features=1024, out_features=640, bias=False)
              (to_v): Linear(in_features=1024, out_features=640, bias=False)
              (to_out): ModuleList(
                (0): Linear(in_features=640, out_features=640, bias=True)
                (1): Dropout(p=0.0, inplace=False)
              )
            )
            (norm3): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
            (ff): FeedForward(
              (net): ModuleList(
                (0): GEGLU(
                  (proj): Linear(in_features=640, out_features=5120, bias=True)
                )
                (1): Dropout(p=0.0, inplace=False)
                (2): Linear(in_features=2560, out_features=640, bias=True)
              )
            )
          )
        )
        (proj_out): Linear(in_features=640, out_features=640, bias=True)
      )
      (1): Transformer2DModel(
        (norm): GroupNorm(32, 640, eps=1e-06, affine=True)
        (proj_in): Linear(in_features=640, out_features=640, bias=True)
        (transformer_blocks): ModuleList(
          (0): BasicTransformerBlock(
            (norm1): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
            (attn1): Attention(
              (to_q): Linear(in_features=640, out_features=640, bias=False)
              (to_k): Linear(in_features=640, out_features=640, bias=False)
              (to_v): Linear(in_features=640, out_features=640, bias=False)
              (to_out): ModuleList(
                (0): Linear(in_features=640, out_features=640, bias=True)
                (1): Dropout(p=0.0, inplace=False)
              )
            )
            (norm2): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
            (attn2): Attention(
              (to_q): Linear(in_features=640, out_features=640, bias=False)
              (to_k): Linear(in_features=1024, out_features=640, bias=False)
              (to_v): Linear(in_features=1024, out_features=640, bias=False)
              (to_out): ModuleList(
                (0): Linear(in_features=640, out_features=640, bias=True)
                (1): Dropout(p=0.0, inplace=False)
              )
            )
            (norm3): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
            (ff): FeedForward(
              (net): ModuleList(
                (0): GEGLU(
                  (proj): Linear(in_features=640, out_features=5120, bias=True)
                )
                (1): Dropout(p=0.0, inplace=False)
                (2): Linear(in_features=2560, out_features=640, bias=True)
              )
            )
          )
        )
        (proj_out): Linear(in_features=640, out_features=640, bias=True)
      )
      (2): Transformer2DModel(
        (norm): GroupNorm(32, 640, eps=1e-06, affine=True)
        (proj_in): Linear(in_features=640, out_features=640, bias=True)
        (transformer_blocks): ModuleList(
          (0): BasicTransformerBlock(
            (norm1): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
            (attn1): Attention(
              (to_q): Linear(in_features=640, out_features=640, bias=False)
              (to_k): Linear(in_features=640, out_features=640, bias=False)
              (to_v): Linear(in_features=640, out_features=640, bias=False)
              (to_out): ModuleList(
                (0): Linear(in_features=640, out_features=640, bias=True)
                (1): Dropout(p=0.0, inplace=False)
              )
            )
            (norm2): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
            (attn2): Attention(
              (to_q): Linear(in_features=640, out_features=640, bias=False)
              (to_k): Linear(in_features=1024, out_features=640, bias=False)
              (to_v): Linear(in_features=1024, out_features=640, bias=False)
              (to_out): ModuleList(
                (0): Linear(in_features=640, out_features=640, bias=True)
                (1): Dropout(p=0.0, inplace=False)
              )
            )
            (norm3): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
            (ff): FeedForward(
              (net): ModuleList(
                (0): GEGLU(
                  (proj): Linear(in_features=640, out_features=5120, bias=True)
                )
                (1): Dropout(p=0.0, inplace=False)
                (2): Linear(in_features=2560, out_features=640, bias=True)
              )
            )
          )
        )
        (proj_out): Linear(in_features=640, out_features=640, bias=True)
      )
    )
    (resnets): ModuleList(
      (0): ResnetBlock2D(
        (norm1): GroupNorm(32, 1920, eps=1e-05, affine=True)
        (conv1): Conv2d(1920, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (time_emb_proj): Linear(in_features=1280, out_features=640, bias=True)
        (norm2): GroupNorm(32, 640, eps=1e-05, affine=True)
        (dropout): Dropout(p=0.0, inplace=False)
        (conv2): Conv2d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (nonlinearity): SiLU()
        (conv_shortcut): Conv2d(1920, 640, kernel_size=(1, 1), stride=(1, 1))
      )
      (1): ResnetBlock2D(
        (norm1): GroupNorm(32, 1280, eps=1e-05, affine=True)
        (conv1): Conv2d(1280, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (time_emb_proj): Linear(in_features=1280, out_features=640, bias=True)
        (norm2): GroupNorm(32, 640, eps=1e-05, affine=True)
        (dropout): Dropout(p=0.0, inplace=False)
        (conv2): Conv2d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (nonlinearity): SiLU()
        (conv_shortcut): Conv2d(1280, 640, kernel_size=(1, 1), stride=(1, 1))
      )
      (2): ResnetBlock2D(
        (norm1): GroupNorm(32, 960, eps=1e-05, affine=True)
        (conv1): Conv2d(960, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (time_emb_proj): Linear(in_features=1280, out_features=640, bias=True)
        (norm2): GroupNorm(32, 640, eps=1e-05, affine=True)
        (dropout): Dropout(p=0.0, inplace=False)
        (conv2): Conv2d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (nonlinearity): SiLU()
        (conv_shortcut): Conv2d(960, 640, kernel_size=(1, 1), stride=(1, 1))
      )
    )
    (upsamplers): ModuleList(
      (0): Upsample2D(
        (conv): Conv2d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
    )
  )
```

#### (3): CrossAttnUpBlock2D

``` 
(3): CrossAttnUpBlock2D(
    (attentions): ModuleList(
      (0): Transformer2DModel(
        (norm): GroupNorm(32, 320, eps=1e-06, affine=True)
        (proj_in): Linear(in_features=320, out_features=320, bias=True)
        (transformer_blocks): ModuleList(
          (0): BasicTransformerBlock(
            (norm1): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
            (attn1): Attention(
              (to_q): Linear(in_features=320, out_features=320, bias=False)
              (to_k): Linear(in_features=320, out_features=320, bias=False)
              (to_v): Linear(in_features=320, out_features=320, bias=False)
              (to_out): ModuleList(
                (0): Linear(in_features=320, out_features=320, bias=True)
                (1): Dropout(p=0.0, inplace=False)
              )
            )
            (norm2): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
            (attn2): Attention(
              (to_q): Linear(in_features=320, out_features=320, bias=False)
              (to_k): Linear(in_features=1024, out_features=320, bias=False)
              (to_v): Linear(in_features=1024, out_features=320, bias=False)
              (to_out): ModuleList(
                (0): Linear(in_features=320, out_features=320, bias=True)
                (1): Dropout(p=0.0, inplace=False)
              )
            )
            (norm3): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
            (ff): FeedForward(
              (net): ModuleList(
                (0): GEGLU(
                  (proj): Linear(in_features=320, out_features=2560, bias=True)
                )
                (1): Dropout(p=0.0, inplace=False)
                (2): Linear(in_features=1280, out_features=320, bias=True)
              )
            )
          )
        )
        (proj_out): Linear(in_features=320, out_features=320, bias=True)
      )
      (1): Transformer2DModel(
        (norm): GroupNorm(32, 320, eps=1e-06, affine=True)
        (proj_in): Linear(in_features=320, out_features=320, bias=True)
        (transformer_blocks): ModuleList(
          (0): BasicTransformerBlock(
            (norm1): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
            (attn1): Attention(
              (to_q): Linear(in_features=320, out_features=320, bias=False)
              (to_k): Linear(in_features=320, out_features=320, bias=False)
              (to_v): Linear(in_features=320, out_features=320, bias=False)
              (to_out): ModuleList(
                (0): Linear(in_features=320, out_features=320, bias=True)
                (1): Dropout(p=0.0, inplace=False)
              )
            )
            (norm2): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
            (attn2): Attention(
              (to_q): Linear(in_features=320, out_features=320, bias=False)
              (to_k): Linear(in_features=1024, out_features=320, bias=False)
              (to_v): Linear(in_features=1024, out_features=320, bias=False)
              (to_out): ModuleList(
                (0): Linear(in_features=320, out_features=320, bias=True)
                (1): Dropout(p=0.0, inplace=False)
              )
            )
            (norm3): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
            (ff): FeedForward(
              (net): ModuleList(
                (0): GEGLU(
                  (proj): Linear(in_features=320, out_features=2560, bias=True)
                )
                (1): Dropout(p=0.0, inplace=False)
                (2): Linear(in_features=1280, out_features=320, bias=True)
              )
            )
          )
        )
        (proj_out): Linear(in_features=320, out_features=320, bias=True)
      )
      (2): Transformer2DModel(
        (norm): GroupNorm(32, 320, eps=1e-06, affine=True)
        (proj_in): Linear(in_features=320, out_features=320, bias=True)
        (transformer_blocks): ModuleList(
          (0): BasicTransformerBlock(
            (norm1): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
            (attn1): Attention(
              (to_q): Linear(in_features=320, out_features=320, bias=False)
              (to_k): Linear(in_features=320, out_features=320, bias=False)
              (to_v): Linear(in_features=320, out_features=320, bias=False)
              (to_out): ModuleList(
                (0): Linear(in_features=320, out_features=320, bias=True)
                (1): Dropout(p=0.0, inplace=False)
              )
            )
            (norm2): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
            (attn2): Attention(
              (to_q): Linear(in_features=320, out_features=320, bias=False)
              (to_k): Linear(in_features=1024, out_features=320, bias=False)
              (to_v): Linear(in_features=1024, out_features=320, bias=False)
              (to_out): ModuleList(
                (0): Linear(in_features=320, out_features=320, bias=True)
                (1): Dropout(p=0.0, inplace=False)
              )
            )
            (norm3): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
            (ff): FeedForward(
              (net): ModuleList(
                (0): GEGLU(
                  (proj): Linear(in_features=320, out_features=2560, bias=True)
                )
                (1): Dropout(p=0.0, inplace=False)
                (2): Linear(in_features=1280, out_features=320, bias=True)
              )
            )
          )
        )
        (proj_out): Linear(in_features=320, out_features=320, bias=True)
      )
    )
    (resnets): ModuleList(
      (0): ResnetBlock2D(
        (norm1): GroupNorm(32, 960, eps=1e-05, affine=True)
        (conv1): Conv2d(960, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (time_emb_proj): Linear(in_features=1280, out_features=320, bias=True)
        (norm2): GroupNorm(32, 320, eps=1e-05, affine=True)
        (dropout): Dropout(p=0.0, inplace=False)
        (conv2): Conv2d(320, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (nonlinearity): SiLU()
        (conv_shortcut): Conv2d(960, 320, kernel_size=(1, 1), stride=(1, 1))
      )
      (1): ResnetBlock2D(
        (norm1): GroupNorm(32, 640, eps=1e-05, affine=True)
        (conv1): Conv2d(640, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (time_emb_proj): Linear(in_features=1280, out_features=320, bias=True)
        (norm2): GroupNorm(32, 320, eps=1e-05, affine=True)
        (dropout): Dropout(p=0.0, inplace=False)
        (conv2): Conv2d(320, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (nonlinearity): SiLU()
        (conv_shortcut): Conv2d(640, 320, kernel_size=(1, 1), stride=(1, 1))
      )
      (2): ResnetBlock2D(
        (norm1): GroupNorm(32, 640, eps=1e-05, affine=True)
        (conv1): Conv2d(640, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (time_emb_proj): Linear(in_features=1280, out_features=320, bias=True)
        (norm2): GroupNorm(32, 320, eps=1e-05, affine=True)
        (dropout): Dropout(p=0.0, inplace=False)
        (conv2): Conv2d(320, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (nonlinearity): SiLU()
        (conv_shortcut): Conv2d(640, 320, kernel_size=(1, 1), stride=(1, 1))
      )
    )
  )
)
```

### By products

```  
for resnet, attn in zip(self.resnets, self.attentions):
    # Pop res hidden states. This is the residual connection in UNet.
    res_hidden_states = res_hidden_states_tuple[-1]
    res_hidden_states_tuple = res_hidden_states_tuple[:-1]
    # ... a FreeU block here, skip.
    hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
    hidden_states = resnet(hidden_states, temb)
```
here is the Residual Connection of UNet. Expand channel, and then use resnet to shrink it.