# What do all the settings do?
This document hopes to explain what the various settings are. Some of them we're still figuring out. :)
Note that a few of the settings can be randomly chosen -- see the section below for details.

| *Setting name* | *Default in settings.json* | *Explanation*
| -------------------------|------------------------|:---
| **batch_name** | "Default" | The directory within images_out to store your results
| **text_prompts** | "The Big Sur Coast, by Asher Brown Durand, featured on ArtStation." | The phrase(s) to use for generating an image. More details below
| **n_batches** | 1 | How many images to generate
| **steps** | 250 | Generally, the more steps you run, the more detailed the results, however the returns start to diminish after 350
| **display_rate** | 50 | How often (in steps) to update the progress.png image
| **width** | 832 | Image output width in pixels - must be a multiple of 64
| **height** | 512 | Image output height in pixels - must be a multiple of 64
| **set_seed** | "random_seed" | If set to random_seed it will generate a new seed. Replace this with a specific number to elimate randomness in the start. Additional images in a batch are always the seed from the previous image - 1
| **image_prompts** | {} | For using images instead of words for prompts. Specifiy the file name of the init image.
| **clip_guidance_scale** | "auto" | Controls how much the image should look like the prompt. Affected by resolution, so "auto" will try to calculate a good value. Scheduled value supported.
| **tv_scale** | 0 | Controls the smoothness of the final output. tests have shown minimal impact when changing this.
| **range_scale** | 150 | Controls how far out of range RGB values are allowed to be.
| **sat_scale** | 0 | Controls how much saturation is allowed.
| **cutn_batches** | 8 | How many batches of cut_overview and cut_innercut to run per step.
| **cutn_batches_final** | "None" | If set, the image will start with cutn_batches and progress to this number by the final step
| **max_frames** | 10000 | Maximum size of a batch
| **interp_spline** | "Linear" | Do not change, currently will not look good.
| **init_image** | null | The starting image to use. Usualy leave this blank and it will start with randomness
| **init_masked** | null | Optional second init - include the bits you want to render against and leave the rest black - leave null if you want perlin noise instead. Use in conjunction with render_mask
| **render_mask** | null | A black and white image that tells the compositor what to keep (white) or leave original (black) - used in conjunction with init_image and init_masked.
| **init_scale** | 1000 | This enhances the effect of the init image, a good value is 1000
| **skip_steps** | 0 | How many steps in the overall process to skip. Generally leave this at 0, though if using an init_image it is recommended to be 50% of overall steps
| **perlin_init** | false | Option to start with random perlin noise
| **perlin_mode** | "mixed" | Other options are "grey" or "color", what they do I'm not sure
| **perlin_contrast** | 1.0 | Amount of contrast adjustment for perlin_init
| **perlin_brightness** | 1.0 | Amount of brightness adjustment for perlin_init
| **skip_augs** | false | Controls whether to skip torchvision augmentations
| **randomize_class** | true | Controls whether the imagenet class is randomly changed each iteration
| **clip_denoised** | false | Determines whether CLIP discriminates a noisy or denoised image
| **clamp_grad** | true | Experimental: Using adaptive clip grad in the cond_fn
| **clamp_max** | "auto" | Lower values (0.01) can help keep colors muted. Higher values (0.3) allow for more vibrancy. However it is affected by steps, so "auto" will try to calculate a good value. Scheduled value supported.
| **fuzzy_prompt** | false | Controls whether to add multiple noisy prompts to the prompt losses
| **rand_mag** | 0.05 | Controls how far it can stray from your prompt - not used unless either fuzzy_prompt is true, or an init image is used
| **eta** | "auto" | Has to do with how much the generator can stray from your prompt. Affected by steps, so "auto" will calculate a good value.
| **diffusion_model** | "512x512_diffusion_uncond_finetune_008100" | Additional options are: "256x256_openai_comics_faces_by_alex_spirin", "256x256_diffusion_uncond", "pixel_art_diffusion_hard_256", "pixel_art_diffusion_soft_256", "pixelartdiffusion4k", "portrait_generator_v001", "watercolordiffusion", "watercolordiffusion_2", "PulpSciFiDiffusion", "FeiArt_Handpainted_CG_Diffusion", "IsometricDiffusionRevrart512px"
| **use_secondary_model** | true | Reduces memory and improves speed, potentially at a loss of quality
| **diffusion_steps** | 1000 | Note: The code seems to calculate this no matter what you put in, so might as well leave it
| **sampling_mode** | "ddim"  | Options are "plms" or "ddim" - plms can reach a nice image in fewer steps, but may not look as good as ddim.
| **ViTB32** | 0.0 | Enable (1.0) or Disable (0.0) this CLIP Model. Between 0.0 and 1.0 is a weight factor
| **ViTB16** | 0.0 | Enable (1.0) or Disable (0.0) this CLIP Model. Between 0.0 and 1.0 is a weight factor
| **ViTL14** | 0.0 | Enable (1.0) or Disable (0.0) this CLIP Model. Between 0.0 and 1.0 is a weight factor
| **ViTL14_336** | 0.0 | Enable (1.0) or Disable (0.0) this CLIP Model. Between 0.0 and 1.0 is a weight factor
| **RN101** | 0.0 | Enable (1.0) or Disable (0.0) this CLIP Model. Between 0.0 and 1.0 is a weight factor
| **RN50** | 0.0 | Enable (1.0) or Disable (0.0) this CLIP Model. Between 0.0 and 1.0 is a weight factor
| **RN50x4** | 0.0 | Enable (1.0) or Disable (0.0) this CLIP Model. Between 0.0 and 1.0 is a weight factor
| **RN50x16** | 0.0 | Enable (1.0) or Disable (0.0) this CLIP Model. Between 0.0 and 1.0 is a weight factor
| **RN50x64** | 0.0 | Enable (1.0) or Disable (0.0) this CLIP Model. Between 0.0 and 1.0 is a weight factor
| **ViTB32_laion2b_e16** | 0.0 | Enable (1.0) or Disable (0.0) this OpenClip Model. Between 0.0 and 1.0 is a weight factor
| **ViTB32_laion400m_e31** | 0.0 | Enable (1.0) or Disable (0.0) this OpenClip Model. Between 0.0 and 1.0 is a weight factor
| **ViTB32_laion400m_32** | 0.0 | Enable (1.0) or Disable (0.0) this OpenClip Model. Between 0.0 and 1.0 is a weight factor
| **ViTB32quickgelu_laion400m_e31** | 0.0 | Enable (1.0) or Disable (0.0) this OpenClip Model. Between 0.0 and 1.0 is a weight factor
| **ViTB32quickgelu_laion400m_e32** | 0.0 | Enable (1.0) or Disable (0.0) this OpenClip Model. Between 0.0 and 1.0 is a weight factor
| **ViTB16_laion400m_e31** | 0.0 | Enable (1.0) or Disable (0.0) this OpenClip Model. Between 0.0 and 1.0 is a weight factor
| **ViTB16_laion400m_e32** | 0.0 | Enable (1.0) or Disable (0.0) this OpenClip Model. Between 0.0 and 1.0 is a weight factor
| **RN50_yffcc15m** | 0.0 | Enable (1.0) or Disable (0.0) this OpenClip Model. Between 0.0 and 1.0 is a weight factor
| **RN50_cc12m** | 0.0 | Enable (1.0) or Disable (0.0) this OpenClip Model. Between 0.0 and 1.0 is a weight factor
| **RN50_quickgelu_yfcc15m** | 0.0 | Enable (1.0) or Disable (0.0) this OpenClip Model. Between 0.0 and 1.0 is a weight factor
| **RN50_quickgelu_cc12m** | 0.0 | Enable (1.0) or Disable (0.0) this OpenClip Model. Between 0.0 and 1.0 is a weight factor
| **RN101_yfcc15m** | 0.0 | Enable (1.0) or Disable (0.0) this OpenClip Model. Between 0.0 and 1.0 is a weight factor
| **RN101_quickgelu_yfcc15m** | 0.0 | Enable (1.0) or Disable (0.0) this OpenClip Model. Between 0.0 and 1.0 is a weight factor
| **cut_overview** | "[5]\*400+[1]\*600" | How many "big picture" passes to do. More towards the start, less later, is the general idea
| **cut_innercut** | "[1]\*400+[5]\*600" | Conversely, how many detail passes to do. Fewer at the start, then get more detailed
| **cut_ic_pow** | 1 | A higher number can add more detail, but may create unwanted fine lines (value range: 0.0 to 100). NOTE: Can be scheduled like cut_overview instead (in which case cut_ic_pow_final is ignored)
| **cut_ic_pow_final** | "None" |If set, image will start at cut_ic_pow and continue toward cut_ic_pow_final by the end, unless cut_ic_pow is a schedule
| **cut_icgray_p** | "[0.2]\*400+[0]\*600" | Percent of cuts to do as grayscale, which evidently can help add detail
| **cut_heatmaps** | false | Helps ensure innercuts are evenly dispersed, at the expense of some render time
| **smooth_schedules** | false | When true, will smooth the transitions within schedule-style values like cut_overview
| **gobig_orientation** | "vertical" | Which direction to do slices for gobig mode. Options are vertical or horizontal, but vertical is best in most cases
| **gobig_scale** | 2 | Amount to scale original image by (2 is recommended, 3 or 4 is getting nuts)
| **gobig_skip_ratio** | 0.6 | Amount of steps to skip (60% is usually good - too high and there's not enough time for gobig to work -- too low and gobig will add unwanted detail)
| **gobig_overlap** | 64 | Size in pixels for how much to overlap (blend) images in gobig mode, to hide seams
| **gobig_maximize** | false | If true, the final render size for gobig will be extended to use areas that would otherwise not be used, effectively increasing the final image size by 10 to 30% for free!
| **intermediate_saves**   | 0  | Save in progress. A value of `2` will save a copy at 33% and 66%. 0 will save none. A value of `[5, 9, 34, 45]` will save at steps 5, 9, 34, and 45. A value of `[.5, .7, .9]` will save at 50%, 70%, and 90% of total steps (Make sure to include the brackets)
| **symmetry_loss_v** | false | Set this to "true" to enable left/right symmetry during the render
| **symmetry_loss_h** | false | Set this to "true" to enable top/bottom symmetry during the render
| **symm_loss_scale** |  20000 | helps control how closely each side should match during symmetry. Definitely play with this number to get the results you're looking for.
| **symm_switch** | 45 | what step to stop doing symmetry mode
| **simple_symmetry** | [0] | Alternate version of symmetry (beta). This will simply mirror the left side of the image to the right at whatever steps are listed (I recommend only doing this once or twice, and during the first two-thirds of the image. For 200 steps, maybe [25,100] is a good starting point). Note: only triggers if symmetry_loss_v and symmetry_loss_h are false. Note that you can also do values like .5 which will calculate to 50% of total steps.
| **stop_early** | 0 | stop processing your image at a certain step
| **use_jpg** | false | Save images as jpg rather than PNG (save disk space at cost of image quality)
| **cool_down** | 0 | total # of idle seconds to add to your render, which will be divided between the steps, to let your GPU cool off.

## Text Prompts
There are a handful of techniques available within Text Prompts. Here are a few examples:

### Multiple prompts for a single image
Dividing prompts up into separate prompts can sometimes help with image quality.
```
    "text_prompts": {
        "0": [
            "A castle in the Scottish Highlands, by Beeple", "sunset, autumn leaves"
        ],
    },
```

### Weights
You can add a weight to the end of any prompt to tell the system how important that prompt is compared to the others. Negatives are supported, too.
Please note the total weight must not be zero.
```
    "text_prompts": {
        "0": [
            "A castle in the Scottish Highlands, by Beeple:2", "sunset, autumn leaves:-1"
        ],
    },
```

### A prompt series, for use with n_batches
This setting will use the first prompt for the first image in a batch, and the second prompt for the second image, etc.
```
    "text_prompts": {
        "0": [
            "A castle in the Scottish Highlands, by Beeple:2", "sunset, autumn leaves:-1"
        ],
        "1": [
            "An inn in the Scottish Lowlands, by RHADS"
        ],
    },
```
If you do a batch with only one prompt, PRD will use that prompt for all images.
You can also set the next image to, say, "10" and PRD would start using that prompt at the tenth image of the batch.

### Randomization
Prompts can be randomized in two ways. First, if you surround a word with the _ character, and there is a matching text file in the settings folder, a random line from that text file will replace the surrounded word.
For example:
```
    "text_prompts": {
        "0": [
            "A castle in the Scottish Highlands, by _artist_"
        ],
    },
```
This would look for artist.txt, randomly select a line from that file, and replace ```_artist_``` with it. This can be done as many times as you want within a prompt, and you can add whatever text files you wish to the settings folder. Subjects, adjectives, artstyle, and so on. Some of these are provided, but feel free to create your own!

Prompts can also be randomized directly for quicker experimentation. This is done by adding a set of dynamic options within the prompt.
Example:
```
    "text_prompts": {
        "0": [
            "A <castle|home|fort|inn> in the Scottish Highlands, by _artist_"
        ],
    },
```
This would select one of the words between the <> characters, and discard the rest. 
Note: This is not done for every image in a batch. If you want multiple randomizations, you would need to run prd multiple times, or use the prompt series above.

One additional trick for directly randomized prompts is selecting how many words you want back. By default it's one, but you can ask for more this way:
```
    "text_prompts": {
        "0": [
            "A castle in the Scottish Highlands, by <^^2|sparth|RHADS|Beeple|Asher Brown Durand>"
        ],
    },
```
In this case, ^^2 means two artists would be selected from the four options.

## Randomizable settings
The following settings can be set to "random" (with the quotes), which will tell the code to pick a random value within their expected boundaries:

**clip_guidance_scale**
**tv_scale**
**range_scale**
**sat_scale**
**clamp_max**
**rand_mag**
**eta**
**cut_ic_pow**
**diffusion_model** (note: it might select one you don't have enough VRAM for...)

In addition, init_image can be randomized. Create a folder inside "init_images" and place as many images inside it was you want.
Let's say for example we have a folder called "space". 
Now, in your settings file, set your init_image to:
```
    "init_image": "_space_"
```
A random image from init_images/space will be chosen. Be sure to set your skip_steps as well.

## Dynamic settings
In addition to text prompts, certain other settings can be provided as a dynamic selection set from which the code will randomly choose a value.
For example:
```
    "eta": "<0.2|0.5|0.75|0.99>"
```
would pick one of those values for eta. Note you have to put the values in quotes when using this feature. 
Currently these values support dynamic mode:

**tv_scale**
**range_scale**
**sat_scale**
**skip_steps**
**eta**
**steps**
**ViTB32** note: for these clip models, use numeric values instead of true / false. 1.0 is fully on, 0.0 is off, 0.5 means half on, etc.
**ViTB16**
**ViTL14**
**ViTL14_336**
**RN101**
**RN50**
**RN50x4**
**RN50x16**
**RN50x64**
**symm_loss_scale** note: only when using a schedule-style setting
**cut_overview**
**cut_innercut**
**cut_ic_pow** note: only when using a schedule-style setting
**clip_guiance_scale** note: only when using a schedule-style setting
**clamp_max** note: only when using a schedule-style setting
**cutn_batches** note: only when using a schedule-style setting

A scheduled setting is like this: ```"[5]*1000"```. See cut_overview and cut_innercut for examples.
The first number there is the value to use, 5. It's saying to use 5 for all 1000 diffusion steps. 