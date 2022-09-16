import os
import json5 as json
import random
import logging

logger = logging.getLogger(__name__)

class Settings:
    batch_name = "Default"
    text_prompts = []
    image_prompt = []
    clip_guidance_scale = "auto"
    tv_scale = 0
    range_scale = 150
    sat_scale = 0
    n_batches = 1
    display_rate = 20
    cutn_batches = 4
    cutn_batches_final = None
    max_frames = 10000
    interp_spline = "Linear"
    init_image = None
    init_masked = None
    init_scale = 1000
    skip_steps = 0
    skip_steps_ratio = 0.0
    frames_scale = 1500
    frames_skip_steps = "60%"
    perlin_init = False
    perlin_mode = "mixed"
    perlin_contrast = 1.0
    perlin_brightness = 1.0
    skip_augs = False
    randomize_class = True
    clip_denoised = False
    clamp_grad = True
    clamp_max = "auto"
    set_seed = "random_seed"
    fuzzy_prompt = False
    rand_mag = 0.05
    eta = "auto"
    width_height = [832, 512]
    width_height_scale = 1
    diffusion_model = "512x512_diffusion_uncond_finetune_008100"
    use_secondary_model = True
    steps = 250
    sampling_mode = "ddim"
    diffusion_steps = 1000
    ViTB32 = 1.0
    ViTB16 = 0.0
    ViTL14 = 0.0
    ViTL14_336 = 0.0
    RN101 = 0.0
    RN50 = 0.0
    RN50x4 = 0.0
    RN50x16 = 0.0
    RN50x64 = 0.0
    ViTB32_laion2b_e16 = 1.0
    ViTB32_laion400m_e31 = 0.0
    ViTB32_laion400m_32 = 0.0
    ViTB32quickgelu_laion400m_e31 = 0.0
    ViTB32quickgelu_laion400m_e32 = 0.0
    ViTB16_laion400m_e31 = 0.0
    ViTB16_laion400m_e32 = 0.0
    ViT_H_14_laion2b = 0.0
    ViT_g_14_laion2b = 0.0
    RN50_yffcc15m = 0.0
    RN50_cc12m = 0.0
    RN50_quickgelu_yfcc15m = 0.0
    RN50_quickgelu_cc12m = 0.0
    RN101_yfcc15m = 0.0
    RN101_quickgelu_yfcc15m = 0.0
    cut_overview = "[12]*400+[4]*600"
    cut_innercut = "[4]*400+[12]*600"
    cut_ic_pow = "[1]*500+[10]*500"
    cut_ic_pow_final = None
    cut_icgray_p = "[0.2]*400+[0]*600"
    cut_heatmaps = False
    smooth_schedules = False
    intermediate_saves = 0
    add_metadata = True
    stop_early = 0
    fix_brightness_contrast = True
    adjustment_interval = 10
    sharpen_preset = 'Off'  # @param ['Off', 'Faster', 'Fast', 'Slow', 'Very Slow']
    keep_unsharp = False  # @param{type: 'boolean'}
    animation_mode = "None" 
    gobig_orientation = "vertical"
    gobig_scale = 2
    gobig_skip_ratio = 0.6
    gobig_overlap = 64
    gobig_maximize = False
    symmetry_loss_v = False
    symmetry_loss_h = False
    symm_loss_scale = "[2500]*1000"
    symm_switch = 45
    simple_symmetry = [0]
    use_jpg = False
    render_mask = None
    cool_down = 0

    #def __init__(self):
    #    # Setting default values for everything, which can then be overridden by settings files.
    def apply_settings_file(self, filename, settings_file, skip_checks=False):
        print(f'apply settings file: {filename}')
        # If any of these are in this settings file they'll be applied, overwriting any previous value.
        # Some are passed through clampval first to make sure they are within bounds (or randomized if desired)
        if is_json_key_present(settings_file, 'batch_name'):
            self.batch_name = (settings_file['batch_name'])
        if is_json_key_present(settings_file, 'text_prompts'):
            self.text_prompts = (settings_file['text_prompts'])
        if is_json_key_present(settings_file, 'image_prompts'):
            self.image_prompts = (settings_file['image_prompts'])
        if is_json_key_present(settings_file, 'clip_guidance_scale'):
            if (type(settings_file['clip_guidance_scale']) == str) and ((settings_file['clip_guidance_scale']) != "random"):
                self.clip_guidance_scale = dynamic_value(settings_file['clip_guidance_scale'])
            else:
                self.clip_guidance_scale = clampval('clip_guidance_scale', 1500, (settings_file['clip_guidance_scale']), 100000, skip_checks)
        if is_json_key_present(settings_file, 'tv_scale'):
            if (settings_file['tv_scale']) != "auto" and (settings_file['tv_scale']) != "random":
                self.tv_scale = int(dynamic_value(settings_file['tv_scale']))
            self.tv_scale = clampval('tv_scale', 0, self.tv_scale, 1000, skip_checks)
        if is_json_key_present(settings_file, 'range_scale'):
            if (settings_file['range_scale']) != "auto" and (settings_file['range_scale']) != "random":
                self.range_scale = int(dynamic_value(settings_file['range_scale']))
            self.range_scale = clampval('range_scale', 0, self.range_scale, 1000, skip_checks)
        if is_json_key_present(settings_file, 'sat_scale'):
            if (settings_file['sat_scale']) != "auto" and (settings_file['sat_scale']) != "random":
                self.sat_scale = int(dynamic_value(settings_file['sat_scale']))
            self.sat_scale = clampval('sat_scale', 0, self.sat_scale, 20000, skip_checks)
        if is_json_key_present(settings_file, 'n_batches'):
            self.n_batches = (settings_file['n_batches'])
        if is_json_key_present(settings_file, 'display_rate'):
            self.display_rate = (settings_file['display_rate'])
        if is_json_key_present(settings_file, 'cutn_batches'):
            if type(settings_file['cutn_batches']) == str:
                self.cutn_batches = dynamic_value(settings_file['cutn_batches'])
            else:
                self.cutn_batches = (settings_file['cutn_batches'])
        if is_json_key_present(settings_file, 'cutn_batches_final'):
            self.cutn_batches_final = (settings_file['cutn_batches_final'])
        if is_json_key_present(settings_file, 'max_frames'):
            self.max_frames = (settings_file['max_frames'])
        if is_json_key_present(settings_file, 'interp_spline'):
            self.interp_spline = (settings_file['interp_spline'])
        if is_json_key_present(settings_file, 'init_image'):
            self.init_image = (settings_file['init_image'])
        if is_json_key_present(settings_file, 'init_masked'):
            self.init_masked = (settings_file['init_masked'])
        if is_json_key_present(settings_file, 'init_scale'):
            self.init_scale = (settings_file['init_scale'])
        if is_json_key_present(settings_file, 'skip_steps'):
            self.skip_steps = int(dynamic_value(settings_file['skip_steps']))
        if is_json_key_present(settings_file, 'skip_steps_ratio'):
            self.skip_steps_ratio = (settings_file['skip_steps_ratio'])
        if is_json_key_present(settings_file, 'stop_early'):
            self.stop_early = (settings_file['stop_early'])
        if is_json_key_present(settings_file, 'frames_scale'):
            self.frames_scale = (settings_file['frames_scale'])
        if is_json_key_present(settings_file, 'frames_skip_steps'):
            self.frames_skip_steps = (settings_file['frames_skip_steps'])
        if is_json_key_present(settings_file, 'perlin_init'):
            self.perlin_init = (settings_file['perlin_init'])
        if is_json_key_present(settings_file, 'perlin_mode'):
            self.perlin_mode = (settings_file['perlin_mode'])
        if is_json_key_present(settings_file, 'perlin_contrast'):
            self.perlin_contrast = (settings_file['perlin_contrast'])
        if is_json_key_present(settings_file, 'perlin_brightness'):
            self.perlin_brightness = (settings_file['perlin_brightness'])
        if is_json_key_present(settings_file, 'skip_augs'):
            self.skip_augs = (settings_file['skip_augs'])
        if is_json_key_present(settings_file, 'randomize_class'):
            self.randomize_class = (settings_file['randomize_class'])
        if is_json_key_present(settings_file, 'clip_denoised'):
            self.clip_denoised = (settings_file['clip_denoised'])
        if is_json_key_present(settings_file, 'clamp_grad'):
            self.clamp_grad = (settings_file['clamp_grad'])
        if is_json_key_present(settings_file, 'clamp_max'):
            if (type(settings_file['clamp_max']) == str) and ((settings_file['clamp_max']) != "random"):
                self.clamp_max = dynamic_value(settings_file['clamp_max'])
            else:
                self.clamp_max = clampval('clamp_max', 0.001, settings_file['clamp_max'], 0.3, skip_checks)
        if is_json_key_present(settings_file, 'set_seed'):
            self.set_seed = (settings_file['set_seed'])
        if is_json_key_present(settings_file, 'fuzzy_prompt'):
            self.fuzzy_prompt = (settings_file['fuzzy_prompt'])
        if is_json_key_present(settings_file, 'rand_mag'):
            self.rand_mag = clampval('rand_mag', 0.0, (settings_file['rand_mag']), 0.999, skip_checks)
        if is_json_key_present(settings_file, 'eta'):
            if (settings_file['eta']) != "auto" and (settings_file['eta']) != "random":
                self.eta = float(dynamic_value(settings_file['eta']))
            self.eta = clampval('eta', 0.0, self.eta, 0.999, skip_checks)
        if is_json_key_present(settings_file, 'width'):
            self.width_height = [(settings_file['width']),
                            (settings_file['height'])]
        if is_json_key_present(settings_file, 'width_height_scale'):
            self.width_height_scale = (settings_file['width_height_scale'])
        if is_json_key_present(settings_file, 'diffusion_model'):
            self.diffusion_model = (settings_file['diffusion_model'])
        if is_json_key_present(settings_file, 'use_secondary_model'):
            self.use_secondary_model = (settings_file['use_secondary_model'])
        if is_json_key_present(settings_file, 'steps'):
            self.steps = int(dynamic_value(settings_file['steps']))
        if is_json_key_present(settings_file, 'sampling_mode'):
            self.sampling_mode = (settings_file['sampling_mode'])
        if is_json_key_present(settings_file, 'diffusion_steps'):
            self.diffusion_steps = (settings_file['diffusion_steps'])
        if is_json_key_present(settings_file, 'ViTB32'):
            self.ViTB32 = float(dynamic_value(settings_file['ViTB32']))
        if is_json_key_present(settings_file, 'ViTB16'):
            self.ViTB16 = float(dynamic_value(settings_file['ViTB16']))
        if is_json_key_present(settings_file, 'ViTL14'):
            self.ViTL14 = float(dynamic_value(settings_file['ViTL14']))
        if is_json_key_present(settings_file, 'ViTL14_336'):
            self.ViTL14_336 = float(dynamic_value(settings_file['ViTL14_336']))
        if is_json_key_present(settings_file, 'RN101'):
            self.RN101 = float(dynamic_value(settings_file['RN101']))
        if is_json_key_present(settings_file, 'RN50'):
            self.RN50 = float(dynamic_value(settings_file['RN50']))
        if is_json_key_present(settings_file, 'RN50x4'):
            self.RN50x4 = float(dynamic_value(settings_file['RN50x4']))
        if is_json_key_present(settings_file, 'RN50x16'):
            self.RN50x16 = float(dynamic_value(settings_file['RN50x16']))
        if is_json_key_present(settings_file, 'RN50x64'):
            self.RN50x64 = float(dynamic_value(settings_file['RN50x64']))
        if is_json_key_present(settings_file, 'ViTB32_laion2b_e16'):
            self.ViTB32_laion2b_e16 = float(dynamic_value(settings_file['ViTB32_laion2b_e16']))
        if is_json_key_present(settings_file, 'ViTB32_laion400m_e31'):
            self.ViTB32_laion400m_e31 = float(dynamic_value(settings_file['ViTB32_laion400m_e31']))
        if is_json_key_present(settings_file, 'ViTB32_laion400m_32'):
            self.ViTB32_laion400m_32 = float(dynamic_value(settings_file['ViTB32_laion400m_32']))
        if is_json_key_present(settings_file, 'ViTB32quickgelu_laion400m_e31'):
            self.ViTB32quickgelu_laion400m_e31 = float(dynamic_value(settings_file['ViTB32quickgelu_laion400m_e31']))
        if is_json_key_present(settings_file, 'ViTB32quickgelu_laion400m_e32'):
            self.ViTB32quickgelu_laion400m_e32 = float(dynamic_value(settings_file['ViTB32quickgelu_laion400m_e32']))
        if is_json_key_present(settings_file, 'ViTB16_laion400m_e31'):
            self.ViTB16_laion400m_e31 = float(dynamic_value(settings_file['ViTB16_laion400m_e31']))
        if is_json_key_present(settings_file, 'ViTB16_laion400m_e32'):
            self.ViTB16_laion400m_e32 = float(dynamic_value(settings_file['ViTB16_laion400m_e32']))
        if is_json_key_present(settings_file, 'ViT_H_14_laion2b'):
            self.ViT_H_14_laion2b = float(dynamic_value(settings_file['ViT_H_14_laion2b']))
        if is_json_key_present(settings_file, 'ViT_g_14_laion2b'):
            self.ViT_g_14_laion2b = float(dynamic_value(settings_file['ViT_g_14_laion2b']))
        if is_json_key_present(settings_file, 'RN50_yffcc15m'):
            self.RN50_yffcc15m = float(dynamic_value(settings_file['RN50_yffcc15m']))
        if is_json_key_present(settings_file, 'RN50_cc12m'):
            self.RN50_cc12m = float(dynamic_value(settings_file['RN50_cc12m']))
        if is_json_key_present(settings_file, 'RN50_quickgelu_yfcc15m'):
            self.RN50_quickgelu_yfcc15m = float(dynamic_value(settings_file['RN50_quickgelu_yfcc15m']))
        if is_json_key_present(settings_file, 'RN50_quickgelu_cc12m'):
            self.RN50_quickgelu_cc12m = float(dynamic_value(settings_file['RN50_quickgelu_cc12m']))
        if is_json_key_present(settings_file, 'RN101_yfcc15m'):
            self.RN101_yfcc15m = float(dynamic_value(settings_file['RN101_yfcc15m']))
        if is_json_key_present(settings_file, 'RN101_quickgelu_yfcc15m'):
            self.RN101_quickgelu_yfcc15m = float(dynamic_value(settings_file['RN101_quickgelu_yfcc15m']))
        if is_json_key_present(settings_file, 'cut_overview'):
            self.cut_overview = dynamic_value(settings_file['cut_overview'])
        if is_json_key_present(settings_file, 'cut_innercut'):
            self.cut_innercut = dynamic_value(settings_file['cut_innercut'])
        if is_json_key_present(settings_file, 'cut_ic_pow'):
            if (type(settings_file['cut_ic_pow']) == str) and ((settings_file['cut_ic_pow']) != "random"):
                self.cut_ic_pow = dynamic_value(settings_file['cut_ic_pow'])
            else:
                self.cut_ic_pow = clampval('cut_ic_pow', 0.0, (settings_file['cut_ic_pow']), 100, skip_checks)
        if is_json_key_present(settings_file, 'cut_ic_pow_final'):
            self.cut_ic_pow_final = clampval('cut_ic_pow_final', 0.5, (settings_file['cut_ic_pow_final']), 100, skip_checks)
        if is_json_key_present(settings_file, 'cut_icgray_p'):
            self.cut_icgray_p = (settings_file['cut_icgray_p'])
        if is_json_key_present(settings_file, 'cut_heatmaps'):
            self.cut_heatmaps = (settings_file['cut_heatmaps'])
        if is_json_key_present(settings_file, 'smooth_schedules'):
            self.smooth_schedules = (settings_file['smooth_schedules'])
        if is_json_key_present(settings_file, 'intermediate_saves'):
            self.intermediate_saves = (settings_file['intermediate_saves'])
        if is_json_key_present(settings_file, 'fix_brightness_contrast'):
            self.fix_brightness_contrast = (settings_file['fix_brightness_contrast'])
        if is_json_key_present(settings_file, 'adjustment_interval'):
            self.adjustment_interval = (settings_file['adjustment_interval'])
        if is_json_key_present(settings_file, 'sharpen_preset'):
            self.sharpen_preset = (settings_file['sharpen_preset'])
        if is_json_key_present(settings_file, 'keep_unsharp'):
            self.keep_unsharp = (settings_file['keep_unsharp'])
        if is_json_key_present(settings_file, 'animation_mode'):
            self.animation_mode = (settings_file['animation_mode'])
        if is_json_key_present(settings_file, 'gobig_scale'):
            self.gobig_scale = int(settings_file['gobig_scale'])
        if is_json_key_present(settings_file, 'gobig_skip_ratio'):
            self.gobig_skip_ratio = (settings_file['gobig_skip_ratio'])
        if is_json_key_present(settings_file, 'gobig_overlap'):
            self.gobig_overlap = (settings_file['gobig_overlap'])
        if is_json_key_present(settings_file, 'gobig_maximize'):
            self.gobig_maximize = (settings_file['gobig_maximize'])
        if is_json_key_present(settings_file, 'symmetry_loss'):
            self.symmetry_loss_v = (settings_file['symmetry_loss'])
            print("symmetry_loss was depracated, please use symmetry_loss_v in the future")
        if is_json_key_present(settings_file, 'symmetry_loss_v'):
            self.symmetry_loss_v = (settings_file['symmetry_loss_v'])
        if is_json_key_present(settings_file, 'symmetry_loss_h'):
            self.symmetry_loss_h = (settings_file['symmetry_loss_h'])
        if is_json_key_present(settings_file, 'sloss_scale'):
            print('"sloss_scale" is deprecated. Please update your settings to use "symm_loss_scale"')
            self.symm_loss_scale = (settings_file['sloss_scale'])
        if is_json_key_present(settings_file, 'symm_loss_scale'):
            if type(settings_file['symm_loss_scale']) == str:
                symm_loss_scale = dynamic_value(settings_file['symm_loss_scale'])
            else:
                symm_loss_scale = (settings_file['symm_loss_scale'])
        if is_json_key_present(settings_file, 'symm_switch'):
            self.symm_switch = int(clampval('symm_switch', 1, (settings_file['symm_switch']), self.steps, skip_checks))
        if is_json_key_present(settings_file, 'simple_symmetry'):
            self.simple_symmetry = (settings_file['simple_symmetry'])
        if is_json_key_present(settings_file, 'use_jpg'):
            self.use_jpg = (settings_file['use_jpg'])
        if is_json_key_present(settings_file, 'render_mask'):
            self.render_mask = (settings_file['render_mask'])
        if is_json_key_present(settings_file, 'cool_down'):
            self.cool_down = (settings_file['cool_down'])
    
    def save_settings(self, extra_settings):
        setting_list = {
            'batch_name': self.batch_name,
            'text_prompts': self.text_prompts,
            'n_batches': self.n_batches,
            'steps': self.steps,
            'display_rate': self.display_rate,
            'width_height_scale': self.width_height_scale,
            'width': int(self.width_height[0] / self.width_height_scale),
            'height': int(self.width_height[1] / self.width_height_scale),
            'set_seed': extra_settings['seed'],
            'image_prompts': self.image_prompts,
            'clip_guidance_scale': self.clip_guidance_scale,
            'tv_scale': self.tv_scale,
            'range_scale': self.range_scale,
            'sat_scale': self.sat_scale,
            # 'cutn': cutn,
            'cutn_batches': self.cutn_batches,
            'cutn_batches_final': self.cutn_batches_final,
            'max_frames': self.max_frames,
            'interp_spline': self.interp_spline,
            # 'rotation_per_frame': rotation_per_frame,
            'init_image': self.init_image,
            'init_masked': self.init_masked,
            'render_mask': self.render_mask,
            'init_scale': self.init_scale,
            'skip_steps': self.skip_steps,
            'skip_steps_ratio': self.skip_steps_ratio,
            # 'zoom_per_frame': zoom_per_frame,
            'frames_scale': self.frames_scale,
            'frames_skip_steps': self.frames_skip_steps,
            'perlin_init': self.perlin_init,
            'perlin_mode': self.perlin_mode,
            'skip_augs': self.skip_augs,
            'randomize_class': self.randomize_class,
            'clip_denoised': self.clip_denoised,
            'clamp_grad': self.clamp_grad,
            'clamp_max': self.clamp_max,
            'fuzzy_prompt': self.fuzzy_prompt,
            'rand_mag': self.rand_mag,
            'eta': self.eta,
            'diffusion_model': extra_settings["diffusion_model_name"],
            'use_secondary_model': self.use_secondary_model,
            'diffusion_steps': self.diffusion_steps,
            'sampling_mode': self.sampling_mode,
            'ViTB32': self.ViTB32,
            'ViTB16': self.ViTB16,
            'ViTL14': self.ViTL14,
            'ViTL14_336': self.ViTL14_336,
            'RN101': self.RN101,
            'RN50': self.RN50,
            'RN50x4': self.RN50x4,
            'RN50x16': self.RN50x16,
            'RN50x64': self.RN50x64,
            'ViTB32_laion2b_e16': self.ViTB32_laion2b_e16,
            'ViTB32_laion400m_e31': self.ViTB32_laion400m_e31,
            'ViTB32_laion400m_32': self.ViTB32_laion400m_32,
            'ViTB32quickgelu_laion400m_e31': self.ViTB32quickgelu_laion400m_e31,
            'ViTB32quickgelu_laion400m_e32': self.ViTB32quickgelu_laion400m_e32,
            'ViTB16_laion400m_e31': self.ViTB16_laion400m_e31,
            'ViTB16_laion400m_e32': self.ViTB16_laion400m_e32,
            'ViT_H_14_laion2b': self.ViT_H_14_laion2b,
            'ViT_g_14_laion2b': self.ViT_g_14_laion2b,
            'RN50_yffcc15m': self.RN50_yffcc15m,
            'RN50_cc12m': self.RN50_cc12m,
            'RN50_quickgelu_yfcc15m': self.RN50_quickgelu_yfcc15m,
            'RN50_quickgelu_cc12m': self.RN50_quickgelu_cc12m,
            'RN101_yfcc15m': self.RN101_yfcc15m,
            'RN101_quickgelu_yfcc15m': self.RN101_quickgelu_yfcc15m,
            'cut_overview': str(self.cut_overview),
            'cut_innercut': str(self.cut_innercut),
            'cut_ic_pow': extra_settings["og_cut_ic_pow"],
            'cut_ic_pow_final': self.cut_ic_pow_final,
            'cut_icgray_p': str(self.cut_icgray_p),
            'cut_heatmaps': self.cut_heatmaps,
            'smooth_schedules': self.smooth_schedules,
            'animation_mode': self.animation_mode,
            'stop_early': self.stop_early,
            'fix_brightness_contrast': self.fix_brightness_contrast,
            'adjustment_interval': self.adjustment_interval,
            'sharpen_preset': self.sharpen_preset,
            'keep_unsharp': self.keep_unsharp,
            'gobig_scale': self.gobig_scale,
            'gobig_skip_ratio': self.gobig_skip_ratio,
            'gobig_overlap': self.gobig_overlap,
            'gobig_maximize': self.gobig_maximize,
            'symmetry_loss_v': self.symmetry_loss_v,
            'symmetry_loss_h': self.symmetry_loss_h,
            'symm_loss_scale': self.symm_loss_scale,
            'symm_switch': self.symm_switch,
            'simple_symmetry': self.simple_symmetry,
            'perlin_brightness': self.perlin_brightness,
            'perlin_contrast': self.perlin_contrast,
            'use_jpg': self.use_jpg
        }
        return setting_list

# Simple check to see if a key is present in the settings file
def is_json_key_present(json, key, subkey="none"):
    try:
        if subkey != "none":
            buf = json[key][subkey]
        else:
            buf = json[key]
    except KeyError:
        return False
    if type(buf) == type(None):
        return False
    return True


# A simple way to ensure values are in an accceptable range, and also return a random value if desired
def clampval(var_name, minval, val, maxval, skip_checks):
    if val == "random":
        try:
            val = random.randint(minval, maxval)
        except:
            val = random.uniform(minval, maxval)
        return val
    # Auto is handled later, so we just return it back as is
    elif val == "auto":
        return val
    elif type(val) == str:
        return val
    elif val < minval and not skip_checks:
        print(f'Warning: {var_name} is below {minval} - if you get bad results, consider adjusting.')
        return val
    elif val > maxval and not skip_checks:
        print(f'Warning: {var_name} is above {maxval} - if you get bad results, consider adjusting.')
        return val
    else:
        return val


# Dynamic value - takes ready-made possible options within a string and returns the string with an option randomly selected
# Format is "I will return <Value1|Value2|Value3> in this string"
# Which would come back as "I will return Value2 in this string" (for example)
# Optionally if a value of ^^# is first, it means to return that many dynamic values,
# so <^^2|Value1|Value2|Value3> in the above example would become:
# "I will return Value3 Value2 in this string"
# note: for now assumes a string for return. TODO return a desired type
def dynamic_value(incoming):
    if type(incoming) == str:  # we only need to do something if it's a string...
        if incoming == "auto" or incoming == "random":
            return incoming
        elif "<" in incoming:   # ...and if < is in the string...
            text = incoming
            logger.debug(f'Original value: {text}')
            while "<" in text:
                start = text.index('<')
                end = text.index('>')
                swap = text[(start + 1):end]
                value = ""
                count = 1
                values = swap.split('|')
                if "^^" in values[0]:
                    count = values[0]
                    values.pop(0)
                    count = int(count[2:])
                random.shuffle(values)
                for i in range(count):
                    value = value + values[i] + " "
                value = value[:-1]  # remove final space
                text = text.replace(f'<{swap}>', value)
            logger.debug(f'Dynamic value: {text}')
            return text
        else:
            return incoming
    else:
        return incoming

# Linear interpolation. Return y between y1 and y2 for the same position x is bettewen x1 and x2
def val_interpolate(x1, y1, x2, y2, x):
    d = [[x1, y1], [x2, y2]]
    output = d[0][1] + (x - d[0][0]) * ((d[1][1] - d[0][1])/(d[1][0] - d[0][0]))
    if type(y1) == int:
        output = int(output)  # return the proper type
    return(output)

# take a single number and turn it into a string-style schedule, with support for interpolated
def num_to_schedule(input, final=-9999):
    if final != -9999:
        output = (f"[{input}]*1+")
        for i in range(1, 1000):
            val = val_interpolate(1, input, 1000, final, i)
            output = output + (f"[{val}]*1+")
        output = output[:-1]  # remove the final plus character
    else:
        output = (f'[{input}]*1000')
    return(output)

def randomizer(category):
    random.seed()
    randomizers = []
    with open(f'settings/{category}.txt', encoding="utf-8") as f:
        for line in f:
            randomizers.append(line.strip())
    random_item = random.choice(randomizers)
    return(random_item)

def randomize_prompts(prompts):
    # take a list of prompts and handle any _random_ elements
    newprompts = []
    for prompt in prompts:
        if "<" in prompt:
            newprompt = dynamic_value(prompt)
        else:
            newprompt = prompt
        if "_" in newprompt:
            while "_" in newprompt:
                start = newprompt.index('_')
                end = newprompt.index('_', start+1)
                swap = newprompt[(start + 1):end]
                swapped = randomizer(swap)
                newprompt = newprompt.replace(f'_{swap}_', swapped, 1)
        newprompts.append(newprompt)
    return newprompts

# randomly pick a file name from a directory:
def random_file(rootdir, subdir):
    files = []
    files = os.listdir(f'{rootdir}/{subdir}')
    file = random.choice(files)
    return(file)