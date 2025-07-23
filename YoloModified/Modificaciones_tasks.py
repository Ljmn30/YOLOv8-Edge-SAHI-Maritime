
# Find the parse_model function and add the imports to the top of the file:

# Add these imports to the top of the tasks.py file (after the existing imports):
"""
# Enhanced modules import
try:
    from ultralytics.nn.modules.enhanced import (
        DyConv, C2f_ECA, FeatureFusion, SPPF_Enhanced, DetectEnhanced, ECAAttention
    )
    ENHANCED_AVAILABLE = True
except ImportError:
    ENHANCED_AVAILABLE = False
    print("Warning: Enhanced modules not available, falling back to standard modules")
    # Fallback imports
    from ultralytics.nn.modules.conv import Conv as DyConv
    from ultralytics.nn.modules.block import C2f as C2f_ECA
    from ultralytics.nn.modules.conv import Conv as FeatureFusion
    from ultralytics.nn.modules.block import SPPF as SPPF_Enhanced
    from ultralytics.nn.modules.head import Detect as DetectEnhanced
    from ultralytics.nn.modules.conv import Conv as ECAAttention
"""

# Then, in the parse_model function, find where the modules are defined and modify:

def parse_model(d, ch, verbose=True):
    """Parse a YOLO model.yaml dictionary into a PyTorch model."""
    import ast
    import contextlib
    import json
    import platform
    import zipfile
    from copy import deepcopy
    from pathlib import Path

    from ultralytics.nn.modules import (
        AIFI,
        C1,
        C2,
        C3,
        C3TR,
        SPP,
        SPPF,
        Bottleneck,
        BottleneckCSP,
        C2f,
        C2fAttn,
        C3Ghost,
        C3x,
        Classify,
        Concat,
        Conv,
        Conv2,
        ConvTranspose,
        Detect,
        DWConv,
        DWConvTranspose2d,
        Focus,
        GhostBottleneck,
        GhostConv,
        HGBlock,
        HGStem,
        Pose,
        RepC3,
        RepConv,
        RepNCSPELAN4,
        RepNCSPELAN4,
        ResNetLayer,
        RTDETRDecoder,
        Segment,
        WorldDetect,
        v10Detect,
    )
    
    # Add enhanced modules to the dictionary
    enhanced_modules = {}
    if ENHANCED_AVAILABLE:
        enhanced_modules.update({
            'DyConv': DyConv,
            'C2f_ECA': C2f_ECA,
            'FeatureFusion': FeatureFusion,
            'SPPF_Enhanced': SPPF_Enhanced,
            'DetectEnhanced': DetectEnhanced,
            'ECAAttention': ECAAttention,
        })
    
    # Rest of the parse_model code...
    max_channels = float("inf")
    nc, act, scales = (d.get(x) for x in ("nc", "activation", "scales"))
    depth, width, kpt_shape = (d.get(x, 1.0) for x in ("depth_multiple", "width_multiple", "kpt_shape"))
    if scales:
        scale = d.get("scale")
        if not scale:
            scale = tuple(scales.keys())[0]
            LOGGER.warning(f"WARNING ⚠️ no model scale passed. Assuming scale='{scale}'.")
        depth, width, max_channels = scales[scale]

    if act:
        Conv.default_act = eval(act)  # redefine default activation, i.e. Conv.default_act = nn.SiLU()
        if verbose:
            LOGGER.info(f"{colorstr('activation:')} {act}")  # print

    if verbose:
        LOGGER.info(f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<45}{'arguments':<30}")
    ch = [ch]
    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):  # from, number, module, args
        if isinstance(m, str):
            if m in enhanced_modules:
                m = enhanced_modules[m]
            else:
                m = getattr(torch.nn, m[3:]) if "nn." in m else globals()[m]  # get module
        
        for j, a in enumerate(args):
            if isinstance(a, str):
                with contextlib.suppress(ValueError):
                    args[j] = locals()[a] if a in locals() else ast.literal_eval(a)

        n = n_ = max(round(n * depth), 1) if n > 1 else n  # depth gain
        if m in {
            Conv,
            GhostConv,
            Bottleneck,
            GhostBottleneck,
            SPP,
            SPPF,
            DWConv,
            Focus,
            BottleneckCSP,
            C1,
            C2,
            C2f,
            C2fAttn,
            C3,
            C3TR,
            C3Ghost,
            RepC3,
            C3x,
            RepConv,
            DWConvTranspose2d,
            ConvTranspose,
            RepNCSPELAN4,
            AIFI,
            HGStem,
            HGBlock,
            ResNetLayer,
            # Add enhanced modules
            DyConv,
            C2f_ECA,
            SPPF_Enhanced,
            ECAAttention,
        }:
            c1, c2 = ch[f], args[0]
            if c2 != nc:  # if c2 not equal to number of classes (i.e. for Classify() output)
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            if m is RepNCSPELAN4:
                args[1] = make_divisible(min(args[1], max_channels) * width, 8)
            args = [c1, c2, *args[1:]]
            if m in {C2f, C2fAttn, C3, C3TR, C3Ghost, C3x, RepC3, C2f_ECA}:
                args.insert(2, n)
                n = 1
        elif m is AIFI:
            args = [ch[f], *args]
        elif m in {HGStem, HGBlock}:
            c1, c2 = ch[f], args[0]
            c2 = make_divisible(min(c2, max_channels) * width, 8)
            args = [c1, c2, *args[1:]]
        elif m is ResNetLayer:
            c2 = args[1] if args[3] else args[1] * 4
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif m in {Detect, WorldDetect, Segment, Pose, v10Detect, DetectEnhanced}:
            args.append([ch[x] for x in f])
            if m is Segment:
                args[2] = make_divisible(min(args[2], max_channels) * width, 8)
        elif m is RTDETRDecoder: 
            args.insert(1, [ch[x] for x in f])
        elif m is FeatureFusion:
            c1 = ch[f] if isinstance(f, int) else ch[f[0]]
            c2 = args[0] if args else c1
            args = [c1, c2]
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace("__main__.", "")  # module type
        m_.np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type = i, f, t  # attach index, 'from' index, type
        if verbose:
            LOGGER.info(f"{i:>3}{str(f):>20}{n_:>3}{m_.np:10.0f}  {t:<45}{str(args):<30}")  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)