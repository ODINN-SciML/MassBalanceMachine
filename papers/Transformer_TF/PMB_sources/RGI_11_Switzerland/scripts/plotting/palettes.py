import matplotlib.colors as mcolors


def get_color_maps(vmin, vmax):
    if vmin < 0 and vmax > 0:
        norm_ann = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=-vmin)
        cmap_ann = "coolwarm_r"
    elif vmin < 0 and vmax < 0:
        norm_ann = mcolors.Normalize(vmin=vmin, vmax=vmax)
        cmap_ann = "Reds_r"
    else:
        norm_ann = mcolors.Normalize(vmin=vmin, vmax=vmax)
        cmap_ann = "Blues"
    return cmap_ann, norm_ann
