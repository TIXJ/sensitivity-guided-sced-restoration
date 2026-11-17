"""Shared Times New Roman styling and an expanded contingency-error display."""
from pathlib import Path
import math
import unicodedata


def configure_matplotlib():
    import matplotlib as mpl
    mpl.rcParams.update({'font.family': 'Times New Roman',
                         'font.serif': ['Times New Roman'], 'pdf.fonttype': 42,
                         'mathtext.fontset': 'custom',
                         'mathtext.rm': 'Times New Roman',
                         'mathtext.it': 'Times New Roman:italic',
                         'mathtext.bf': 'Times New Roman:bold'})


def plot_contingency_errors(rows, destination):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.cbook import boxplot_stats
    configure_matplotlib()
    levels=sorted({r['demand_mw'] for r in rows if r['converged']})
    fig,axes=plt.subplots(1,len(levels),figsize=(3.55,2.65),layout='constrained')
    for ax,level in zip(axes,levels):
        values=np.array([r['hybrid_mae_mw'] for r in rows if r['converged'] and r['demand_mw']==level])
        stats=boxplot_stats(values,whis=1.5)[0]
        ax.bxp([stats],positions=[1],widths=.55,showfliers=False,patch_artist=True,
               boxprops={'facecolor':'#bbd5e5','edgecolor':'#244b65','linewidth':1.1},
               medianprops={'color':'#9e302b','linewidth':1.5},
               whiskerprops={'color':'#244b65','linewidth':1.1},
               capprops={'color':'#244b65','linewidth':1.1})
        pad=max((stats['whishi']-stats['whislo'])*.18,.08)
        ax.set_ylim(stats['whislo']-pad,stats['whishi']+pad)
        ax.set_xlim(.5,1.5)
        ax.set_xticks([])
        ax.set_title(f'{level/1000:.1f} GW\n(n = {len(values)})',fontsize=8)
        ax.tick_params(axis='y',labelsize=7)
        ax.yaxis.set_major_locator(plt.MaxNLocator(4))
        ax.grid(axis='y',alpha=.22,linewidth=.5)
        ax.set_axisbelow(True)
    axes[0].set_ylabel('Post-outage mean absolute flow error (MW)',fontsize=8)
    fig.savefig(destination)
    fig.savefig(Path(destination).with_suffix('.png'),dpi=300)
    plt.close(fig)


def normalize_vector_fonts(source, destination):
    """Replace PDF text glyphs only; preserve paths, images and text origins."""
    import fitz
    doc=fitz.open(source)
    fonts={'regular':'times.ttf','bold':'timesbd.ttf','italic':'timesi.ttf','bolditalic':'timesbi.ttf'}
    for page in doc:
        lines=[line for block in page.get_text('dict')['blocks'] if 'lines' in block for line in block['lines']]
        spans=[dict(s,rotation=round(math.degrees(math.atan2(-line['dir'][1],line['dir'][0])))%360)
               for line in lines for s in line['spans']]
        if all('Times' in s['font'] for s in spans):continue
        # Reinsert all spans so overlapping mathematical superscripts survive redaction.
        for s in spans:
            page.add_redact_annot(fitz.Rect(s['bbox']),fill=False)
        page.apply_redactions(images=0,graphics=0)
        inserted=set()
        for s in spans:
            text=unicodedata.normalize('NFKC',s['text']).replace('\u2217','*')
            bold=bool(s['flags'] & 16) or any('BOLD' in unicodedata.name(c,'') for c in s['text'])
            italic=bool(s['flags'] & 2) or 'Math' in s['font'] or 'Italic' in s['font']
            style=('bold' if bold else '')+('italic' if italic else '') or 'regular'
            if style not in inserted:
                page.insert_font(fontname='TNR'+style,fontfile='external/Windows/Fonts/'+fonts[style])
                inserted.add(style)
            font=fitz.Font(fontfile='external/Windows/Fonts/'+fonts[style])
            assert all(c.isspace() or font.has_glyph(ord(c)) for c in text),(text,style)
            width=font.text_length(text,fontsize=s['size'])
            available=(s['bbox'][3]-s['bbox'][1] if s['rotation'] in (90,270)
                       else s['bbox'][2]-s['bbox'][0])
            size=s['size']*min(1,available/width) if width else s['size']
            color=tuple(((s['color']>>shift)&255)/255 for shift in (16,8,0))
            page.insert_text(s['origin'],text,fontname='TNR'+style,fontsize=size,color=color,rotate=s['rotation'])
    doc.subset_fonts()
    doc.save(destination,garbage=4,deflate=True)
    doc.close()
