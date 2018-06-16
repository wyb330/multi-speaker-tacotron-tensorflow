import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from matplotlib.pyplot import rc
from jamo import h2j, j2hcj
from text import PAD, EOS
from text.korean import normalize

FONT_NAME = "NanumBarunGothic"


def check_font():
    flist = font_manager.findSystemFonts()
    names = [font_manager.FontProperties(fname=fname).get_name() for fname in flist]
    if not (FONT_NAME in names):
        font_manager._rebuild()


check_font()
rc('font', family=FONT_NAME)


def plot(alignment, info, text, isKorean=True):
    char_len, audio_len = alignment.shape  # 145, 200

    fig, ax = plt.subplots(figsize=(char_len / 5, 5))
    im = ax.imshow(
        alignment.T,
        aspect='auto',
        origin='lower',
        interpolation='none')

    xlabel = 'Encoder timestep'
    ylabel = 'Decoder timestep'

    if info is not None:
        xlabel += '\n{}'.format(info)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if text:
        if isKorean:
            jamo_text = j2hcj(h2j(normalize(text)))
        else:
            jamo_text = text
        pad = [PAD] * (char_len - len(jamo_text) - 1)

        plt.xticks(range(char_len),
                   [tok for tok in jamo_text] + [EOS] + pad)

    if text is not None:
        while True:
            if text[-1] in [EOS, PAD]:
                text = text[:-1]
            else:
                break
        plt.title(text)

    plt.tight_layout()


def plot_alignment(
        alignment, path, info=None, text=None, isKorean=True):
    if text:
        tmp_alignment = alignment[:len(h2j(text)) + 2]

        plot(tmp_alignment, info, text, isKorean)
        plt.savefig(path, format='png')
    else:
        plot(alignment, info, text, isKorean)
        plt.savefig(path, format='png')

    print(" [*] Plot saved: {}".format(path))

