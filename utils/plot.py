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

    # fig.colorbar(im, ax=ax)
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


if __name__ == '__main__':
    # guided alignment test
    import numpy as np
    max_N = 90
    max_T = 200
    g = 0.2
    W = np.zeros((max_N, max_T), dtype=np.float32)
    for n_pos in range(W.shape[0]):
        for t_pos in range(W.shape[1]):
            W[n_pos, t_pos] = 1 - np.exp(-(t_pos / float(max_T) - n_pos / float(max_N)) ** 2 / (2 * g * g))
    # plot(W, None, None, False)

    alignment = np.zeros((max_N, max_T), dtype=np.float32)
    for n_pos in range(alignment.shape[0]):
        for t_pos in range(alignment.shape[1]):
            alignment[n_pos, t_pos] = 1 / (1 + abs(n_pos * (max_T / max_N) - t_pos))
    # plot(alignment, None, None, False)

    attention = alignment * W
    # plot(attention, None, None, False)

    plt.subplot(1, 3, 1), plt.imshow(W, origin='lower'), plt.title('weight'), plt.xticks([]), plt.yticks([])
    plt.subplot(1, 3, 2), plt.imshow(alignment, origin='lower'), plt.title('alignment'), plt.xticks([]), plt.yticks([])
    plt.subplot(1, 3, 3), plt.imshow(attention, origin='lower'), plt.title('attention'), plt.xticks([]), plt.yticks([])
    plt.show()
