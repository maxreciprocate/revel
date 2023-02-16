from blob import *
import freetype

def renderchar(fontpath: str, char: str) -> np.ndarray:
    face = freetype.Face(fontpath)
    face.set_char_size(32*32)
    face.load_char(char, freetype.FT_LOAD_RENDER | freetype.FT_LOAD_TARGET_MONO)
    bitmap = face.glyph.bitmap

    h, w = bitmap.rows, bitmap.width
    data = zeros(h * w, np.ubyte)

    for y in range(h):
        for ind in range(bitmap.pitch):
            value = bitmap.buffer[y * bitmap.pitch + ind]

            rowstart = y * w + ind * 8
            endbitind = ind * 8

            for bitind in range(min(8, w - endbitind)):
                bit = value & (1 << (7 - bitind))

                data[rowstart + bitind] = 1 if bit else 0

    glyph = data.reshape(h, w)
    return glyph

def starshow(glyph: np.ndarray) -> None:
    h, w = glyph.shape
    out = ""
    for ind, x in enumerate(glyph.ravel()):
        out += "*" if x else " "
        if not((ind + 1) % w):
            out += "\n"

    print(out)

def renderalphabet(fontpath: str, canvassize: tuple) -> np.ndarray:
    "renders lowercase letters and fits to common canvas size"
    maxh, maxw = 0, 0
    glyphs = []

    for char in range(ord('a'), ord('z')+1):
        glyph = renderchar(fontpath, chr(char))
        h, w = glyph.shape
        maxh = max(maxh, h)
        maxw = max(maxw, w)

        glyphs.append(glyph)

    for ind, glyph in enumerate(glyphs):
        canvas = zeros((maxh, maxw), np.uint8)
        canvas[:glyph.shape[0], :glyph.shape[1]] = glyph
        glyphs[ind] = canvas

    alfbet = zeros((len(glyphs), *canvassize), np.uint8)
    alfbet[:, :maxh, :maxw] = array(glyphs)

    return alfbet

if __name__ == '__main__':
    alfbet = renderalphabet('geo/Geo-Regular.ttf', (13, 13))
    starshow(alfbet[6])
    iimshow(alfbet[6])
